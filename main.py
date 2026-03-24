"""
SHAP-E FastAPI Service
Generiert 3D-Modelle (PLY/GLB) aus Text- oder Bildeingaben.
"""

import io
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import trimesh
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_mesh
from shap_e.util.image_util import load_image

# ── Konfiguration ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Maximale Aufbewahrungszeit für generierte Dateien (Sekunden)
FILE_TTL = 3600  # 1 Stunde

# ── App & Modell-Initialisierung ─────────────────────────────────────────────

app = FastAPI(
    title="SHAP-E 3D Generation Service",
    description="Generiert 3D-Modelle aus Text oder Bildern mittels OpenAIs SHAP-E.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelle werden beim Start einmalig geladen
print(f"Lade SHAP-E Modelle auf {DEVICE} ...")
xm = load_model("transmitter", device=DEVICE)
text_model = load_model("text300M", device=DEVICE)
image_model = load_model("image300M", device=DEVICE)
diffusion = diffusion_from_config(load_config("diffusion"))
print("Modelle bereit.")

# ── Schemas ───────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    prompt: str = Field(..., example="a futuristic helmet", description="Textbeschreibung des 3D-Objekts")
    guidance_scale: float = Field(15.0, ge=1.0, le=30.0, description="Stärke des Guidance (höher = mehr Prompt-Treue)")
    num_steps: int = Field(64, ge=16, le=256, description="Anzahl Diffusionsschritte (mehr = besser, langsamer)")
    output_format: str = Field("ply", pattern="^(ply|obj)$", description="Ausgabeformat: ply oder obj")


class ImageRequest(BaseModel):
    guidance_scale: float = Field(3.0, ge=1.0, le=15.0)
    num_steps: int = Field(64, ge=16, le=256)
    output_format: str = Field("ply", pattern="^(ply|obj)$")


class GenerationResponse(BaseModel):
    job_id: str
    download_url: str
    format: str
    duration_seconds: float
    prompt: Optional[str] = None


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def latents_to_mesh(latents, output_format: str) -> bytes:
    """Konvertiert SHAP-E-Latents in ein Mesh und gibt Bytes zurück."""
    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()

    buf = io.BytesIO()
    if output_format == "ply":
        mesh.write_ply(buf)
    else:  # obj
        mesh.write_obj(buf)
    return buf.getvalue()


def cleanup_old_files():
    """Löscht Ausgabedateien, die älter als FILE_TTL Sekunden sind."""
    now = time.time()
    for f in OUTPUT_DIR.glob("*"):
        if now - f.stat().st_mtime > FILE_TTL:
            f.unlink(missing_ok=True)


# ── Endpunkte ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
def health():
    """Gibt den Status des Services zurück."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/generate_shape/text", response_model=GenerationResponse, tags=["Generierung"])
def generate_from_text(req: TextRequest, background_tasks: BackgroundTasks):
    """
    Generiert ein 3D-Modell aus einem Textprompt.

    - **prompt**: Beschreibung des gewünschten Objekts (Englisch empfohlen)
    - **guidance_scale**: Höhere Werte = mehr Prompt-Treue, weniger Vielfalt
    - **num_steps**: Mehr Schritte = höhere Qualität, längere Laufzeit
    - **output_format**: `ply` (empfohlen) oder `obj`
    """
    t0 = time.time()
    try:
        with torch.no_grad():
            latents = sample_latents(
                batch_size=1,
                model=text_model,
                diffusion=diffusion,
                guidance_scale=req.guidance_scale,
                model_kwargs={"texts": [req.prompt]},
                progress=False,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=req.num_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
        mesh_bytes = latents_to_mesh(latents, req.output_format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generierung fehlgeschlagen: {e}")

    job_id = uuid.uuid4().hex
    out_path = OUTPUT_DIR / f"{job_id}.{req.output_format}"
    out_path.write_bytes(mesh_bytes)

    background_tasks.add_task(cleanup_old_files)

    return GenerationResponse(
        job_id=job_id,
        download_url=f"/download/{job_id}.{req.output_format}",
        format=req.output_format,
        duration_seconds=round(time.time() - t0, 2),
        prompt=req.prompt,
    )


@app.post("/generate_shape/image", response_model=GenerationResponse, tags=["Generierung"])
async def generate_from_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Bild des Objekts (PNG/JPEG, weißer Hintergrund empfohlen)"),
    guidance_scale: float = 3.0,
    num_steps: int = 64,
    output_format: str = "ply",
):
    """
    Generiert ein 3D-Modell aus einem hochgeladenen Bild.

    Tipps für beste Ergebnisse:
    - Weißer oder transparenter Hintergrund
    - Objekt zentriert & vollständig sichtbar
    - Quadratisches Format (z.B. 256×256)
    """
    if output_format not in ("ply", "obj"):
        raise HTTPException(status_code=422, detail="output_format muss 'ply' oder 'obj' sein.")

    t0 = time.time()
    img_bytes = await file.read()

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        image = load_image(tmp_path)
        os.unlink(tmp_path)

        with torch.no_grad():
            latents = sample_latents(
                batch_size=1,
                model=image_model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs={"images": [image]},
                progress=False,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=num_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
        mesh_bytes = latents_to_mesh(latents, output_format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generierung fehlgeschlagen: {e}")

    job_id = uuid.uuid4().hex
    out_path = OUTPUT_DIR / f"{job_id}.{output_format}"
    out_path.write_bytes(mesh_bytes)

    background_tasks.add_task(cleanup_old_files)

    return GenerationResponse(
        job_id=job_id,
        download_url=f"/download/{job_id}.{output_format}",
        format=output_format,
        duration_seconds=round(time.time() - t0, 2),
    )


@app.get("/download/{filename}", tags=["Infra"])
def download_file(filename: str):
    """Lädt eine generierte 3D-Datei herunter."""
    # Sicherheitscheck: kein Path-Traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Ungültiger Dateiname.")

    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Datei nicht gefunden oder abgelaufen.")

    media_type = "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.delete("/download/{filename}", tags=["Infra"])
def delete_file(filename: str):
    """Löscht eine generierte Datei manuell."""
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Ungültiger Dateiname.")

    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Datei nicht gefunden.")
    path.unlink()
    return {"deleted": filename}
