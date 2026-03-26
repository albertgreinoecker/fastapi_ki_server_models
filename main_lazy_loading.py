"""
SHAP-E FastAPI Service
Generiert 3D-Modelle (PLY/OBJ) aus Text- oder Bildeingaben.
Modelle werden lazy geladen und nach 30 min Idle automatisch entladen.
"""

import io
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image

# ── Konfiguration ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FILE_TTL = 3600         # Ausgabedateien nach 1 h löschen
IDLE_TIMEOUT = 1800     # Modelle nach 30 min Idle entladen
WATCHDOG_INTERVAL = 60  # Watchdog prüft jede Minute


# ── Modell-Manager ────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.xm = None
        self.text_model = None
        self.image_model = None
        self.diffusion = None
        self._lock = threading.Lock()
        self._last_used: float = 0
        self._loaded: bool = False

    # ── interne Methoden (immer unter Lock aufrufen) ──────────────────────────

    def _load(self):
        print(f"[ModelManager] Lade Modelle auf {DEVICE} ...")
        self.xm = load_model("transmitter", device=DEVICE)
        self.text_model = load_model("text300M", device=DEVICE)
        self.image_model = load_model("image300M", device=DEVICE)
        self.diffusion = diffusion_from_config(load_config("diffusion"))
        self._loaded = True
        print("[ModelManager] Modelle bereit.")

    def _unload(self):
        print("[ModelManager] Entlade Modelle (Idle-Timeout) ...")
        self.xm = None
        self.text_model = None
        self.image_model = None
        self.diffusion = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ModelManager] Modelle entladen.")

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _watchdog_loop(self):
        while True:
            time.sleep(WATCHDOG_INTERVAL)
            with self._lock:
                if self._loaded and (time.time() - self._last_used) > IDLE_TIMEOUT:
                    self._unload()

    def start_watchdog(self):
        t = threading.Thread(target=self._watchdog_loop, daemon=True)
        t.start()
        print("[ModelManager] Watchdog gestartet.")

    # ── öffentliche API ───────────────────────────────────────────────────────

    def get(self):
        """Gibt (xm, text_model, image_model, diffusion) zurück.
        Lädt die Modelle bei Bedarf (Lazy Load)."""
        with self._lock:
            if not self._loaded:
                self._load()
            self._last_used = time.time()
            return self.xm, self.text_model, self.image_model, self.diffusion

    @property
    def status(self) -> dict:
        with self._lock:
            idle = round(time.time() - self._last_used, 1) if self._loaded else None
            return {
                "models_loaded": self._loaded,
                "device": str(DEVICE),
                "cuda_available": torch.cuda.is_available(),
                "idle_seconds": idle,
                "unload_in_seconds": round(IDLE_TIMEOUT - idle, 1) if idle is not None else None,
            }


models = ModelManager()


# ── Lifespan (kein Vorladen mehr) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.start_watchdog()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHAP-E 3D Generation Service",
    description=(
        "Generiert 3D-Modelle aus Text oder Bildern mittels OpenAIs SHAP-E.\n\n"
        "Modelle werden beim ersten Request geladen und nach 30 min Idle automatisch entladen."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    prompt: str = Field(..., example="a futuristic helmet")
    guidance_scale: float = Field(15.0, ge=1.0, le=30.0)
    num_steps: int = Field(64, ge=16, le=256)
    output_format: str = Field("ply", pattern="^(ply|obj)$")


class GenerationResponse(BaseModel):
    job_id: str
    download_url: str
    format: str
    duration_seconds: float
    prompt: Optional[str] = None


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def latents_to_mesh(xm, latents, output_format: str) -> bytes:
    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    buf = io.BytesIO()
    if output_format == "ply":
        mesh.write_ply(buf)
    else:
        mesh.write_obj(buf)
    return buf.getvalue()


def build_download_url(request: Request, filename: str) -> str:
    return str(request.base_url) + f"download/{filename}"


def cleanup_old_files():
    now = time.time()
    for f in OUTPUT_DIR.glob("*"):
        if now - f.stat().st_mtime > FILE_TTL:
            f.unlink(missing_ok=True)


def sample(model, diffusion, model_kwargs, guidance_scale, num_steps):
    return sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        progress=False,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=num_steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )


# ── Endpunkte ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
def health():
    """Schneller Liveness-Check (lädt keine Modelle)."""
    return {"status": "ok"}


@app.get("/status", tags=["Infra"])
def status():
    """Gibt den aktuellen Modell-Ladestatus und Idle-Zeit zurück."""
    return models.status


@app.post("/generate_shape/text", response_model=GenerationResponse, tags=["Generierung"])
def generate_from_text(req: TextRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Generiert ein 3D-Modell aus einem Textprompt.

    Beim ersten Aufruf (oder nach Idle-Timeout) werden die Modelle
    zuerst geladen (~30–60 s), danach sind sie warm für folgende Requests.
    """
    t0 = time.time()
    xm, text_model, _, diffusion = models.get()

    try:
        with torch.no_grad():
            latents = sample(text_model, diffusion, {"texts": [req.prompt]},
                             req.guidance_scale, req.num_steps)
        mesh_bytes = latents_to_mesh(xm, latents, req.output_format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generierung fehlgeschlagen: {e}")

    job_id = uuid.uuid4().hex
    filename = f"{job_id}.{req.output_format}"
    (OUTPUT_DIR / filename).write_bytes(mesh_bytes)
    background_tasks.add_task(cleanup_old_files)

    return GenerationResponse(
        job_id=job_id,
        download_url=build_download_url(request, filename),
        format=req.output_format,
        duration_seconds=round(time.time() - t0, 2),
        prompt=req.prompt,
    )


@app.post("/generate_shape/image", response_model=GenerationResponse, tags=["Generierung"])
async def generate_from_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PNG/JPEG, weißer Hintergrund empfohlen"),
    guidance_scale: float = 3.0,
    num_steps: int = 64,
    output_format: str = "ply",
):
    """
    Generiert ein 3D-Modell aus einem hochgeladenen Bild.

    Tipps: weißer/transparenter Hintergrund, Objekt zentriert, quadratisch (256x256).
    """
    if output_format not in ("ply", "obj"):
        raise HTTPException(status_code=422, detail="output_format muss 'ply' oder 'obj' sein.")

    t0 = time.time()
    img_bytes = await file.read()

    xm, _, image_model, diffusion = models.get()

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        image = load_image(tmp_path)
        os.unlink(tmp_path)

        with torch.no_grad():
            latents = sample(image_model, diffusion, {"images": [image]},
                             guidance_scale, num_steps)
        mesh_bytes = latents_to_mesh(xm, latents, output_format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generierung fehlgeschlagen: {e}")

    job_id = uuid.uuid4().hex
    filename = f"{job_id}.{output_format}"
    (OUTPUT_DIR / filename).write_bytes(mesh_bytes)
    background_tasks.add_task(cleanup_old_files)

    return GenerationResponse(
        job_id=job_id,
        download_url=build_download_url(request, filename),
        format=output_format,
        duration_seconds=round(time.time() - t0, 2),
    )


@app.get("/download/{filename}", tags=["Infra"])
def download_file(filename: str):
    """Lädt eine generierte 3D-Datei herunter (verfügbar für 1 Stunde)."""
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Ungültiger Dateiname.")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Datei nicht gefunden oder abgelaufen.")
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


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