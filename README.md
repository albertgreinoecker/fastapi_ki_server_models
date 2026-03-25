# SHAP-E FastAPI Server

This project provides a small FastAPI service that generates 3D meshes from either:

- a text prompt
- an uploaded image

The server uses OpenAI's SHAP-E models and exposes HTTP endpoints for generation, download, health checks, and cleanup of generated files.

## What The Server Does

`main.py` starts a FastAPI application that loads three SHAP-E components on startup:

- `transmitter`
- `text300M`
- `image300M`

The service automatically selects `cuda` when available, otherwise it runs on `cpu`.

Generated meshes are written to `outputs/` and are kept for `3600` seconds by default. After that, old files are removed by the background cleanup task.

## Endpoints

### `GET /health`

Returns basic runtime information:

- service status
- selected device
- whether CUDA is available

### `POST /generate_shape/text`

Generates a 3D mesh from a JSON request body.

Example request body:

```json
{
  "prompt": "a futuristic helmet",
  "guidance_scale": 15.0,
  "num_steps": 64,
  "output_format": "ply"
}
```

Fields:

- `prompt`: Text description of the object
- `guidance_scale`: Higher values usually follow the prompt more closely
- `num_steps`: More steps usually improve quality but take longer
- `output_format`: `ply` or `obj`

### `POST /generate_shape/image`

Generates a 3D mesh from an uploaded image using `multipart/form-data`.

Query parameters:

- `guidance_scale`
- `num_steps`
- `output_format`

Uploaded file field:

- `file`

Image input works best when the object is centered and clearly visible, ideally on a white or transparent background.

### `GET /download/{filename}`

Downloads a generated mesh file from `outputs/`.

### `DELETE /download/{filename}`

Deletes a generated mesh file manually.

## Installation

Install the Python dependencies from the provided requirements file:

```bash
pip install -r ____requirements.txt
pip install git+https://github.com/openai/shap-e.git
```

## Running The Server

Start the FastAPI app with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 2222
```

Once running, the API will be available at:

- `http://localhost:2222`
- Swagger UI: `http://localhost:2222/docs`

## Example 1: Text Generation

`shape_text_sample_call.py` is a minimal client for the text endpoint.

What it does:

- sends a JSON `POST` request to `/generate_shape/text`
- passes a prompt and `num_steps`
- prints the HTTP status code
- prints the JSON response as text

Current example payload:

```json
{
  "prompt": "man with banana in the hand",
  "num_steps": 64
}
```

The server response includes:

- `job_id`
- `download_url`
- `format`
- `duration_seconds`
- `prompt`

You can use the returned `download_url` to fetch the generated `.ply` or `.obj` file.

## Example 2: Image Generation

`shape_image_sample_call.py` is a minimal client for the image endpoint.

What it does:

- opens a local image file
- uploads it to `/generate_shape/image`
- sends generation settings as query parameters
- prints the returned JSON response

Current example settings:

- `guidance_scale=3.0`
- `num_steps=64`
- `output_format=ply`

The script expects the image file to exist at:

```text
/home/albert/HTL/htl-logo_rauten.jpg
```

If you want to reuse the script on another machine, update that path and the server URL.

## Output Files

Generated files are stored in:

```text
outputs/
```

Each file is named with a generated job ID, for example:

```text
outputs/3cfc116d938f40e2b0695ce752cbf8cb.ply
```

## Notes

- The sample scripts currently point to `http://10.10.11.11:2222`
- `output_format` supports `ply` and `obj`
- Model loading happens when the server starts, so first startup can take a while
