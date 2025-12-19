import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from libs.lut_api_final import apply_luts_as_data_urls, find_luts  # noqa: E402

LUT_DIR = BASE_DIR / "ml_luts"
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
MAX_PREVIEW_LUTS = 120
MAX_PREVIEW_SIZE = 1280

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("lut-api")


class SelectionItem(BaseModel):
    label: str
    path: str


class ExportRequest(BaseModel):
    image: str
    selection: List[Union[str, SelectionItem]]


class Variant(BaseModel):
    label: str
    path: str
    data_url: str


class PreviewResponse(BaseModel):
    image: str
    variants: List[Variant]


app = FastAPI(title="LUT Mock API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---
def list_lut_labels(limit: int | None = None) -> list[str]:
    luts = find_luts(LUT_DIR, limit=limit)
    return [p.name for p in luts]


def _load_image_from_path(path_str: str) -> Image.Image:
    path = (BASE_DIR / path_str).resolve() if not Path(path_str).is_absolute() else Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _read_image(upload: UploadFile | None, image_path: str | None) -> tuple[Image.Image, str]:
    if upload:
        image = Image.open(upload.file).convert("RGB")
        name = upload.filename or "uploaded_image"
        return image, name
    if image_path:
        image = _load_image_from_path(image_path)
        return image, Path(image_path).name
    raise HTTPException(status_code=400, detail="Provide image_file or image_path")


def _select_luts(limit: Optional[int]) -> list[Path]:
    luts = find_luts(LUT_DIR, limit=limit or MAX_PREVIEW_LUTS)
    if not luts:
        raise HTTPException(status_code=404, detail="No LUT files found in LUT_DIR")
    return luts


def _normalize_selection_item(raw: Union[str, SelectionItem, dict]) -> dict:
    """Ensure each selection entry has label/path; fix messy string inputs."""
    label = ""
    path = ""

    if isinstance(raw, SelectionItem):
        label = raw.label
        path = raw.path
    elif isinstance(raw, str):
        label = raw
    elif isinstance(raw, dict):
        label = str(raw.get("label", "")).strip()
        path = str(raw.get("path", "")).strip()
    else:
        label = str(raw)

    # Handle serialized repr like "label='foo' path='/abs/bar.cube'"
    if "path=" in label and "label=" in label:
        match_label = re.search(r"label='([^']+)'", label)
        match_path = re.search(r"path='([^']+)'", label)
        if match_label:
            label = match_label.group(1)
        if match_path:
            path = match_path.group(1)

    if not label and path:
        label = Path(path).name
    # Trim to filename for readability
    simple_label = Path(label).name if label else label

    if not path and label:
        candidate = LUT_DIR / label
        path = str(candidate) if candidate.exists() else ""

    return {"label": simple_label, "path": path}


# --- Routes ---
@app.get("/")
def health() -> dict:
    return {"health": "100% ok"}

@app.get("/luts")
def get_luts() -> dict:
    labels = list_lut_labels()
    return {"labels": labels, "count": len(labels)}


@app.post("/preview", response_model=PreviewResponse)
async def preview(
    image_file: UploadFile | None = File(default=None),
    image_path: str | None = Form(default=None),
    limit: int | None = Form(default=None),
):
    try:
        image, image_name = _read_image(image_file, image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Unable to read image: {exc}") from exc

    luts = _select_luts(limit)
    logger.info(
        "Starting preview generation: image=%s, luts=%d, limit=%s",
        image_name,
        len(luts),
        limit,
    )

    variants_payload = apply_luts_as_data_urls(
        image=image,
        lut_paths=luts,
        max_size=MAX_PREVIEW_SIZE,
    )
    variants = [
        Variant(label=item["lut"], path=str(lut_path), data_url=item["data_url"])
        for lut_path, item in zip(luts, variants_payload)
    ]

    logger.info("Finished preview generation for %s with %d variants", image_name, len(variants))
    return PreviewResponse(image=image_name, variants=variants)


@app.post("/export")
def export(req: ExportRequest) -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Normalize selection to list of dicts with label/path
    normalized: list[dict] = [_normalize_selection_item(item) for item in req.selection]

    payload = {"image": req.image, "selection": normalized}
    export_path = EXPORT_DIR / "selection.json"
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Exported selection for %s to %s (%d labels)",
        req.image,
        export_path,
        len(normalized),
    )
    return {"written_to": str(export_path), "payload": payload}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
