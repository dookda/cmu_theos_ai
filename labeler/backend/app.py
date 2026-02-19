import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import TILES_DIR, LABELS_DIR, TILE_SIZE, load_classes, save_classes
from .label_io import (
    load_label, save_label, mask_to_base64, base64_to_mask, list_labeled_files,
    mask_to_yolo_detect, mask_to_yolo_segment, save_yolo_detect, save_yolo_segment,
)
from .sam_engine import sam_engine

app = FastAPI(title="Dabeler")


# --- Folder state ---

def _list_folders():
    """List available tile folders. Includes root ('') if PNGs exist directly in TILES_DIR."""
    if not os.path.isdir(TILES_DIR):
        return []

    folders = []

    # Include root if PNGs exist directly in TILES_DIR
    root_has_png = any(
        f.endswith(".png") for f in os.listdir(TILES_DIR)
        if os.path.isfile(os.path.join(TILES_DIR, f))
    )
    if root_has_png:
        folders.append("")

    # Also include subdirectories containing PNGs
    for entry in sorted(os.listdir(TILES_DIR)):
        subpath = os.path.join(TILES_DIR, entry)
        if os.path.isdir(subpath):
            has_png = any(f.endswith(".png") for f in os.listdir(subpath))
            if has_png:
                folders.append(entry)

    return folders


_current_folder: str | None = None


def _get_folder() -> str:
    global _current_folder
    if _current_folder is None:
        folders = _list_folders()
        _current_folder = folders[0] if folders else ""
    return _current_folder


def _tiles_dir() -> str:
    folder = _get_folder()
    if folder:
        return os.path.join(TILES_DIR, folder)
    return TILES_DIR


# --- Export format state ---

_export_formats: set[str] = {"semantic"}
_VALID_FORMATS = {"semantic", "detect", "segment"}


# --- Pydantic models ---

class SaveLabelRequest(BaseModel):
    mask: str  # base64-encoded PNG


class SAMPredictRequest(BaseModel):
    filename: str
    points: list[list[float]] | None = None
    point_labels: list[int] | None = None
    box: list[float] | None = None


class SelectFolderRequest(BaseModel):
    folder: str


class ExportFormatsRequest(BaseModel):
    formats: list[str]  # subset of ["semantic", "detect", "segment"]


class ClassDef(BaseModel):
    name: str
    color: list[int]  # [R, G, B]


class UpdateClassesRequest(BaseModel):
    classes: list[ClassDef]


# --- Folder endpoints ---

@app.get("/api/folders")
def get_folders():
    folders = _list_folders()
    return {
        "folders": folders,
        "current": _get_folder(),
    }


@app.post("/api/folders/select")
def select_folder(req: SelectFolderRequest):
    global _current_folder
    folders = _list_folders()
    if req.folder not in folders:
        raise HTTPException(400, f"Folder '{req.folder}' not found")
    _current_folder = req.folder
    return {"status": "ok", "current": _current_folder}


# --- Export format endpoints ---

@app.get("/api/export-formats")
def get_export_formats():
    return {"formats": sorted(_export_formats)}


@app.post("/api/export-formats")
def set_export_formats(req: ExportFormatsRequest):
    global _export_formats
    invalid = set(req.formats) - _VALID_FORMATS
    if invalid:
        raise HTTPException(400, f"Invalid formats: {invalid}")
    _export_formats = set(req.formats) if req.formats else {"semantic"}
    return {"formats": sorted(_export_formats)}


# --- Tile endpoints ---

@app.get("/api/tiles")
def get_tiles():
    td = _tiles_dir()
    if not os.path.isdir(td):
        return {"tiles": [], "total": 0, "labeled": 0}
    tile_files = sorted([
        f for f in os.listdir(td)
        if f.endswith(".png") and "nir" not in f
    ])
    labeled = list_labeled_files(_get_folder())
    return {
        "tiles": tile_files,
        "total": len(tile_files),
        "labeled": len(labeled & set(tile_files)),
    }


@app.get("/api/tiles/{filename}")
def get_tile(filename: str):
    path = os.path.join(_tiles_dir(), filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Tile not found")
    return FileResponse(path, media_type="image/png")


# --- Label endpoints ---

@app.get("/api/labels/{filename}")
def get_label(filename: str):
    mask = load_label(filename, _get_folder())
    if mask is None:
        return {"exists": False, "mask": None}
    return {"exists": True, "mask": mask_to_base64(mask)}


@app.post("/api/labels/{filename}")
def post_label(filename: str, req: SaveLabelRequest):
    mask = base64_to_mask(req.mask)
    if mask.shape != (TILE_SIZE, TILE_SIZE):
        raise HTTPException(400, f"Mask must be {TILE_SIZE}x{TILE_SIZE}")
    num_classes = len(load_classes())
    if mask.max() >= num_classes:
        raise HTTPException(400, f"Mask values must be 0-{num_classes - 1}")
    folder = _get_folder()

    # Always save the semantic mask (source of truth)
    save_label(filename, mask, folder)

    # Export YOLO formats if selected
    if "detect" in _export_formats:
        lines = mask_to_yolo_detect(mask)
        save_yolo_detect(filename, lines, folder)
    if "segment" in _export_formats:
        lines = mask_to_yolo_segment(mask)
        save_yolo_segment(filename, lines, folder)

    return {"status": "saved", "formats": sorted(_export_formats)}


@app.get("/api/progress")
def get_progress():
    td = _tiles_dir()
    if not os.path.isdir(td):
        return {"total": 0, "labeled": 0}
    tile_files = [
        f for f in os.listdir(td)
        if f.endswith(".png") and "nir" not in f
    ]
    labeled = list_labeled_files(_get_folder())
    return {
        "total": len(tile_files),
        "labeled": len(labeled & set(tile_files)),
    }


# --- SAM endpoints ---

@app.post("/api/sam/embed/{filename}")
def embed_tile(filename: str):
    path = os.path.join(_tiles_dir(), filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Tile not found")
    newly_computed = sam_engine.embed(filename, _get_folder())
    return {"status": "computed" if newly_computed else "cached"}


@app.post("/api/sam/predict")
def sam_predict(req: SAMPredictRequest):
    path = os.path.join(_tiles_dir(), req.filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Tile not found")

    results = sam_engine.predict(
        filename=req.filename,
        points=req.points,
        point_labels=req.point_labels,
        box=req.box,
        folder=_get_folder(),
    )
    return {"masks": results}


@app.post("/api/sam/precompute")
def precompute_embeddings(background_tasks: BackgroundTasks):
    folder = _get_folder()
    background_tasks.add_task(sam_engine.precompute_all, folder)
    return {"status": "started"}


# --- Class management endpoints ---

@app.post("/api/classes")
def update_classes(req: UpdateClassesRequest):
    if len(req.classes) < 1:
        raise HTTPException(400, "Must have at least 1 class (background)")
    if req.classes[0].name != "background":
        raise HTTPException(400, "First class must be 'background'")
    for cls in req.classes:
        if len(cls.color) != 3 or not all(0 <= c <= 255 for c in cls.color):
            raise HTTPException(400, f"Invalid color for class '{cls.name}'")

    classes = [{"name": c.name, "color": c.color} for c in req.classes]
    save_classes([{"index": i, **c} for i, c in enumerate(classes)])
    return {"classes": load_classes()}


@app.delete("/api/classes/{index}")
def delete_class(index: int):
    classes = load_classes()
    if index == 0:
        raise HTTPException(400, "Cannot delete background class")
    if index >= len(classes):
        raise HTTPException(404, "Class not found")
    classes.pop(index)
    # Re-index
    for i, c in enumerate(classes):
        c["index"] = i
    save_classes(classes)
    return {"classes": load_classes()}


# --- Config endpoint ---

@app.get("/api/config")
def get_config():
    return {
        "classes": load_classes(),
        "tile_size": TILE_SIZE,
    }


# --- Serve frontend ---

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
