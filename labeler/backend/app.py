import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import TILES_DIR, LABELS_DIR, EMBEDDINGS_DIR, TILE_SIZE, load_classes, save_classes
from .label_io import (
    load_label, save_label, mask_to_base64, base64_to_mask, list_labeled_files,
    mask_to_yolo_detect, mask_to_yolo_segment, save_yolo_detect, save_yolo_segment,
    delete_tile_files, run_kmeans, augment_tile, mask_to_coco_annotations,
    get_class_stats, VALID_TRANSFORMS,
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

_export_formats: set[str] = {"semantic", "detect", "segment"}
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


class SplitRequest(BaseModel):
    train_ratio: float   # 0.0–1.0
    val_ratio: float     # 0.0–1.0  (test = 1 - train - val)


class KMeansRequest(BaseModel):
    n_clusters: int


class AugmentRequest(BaseModel):
    transforms: list[str]   # subset of VALID_TRANSFORMS
    n_random: int = 3        # copies per random transform
    labeled_only: bool = True


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

@app.get("/api/splits")
def get_splits():
    """Return train/val/test split sets from splits.json in the current tiles dir."""
    import json
    splits_path = os.path.join(_tiles_dir(), "splits.json")
    if not os.path.exists(splits_path):
        return {"splits": None}
    try:
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        return {"splits": splits}
    except Exception:
        return {"splits": None}


@app.post("/api/splits")
def create_splits(req: SplitRequest):
    """Generate train/val/test splits from labeled tiles only.

    Only original (non-augmented) tiles are split randomly into train/val/test.
    Augmented tiles (_aug_*) are always assigned to train to prevent evaluation leakage.
    Unlabeled tiles are excluded entirely.
    """
    import json, random
    if req.train_ratio <= 0 or req.val_ratio <= 0 or req.train_ratio + req.val_ratio >= 1.0:
        raise HTTPException(400, "train_ratio and val_ratio must be positive and sum to less than 1.0")

    td = _tiles_dir()
    folder = _get_folder()
    labeled = list_labeled_files(folder)

    # Only include labeled tiles that exist on disk; skip NIR sidecars
    all_labeled = sorted([
        f for f in os.listdir(td)
        if f.endswith(".png") and "nir" not in f
        and os.path.isfile(os.path.join(td, f))
        and f in labeled
    ])
    if not all_labeled:
        raise HTTPException(400, "No labeled tiles found. Label some tiles before generating splits.")

    # Separate originals from augmented tiles
    originals = [f for f in all_labeled if "_aug_" not in f]
    aug_tiles  = [f for f in all_labeled if "_aug_" in f]

    if not originals:
        raise HTTPException(400, "No original labeled tiles found (only augmented tiles exist).")

    # Randomly split originals into train / val / test
    random.shuffle(originals)
    n = len(originals)
    n_train = max(1, round(n * req.train_ratio))
    n_val   = max(1, round(n * req.val_ratio))

    train_orig = originals[:n_train]
    val_tiles  = originals[n_train:n_train + n_val]
    test_tiles = originals[n_train + n_val:]

    # Augmented tiles always go to train (no val/test contamination)
    splits = {
        "train": sorted(train_orig + aug_tiles),
        "val":   sorted(val_tiles),
        "test":  sorted(test_tiles),
    }

    splits_path = os.path.join(td, "splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return {
        "status": "ok",
        "total": len(all_labeled),
        "train": len(splits["train"]),
        "val":   len(splits["val"]),
        "test":  len(splits["test"]),
        "splits": splits,
    }




@app.get("/api/tiles")
def get_tiles():
    td = _tiles_dir()
    if not os.path.isdir(td):
        return {"tiles": [], "total": 0, "labeled": 0, "labeled_files": []}
    tile_files = sorted([
        f for f in os.listdir(td)
        if f.endswith(".png") and "nir" not in f
    ])
    labeled = list_labeled_files(_get_folder())
    labeled_in_tiles = labeled & set(tile_files)
    return {
        "tiles": tile_files,
        "total": len(tile_files),
        "labeled": len(labeled_in_tiles),
        "labeled_files": sorted(labeled_in_tiles),
    }


@app.get("/api/tiles/{filename}")
def get_tile(filename: str):
    path = os.path.join(_tiles_dir(), filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Tile not found")
    return FileResponse(path, media_type="image/png")


@app.delete("/api/tiles/{filename}")
def delete_tile(filename: str):
    path = os.path.join(_tiles_dir(), filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Tile not found")
    try:
        result = delete_tile_files(filename, _get_folder(), _tiles_dir(), EMBEDDINGS_DIR)
    except OSError as e:
        raise HTTPException(500, f"Failed to delete: {e}")
    return {"status": "deleted", "filename": filename, "deleted": result["deleted"]}


# --- Upload endpoint ---

@app.post("/api/upload-tiles")
async def upload_tiles(files: list[UploadFile] = File(...)):
    """Upload one or more tile images (PNG/JPG) to the current tiles directory.

    All images are converted to PNG and resized to TILE_SIZE×TILE_SIZE.
    """
    import io as _io
    from PIL import Image as PILImage

    td = _tiles_dir()
    os.makedirs(td, exist_ok=True)
    allowed = {".png", ".jpg", ".jpeg"}
    uploaded, skipped = [], []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in allowed:
            skipped.append(f.filename)
            continue
        # Always save as .png so the tile listing can find it
        stem = os.path.splitext(os.path.basename(f.filename))[0]
        dest_name = f"{stem}.png"
        dest = os.path.join(td, dest_name)
        data = await f.read()
        try:
            img = PILImage.open(_io.BytesIO(data)).convert("RGB")
            if img.size != (TILE_SIZE, TILE_SIZE):
                img = img.resize((TILE_SIZE, TILE_SIZE), PILImage.LANCZOS)
            img.save(dest, format="PNG")
        except Exception as e:
            skipped.append(f.filename)
            continue
        uploaded.append(dest_name)
    return {"uploaded": uploaded, "skipped": skipped}


# --- Augmentation endpoint ---

def _clear_aug_tiles(folder: str, td: str) -> int:
    """Delete all _aug_ tiles and their associated files. Returns count of deleted tiles."""
    aug_tiles = [
        f for f in os.listdir(td)
        if f.endswith(".png") and "_aug_" in f and os.path.isfile(os.path.join(td, f))
    ] if os.path.isdir(td) else []
    deleted = 0
    for filename in aug_tiles:
        try:
            delete_tile_files(filename, folder, td, EMBEDDINGS_DIR)
            deleted += 1
        except Exception:
            continue
    return deleted


@app.post("/api/augment")
def augment_dataset(req: AugmentRequest):
    """Clear existing aug tiles then stream progress as SSE while augmenting."""
    import json as _json

    invalid = set(req.transforms) - VALID_TRANSFORMS
    if invalid:
        raise HTTPException(400, f"Unknown transforms: {sorted(invalid)}. Valid: {sorted(VALID_TRANSFORMS)}")
    if not req.transforms:
        raise HTTPException(400, "No transforms specified")

    folder = _get_folder()
    td = _tiles_dir()

    def generate():
        # Step 1 — clear old aug tiles
        _clear_aug_tiles(folder, td)

        labeled = list_labeled_files(folder)
        if req.labeled_only:
            candidates = sorted(labeled)
        else:
            candidates = sorted([
                f for f in os.listdir(td)
                if f.endswith(".png") and "nir" not in f and "_aug_" not in f
                and os.path.isfile(os.path.join(td, f))
            ])

        total = len(candidates)
        yield f"data: {_json.dumps({'type': 'start', 'total': total})}\n\n"

        all_created = []
        for i, filename in enumerate(candidates):
            mask = load_label(filename, folder)
            if mask is not None:
                try:
                    result = augment_tile(filename, mask, req.transforms, td, folder, req.n_random)
                    all_created.extend(result["created"])
                except Exception:
                    pass
            yield f"data: {_json.dumps({'type': 'progress', 'done': i + 1, 'total': total})}\n\n"

        yield f"data: {_json.dumps({'type': 'done', 'augmented': total, 'created': len(all_created)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/augment")
def clear_augmentation():
    """Delete all augmented tiles (_aug_*) and their associated files from the current folder."""
    folder = _get_folder()
    td = _tiles_dir()
    deleted_tiles = _clear_aug_tiles(folder, td)
    return {"status": "ok", "deleted_tiles": deleted_tiles, "deleted_files": deleted_tiles}


# --- Stats endpoint ---

@app.get("/api/stats")
def get_stats():
    """Return per-class pixel distribution across all labeled tiles in the current folder."""
    folder = _get_folder()
    td = _tiles_dir()
    classes = load_classes()

    raw = get_class_stats(folder)
    total_px = raw["total_pixels"]

    n_total = sum(
        1 for f in os.listdir(td)
        if f.endswith(".png") and "nir" not in f and os.path.isfile(os.path.join(td, f))
    ) if os.path.isdir(td) else 0

    result = []
    for c in classes:
        idx = c["index"]
        px = raw["class_pixels"].get(idx, 0)
        pct = round(px / total_px * 100, 1) if total_px > 0 else 0.0
        result.append({
            "index": idx,
            "name": c["name"],
            "color": c["color"],
            "pixels": px,
            "pct": pct,
        })

    return {
        "classes": result,
        "labeled": raw["labeled"],
        "total": n_total,
    }


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


@app.post("/api/reexport-yolo")
def reexport_yolo():
    """Re-export YOLO detect/segment labels for all existing semantic masks in the current folder."""
    folder = _get_folder()
    labeled = list_labeled_files(folder)
    count = 0
    for filename in sorted(labeled):
        mask = load_label(filename, folder)
        if mask is None:
            continue
        if "detect" in _export_formats:
            save_yolo_detect(filename, mask_to_yolo_detect(mask), folder)
        if "segment" in _export_formats:
            save_yolo_segment(filename, mask_to_yolo_segment(mask), folder)
        count += 1
    return {"status": "ok", "reexported": count, "formats": sorted(_export_formats)}


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


# --- K-Means endpoint ---

@app.post("/api/kmeans/{filename}")
def kmeans_segment(filename: str, req: KMeansRequest):
    if not 2 <= req.n_clusters <= 16:
        raise HTTPException(400, "n_clusters must be between 2 and 16")
    path = os.path.join(_tiles_dir(), filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Tile not found")
    stem = os.path.splitext(filename)[0]
    nir_path = os.path.join(_tiles_dir(), f"{stem}_nir.png")
    nir_path = nir_path if os.path.exists(nir_path) else None
    try:
        result = run_kmeans(path, req.n_clusters, nir_path)
    except Exception as e:
        raise HTTPException(500, f"K-Means failed: {e}")
    return result


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
    try:
        save_classes([{"index": i, **c} for i, c in enumerate(classes)])
    except OSError as e:
        raise HTTPException(500, f"Failed to save config: {e}")
    return {"classes": load_classes()}


@app.delete("/api/classes/{index}")
def delete_class(index: int):
    classes = load_classes()
    if index == 0:
        raise HTTPException(400, "Cannot delete background class")
    if index >= len(classes):
        raise HTTPException(404, "Class not found")
    classes.pop(index)
    for i, c in enumerate(classes):
        c["index"] = i
    try:
        save_classes(classes)
    except OSError as e:
        raise HTTPException(500, f"Failed to save config: {e}")
    return {"classes": load_classes()}


# --- Config endpoint ---

@app.get("/api/config")
def get_config():
    return {
        "classes": load_classes(),
        "tile_size": TILE_SIZE,
    }


# --- Export ZIP endpoint ---

_export_tmp: dict = {}   # format -> temp file path (single-user tool)


def _build_export_zip(format: str, td: str, splits: dict, folder: str,
                      label_base: str, label_classes: list, classes: list) -> str:
    """Build the export ZIP to a temp file and return its path."""
    import tempfile, io as _io, zipfile

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        if format == "coco":
            import json as _json
            categories = [{"id": c["index"], "name": c["name"]} for c in label_classes]
            for split_name in ("train", "val", "test"):
                coco_images, coco_anns = [], []
                ann_id = 1
                for img_id, filename in enumerate(splits.get(split_name, []), start=1):
                    img_path = os.path.join(td, filename)
                    if os.path.exists(img_path):
                        zf.write(img_path, f"{split_name}/images/{filename}")
                    coco_images.append({"id": img_id, "file_name": filename,
                                        "width": TILE_SIZE, "height": TILE_SIZE})
                    mask = load_label(filename, folder)
                    if mask is not None:
                        anns = mask_to_coco_annotations(mask, img_id, ann_id)
                        coco_anns.extend(anns)
                        ann_id += len(anns)
                zf.writestr(f"annotations/{split_name}.json", _json.dumps({
                    "info": {"description": "Exported from Dabeler", "version": "1.0"},
                    "categories": categories, "images": coco_images, "annotations": coco_anns,
                }, indent=2))
        else:
            for split_name in ("train", "val", "test"):
                for filename in splits.get(split_name, []):
                    stem = os.path.splitext(filename)[0]
                    img_path = os.path.join(td, filename)
                    if os.path.exists(img_path):
                        zf.write(img_path, f"{split_name}/images/{filename}")
                    if format == "semantic":
                        lbl = os.path.join(label_base, "semantic", filename)
                        if os.path.exists(lbl):
                            zf.write(lbl, f"{split_name}/masks/{filename}")
                    elif format == "yolo_detect":
                        mask = load_label(filename, folder)
                        if mask is not None:
                            lines = mask_to_yolo_detect(mask)
                            zf.writestr(f"{split_name}/labels/{stem}.txt", "\n".join(lines) + "\n" if lines else "")
                    elif format == "yolo_segment":
                        mask = load_label(filename, folder)
                        if mask is not None:
                            lines = mask_to_yolo_segment(mask)
                            zf.writestr(f"{split_name}/labels/{stem}.txt", "\n".join(lines) + "\n" if lines else "")
            if format in ("yolo_detect", "yolo_segment"):
                names = [c["name"] for c in label_classes]
                zf.writestr("data.yaml", "\n".join([
                    "train: ../train/images", "val:   ../val/images", "test:  ../test/images",
                    "", f"nc: {len(names)}", f"names: {names}",
                ]) + "\n")
    return tmp.name


@app.post("/api/export-zip")
def export_zip_stream(format: str = "yolo_detect"):
    """Stream export progress via SSE, build ZIP, store for download."""
    import json as _json

    valid_formats = {"semantic", "yolo_detect", "yolo_segment", "coco"}
    if format not in valid_formats:
        raise HTTPException(400, f"format must be one of {sorted(valid_formats)}")

    td = _tiles_dir()
    splits_path = os.path.join(td, "splits.json")
    if not os.path.exists(splits_path):
        raise HTTPException(400, "No splits found. Generate splits first.")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = _json.load(f)

    folder   = _get_folder()
    label_base = os.path.join(LABELS_DIR, folder) if folder else LABELS_DIR
    classes  = load_classes()
    label_classes = [c for c in classes if c["index"] != 0]

    all_files = [f for s in ("train", "val", "test") for f in splits.get(s, [])]
    total = len(all_files)

    def generate():
        yield f"data: {_json.dumps({'type': 'start', 'total': total})}\n\n"
        # Build ZIP (progress tracked per tile via generator wrapper)
        import tempfile, zipfile

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.close()
        done = 0

        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            if format == "coco":
                categories = [{"id": c["index"], "name": c["name"]} for c in label_classes]
                for split_name in ("train", "val", "test"):
                    coco_images, coco_anns, ann_id = [], [], 1
                    for img_id, filename in enumerate(splits.get(split_name, []), start=1):
                        img_path = os.path.join(td, filename)
                        if os.path.exists(img_path):
                            zf.write(img_path, f"{split_name}/images/{filename}")
                        coco_images.append({"id": img_id, "file_name": filename,
                                            "width": TILE_SIZE, "height": TILE_SIZE})
                        mask = load_label(filename, folder)
                        if mask is not None:
                            anns = mask_to_coco_annotations(mask, img_id, ann_id)
                            coco_anns.extend(anns); ann_id += len(anns)
                        done += 1
                        yield f"data: {_json.dumps({'type': 'progress', 'done': done, 'total': total})}\n\n"
                    zf.writestr(f"annotations/{split_name}.json", _json.dumps({
                        "info": {"description": "Exported from Dabeler", "version": "1.0"},
                        "categories": categories, "images": coco_images, "annotations": coco_anns,
                    }, indent=2))
            else:
                for split_name in ("train", "val", "test"):
                    for filename in splits.get(split_name, []):
                        stem = os.path.splitext(filename)[0]
                        img_path = os.path.join(td, filename)
                        if os.path.exists(img_path):
                            zf.write(img_path, f"{split_name}/images/{filename}")
                        if format == "semantic":
                            lbl = os.path.join(label_base, "semantic", filename)
                            if os.path.exists(lbl):
                                zf.write(lbl, f"{split_name}/masks/{filename}")
                        elif format == "yolo_detect":
                            mask = load_label(filename, folder)
                            if mask is not None:
                                lines = mask_to_yolo_detect(mask)
                                zf.writestr(f"{split_name}/labels/{stem}.txt",
                                            "\n".join(lines) + "\n" if lines else "")
                        elif format == "yolo_segment":
                            mask = load_label(filename, folder)
                            if mask is not None:
                                lines = mask_to_yolo_segment(mask)
                                zf.writestr(f"{split_name}/labels/{stem}.txt",
                                            "\n".join(lines) + "\n" if lines else "")
                        done += 1
                        yield f"data: {_json.dumps({'type': 'progress', 'done': done, 'total': total})}\n\n"
                if format in ("yolo_detect", "yolo_segment"):
                    names = [c["name"] for c in label_classes]
                    zf.writestr("data.yaml", "\n".join([
                        "train: ../train/images", "val:   ../val/images", "test:  ../test/images",
                        "", f"nc: {len(names)}", f"names: {names}",
                    ]) + "\n")

        _export_tmp[format] = tmp.name
        yield f"data: {_json.dumps({'type': 'done', 'format': format})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/export-zip/download")
def export_zip_download(format: str = "yolo_detect"):
    """Serve the previously built export ZIP file."""
    import tempfile
    path = _export_tmp.get(format)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Export not ready. Run export first.")
    return StreamingResponse(
        open(path, "rb"),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="dataset_{format}.zip"',
            "Content-Length": str(os.path.getsize(path)),
        },
    )


# --- Serve frontend ---

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
