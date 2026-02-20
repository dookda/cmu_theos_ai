import os
import base64
import io
import numpy as np
import cv2
from PIL import Image

from .config import LABELS_DIR, TILE_SIZE


def _folder_dir(folder: str = "") -> str:
    """Get the labels directory for a given folder."""
    if folder:
        return os.path.join(LABELS_DIR, folder)
    return LABELS_DIR


def _semantic_dir(folder: str = "") -> str:
    """Get the semantic mask subdirectory for a given folder."""
    return os.path.join(_folder_dir(folder), "semantic")


def list_labeled_files(folder: str = "") -> set:
    """Return set of filenames that have semantic labels."""
    d = _semantic_dir(folder)
    if not os.path.isdir(d):
        return set()
    return {f for f in os.listdir(d) if f.endswith(".png")}


def load_label(filename: str, folder: str = "") -> np.ndarray | None:
    """Load a label mask. Returns None if not found."""
    path = os.path.join(_semantic_dir(folder), filename)
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask


def save_label(filename: str, mask: np.ndarray, folder: str = ""):
    """Save a label mask as uint8 grayscale PNG."""
    d = _semantic_dir(folder)
    os.makedirs(d, exist_ok=True)
    assert mask.shape == (TILE_SIZE, TILE_SIZE), f"Expected ({TILE_SIZE},{TILE_SIZE}), got {mask.shape}"
    assert mask.dtype == np.uint8
    path = os.path.join(d, filename)
    cv2.imwrite(path, mask)


def mask_to_base64(mask: np.ndarray) -> str:
    """Encode a grayscale mask to base64 PNG."""
    img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_mask(b64: str) -> np.ndarray:
    """Decode a base64 PNG to a grayscale numpy array."""
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img, dtype=np.uint8)


def mask_to_yolo_detect(mask: np.ndarray) -> list[str]:
    """Convert a semantic mask to YOLOv8 detection format (class cx cy w h, normalized)."""
    h, w = mask.shape
    lines = []
    unique_classes = np.unique(mask)

    for cls in unique_classes:
        if cls == 0:
            continue
        binary = (mask == cls).astype(np.uint8)
        num_labels, labels_map = cv2.connectedComponents(binary)

        for label_id in range(1, num_labels):
            component = (labels_map == label_id).astype(np.uint8)
            coords = cv2.findNonZero(component)
            if coords is None:
                continue
            x1, y1, bw, bh = cv2.boundingRect(coords)
            cx = (x1 + bw / 2) / w
            cy = (y1 + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{int(cls) - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def mask_to_yolo_segment(mask: np.ndarray) -> list[str]:
    """Convert a semantic mask to YOLOv8 segmentation format (class x1 y1 x2 y2 ... xn yn, normalized)."""
    h, w = mask.shape
    lines = []
    unique_classes = np.unique(mask)

    for cls in unique_classes:
        if cls == 0:
            continue
        binary = (mask == cls).astype(np.uint8)
        num_labels, labels_map = cv2.connectedComponents(binary)

        for label_id in range(1, num_labels):
            component = (labels_map == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            # Use the largest contour
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            # Simplify polygon
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                approx = contour
            points = approx.reshape(-1, 2)
            coords_str = " ".join(f"{px / w:.6f} {py / h:.6f}" for px, py in points)
            lines.append(f"{int(cls) - 1} {coords_str}")

    return lines


def save_yolo_detect(filename: str, lines: list[str], folder: str = ""):
    """Save YOLO detection labels."""
    d = os.path.join(_folder_dir(folder), "yolo_detect")
    os.makedirs(d, exist_ok=True)
    txt_name = os.path.splitext(filename)[0] + ".txt"
    path = os.path.join(d, txt_name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def save_yolo_segment(filename: str, lines: list[str], folder: str = ""):
    """Save YOLO segmentation labels."""
    d = os.path.join(_folder_dir(folder), "yolo_segment")
    os.makedirs(d, exist_ok=True)
    txt_name = os.path.splitext(filename)[0] + ".txt"
    path = os.path.join(d, txt_name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def run_kmeans(path: str, n_clusters: int, nir_path: str | None = None) -> dict:
    """Run K-means clustering on a tile image (RGB or RGB+NIR).

    Returns cluster mask as base64 PNG and representative RGB colors per cluster.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = img_rgb.shape[:2]

    if nir_path and os.path.exists(nir_path):
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        nir = nir[:h, :w]  # ensure same size
        channels = np.concatenate([img_rgb, nir[:, :, np.newaxis]], axis=2)
    else:
        channels = img_rgb

    pixels = channels.reshape(-1, channels.shape[2])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    cluster_mask = labels.reshape(h, w).astype(np.uint8)
    mask_img = Image.fromarray(cluster_mask, mode='L')
    buf = io.BytesIO()
    mask_img.save(buf, format='PNG')
    mask_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Representative RGB colors (first 3 channels of centers)
    rgb_centers = centers[:, :3].astype(int).clip(0, 255).tolist()

    return {
        "mask": mask_b64,
        "n_clusters": n_clusters,
        "centers": rgb_centers,
    }


def delete_tile_files(filename: str, folder: str, tiles_dir: str, embeddings_dir: str) -> dict:
    """Delete a tile and all its associated files (NIR, label, YOLO, embedding)."""
    stem = os.path.splitext(filename)[0]
    folder_suffix = folder if folder else ""

    candidates = [
        # Tile image
        os.path.join(tiles_dir, filename),
        # NIR band
        os.path.join(tiles_dir, f"{stem}_nir.png"),
        # Semantic label
        os.path.join(_semantic_dir(folder_suffix), filename),
        # YOLO detect label
        os.path.join(_folder_dir(folder_suffix), "yolo_detect", f"{stem}.txt"),
        # YOLO segment label
        os.path.join(_folder_dir(folder_suffix), "yolo_segment", f"{stem}.txt"),
        # SAM embedding
        os.path.join(embeddings_dir, folder_suffix, f"{stem}.npy") if folder_suffix
        else os.path.join(embeddings_dir, f"{stem}.npy"),
    ]

    deleted = []
    for path in candidates:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)

    return {"deleted": deleted}
