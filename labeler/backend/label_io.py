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


def list_labeled_files(folder: str = "") -> set:
    """Return set of filenames that have labels."""
    d = _folder_dir(folder)
    if not os.path.isdir(d):
        return set()
    return {f for f in os.listdir(d) if f.endswith(".png")}


def load_label(filename: str, folder: str = "") -> np.ndarray | None:
    """Load a label mask. Returns None if not found."""
    path = os.path.join(_folder_dir(folder), filename)
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask


def save_label(filename: str, mask: np.ndarray, folder: str = ""):
    """Save a label mask as uint8 grayscale PNG."""
    d = _folder_dir(folder)
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
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

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
            lines.append(f"{int(cls)} {coords_str}")

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
