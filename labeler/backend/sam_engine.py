import os
import numpy as np
import cv2
import torch
import base64
import io
from PIL import Image

from .config import CHECKPOINT_PATH, EMBEDDINGS_DIR, TILES_DIR


class SAMEngine:
    def __init__(self):
        self.model = None
        self.predictor = None
        self._loaded = False

    def load(self):
        """Load MobileSAM model."""
        if self._loaded:
            return

        from mobile_sam import sam_model_registry, SamPredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT_PATH)
        sam.to(device)
        sam.eval()

        self.predictor = SamPredictor(sam)
        self._loaded = True
        print(f"MobileSAM loaded on {device}")

    def _tiles_path(self, filename: str, folder: str = "") -> str:
        if folder:
            return os.path.join(TILES_DIR, folder, filename)
        return os.path.join(TILES_DIR, filename)

    def _embedding_path(self, filename: str, folder: str = "") -> str:
        if folder:
            d = os.path.join(EMBEDDINGS_DIR, folder)
        else:
            d = EMBEDDINGS_DIR
        os.makedirs(d, exist_ok=True)
        name = os.path.splitext(filename)[0]
        return os.path.join(d, f"{name}.npy")

    def embed(self, filename: str, folder: str = "") -> bool:
        """Compute and cache embedding for a tile. Returns True if newly computed."""
        self.load()
        cache_path = self._embedding_path(filename, folder)
        if os.path.exists(cache_path):
            return False

        img_path = self._tiles_path(filename, folder)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)
        embedding = self.predictor.get_image_embedding().cpu().numpy()
        np.save(cache_path, embedding)
        return True

    def predict(self, filename: str, points=None, point_labels=None, box=None, folder: str = ""):
        """Run SAM prediction with cached embedding."""
        self.load()

        img_path = self._tiles_path(filename, folder)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cache_path = self._embedding_path(filename, folder)
        if os.path.exists(cache_path):
            self.predictor.set_image(image)
            cached = np.load(cache_path)
            self.predictor.features = torch.from_numpy(cached).to(self.predictor.device)
        else:
            self.predictor.set_image(image)
            embedding = self.predictor.get_image_embedding().cpu().numpy()
            np.save(cache_path, embedding)

        point_coords = None
        point_lab = None
        box_np = None

        if points and point_labels:
            point_coords = np.array(points, dtype=np.float32)
            point_lab = np.array(point_labels, dtype=np.int32)

        if box:
            box_np = np.array(box, dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_lab,
            box=box_np,
            multimask_output=True,
        )

        results = []
        for mask, score in zip(masks, scores):
            binary = (mask.astype(np.uint8)) * 255
            img = Image.fromarray(binary, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            results.append({"mask": b64, "score": float(score)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def precompute_all(self, folder: str = ""):
        """Precompute embeddings for all tiles in a folder."""
        self.load()
        if folder:
            tiles_dir = os.path.join(TILES_DIR, folder)
        else:
            tiles_dir = TILES_DIR
        tile_files = sorted([
            f for f in os.listdir(tiles_dir)
            if f.endswith(".png") and "nir" not in f
        ])
        total = len(tile_files)
        computed = 0
        for f in tile_files:
            if self.embed(f, folder):
                computed += 1
        return {"total": total, "computed": computed}


# Singleton
sam_engine = SAMEngine()
