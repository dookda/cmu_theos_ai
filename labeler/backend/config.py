import os
import yaml

TILES_DIR = os.environ.get("TILES_DIR", "../data/tiles")
LABELS_DIR = os.environ.get("LABELS_DIR", "../data/labels")
EMBEDDINGS_DIR = os.environ.get("EMBEDDINGS_DIR", "./embeddings")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "../configs/config.yaml")
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "/app/checkpoints/mobile_sam.pt")

TILE_SIZE = 512


def load_classes():
    """Load class definitions from config.yaml."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        classes = []
        for i, cls in enumerate(config.get("classes", [])):
            classes.append({
                "index": i,
                "name": cls["name"],
                "color": cls["color"],
            })
        return classes
    except FileNotFoundError:
        return [
            {"index": 0, "name": "background", "color": [0, 0, 0]},
            {"index": 1, "name": "vegetation", "color": [0, 255, 0]},
            {"index": 2, "name": "water", "color": [0, 0, 255]},
            {"index": 3, "name": "urban", "color": [255, 0, 0]},
            {"index": 4, "name": "agriculture", "color": [255, 255, 0]},
            {"index": 5, "name": "bare_soil", "color": [139, 69, 19]},
            {"index": 6, "name": "road", "color": [128, 128, 128]},
        ]
