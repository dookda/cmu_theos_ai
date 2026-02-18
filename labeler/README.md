# THEOS Labeler - SAM Annotation Tool

Web-based labeling tool with MobileSAM auto-segmentation for THEOS satellite tiles.

## Quick Start

```bash
cd labeler
docker-compose up --build
```

Open **http://localhost:8000**

## Features

- **SAM Auto-Segment**: Click or drag a box to auto-segment regions
- **Brush/Eraser**: Manual pixel-level editing
- **7 Classes**: background, vegetation, water, urban, agriculture, bare_soil, road
- **Undo**: Up to 20 levels
- **Keyboard Shortcuts**: 1-7 (class), S (save), Z (undo), Arrow keys (navigate)

## How to Label

1. Select a class from the left panel (or press 1-7)
2. **SAM mode (default)**: Click on a region to auto-segment, or drag a box
3. Press **Enter** to accept the SAM mask, **Escape** to reject
4. Right-click to add negative points (exclude areas)
5. Switch to **Brush** for manual corrections
6. Press **S** to save, then navigate to next tile

## Output

Labels are saved as grayscale PNG in `data/labels/` (pixel value = class index 0-6),
compatible with the training pipeline in `utils/dataset.py`.

## Requirements

- Docker + Docker Compose
- ~2 GB disk space (for MobileSAM + PyTorch in container)
