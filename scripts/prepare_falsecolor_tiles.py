"""
===========================================================
สคริปต์สร้าง False Color Tiles จากภาพดาวเทียม THEOS2
===========================================================
ขั้นตอน:
1. อ่านภาพ GeoTIFF (bands: R=1, G=2, B=3, NIR=4)
2. สร้าง False Color Composite (NIR, R, G)
3. ตัดเป็น tiles ขนาด 512x512 พิกเซล
4. กรอง tiles ที่เป็น nodata ออก
5. แบ่ง train/val/test (70/15/15%)
6. บันทึก metadata (พิกัด, bounding box)
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import json
from tqdm import tqdm
import yaml
import cv2


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def normalize_band(band, percentile_clip=(2, 98)):
    """Normalize band to 0-255 using percentile clipping for better contrast."""
    band = band.astype(np.float32)
    valid = band[band > 0]  # exclude nodata (0)
    if len(valid) == 0:
        return np.zeros_like(band, dtype=np.uint8)

    low = np.percentile(valid, percentile_clip[0])
    high = np.percentile(valid, percentile_clip[1])
    band = (band - low) / (high - low + 1e-8)
    band = np.clip(band * 255, 0, 255).astype(np.uint8)
    return band


def create_falsecolor_tiles(
    tif_path: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    min_valid_ratio: float = 0.5,
    nodata_value: float = None,
):
    """
    สร้าง False Color Tiles จากภาพ THEOS2

    False Color Composite: NIR(band4) → Red, R(band1) → Green, G(band2) → Blue
    ทำให้พืชพรรณปรากฏเป็นสีแดง เห็นชัดเจนกว่า True Color

    Parameters:
    -----------
    tif_path : str - พาธไปยังไฟล์ GeoTIFF
    output_dir : str - โฟลเดอร์สำหรับเก็บ tiles
    tile_size : int - ขนาดของแต่ละ tile (พิกเซล)
    overlap : int - จำนวนพิกเซลที่ซ้อนทับ
    min_valid_ratio : float - สัดส่วนขั้นต่ำของพิกเซลที่มีค่า
    nodata_value : float - ค่า nodata (None = อ่านจากไฟล์)
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []

    with rasterio.open(tif_path) as src:
        print(f"  เปิดไฟล์: {tif_path}")
        print(f"   ขนาด: {src.width} x {src.height} พิกเซล")
        print(f"   จำนวน bands: {src.count}")
        print(f"   ระบบพิกัด: {src.crs}")
        print(f"   ชนิดข้อมูล: {src.dtypes}")

        if nodata_value is None:
            nodata_value = src.nodata or 0
        print(f"   Nodata value: {nodata_value}")

        # THEOS2 bands (1-indexed in rasterio): R=1, G=2, B=3, NIR=4
        # False Color Composite: NIR, R, G (displayed as RGB)
        false_color_bands = [4, 1, 2]  # NIR → R channel, R → G channel, G → B channel
        print(f"\n   False Color: NIR(band4)->R, R(band1)->G, G(band2)->B")

        # Pre-read full bands to compute global percentile for normalization
        print("\n   คำนวณค่า normalization (percentile 2-98)...")
        band_stats = {}
        for band_idx in false_color_bands:
            data = src.read(band_idx)
            valid = data[(data > 0) & (data != nodata_value)]
            if len(valid) > 0:
                band_stats[band_idx] = {
                    "low": float(np.percentile(valid, 2)),
                    "high": float(np.percentile(valid, 98)),
                }
            else:
                band_stats[band_idx] = {"low": 0, "high": 1}
            print(f"   Band {band_idx}: low={band_stats[band_idx]['low']:.1f}, high={band_stats[band_idx]['high']:.1f}")
            del data

        step = tile_size - overlap
        n_cols = (src.width - overlap) // step
        n_rows = (src.height - overlap) // step
        total_tiles = n_cols * n_rows

        print(f"\n   จะตัดเป็น tiles ขนาด {tile_size}x{tile_size}")
        print(f"   overlap: {overlap} พิกเซล")
        print(f"   จำนวน tiles ทั้งหมด (ก่อนกรอง): {total_tiles}")

        tile_idx = 0
        valid_count = 0

        for row in tqdm(range(n_rows), desc="แถว"):
            for col in range(n_cols):
                x_off = col * step
                y_off = row * step

                if x_off + tile_size > src.width or y_off + tile_size > src.height:
                    continue

                window = Window(x_off, y_off, tile_size, tile_size)

                # อ่าน false color bands
                tile_data = src.read(false_color_bands, window=window)

                # ตรวจสอบ nodata
                # ใช้ band แรก (NIR) ตรวจสอบ — ถ้า NIR เป็น 0 หรือ nodata ถือว่าไม่มีข้อมูล
                if nodata_value != 0:
                    valid_mask = (tile_data[0] != nodata_value) & (tile_data[0] > 0)
                else:
                    valid_mask = tile_data[0] > 0

                valid_ratio = np.sum(valid_mask) / (tile_size * tile_size)

                if valid_ratio < min_valid_ratio:
                    continue

                # Normalize แต่ละ band ด้วย global percentile
                tile_normalized = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
                for i, band_idx in enumerate(false_color_bands):
                    stats = band_stats[band_idx]
                    band = tile_data[i].astype(np.float32)
                    band = (band - stats["low"]) / (stats["high"] - stats["low"] + 1e-8)
                    tile_normalized[i] = np.clip(band * 255, 0, 255).astype(np.uint8)

                # CHW -> HWC, then RGB -> BGR for OpenCV
                tile_hwc = np.transpose(tile_normalized, (1, 2, 0))
                bgr = tile_hwc[:, :, [2, 1, 0]]

                tile_filename = f"tile_{tile_idx:05d}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                cv2.imwrite(tile_path, bgr)

                # เก็บ metadata
                transform = src.window_transform(window)
                bounds = rasterio.windows.bounds(window, src.transform)
                metadata_list.append({
                    "tile_id": tile_idx,
                    "filename": tile_filename,
                    "x_off": x_off,
                    "y_off": y_off,
                    "bounds": {
                        "left": bounds[0],
                        "bottom": bounds[1],
                        "right": bounds[2],
                        "top": bounds[3],
                    },
                    "valid_ratio": round(valid_ratio, 4),
                    "composite": "false_color_NIR_R_G",
                })

                tile_idx += 1
                valid_count += 1

        print(f"\n   สร้าง tiles สำเร็จ: {valid_count} tiles (จาก {total_tiles} ทั้งหมด)")
        print(f"   บันทึกที่: {output_dir}")

    # บันทึก metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"   Metadata: {meta_path}")

    return metadata_list


def split_dataset(
    tiles_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """แบ่งข้อมูลเป็น train/val/test"""
    from sklearn.model_selection import train_test_split

    meta_path = os.path.join(tiles_dir, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    indices = list(range(len(metadata)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_size), random_state=seed
    )

    splits = {
        "train": [metadata[i]["filename"] for i in train_idx],
        "val": [metadata[i]["filename"] for i in val_idx],
        "test": [metadata[i]["filename"] for i in test_idx],
    }

    split_path = os.path.join(tiles_dir, "splits.json")
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"\n   แบ่งข้อมูล:")
    print(f"   Train: {len(train_idx)} tiles ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(val_idx)} tiles ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(test_idx)} tiles ({test_ratio*100:.0f}%)")
    print(f"   บันทึกที่: {split_path}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="สร้าง False Color Tiles จากภาพดาวเทียม THEOS2")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="พาธไปยังไฟล์ config")
    parser.add_argument("--tif", type=str, default=None,
                        help="พาธไปยังไฟล์ GeoTIFF (override config)")
    parser.add_argument("--output", type=str, default="data/tiles/data_falsecolor",
                        help="โฟลเดอร์เก็บ tiles")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="ขนาด tile")
    parser.add_argument("--overlap", type=int, default=None,
                        help="จำนวนพิกเซลที่ซ้อนทับ (override config)")
    parser.add_argument("--min-valid", type=float, default=0.5,
                        help="สัดส่วนขั้นต่ำของพิกเซลที่มีค่า")
    parser.add_argument("--no-split", action="store_true",
                        help="ไม่ต้องแบ่ง train/val/test")

    args = parser.parse_args()
    config = load_config(args.config)

    tif_path = args.tif or config["data"]["source_tif"]
    output_dir = args.output
    tile_size = args.tile_size
    overlap = args.overlap or config["data"]["overlap"]

    print("=" * 60)
    print("  สร้าง False Color Tiles จากภาพ THEOS2")
    print("  (NIR-R-G Composite)")
    print("=" * 60)

    # สร้าง false color tiles
    metadata = create_falsecolor_tiles(
        tif_path=tif_path,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        min_valid_ratio=args.min_valid,
    )

    # แบ่ง train/val/test
    if not args.no_split and len(metadata) > 0:
        split_dataset(
            tiles_dir=output_dir,
            train_ratio=config["data"]["train_ratio"],
            val_ratio=config["data"]["val_ratio"],
            test_ratio=config["data"]["test_ratio"],
        )

    print("\n   เตรียมข้อมูล False Color เสร็จสมบูรณ์!")
