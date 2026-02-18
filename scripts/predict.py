"""
===========================================================
สคริปต์ Inference สำหรับ Semantic Segmentation
ทำนาย Land Cover บนภาพดาวเทียม THEOS ทั้งภาพ
===========================================================
การใช้งาน:
    python scripts/predict.py --model deeplabv3 --input theos/theos_4326.tif --output results/prediction.tif
"""

from models.hrnet import create_hrnet
from models.unet import create_unet
from models.deeplabv3 import create_deeplabv3
import os
import sys
import argparse
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import yaml
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(model_name: str, checkpoint_path: str, config: dict, device: torch.device):
    """โหลดโมเดลจาก checkpoint"""
    num_classes = config["num_classes"]
    seg_config = config["segmentation"]

    if model_name == "deeplabv3":
        model = create_deeplabv3(
            encoder_name=seg_config["deeplabv3"]["encoder"],
            encoder_weights=None,
            num_classes=num_classes,
        )
    elif model_name == "unet":
        model = create_unet(
            encoder_name=seg_config["unet"]["encoder"],
            encoder_weights=None,
            num_classes=num_classes,
        )
    elif model_name == "hrnet":
        model = create_hrnet(
            encoder_name=seg_config["hrnet"]["encoder"],
            encoder_weights=None,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"ไม่รู้จักโมเดล: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ โหลดโมเดล {model_name} จาก: {checkpoint_path}")
    return model


def normalize_tile(tile_data):
    """Normalize tile สำหรับ inference"""
    # แปลง UInt16 เป็น float32 [0,1]
    tile = tile_data.astype(np.float32)
    for i in range(tile.shape[0]):
        band = tile[i]
        min_v, max_v = np.nanmin(band), np.nanmax(band)
        if max_v > min_v:
            tile[i] = (band - min_v) / (max_v - min_v)
        else:
            tile[i] = 0

    # ImageNet normalization (3 channels)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    tile[:3] = (tile[:3] - mean) / std

    return tile


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    tif_path: str,
    output_path: str,
    tile_size: int = 512,
    overlap: int = 64,
    device: torch.device = None,
    num_classes: int = 7,
    batch_size: int = 4,
):
    """
    ทำนาย Land Cover บนภาพทั้งหมดแบบ sliding window

    Parameters:
    -----------
    model : nn.Module - โมเดลที่เทรนแล้ว
    tif_path : str - ไฟล์ GeoTIFF input
    output_path : str - ไฟล์ GeoTIFF output (prediction map)
    tile_size : int - ขนาด tile
    overlap : int - ขนาด overlap
    device : torch.device
    num_classes : int - จำนวน classes
    batch_size : int - batch size สำหรับ inference
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        print(f"📡 ภาพ: {width} x {height} พิกเซล")

        # สร้าง output array
        prediction = np.zeros((height, width), dtype=np.uint8)
        count_map = np.zeros((height, width), dtype=np.float32)

        step = tile_size - overlap
        n_cols = (width - overlap + step - 1) // step
        n_rows = (height - overlap + step - 1) // step
        total = n_cols * n_rows

        print(f"🔲 ทั้งหมด ~{total} tiles ({n_rows} แถว x {n_cols} คอลัมน์)")

        # Collect tiles
        tiles_batch = []
        positions_batch = []

        pbar = tqdm(total=total, desc="Predicting")

        for row in range(n_rows):
            for col in range(n_cols):
                x_off = min(col * step, width - tile_size)
                y_off = min(row * step, height - tile_size)

                if x_off < 0 or y_off < 0:
                    continue

                window = Window(x_off, y_off, tile_size, tile_size)

                # อ่านข้อมูล (THEOS bands: R=1, G=2, B=3)
                tile_data = src.read([1, 2, 3], window=window)

                # ตรวจสอบ valid pixels
                if np.sum(tile_data[0] > 0) < tile_size * tile_size * 0.1:
                    pbar.update(1)
                    continue

                # Normalize
                tile_norm = normalize_tile(tile_data)

                tiles_batch.append(tile_norm)
                positions_batch.append((x_off, y_off))

                # Process batch
                if len(tiles_batch) >= batch_size:
                    batch = torch.from_numpy(
                        np.stack(tiles_batch)).float().to(device)
                    outputs = model(batch)
                    preds = outputs.argmax(dim=1).cpu().numpy()

                    for pred, (px, py) in zip(preds, positions_batch):
                        prediction[py:py+tile_size, px:px+tile_size] = pred
                        count_map[py:py+tile_size, px:px+tile_size] += 1

                    tiles_batch = []
                    positions_batch = []

                pbar.update(1)

        # Process remaining tiles
        if tiles_batch:
            batch = torch.from_numpy(np.stack(tiles_batch)).float().to(device)
            outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for pred, (px, py) in zip(preds, positions_batch):
                prediction[py:py+tile_size, px:px+tile_size] = pred
                count_map[py:py+tile_size, px:px+tile_size] += 1

        pbar.close()

        # บันทึกผลลัพธ์เป็น GeoTIFF
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype='uint8',
            compress='lzw',
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)

        print(f"\n✅ บันทึกผลทำนายที่: {output_path}")

        # สร้างภาพสีสำหรับ visualization
        create_color_map(prediction, output_path.replace('.tif', '_color.png'))

    return prediction


def create_color_map(prediction: np.ndarray, output_path: str):
    """สร้างภาพสีจาก prediction map"""
    # กำหนดสี
    colors = np.array([
        [0, 0, 0],       # background - ดำ
        [0, 255, 0],     # vegetation - เขียว
        [0, 0, 255],     # water - น้ำเงิน
        [255, 0, 0],     # urban - แดง
        [255, 255, 0],   # agriculture - เหลือง
        [139, 69, 19],   # bare_soil - น้ำตาล
        [128, 128, 128],  # road - เทา
    ], dtype=np.uint8)

    h, w = prediction.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(len(colors)):
        mask = prediction == class_id
        color_map[mask] = colors[class_id]

    # ลดขนาดสำหรับ visualization
    max_dim = 4096
    scale = min(max_dim / h, max_dim / w, 1.0)
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        color_map = cv2.resize(color_map, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
    print(f"🎨 ภาพสี: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ทำนาย Land Cover จากภาพดาวเทียม")
    parser.add_argument("--model", type=str, default="deeplabv3",
                        choices=["deeplabv3", "unet", "hrnet"],
                        help="เลือกโมเดล")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="พาธไปยัง checkpoint (.pth)")
    parser.add_argument("--input", type=str, default="theos/theos_4326.tif",
                        help="ไฟล์ GeoTIFF input")
    parser.add_argument("--output", type=str, default=None,
                        help="ไฟล์ GeoTIFF output")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # หา checkpoint
    checkpoint_path = args.checkpoint or f"models/{args.model}_best.pth"
    output_path = args.output or f"results/{args.model}_prediction.tif"

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    model = load_model(args.model, checkpoint_path, config, device)

    predict_full_image(
        model=model,
        tif_path=args.input,
        output_path=output_path,
        tile_size=args.tile_size,
        device=device,
        num_classes=config["num_classes"],
        batch_size=args.batch_size,
    )
