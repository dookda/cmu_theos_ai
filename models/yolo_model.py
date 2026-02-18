"""
===========================================================
YOLO Model สำหรับ Object Detection & Instance Segmentation
บนภาพดาวเทียม
===========================================================
YOLOv8 รองรับ:
- Object Detection: ตรวจจับวัตถุ (bounding box)
- Instance Segmentation: แบ่งส่วนวัตถุแต่ละชิ้น (polygon mask)
- เหมาะสำหรับตรวจจับอาคาร, ยานพาหนะ, แปลงเกษตร ฯลฯ
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO


def setup_yolo_dataset(
    tiles_dir: str,
    labels_dir: str,
    output_dir: str,
    classes: list,
    splits_file: str = None,
):
    """
    จัดเตรียมข้อมูลในรูปแบบ YOLO

    โครงสร้าง YOLO dataset:
    output_dir/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        data.yaml

    Parameters:
    -----------
    tiles_dir : str - โฟลเดอร์ tiles (ภาพ)
    labels_dir : str - โฟลเดอร์ YOLO labels (.txt)
    output_dir : str - โฟลเดอร์ output
    classes : list - รายชื่อ classes
    splits_file : str - พาธไปยัง splits.json
    """
    import json

    # สร้างโครงสร้างโฟลเดอร์
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # โหลด splits
    if splits_file and os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            splits = json.load(f)
    else:
        # ถ้าไม่มี splits ให้ใช้ default
        all_files = sorted([f for f in os.listdir(tiles_dir)
                           if f.endswith('.png') and 'nir' not in f])
        n = len(all_files)
        splits = {
            "train": all_files[:int(n*0.7)],
            "val": all_files[int(n*0.7):int(n*0.85)],
            "test": all_files[int(n*0.85):],
        }

    # คัดลอกไฟล์ไปยังโครงสร้าง YOLO
    for split, files in splits.items():
        for filename in files:
            # คัดลอกภาพ
            src_img = os.path.join(tiles_dir, filename)
            dst_img = os.path.join(output_dir, "images", split, filename)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            # คัดลอก label
            label_name = filename.replace('.png', '.txt')
            src_label = os.path.join(labels_dir, label_name)
            dst_label = os.path.join(output_dir, "labels", split, label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

    # สร้าง data.yaml
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ จัดเตรียมข้อมูล YOLO เสร็จแล้ว")
    print(f"   พาธ: {output_dir}")
    print(f"   Config: {yaml_path}")

    return yaml_path


def train_yolo(
    data_yaml: str,
    model_name: str = "yolov8m-seg",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    patience: int = 20,
    lr: float = 0.01,
    project: str = "results",
    name: str = "yolo_theos",
    device: str = "auto",
):
    """
    เทรนโมเดล YOLO

    Parameters:
    -----------
    data_yaml : str - พาธไปยัง data.yaml
    model_name : str - ชื่อโมเดล YOLO
        Detection: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        Segmentation: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
    epochs : int - จำนวน epochs
    batch_size : int - batch size
    image_size : int - ขนาดภาพ input
    patience : int - early stopping patience
    lr : float - learning rate เริ่มต้น
    project : str - โฟลเดอร์เก็บผลลัพธ์
    name : str - ชื่อ experiment
    device : str - "auto", "cpu", "0", "0,1" (GPU IDs)
    """
    print(f"\n{'='*60}")
    print(f"  🎯 เริ่มเทรน YOLO: {model_name}")
    print(f"{'='*60}")

    # โหลดโมเดล
    model = YOLO(model_name + ".pt")

    # เทรน
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        patience=patience,
        lr0=lr,
        project=project,
        name=name,
        device=device,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Performance
        amp=True,
        workers=4,
        save=True,
        plots=True,
        verbose=True,
    )

    print(f"\n✅ เทรนเสร็จแล้ว!")
    print(f"   ผลลัพธ์: {project}/{name}")

    return model, results


def predict_yolo(
    model_path: str,
    source: str,
    output_dir: str = "results/yolo_predictions",
    conf: float = 0.25,
    iou: float = 0.45,
    image_size: int = 640,
):
    """
    ทำนายด้วยโมเดล YOLO

    Parameters:
    -----------
    model_path : str - พาธไปยังโมเดล (.pt)
    source : str - พาธไปยังภาพหรือโฟลเดอร์
    output_dir : str - โฟลเดอร์เก็บผลลัพธ์
    conf : float - confidence threshold
    iou : float - IoU threshold สำหรับ NMS
    """
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=image_size,
        save=True,
        save_txt=True,
        project=output_dir,
        name="predict",
    )

    print(f"✅ ทำนายเสร็จแล้ว! ผลลัพธ์อยู่ที่: {output_dir}/predict")

    return results


def validate_yolo(model_path: str, data_yaml: str):
    """ประเมินผลโมเดล YOLO"""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    print(f"\n📊 ผลการประเมิน YOLO:")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")

    if hasattr(metrics, 'seg'):
        print(f"   Seg mAP50:    {metrics.seg.map50:.4f}")
        print(f"   Seg mAP50-95: {metrics.seg.map:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO สำหรับภาพดาวเทียม")
    parser.add_argument("--mode", type=str, choices=["setup", "train", "predict", "val"],
                        default="train", help="โหมดการทำงาน")
    parser.add_argument("--data", type=str, default="data/yolo/data.yaml",
                        help="พาธไปยัง data.yaml")
    parser.add_argument("--model", type=str, default="yolov8m-seg",
                        help="ชื่อโมเดลหรือพาธไปยัง .pt file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--source", type=str, default=None,
                        help="พาธภาพสำหรับ prediction")

    args = parser.parse_args()

    if args.mode == "train":
        train_yolo(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
        )
    elif args.mode == "predict":
        predict_yolo(
            model_path=args.model,
            source=args.source or "data/tiles",
        )
    elif args.mode == "val":
        validate_yolo(
            model_path=args.model,
            data_yaml=args.data,
        )
