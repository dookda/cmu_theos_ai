"""
===========================================================
สคริปต์เทรน YOLO สำหรับ Object Detection/Segmentation
บนภาพดาวเทียม THEOS
===========================================================
การใช้งาน:
    python scripts/train_yolo.py --config configs/config.yaml
    python scripts/train_yolo.py --model yolov8l-seg --epochs 200
"""

from models.yolo_model import setup_yolo_dataset, train_yolo, validate_yolo
import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="เทรน YOLO สำหรับภาพดาวเทียม")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="ชื่อโมเดล YOLO (override config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--setup-only", action="store_true",
                        help="จัดเตรียมข้อมูลอย่างเดียว ไม่เทรน")

    args = parser.parse_args()

    # โหลด config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    yolo_config = config["yolo"]

    # Override config
    model_name = args.model or yolo_config["model"]
    epochs = args.epochs or yolo_config["epochs"]
    batch_size = args.batch or yolo_config["batch_size"]
    image_size = args.imgsz or yolo_config["image_size"]

    print(f"{'='*60}")
    print(f"  🎯 YOLO Training Pipeline")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Image: {image_size}")
    print(f"{'='*60}\n")

    # ขั้นตอนที่ 1: จัดเตรียมข้อมูล YOLO
    print("📦 ขั้นตอนที่ 1: จัดเตรียมข้อมูลในรูปแบบ YOLO...")
    yolo_data_dir = os.path.join("data", "yolo")

    data_yaml = setup_yolo_dataset(
        tiles_dir=config["data"]["tiles_dir"],
        labels_dir=config["data"]["yolo_labels_dir"],
        output_dir=yolo_data_dir,
        classes=yolo_config["classes"],
        splits_file=os.path.join(config["data"]["tiles_dir"], "splits.json"),
    )

    if args.setup_only:
        print("\n✅ จัดเตรียมข้อมูลเสร็จแล้ว (--setup-only)")
        return

    # ขั้นตอนที่ 2: เทรน
    print(f"\n🚀 ขั้นตอนที่ 2: เริ่มเทรน {model_name}...")
    model, results = train_yolo(
        data_yaml=data_yaml,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        patience=yolo_config["patience"],
        lr=yolo_config["learning_rate"],
        project=config["output"]["results_dir"],
        name=f"yolo_{model_name}",
    )

    # ขั้นตอนที่ 3: ประเมินผล
    print(f"\n📊 ขั้นตอนที่ 3: ประเมินผลโมเดล...")
    best_model = os.path.join(
        config["output"]["results_dir"],
        f"yolo_{model_name}",
        "weights",
        "best.pt"
    )

    if os.path.exists(best_model):
        validate_yolo(best_model, data_yaml)

    print(f"\n✅ เทรน YOLO เสร็จสมบูรณ์!")


if __name__ == "__main__":
    main()
