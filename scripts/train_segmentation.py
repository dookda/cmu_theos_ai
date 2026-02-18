"""
===========================================================
สคริปต์เทรน Semantic Segmentation (DeepLabV3, UNet, HRNet)
สำหรับ Land Cover Classification จากภาพดาวเทียม THEOS
===========================================================
การใช้งาน:
    python scripts/train_segmentation.py --model deeplabv3 --config configs/config.yaml
    python scripts/train_segmentation.py --model unet
    python scripts/train_segmentation.py --model hrnet
"""

from utils.metrics import SegmentationMetrics
from utils.losses import get_loss_function
from utils.dataset import create_dataloaders
from models.hrnet import create_hrnet
from models.unet import create_unet, create_unetpp
from models.deeplabv3 import create_deeplabv3
import os
import sys
import argparse
import time
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path

# เพิ่ม project root ใน path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_device():
    """เลือก device ที่ดีที่สุด"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️  ใช้ GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🖥️  ใช้ Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print(f"🖥️  ใช้ CPU")
    return device


def create_model(model_name: str, config: dict):
    """สร้างโมเดลตามชื่อ"""
    num_classes = config["num_classes"]
    seg_config = config["segmentation"]
    in_channels = 4 if config["data"].get("use_nir", False) else 3

    if model_name == "deeplabv3":
        model = create_deeplabv3(
            encoder_name=seg_config["deeplabv3"]["encoder"],
            encoder_weights="imagenet" if seg_config["deeplabv3"]["pretrained"] else None,
            num_classes=num_classes,
            in_channels=in_channels,
        )
    elif model_name == "unet":
        model = create_unet(
            encoder_name=seg_config["unet"]["encoder"],
            encoder_weights="imagenet" if seg_config["unet"]["pretrained"] else None,
            num_classes=num_classes,
            in_channels=in_channels,
        )
    elif model_name == "unetpp":
        model = create_unetpp(
            encoder_name=seg_config["unet"]["encoder"],
            encoder_weights="imagenet" if seg_config["unet"]["pretrained"] else None,
            num_classes=num_classes,
            in_channels=in_channels,
        )
    elif model_name == "hrnet":
        model = create_hrnet(
            encoder_name=seg_config["hrnet"]["encoder"],
            encoder_weights="imagenet" if seg_config["hrnet"]["pretrained"] else None,
            num_classes=num_classes,
            in_channels=in_channels,
        )
    else:
        raise ValueError(
            f"ไม่รู้จักโมเดล: {model_name}. เลือกจาก: deeplabv3, unet, unetpp, hrnet")

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"🏗️  สร้างโมเดล {model_name.upper()} สำเร็จ ({total_params:,} parameters)")

    return model


def get_optimizer(model, config: dict):
    """สร้าง optimizer"""
    seg_config = config["segmentation"]
    opt_name = seg_config["optimizer"]
    lr = seg_config["learning_rate"]
    wd = seg_config["weight_decay"]

    if opt_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"ไม่รู้จัก optimizer: {opt_name}")


def get_scheduler(optimizer, config: dict):
    """สร้าง learning rate scheduler"""
    seg_config = config["segmentation"]
    scheduler_name = seg_config["scheduler"]
    epochs = seg_config["epochs"]

    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_name == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    else:
        return None


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """เทรน 1 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"  Epoch {epoch} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, criterion, device, metrics, epoch):
    """ประเมินผลบน validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    metrics.reset()

    pbar = tqdm(val_loader, desc=f"  Epoch {epoch} [Val]  ")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        num_batches += 1

        # อัพเดท metrics
        metrics.update(outputs, masks)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    summary = metrics.get_summary()

    return avg_loss, summary


def train(
    model_name: str,
    config: dict,
    resume_path: str = None,
):
    """
    เทรนโมเดล Semantic Segmentation
    """
    seg_config = config["segmentation"]
    output_dir = config["output"]["models_dir"]
    log_dir = config["output"]["log_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = get_device()

    # สร้างโมเดล
    model = create_model(model_name, config)
    model = model.to(device)

    # Resume
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"📂 โหลดโมเดลจาก: {resume_path} (epoch {start_epoch})")

    # สร้าง DataLoaders
    print("\n📊 กำลังโหลดข้อมูล...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")

    # Loss, Optimizer, Scheduler
    criterion = get_loss_function(seg_config["loss"], config["num_classes"])
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Metrics
    class_names = [c["name"] for c in config["classes"]]
    metrics = SegmentationMetrics(config["num_classes"], class_names)

    # Training loop
    epochs = seg_config["epochs"]
    patience = seg_config["patience"]
    best_miou = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [],
               "miou": [], "oa": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"  🚀 เริ่มเทรน {model_name.upper()}")
    print(
        f"  Epochs: {epochs}, Batch: {seg_config['batch_size']}, LR: {seg_config['learning_rate']}")
    print(
        f"  Loss: {seg_config['loss']}, Optimizer: {seg_config['optimizer']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1)

        # Validate
        val_loss, val_summary = validate(
            model, val_loader, criterion, device, metrics, epoch + 1)

        # Scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # บันทึก history
        miou = val_summary["mean_iou"]
        oa = val_summary["overall_accuracy"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["miou"].append(miou)
        history["oa"].append(oa)
        history["lr"].append(current_lr)

        print(f"\n  📈 Epoch {epoch+1}/{epochs}")
        print(f"     Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"     mIoU: {miou:.4f} | OA: {oa:.4f} | LR: {current_lr:.6f}")

        # บันทึกโมเดลที่ดีที่สุด
        if miou > best_miou:
            best_miou = miou
            patience_counter = 0

            save_path = os.path.join(output_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'config': config,
            }, save_path)
            print(f"     💾 บันทึกโมเดลที่ดีที่สุด! mIoU: {best_miou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n  ⏹️  Early stopping! ไม่มีการปรับปรุงมา {patience} epochs")
                break

        # บันทึก checkpoint ล่าสุด
        save_path = os.path.join(output_dir, f"{model_name}_last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'config': config,
        }, save_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ เทรนเสร็จสมบูรณ์!")
    print(f"  เวลาทั้งหมด: {elapsed/3600:.2f} ชั่วโมง")
    print(f"  Best mIoU: {best_miou:.4f}")
    print(f"  โมเดลที่ดีที่สุด: {output_dir}/{model_name}_best.pth")
    print(f"{'='*60}")

    # บันทึก history
    history_path = os.path.join(log_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # ประเมินผลบน test set
    print(f"\n📊 ประเมินผลบน Test Set...")
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = SegmentationMetrics(config["num_classes"], class_names)
    test_loss, test_summary = validate(
        model, test_loader, criterion, device, test_metrics, "Test")
    test_metrics.print_summary()

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="เทรน Semantic Segmentation สำหรับภาพดาวเทียม")
    parser.add_argument("--model", type=str, default="deeplabv3",
                        choices=["deeplabv3", "unet", "unetpp", "hrnet"],
                        help="เลือกโมเดล: deeplabv3, unet, unetpp, hrnet")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="พาธไปยังไฟล์ config")
    parser.add_argument("--resume", type=str, default=None,
                        help="พาธไปยัง checkpoint สำหรับ resume training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="จำนวน epochs (override config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (override config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (override config)")
    parser.add_argument("--use-nir", action="store_true",
                        help="ใช้ NIR band (4 channels: RGBNIR)")

    args = parser.parse_args()

    # โหลด config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override config
    if args.epochs:
        config["segmentation"]["epochs"] = args.epochs
    if args.batch_size:
        config["segmentation"]["batch_size"] = args.batch_size
    if args.lr:
        config["segmentation"]["learning_rate"] = args.lr
    if args.use_nir:
        config["data"]["use_nir"] = True

    # เทรน
    train(
        model_name=args.model,
        config=config,
        resume_path=args.resume,
    )
