"""
===========================================================
Dataset & DataLoader สำหรับ Semantic Segmentation
รองรับ DeepLabV3, UNet, HRNet
===========================================================
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class SegmentationDataset(Dataset):
    """
    Dataset สำหรับ Semantic Segmentation

    โครงสร้างโฟลเดอร์:
    tiles_dir/
        tile_00000.png
        tile_00001.png
        ...
    labels_dir/
        tile_00000.png   (mask: แต่ละพิกเซลเป็นค่า class index 0-N)
        tile_00001.png
        ...
    """

    def __init__(
        self,
        tiles_dir: str,
        labels_dir: str,
        file_list: list = None,
        num_classes: int = 7,
        transform=None,
        use_nir: bool = False,
    ):
        """
        Parameters:
        -----------
        tiles_dir : str - โฟลเดอร์เก็บ tiles (ภาพ)
        labels_dir : str - โฟลเดอร์เก็บ label masks
        file_list : list - ลิสต์ชื่อไฟล์ (จาก splits.json)
        num_classes : int - จำนวน classes
        transform : albumentations - data augmentation
        use_nir : bool - ใช้ NIR band ด้วยหรือไม่
        """
        self.tiles_dir = tiles_dir
        self.labels_dir = labels_dir
        self.num_classes = num_classes
        self.transform = transform
        self.use_nir = use_nir

        if file_list is not None:
            self.file_list = file_list
        else:
            self.file_list = sorted([
                f for f in os.listdir(tiles_dir)
                if f.endswith('.png') and 'nir' not in f
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # โหลดภาพ
        img_path = os.path.join(self.tiles_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # โหลด NIR (ถ้าใช้)
        if self.use_nir:
            nir_filename = filename.replace('.png', '_nir.png')
            nir_path = os.path.join(self.tiles_dir, nir_filename)
            if os.path.exists(nir_path):
                nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
                image = np.dstack([image, nir])

        # โหลด label mask
        label_path = os.path.join(self.labels_dir, filename)
        if os.path.exists(label_path):
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        else:
            # ถ้าไม่มี label ให้ใช้ dummy mask (สำหรับทดสอบ)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


def get_train_transform(img_size=512, use_nir=False):
    """Augmentation สำหรับ training"""
    return A.Compose([
        A.RandomCrop(width=img_size, height=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=20, val_shift_limit=20, p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406] if not use_nir else [0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225] if not use_nir else [0.229, 0.224, 0.225, 0.25],
        ),
        ToTensorV2(),
    ])


def get_val_transform(img_size=512, use_nir=False):
    """Augmentation สำหรับ validation (ไม่มี random transform)"""
    return A.Compose([
        A.CenterCrop(width=img_size, height=img_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406] if not use_nir else [0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225] if not use_nir else [0.229, 0.224, 0.225, 0.25],
        ),
        ToTensorV2(),
    ])


def create_dataloaders(config: dict):
    """
    สร้าง DataLoaders สำหรับ train/val/test
    """
    tiles_dir = config["data"]["tiles_dir"]
    labels_dir = config["data"]["labels_dir"]
    tile_size = config["data"]["tile_size"]
    batch_size = config["segmentation"]["batch_size"]
    num_classes = config["num_classes"]
    use_nir = config["data"].get("use_nir", False)

    # โหลด splits
    splits_path = os.path.join(tiles_dir, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    else:
        # ถ้าไม่มี splits ให้ใช้ไฟล์ทั้งหมดเป็น train
        all_files = sorted([
            f for f in os.listdir(tiles_dir)
            if f.endswith('.png') and 'nir' not in f
        ])
        splits = {"train": all_files,
                  "val": all_files[:10], "test": all_files[:10]}

    if use_nir:
        print("   NIR band enabled (4-channel input: RGBNIR)")

    # สร้าง datasets
    train_dataset = SegmentationDataset(
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
        file_list=splits["train"],
        num_classes=num_classes,
        transform=get_train_transform(tile_size, use_nir=use_nir),
        use_nir=use_nir,
    )

    val_dataset = SegmentationDataset(
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
        file_list=splits["val"],
        num_classes=num_classes,
        transform=get_val_transform(tile_size, use_nir=use_nir),
        use_nir=use_nir,
    )

    test_dataset = SegmentationDataset(
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
        file_list=splits["test"],
        num_classes=num_classes,
        transform=get_val_transform(tile_size, use_nir=use_nir),
        use_nir=use_nir,
    )

    # สร้าง DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
