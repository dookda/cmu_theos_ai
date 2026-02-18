"""
===========================================================
Loss Functions สำหรับ Semantic Segmentation
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss - เหมาะสำหรับข้อมูลที่ class ไม่สมดุล
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """

    def __init__(self, smooth=1.0, num_classes=7):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(
            target.long(), self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - ลดน้ำหนักของตัวอย่างที่จำแนกได้ง่าย
    เหมาะสำหรับข้อมูลที่ class ไม่สมดุลมาก
    """

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    รวม CrossEntropy + Dice Loss
    ให้ผลดีที่สุดสำหรับ segmentation ในหลายกรณี
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=7):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target.long())
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss_function(loss_name: str, num_classes: int = 7):
    """
    เลือก loss function ตามชื่อ

    Parameters:
    -----------
    loss_name : str - "ce", "dice", "focal", "combo"
    num_classes : int - จำนวน classes
    """
    loss_map = {
        "ce": nn.CrossEntropyLoss(),
        "dice": DiceLoss(num_classes=num_classes),
        "focal": FocalLoss(num_classes=num_classes),
        "combo": CombinedLoss(num_classes=num_classes),
    }

    if loss_name not in loss_map:
        raise ValueError(
            f"ไม่รู้จัก loss: {loss_name}. เลือกจาก: {list(loss_map.keys())}")

    return loss_map[loss_name]
