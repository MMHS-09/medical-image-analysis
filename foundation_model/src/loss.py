import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CombinedLoss(nn.Module):
    """Combined loss function for foundation model training"""
    
    def __init__(self, task: str, num_classes: int = None, weights: Optional[torch.Tensor] = None):
        super(CombinedLoss, self).__init__()
        self.task = task
        self.num_classes = num_classes
        
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        elif task == "segmentation":
            self.criterion = CombinedSegmentationLoss(num_classes=num_classes, weights=weights)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


class CombinedSegmentationLoss(nn.Module):
    """Combined loss for segmentation: Dice + CrossEntropy"""
    
    def __init__(self, num_classes: int = 2, weights: Optional[torch.Tensor] = None, 
                 dice_weight: float = 0.5, ce_weight: float = 0.5):
        super(CombinedSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_loss = DiceLoss(num_classes=num_classes)
    
    def forward(self, predictions, targets):
        ce_loss = self.ce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return combined_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, num_classes: int = 2, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for class_idx in range(self.num_classes):
            pred_class = predictions[:, class_idx]
            target_class = targets_one_hot[:, class_idx]
            
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average Dice score across classes and batch
        dice_score = torch.stack(dice_scores, dim=1).mean()
        
        # Return Dice loss (1 - Dice score)
        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """IoU Loss for segmentation"""
    
    def __init__(self, num_classes: int = 2, smooth: float = 1e-5):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate IoU for each class
        iou_scores = []
        for class_idx in range(self.num_classes):
            pred_class = predictions[:, class_idx]
            target_class = targets_one_hot[:, class_idx]
            
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2)) - intersection
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou)
        
        # Average IoU score across classes and batch
        iou_score = torch.stack(iou_scores, dim=1).mean()
        
        # Return IoU loss (1 - IoU score)
        return 1.0 - iou_score


def get_loss_function(task: str, num_classes: int = None, loss_type: str = "default", 
                     class_weights: Optional[torch.Tensor] = None):
    """Get appropriate loss function based on task and configuration"""
    
    if task == "classification":
        if loss_type == "focal":
            return FocalLoss()
        else:
            return nn.CrossEntropyLoss(weight=class_weights)
    
    elif task == "segmentation":
        if loss_type == "dice":
            return DiceLoss(num_classes=num_classes)
        elif loss_type == "iou":
            return IoULoss(num_classes=num_classes)
        elif loss_type == "focal":
            return FocalLoss()
        else:
            return CombinedSegmentationLoss(num_classes=num_classes, weights=class_weights)
    
    else:
        raise ValueError(f"Unknown task: {task}")


def calculate_class_weights(dataset, num_classes: int, task: str = "classification"):
    """Calculate class weights for handling imbalanced datasets"""
    
    if task == "classification":
        # Count samples per class
        class_counts = torch.zeros(num_classes)
        for sample in dataset:
            class_counts[sample['label']] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(dataset)
        class_weights = total_samples / (num_classes * class_counts)
        
        return class_weights
    
    elif task == "segmentation":
        # For segmentation, calculate weights based on pixel frequencies
        pixel_counts = torch.zeros(num_classes)
        total_pixels = 0
        
        for sample in dataset:
            mask = sample['mask']
            total_pixels += mask.numel()
            
            for class_idx in range(num_classes):
                pixel_counts[class_idx] += (mask == class_idx).sum().item()
        
        # Calculate weights (inverse frequency)
        class_weights = total_pixels / (num_classes * pixel_counts)
        
        return class_weights
    
    else:
        raise ValueError(f"Unknown task: {task}")