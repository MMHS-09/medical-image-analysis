#!/usr/bin/env python3
"""
Advanced Metrics for Medical Image Segmentation

This module provides comprehensive metrics for evaluating segmentation performance,
including per-class metrics, boundary metrics, and statistical measures.
"""

import torch
import numpy as np
from typing import Dict
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator supporting multi-class segmentation.
    """

    def __init__(self, num_classes: int, ignore_background: bool = True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.eps = 1e-7

    def calculate_confusion_matrix(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate confusion matrix for multi-class segmentation.

        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W]

        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        if pred.dim() == 4:  # [B, C, H, W] - logits
            pred = pred.argmax(dim=1)  # [B, H, W]

        # Flatten all dimensions except class
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Create confusion matrix
        valid_mask = (target_flat >= 0) & (target_flat < self.num_classes)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        # Calculate confusion matrix using bincount
        device = pred_flat.device  # Use the same device as input tensors
        confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long, device=device
        )
        indices = self.num_classes * target_flat + pred_flat
        confusion = confusion.view(-1)
        confusion.index_add_(0, indices, torch.ones_like(indices))
        confusion = confusion.view(self.num_classes, self.num_classes)

        return confusion

    def calculate_iou(
        self, pred: torch.Tensor, target: torch.Tensor, per_class: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Intersection over Union (IoU) for each class.

        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W]
            per_class: Whether to return per-class IoU

        Returns:
            Dictionary containing IoU metrics
        """
        confusion = self.calculate_confusion_matrix(pred, target)

        # Calculate IoU for each class
        intersection = torch.diag(confusion)
        union = confusion.sum(dim=1) + confusion.sum(dim=0) - intersection

        iou_per_class = intersection.float() / (union.float() + self.eps)

        results = {}

        if per_class:
            for i in range(self.num_classes):
                class_name = f"IoU_class_{i}"
                if i == 0 and self.ignore_background:
                    class_name = "IoU_background"
                results[class_name] = iou_per_class[i].item()

        # Calculate mean IoU (excluding background if specified)
        if self.ignore_background and self.num_classes > 1:
            mean_iou = iou_per_class[1:].mean().item()
        else:
            mean_iou = iou_per_class.mean().item()

        results["mIoU"] = mean_iou

        return results

    def calculate_dice(
        self, pred: torch.Tensor, target: torch.Tensor, per_class: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Dice coefficient for each class.

        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W]
            per_class: Whether to return per-class Dice

        Returns:
            Dictionary containing Dice metrics
        """
        confusion = self.calculate_confusion_matrix(pred, target)

        # Calculate Dice for each class
        intersection = torch.diag(confusion)
        pred_sum = confusion.sum(dim=0)
        target_sum = confusion.sum(dim=1)

        dice_per_class = (2 * intersection.float()) / (
            pred_sum.float() + target_sum.float() + self.eps
        )

        results = {}

        if per_class:
            for i in range(self.num_classes):
                class_name = f"Dice_class_{i}"
                if i == 0 and self.ignore_background:
                    class_name = "Dice_background"
                results[class_name] = dice_per_class[i].item()

        # Calculate mean Dice (excluding background if specified)
        if self.ignore_background and self.num_classes > 1:
            mean_dice = dice_per_class[1:].mean().item()
        else:
            mean_dice = dice_per_class.mean().item()

        results["mDice"] = mean_dice

        return results

    def calculate_precision_recall_f1(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for each class.

        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W]

        Returns:
            Dictionary containing precision, recall, and F1 metrics
        """
        confusion = self.calculate_confusion_matrix(pred, target)

        # Calculate metrics for each class
        tp = torch.diag(confusion).float()
        fp = confusion.sum(dim=0).float() - tp
        fn = confusion.sum(dim=1).float() - tp

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)

        results = {}

        # Per-class metrics
        for i in range(self.num_classes):
            if not (i == 0 and self.ignore_background):
                results[f"Precision_class_{i}"] = precision[i].item()
                results[f"Recall_class_{i}"] = recall[i].item()
                results[f"F1_class_{i}"] = f1[i].item()

        # Mean metrics (excluding background if specified)
        if self.ignore_background and self.num_classes > 1:
            results["mPrecision"] = precision[1:].mean().item()
            results["mRecall"] = recall[1:].mean().item()
            results["mF1"] = f1[1:].mean().item()
        else:
            results["mPrecision"] = precision.mean().item()
            results["mRecall"] = recall.mean().item()
            results["mF1"] = f1.mean().item()

        return results

    def calculate_hausdorff_distance(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate Hausdorff distance for boundary accuracy assessment.

        Note: This is computationally expensive and should be used sparingly.

        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W]

        Returns:
            Dictionary containing Hausdorff distance metrics
        """
        if pred.dim() == 4:  # [B, C, H, W] - logits
            pred = pred.argmax(dim=1)  # [B, H, W]

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        hausdorff_distances = []

        for batch_idx in range(pred_np.shape[0]):
            batch_distances = []

            for class_id in range(1 if self.ignore_background else 0, self.num_classes):
                pred_mask = (pred_np[batch_idx] == class_id).astype(np.uint8)
                target_mask = (target_np[batch_idx] == class_id).astype(np.uint8)

                # Find boundary points
                pred_boundary = self._get_boundary_points(pred_mask)
                target_boundary = self._get_boundary_points(target_mask)

                if len(pred_boundary) > 0 and len(target_boundary) > 0:
                    # Calculate bidirectional Hausdorff distance
                    dist1 = directed_hausdorff(pred_boundary, target_boundary)[0]
                    dist2 = directed_hausdorff(target_boundary, pred_boundary)[0]
                    hausdorff_dist = max(dist1, dist2)
                    batch_distances.append(hausdorff_dist)

            if batch_distances:
                hausdorff_distances.append(np.mean(batch_distances))

        results = {}
        if hausdorff_distances:
            results["Hausdorff_distance"] = np.mean(hausdorff_distances)
            results["Hausdorff_distance_std"] = np.std(hausdorff_distances)
        else:
            results["Hausdorff_distance"] = 0.0
            results["Hausdorff_distance_std"] = 0.0

        return results

    def _get_boundary_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary points from a binary mask."""
        # Use morphological operations to find boundary
        eroded = ndimage.binary_erosion(mask)
        boundary = mask ^ eroded

        # Get coordinates of boundary points
        boundary_points = np.column_stack(np.where(boundary))
        return boundary_points

    def calculate_volume_similarity(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate volume similarity metrics.

        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W]

        Returns:
            Dictionary containing volume similarity metrics
        """
        if pred.dim() == 4:  # [B, C, H, W] - logits
            pred = pred.argmax(dim=1)  # [B, H, W]

        results = {}

        for class_id in range(1 if self.ignore_background else 0, self.num_classes):
            pred_volume = (pred == class_id).sum().float()
            target_volume = (target == class_id).sum().float()

            if target_volume > 0:
                volume_similarity = 1 - abs(pred_volume - target_volume) / target_volume
                results[f"Volume_similarity_class_{class_id}"] = (
                    volume_similarity.item()
                )
            else:
                results[f"Volume_similarity_class_{class_id}"] = (
                    1.0 if pred_volume == 0 else 0.0
                )

        # Calculate mean volume similarity
        if len(results) > 0:
            results["mVolume_similarity"] = np.mean(list(results.values()))

        return results

    def calculate_all_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, include_hausdorff: bool = False
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            pred: Predicted masks [B, H, W] or [B, C, H, W]
            target: Ground truth masks [B, H, W]
            include_hausdorff: Whether to calculate Hausdorff distance (expensive)

        Returns:
            Dictionary containing all metrics
        """
        all_metrics = {}

        # Basic metrics
        all_metrics.update(self.calculate_iou(pred, target))
        all_metrics.update(self.calculate_dice(pred, target))
        all_metrics.update(self.calculate_precision_recall_f1(pred, target))
        all_metrics.update(self.calculate_volume_similarity(pred, target))

        # Optional expensive metrics
        if include_hausdorff:
            all_metrics.update(self.calculate_hausdorff_distance(pred, target))

        return all_metrics


def fast_metrics_calculation(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> Dict[str, float]:
    """
    Fast calculation of essential metrics for training loops.

    Args:
        pred: Predicted masks [B, H, W] or [B, C, H, W]
        target: Ground truth masks [B, H, W]
        num_classes: Number of classes
        ignore_background: Whether to ignore background class

    Returns:
        Dictionary containing essential metrics
    """
    if pred.dim() == 4:  # [B, C, H, W] - logits
        pred = pred.argmax(dim=1)  # [B, H, W]

    eps = 1e-7
    results = {}

    # Calculate IoU and Dice for each class
    total_iou = 0
    total_dice = 0
    valid_classes = 0

    for class_id in range(num_classes):
        if class_id == 0 and ignore_background:
            continue

        pred_mask = pred == class_id
        target_mask = target == class_id

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()

        # IoU
        iou = intersection / (union + eps) if union > 0 else 1.0

        # Dice
        dice = (
            (2 * intersection) / (pred_sum + target_sum + eps)
            if (pred_sum + target_sum) > 0
            else 1.0
        )

        total_iou += iou
        total_dice += dice
        valid_classes += 1

    # Calculate mean metrics
    if valid_classes > 0:
        results["mIoU"] = (total_iou / valid_classes).item()
        results["mDice"] = (total_dice / valid_classes).item()
    else:
        results["mIoU"] = 0.0
        results["mDice"] = 0.0

    return results


def print_metrics_summary(metrics: Dict[str, float], epoch: int = None):
    """
    Print a formatted summary of metrics.

    Args:
        metrics: Dictionary containing metrics
        epoch: Optional epoch number
    """
    print("\n" + "=" * 60)
    if epoch is not None:
        print(f"SEGMENTATION METRICS - EPOCH {epoch}")
    else:
        print("SEGMENTATION METRICS SUMMARY")
    print("=" * 60)

    # Group metrics by type
    iou_metrics = {k: v for k, v in metrics.items() if "IoU" in k}
    dice_metrics = {k: v for k, v in metrics.items() if "Dice" in k}
    precision_metrics = {k: v for k, v in metrics.items() if "Precision" in k}
    recall_metrics = {k: v for k, v in metrics.items() if "Recall" in k}
    f1_metrics = {k: v for k, v in metrics.items() if "F1" in k}
    other_metrics = {
        k: v
        for k, v in metrics.items()
        if not any(
            metric_type in k
            for metric_type in ["IoU", "Dice", "Precision", "Recall", "F1"]
        )
    }

    # Print main metrics
    if "mIoU" in metrics:
        print(f"Mean IoU:        {metrics['mIoU']:.4f}")
    if "mDice" in metrics:
        print(f"Mean Dice:       {metrics['mDice']:.4f}")
    if "mF1" in metrics:
        print(f"Mean F1:         {metrics['mF1']:.4f}")

    print("-" * 60)

    # Print per-class metrics if available
    if len(iou_metrics) > 1:  # More than just mIoU
        print("Per-Class IoU:")
        for k, v in iou_metrics.items():
            if k != "mIoU":
                print(f"  {k:20s}: {v:.4f}")

    if len(dice_metrics) > 1:  # More than just mDice
        print("Per-Class Dice:")
        for k, v in dice_metrics.items():
            if k != "mDice":
                print(f"  {k:20s}: {v:.4f}")

    # Print other metrics
    if other_metrics:
        print("Additional Metrics:")
        for k, v in other_metrics.items():
            print(f"  {k:20s}: {v:.4f}")

    print("=" * 60)
