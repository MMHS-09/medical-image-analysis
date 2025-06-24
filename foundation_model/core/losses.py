#!/usr/bin/env python3
"""
Advanced Loss Functions for Medical Image Segmentation

This module provides state-of-the-art loss functions specifically designed
for medical image segmentation tasks, addressing class imbalance and improving
boundary delineation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchmetrics


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.

    Focal Loss down-weights easy examples and focuses training on hard examples.
    Particularly effective for datasets with severe class imbalance.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction="none")

        # Calculate p_t
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with additional control over false positives and false negatives.

    When alpha=beta=0.5, it becomes Dice loss.
    When alpha=1, beta=0, it becomes recall-focused.
    When alpha=0, beta=1, it becomes precision-focused.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-7,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities
        if pred.dim() > target.dim():
            pred = F.softmax(pred, dim=1)

        # Handle multi-class segmentation
        if target.dim() == 3:  # [B, H, W]
            target = (
                F.one_hot(target.long(), num_classes=pred.size(1))
                .permute(0, 3, 1, 2)
                .float()
            )

        # Flatten tensors
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
        target_flat = target.view(target.size(0), target.size(1), -1)  # [B, C, H*W]

        # Calculate True Positives, False Positives, False Negatives
        tp = (pred_flat * target_flat).sum(dim=-1)  # [B, C]
        fp = (pred_flat * (1 - target_flat)).sum(dim=-1)  # [B, C]
        fn = ((1 - pred_flat) * target_flat).sum(dim=-1)  # [B, C]

        # Calculate Tversky coefficient
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Calculate loss
        tversky_loss = 1 - tversky

        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines CrossEntropy with Dice/Tversky loss.

    This combination leverages the benefits of both:
    - CrossEntropy: Good for pixel-wise classification
    - Dice/Tversky: Good for region overlap and handling class imbalance
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        use_focal: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        use_tversky: bool = False,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.use_tversky = use_tversky

        # Track loss values for debugging
        self.last_ce_loss = 0
        self.last_region_loss = 0

        # Initialize loss functions
        if use_focal:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        if use_tversky:
            self.region_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        else:
            self.region_loss = nn.Module()  # Placeholder, no-op

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle target format for CE loss
        if target.dim() == 4 and target.size(1) > 1:  # One-hot encoded
            target_for_ce = target.argmax(dim=1)  # Convert to class indices
        elif target.dim() == 4 and target.size(1) == 1:  # Single channel mask
            target_for_ce = target.squeeze(1)
        else:  # Already in the right format
            target_for_ce = target

        # Ensure target_for_ce has the right dtype for CrossEntropyLoss
        target_for_ce = target_for_ce.long()

        # Calculate individual losses
        ce_loss = self.ce_loss(pred, target_for_ce)
        print(f"  CE loss: {ce_loss.item():.4f}")

        # For region loss, if target is not one-hot but pred expects one-hot
        if (
            pred.dim() > target.dim()
            or (pred.dim() == target.dim() and pred.size(1) > 1)
        ) and (target.dim() == 3 or (target.dim() == 4 and target.size(1) == 1)):
            # Target needs to be converted to one-hot
            if target.dim() == 4 and target.size(1) == 1:
                target_cls = target.squeeze(1).long()
            else:
                target_cls = target.long()

            # Create one-hot encoding
            target_for_region = (
                F.one_hot(target_cls, num_classes=pred.size(1))
                .permute(0, 3, 1, 2)
                .float()
            )
        else:
            # Target is already in the right format for region loss
            target_for_region = target

        # For region loss, use torchmetrics Dice if not using Tversky
        if self.use_tversky:
            region_loss = self.region_loss(pred, target_for_region)
        else:
            # Use torchmetrics Dice for region loss
            if not hasattr(self, 'dice_metric'):
                # Lazy init: get number of classes from pred
                num_classes = pred.size(1)
                self.dice_metric = TorchMetricsDice(num_classes=num_classes)
            # For torchmetrics, pred: logits [B, C, H, W], target: class indices [B, H, W]
            if target_for_region.dim() == 4 and target_for_region.size(1) == 1:
                target_metric = target_for_region.squeeze(1).long()
            elif target_for_region.dim() == 4 and target_for_region.size(1) > 1:
                target_metric = target_for_region.argmax(dim=1)
            else:
                target_metric = target_for_region.long()
            region_loss = 1.0 - self.dice_metric(pred, target_metric)  # Dice is a score, so loss = 1 - score
        print(f"  Region loss: {region_loss.item():.4f}")

        # Store for debugging
        self.last_ce_loss = ce_loss.item()
        self.last_region_loss = region_loss.item()

        # Combine losses - ensure we have gradients
        total_loss = self.ce_weight * ce_loss + self.dice_weight * region_loss

        # Final sanity check for gradients
        if not total_loss.requires_grad:
            print("  Warning: total_loss has no gradients!")
        else:
            print(f"  Combined loss value: {total_loss.item():.4f} (with gradients)")

        return total_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better boundary delineation in segmentation.
    Focuses on pixels near object boundaries.
    """

    def __init__(self, theta0: float = 3, theta: float = 5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss.

        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth [B, C, H, W] (one-hot)
        """
        # Convert logits to probabilities
        pred_soft = F.softmax(pred, dim=1)

        # Compute distance transform for target
        target_dist = self._compute_distance_transform(target)

        # Compute boundary loss
        boundary_loss = pred_soft * target_dist

        return boundary_loss.mean()

    def _compute_distance_transform(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute distance transform for boundary loss.
        """
        from scipy.ndimage import distance_transform_edt

        batch_size, num_classes, height, width = target.shape
        dist_maps = torch.zeros_like(target)

        for b in range(batch_size):
            for c in range(num_classes):
                mask = target[b, c].detach().cpu().numpy()
                # Distance from boundary
                if mask.sum() > 0:
                    dist = distance_transform_edt(mask == 0) + distance_transform_edt(
                        mask == 1
                    )
                    dist_maps[b, c] = torch.from_numpy(dist).to(target.device)

        return dist_maps


class TorchMetricsDice(nn.Module):
    """
    Dice Score using torchmetrics implementation.
    """

    def __init__(self, num_classes: int = 2, threshold: float = 0.5, average: str = "macro"):
        super().__init__()
        self.dice = torchmetrics.classification.Dice(num_classes=num_classes, average=average, threshold=threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # For multi-class, pred should be logits [B, C, H, W], target as class indices [B, H, W]
        if pred.shape[1] == 1:
            # Binary case: apply sigmoid and threshold
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).long().squeeze(1)
        else:
            # Multi-class: take argmax
            pred = torch.argmax(pred, dim=1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        return self.dice(pred, target)


class TorchMetricsIoU(nn.Module):
    """
    IoU Score using torchmetrics implementation.
    """

    def __init__(self, num_classes: int = 2, threshold: float = 0.5, average: str = "macro"):
        super().__init__()
        self.iou = torchmetrics.classification.JaccardIndex(num_classes=num_classes, average=average, threshold=threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # For multi-class, pred should be logits [B, C, H, W], target as class indices [B, H, W]
        if pred.shape[1] == 1:
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).long().squeeze(1)
        else:
            pred = torch.argmax(pred, dim=1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        return self.iou(pred, target)


def calculate_class_weights(
    dataset, num_classes: int, method: str = "inverse_freq"
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        dataset: Dataset object with __getitem__ returning (image, mask)
        num_classes: Number of classes
        method: Method for calculating weights ('inverse_freq', 'effective_num')

    Returns:
        Tensor of class weights
    """
    print("Calculating class weights from dataset...")

    class_counts = torch.zeros(num_classes)
    total_pixels = 0

    # Sample a subset of the dataset for efficiency
    sample_size = min(len(dataset), 100)
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    for idx in indices:
        _, mask = dataset[idx]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu()

        # Count pixels for each class
        for class_id in range(num_classes):
            class_counts[class_id] += (mask == class_id).sum().item()

        total_pixels += mask.numel()

    if method == "inverse_freq":
        # Inverse frequency weighting
        class_freq = class_counts / total_pixels
        weights = 1.0 / (class_freq + 1e-7)
        weights = weights / weights.sum() * num_classes  # Normalize

    elif method == "effective_num":
        # Effective number of samples method
        beta = 0.999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # Normalize

    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights ({method}): {weights.tolist()}")

    return weights


def get_loss_function(
    loss_type: str, num_classes: int, class_weights: torch.Tensor = None, **kwargs
):
    """
    Factory function to get the appropriate loss function.

    Args:
        loss_type: Type of loss ('ce', 'dice', 'focal', 'combined', 'iou', 'tversky')
        num_classes: Number of classes
        class_weights: Optional class weights for balancing
        **kwargs: Additional parameters for specific loss functions

    Returns:
        Loss function instance
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "dice":
        return TorchMetricsDice(num_classes=num_classes)

    elif loss_type == "focal":
        return FocalLoss(**kwargs)

    elif loss_type == "iou":
        return TorchMetricsIoU(num_classes=num_classes)

    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)

    elif loss_type == "combined":
        # Default combined loss: CE + Dice
        return CombinedLoss(class_weights=class_weights, **kwargs)

    elif loss_type == "combined_focal":
        # Combined focal + dice
        return CombinedLoss(
            use_focal=True,
            focal_alpha=kwargs.get("focal_alpha", 1.0),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
            **kwargs,
        )

    elif loss_type == "combined_tversky":
        # Combined CE + Tversky
        return CombinedLoss(
            use_tversky=True,
            tversky_alpha=kwargs.get("tversky_alpha", 0.3),
            tversky_beta=kwargs.get("tversky_beta", 0.7),
            class_weights=class_weights,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss function optimized for medical segmentation.
    Combines multiple loss functions with learnable or fixed weights.
    """

    def __init__(
        self,
        loss_weights: dict = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        # Default loss weights
        if loss_weights is None:
            loss_weights = {"focal": 0.3, "dice": 0.4, "tversky": 0.2, "boundary": 0.1}

        self.loss_weights = loss_weights

        # Initialize individual loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = nn.Module()  # Placeholder, no-op
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary_loss = BoundaryLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss.

        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth masks [B, H, W] or [B, C, H, W]

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        # Convert target to proper format if needed
        if target.dim() == 3:  # [B, H, W]
            target_long = target.long()
            target_onehot = (
                F.one_hot(target_long, num_classes=pred.size(1))
                .permute(0, 3, 1, 2)
                .float()
            )
        else:  # [B, C, H, W]
            target_onehot = target.float()
            target_long = target.argmax(dim=1)

        # Compute individual losses
        if "focal" in self.loss_weights:
            losses["focal"] = (
                self.focal_loss(pred, target_long) * self.loss_weights["focal"]
            )

        if "dice" in self.loss_weights:
            losses["dice"] = (
                self.dice_loss(pred, target_onehot) * self.loss_weights["dice"]
            )

        if "tversky" in self.loss_weights:
            losses["tversky"] = (
                self.tversky_loss(pred, target_onehot) * self.loss_weights["tversky"]
            )

        if "boundary" in self.loss_weights:
            losses["boundary"] = (
                self.boundary_loss(pred, target_onehot) * self.loss_weights["boundary"]
            )

        if "ce" in self.loss_weights:
            losses["ce"] = self.ce_loss(pred, target_long) * self.loss_weights.get(
                "ce", 0.1
            )

        # Total loss
        losses["total"] = sum(losses.values())

        return losses


def compute_boundary_loss(
    pred: torch.Tensor, target: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute the boundary loss between predicted and target segmentation maps.

    Args:
        pred: Predicted segmentation maps [B, C, H, W]
        target: Ground truth segmentation maps [B, C, H, W]
        sigma: Gaussian kernel standard deviation for smoothing

    Returns:
        Boundary loss value
    """
    # Convert logits to probabilities
    if pred.dim() > target.dim():
        pred = F.softmax(pred, dim=1)

    # Compute boundaries using Sobel operator
    pred_boundary = compute_sobel_boundary(pred)
    target_boundary = compute_sobel_boundary(target)

    # Compute Gaussian kernel
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(pred.device)

    # Smooth boundaries
    pred_boundary = F.conv2d(
        pred_boundary,
        gaussian_kernel,
        padding=kernel_size // 2,
        groups=pred_boundary.size(1),
    )
    target_boundary = F.conv2d(
        target_boundary,
        gaussian_kernel,
        padding=kernel_size // 2,
        groups=target_boundary.size(1),
    )

    # Compute loss as the L2 distance between predicted and target boundaries
    loss = F.mse_loss(pred_boundary, target_boundary, reduction="mean")

    return loss


def compute_sobel_boundary(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the boundary map using the Sobel operator.

    Args:
        x: Input tensor [B, C, H, W]

    Returns:
        Boundary map [B, C, H, W]
    """
    # Sobel kernels for x and y gradients
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device
    )
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=x.device
    )

    # Reshape and normalize kernels
    sobel_x = sobel_x.view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_y.view(1, 1, 3, 3) / 8.0

    # Compute gradients
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))

    # Combine x and y gradients
    boundary = torch.sqrt(grad_x**2 + grad_y**2)

    return boundary


def create_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """
    Create a Gaussian kernel.

    Args:
        size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian

    Returns:
        Gaussian kernel tensor
    """
    # 1D Gaussian
    x = torch.arange(0, size, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * ((x - size // 2) / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize

    # 2D Gaussian
    kernel_2d = kernel_1d.view(1, -1) * kernel_1d.view(-1, 1)

    return kernel_2d


def calculate_class_weights(
    dataset, num_classes: int, method: str = "inverse_freq"
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        dataset: Dataset object with __getitem__ returning (image, mask)
        num_classes: Number of classes
        method: Method for calculating weights ('inverse_freq', 'effective_num')

    Returns:
        Tensor of class weights
    """
    print("Calculating class weights from dataset...")

    class_counts = torch.zeros(num_classes)
    total_pixels = 0

    # Sample a subset of the dataset for efficiency
    sample_size = min(len(dataset), 100)
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    for idx in indices:
        _, mask = dataset[idx]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu()

        # Count pixels for each class
        for class_id in range(num_classes):
            class_counts[class_id] += (mask == class_id).sum().item()

        total_pixels += mask.numel()

    if method == "inverse_freq":
        # Inverse frequency weighting
        class_freq = class_counts / total_pixels
        weights = 1.0 / (class_freq + 1e-7)
        weights = weights / weights.sum() * num_classes  # Normalize

    elif method == "effective_num":
        # Effective number of samples method
        beta = 0.999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # Normalize

    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights ({method}): {weights.tolist()}")

    return weights


def get_loss_function(
    loss_type: str, num_classes: int, class_weights: torch.Tensor = None, **kwargs
):
    """
    Factory function to get the appropriate loss function.

    Args:
        loss_type: Type of loss ('ce', 'dice', 'focal', 'combined', 'iou', 'tversky')
        num_classes: Number of classes
        class_weights: Optional class weights for balancing
        **kwargs: Additional parameters for specific loss functions

    Returns:
        Loss function instance
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "dice":
        return TorchMetricsDice(num_classes=num_classes)

    elif loss_type == "focal":
        return FocalLoss(**kwargs)

    elif loss_type == "iou":
        return TorchMetricsIoU(num_classes=num_classes)

    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)

    elif loss_type == "combined":
        # Default combined loss: CE + Dice
        return CombinedLoss(class_weights=class_weights, **kwargs)

    elif loss_type == "combined_focal":
        # Combined focal + dice
        return CombinedLoss(
            use_focal=True,
            focal_alpha=kwargs.get("focal_alpha", 1.0),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
            **kwargs,
        )

    elif loss_type == "combined_tversky":
        # Combined CE + Tversky
        return CombinedLoss(
            use_tversky=True,
            tversky_alpha=kwargs.get("tversky_alpha", 0.3),
            tversky_beta=kwargs.get("tversky_beta", 0.7),
            class_weights=class_weights,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss function optimized for medical segmentation.
    Combines multiple loss functions with learnable or fixed weights.
    """

    def __init__(
        self,
        loss_weights: dict = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        # Default loss weights
        if loss_weights is None:
            loss_weights = {"focal": 0.3, "dice": 0.4, "tversky": 0.2, "boundary": 0.1}

        self.loss_weights = loss_weights

        # Initialize individual loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = nn.Module()  # Placeholder, no-op
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary_loss = BoundaryLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined loss.

        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth masks [B, H, W] or [B, C, H, W]

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        # Convert target to proper format if needed
        if target.dim() == 3:  # [B, H, W]
            target_long = target.long()
            target_onehot = (
                F.one_hot(target_long, num_classes=pred.size(1))
                .permute(0, 3, 1, 2)
                .float()
            )
        else:  # [B, C, H, W]
            target_onehot = target.float()
            target_long = target.argmax(dim=1)

        # Compute individual losses
        if "focal" in self.loss_weights:
            losses["focal"] = (
                self.focal_loss(pred, target_long) * self.loss_weights["focal"]
            )

        if "dice" in self.loss_weights:
            losses["dice"] = (
                self.dice_loss(pred, target_onehot) * self.loss_weights["dice"]
            )

        if "tversky" in self.loss_weights:
            losses["tversky"] = (
                self.tversky_loss(pred, target_onehot) * self.loss_weights["tversky"]
            )

        if "boundary" in self.loss_weights:
            losses["boundary"] = (
                self.boundary_loss(pred, target_onehot) * self.loss_weights["boundary"]
            )

        if "ce" in self.loss_weights:
            losses["ce"] = self.ce_loss(pred, target_long) * self.loss_weights.get(
                "ce", 0.1
            )

        # Total loss
        losses["total"] = sum(losses.values())

        return losses
