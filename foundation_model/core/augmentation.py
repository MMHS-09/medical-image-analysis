#!/usr/bin/env python3
"""
Advanced Data Augmentation for Medical Image Segmentation

This module provides sophisticated augmentation techniques specifically designed
for medical image segmentation to improve model generalization and performance.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MedicalSegmentationAugmentation:
    """
    Advanced augmentation pipeline for medical image segmentation.
    Includes both geometric and intensity transformations with careful
    handling of image-mask pairs.
    """

    def __init__(
        self,
        img_size: int = 256,
        geometric_aug_prob: float = 0.8,
        intensity_aug_prob: float = 0.7,
        noise_aug_prob: float = 0.5,
        elastic_deform_prob: float = 0.6,
        rotation_range: int = 20,
        translation_range: float = 0.1,
        scale_range: tuple = (0.85, 1.15),
        brightness_range: tuple = (0.8, 1.2),
        contrast_range: tuple = (0.8, 1.2),
        gamma_range: tuple = (0.8, 1.2),
    ):

        self.img_size = img_size
        self.geometric_aug_prob = geometric_aug_prob
        self.intensity_aug_prob = intensity_aug_prob
        self.noise_aug_prob = noise_aug_prob
        self.elastic_deform_prob = elastic_deform_prob

        # Determine gamma_limit for RandomGamma: use maximum of range, ensure >=1
        if isinstance(gamma_range, tuple) and len(gamma_range) == 2:
            gamma_limit_val = max(max(gamma_range), 1.0)
        else:
            gamma_limit_val = gamma_range if isinstance(gamma_range, (int, float)) and gamma_range >= 1.0 else 1.0

        # Create albumentations transform pipeline
        self.transform = A.Compose(
            [
                # Geometric transformations
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.3),
                    ],
                    p=geometric_aug_prob,
                ),
                # Use Affine transform to replace ShiftScaleRotate
                A.Affine(
                    translate_percent={"x": translation_range, "y": translation_range},
                    scale=scale_range,
                    rotate=rotation_range,
                    interpolation=1,
                    border_mode=0,
                    p=geometric_aug_prob,
                ),
                # Elastic deformation
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    border_mode=0,
                    p=elastic_deform_prob,
                ),
                # Intensity transformations (only applied to image, not mask)
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=brightness_range,
                            contrast_limit=contrast_range,
                            p=1.0,
                        ),
                        A.RandomGamma(gamma_limit=gamma_limit_val, p=1.0),
                        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=intensity_aug_prob,
                ),
                # Noise and blur
                A.OneOf(
                    [
                        A.GaussNoise(p=1.0),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                        A.GaussianBlur(blur_limit=(1, 3), p=1.0),
                    ],
                    p=noise_aug_prob,
                ),
                # Resize to target size
                A.Resize(img_size, img_size, interpolation=1, p=1.0),
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0
                ),
                # Convert to tensor
                ToTensorV2(p=1.0),
            ]
        )

        # Validation transform (no augmentation)
        self.val_transform = A.Compose(
            [
                A.Resize(img_size, img_size, interpolation=1, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0
                ),
                ToTensorV2(p=1.0),
            ]
        )

    def __call__(self, image, mask, is_training=True):
        """
        Apply augmentation to image-mask pair.

        Args:
            image: PIL Image or numpy array
            mask: PIL Image or numpy array
            is_training: Whether to apply training augmentations

        Returns:
            Augmented image and mask tensors
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Apply appropriate transform
        if is_training:
            transformed = self.transform(image=image, mask=mask)
        else:
            transformed = self.val_transform(image=image, mask=mask)

        return transformed["image"], transformed["mask"]


class ImprovedTransforms:
    """
    Traditional PyTorch transforms with medical image optimizations.
    """

    @staticmethod
    def get_training_transforms(img_size=256):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def get_validation_transforms(img_size=256):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def get_mask_transforms(img_size=256):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )


class CutMix:
    """
    CutMix augmentation adapted for segmentation tasks.
    """

    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch_images, batch_masks):
        if random.random() > self.p:
            return batch_images, batch_masks

        batch_size = batch_images.size(0)
        indices = torch.randperm(batch_size)

        lam = np.random.beta(self.alpha, self.alpha)

        # Generate random bounding box
        H, W = batch_images.shape[2:]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        batch_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[
            indices, :, bby1:bby2, bbx1:bbx2
        ]
        batch_masks[:, :, bby1:bby2, bbx1:bbx2] = batch_masks[
            indices, :, bby1:bby2, bbx1:bbx2
        ]

        return batch_images, batch_masks


class MixUp:
    """
    MixUp augmentation for segmentation.
    """

    def __init__(self, alpha=0.2, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch_images, batch_masks):
        if random.random() > self.p:
            return batch_images, batch_masks

        batch_size = batch_images.size(0)
        indices = torch.randperm(batch_size)

        lam = np.random.beta(self.alpha, self.alpha)

        mixed_images = lam * batch_images + (1 - lam) * batch_images[indices]
        mixed_masks = lam * batch_masks + (1 - lam) * batch_masks[indices]

        return mixed_images, mixed_masks


def create_augmentations(task="segmentation", img_size=256, intensity="medium"):
    """
    Create appropriate augmentations based on task and desired intensity.

    Args:
        task: 'segmentation' or 'classification'
        img_size: Output image size
        intensity: 'low', 'medium', or 'high'

    Returns:
        Dictionary of transforms or augmentation instances
    """
    # Set augmentation parameters based on intensity
    if intensity == "low":
        geometric_prob = 0.5
        intensity_prob = 0.3
        noise_prob = 0.2
        elastic_prob = 0.3
        rotation = 10
        translation = 0.05
        scale = (0.9, 1.1)
    elif intensity == "high":
        geometric_prob = 0.9
        intensity_prob = 0.8
        noise_prob = 0.7
        elastic_prob = 0.7
        rotation = 30
        translation = 0.15
        scale = (0.8, 1.2)
    else:  # medium (default)
        geometric_prob = 0.8
        intensity_prob = 0.7
        noise_prob = 0.5
        elastic_prob = 0.6
        rotation = 20
        translation = 0.1
        scale = (0.85, 1.15)

    if task == "segmentation":
        return MedicalSegmentationAugmentation(
            img_size=img_size,
            geometric_aug_prob=geometric_prob,
            intensity_aug_prob=intensity_prob,
            noise_aug_prob=noise_prob,
            elastic_deform_prob=elastic_prob,
            rotation_range=rotation,
            translation_range=translation,
            scale_range=scale,
        )
    else:  # classification
        return {
            "train": ImprovedTransforms.get_training_transforms(img_size),
            "valid": ImprovedTransforms.get_validation_transforms(img_size),
        }
