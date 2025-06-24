import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from typing import List, Tuple, Dict
import glob
import numpy as np
from core.augmentation import ImprovedTransforms, MedicalSegmentationAugmentation
from torch.utils.data import WeightedRandomSampler

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]


class ClassificationDataset(Dataset):
    """Enhanced classification dataset with prompt generation"""

    def __init__(
        self,
        root_dir: str,
        transform=None,
        limit_samples: int = -1,
        modality: str = "MRI",
        limit_samples_fraction: float = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.modality = modality.upper()

        # Use ImageFolder to get samples and classes
        self.dataset = datasets.ImageFolder(root_dir)
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)

        # First try to use limit_samples_fraction if provided
        if limit_samples_fraction is not None and 0.0 < limit_samples_fraction < 1.0:
            num_to_keep = int(len(self.dataset) * limit_samples_fraction)
            if num_to_keep < len(self.dataset):
                indices = random.sample(range(len(self.dataset)), num_to_keep)
                self.samples = [self.dataset.samples[i] for i in indices]
            else:
                self.samples = self.dataset.samples
        # Otherwise use the limit_samples parameter
        elif limit_samples > 0 and limit_samples < len(self.dataset):
            indices = random.sample(range(len(self.dataset)), limit_samples)
            self.samples = [self.dataset.samples[i] for i in indices]
        else:
            self.samples = self.dataset.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Generate prompt based on modality and task
        class_name = self.classes[label]
        prompt = f"Classify this {self.modality} image. Determine if it shows {class_name} or other conditions."

        return {
            "image": image,
            "label": label,
            "prompt": prompt,
            "task": "classification",
            "modality": self.modality,
            "num_classes": self.num_classes,
        }


class SegmentationDataset(Dataset):
    """Segmentation dataset with mask loading and prompt generation"""

    def __init__(
        self,
        root_dir: str,
        transform=None,
        mask_transform=None,
        limit_samples: int = -1,
        modality: str = "MRI",
        mask_suffix: str = "_mask",
        limit_samples_fraction: float = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.modality = modality.upper()
        self.mask_suffix = mask_suffix
        self.limit_samples_fraction = limit_samples_fraction

        # Find all image files with supported extensions
        self.image_paths = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            patterns = [
                os.path.join(root_dir, f"*{ext}"),
                os.path.join(root_dir, "*", f"*{ext}"),
                os.path.join(root_dir, "*", "*", f"*{ext}"),
            ]
            for pattern in patterns:
                self.image_paths.extend(glob.glob(pattern))

        # Filter out mask files from image paths
        self.image_paths = [
            p for p in self.image_paths if mask_suffix not in os.path.basename(p)
        ]

        # Verify corresponding masks exist
        valid_pairs = []
        for img_path in self.image_paths:
            mask_path = self._get_mask_path(img_path)
            if os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))

        self.image_mask_pairs = valid_pairs

        # First try to use limit_samples_fraction if provided
        if (
            self.limit_samples_fraction is not None
            and 0.0 < self.limit_samples_fraction < 1.0
        ):
            num_to_keep = int(len(self.image_mask_pairs) * self.limit_samples_fraction)
            if num_to_keep < len(self.image_mask_pairs):
                self.image_mask_pairs = random.sample(
                    self.image_mask_pairs, num_to_keep
                )
        # Otherwise use the limit_samples parameter (for backward compatibility)
        elif limit_samples > 0 and limit_samples < len(self.image_mask_pairs):
            self.image_mask_pairs = random.sample(self.image_mask_pairs, limit_samples)

    def _get_mask_path(self, img_path: str) -> str:
        """Generate corresponding mask path for an image"""
        # Get the directory, filename and extension
        dir_path = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        # Try same extension first
        mask_path = os.path.join(dir_path, f"{name}{self.mask_suffix}{ext}")
        if os.path.exists(mask_path):
            return mask_path

        # Try other supported extensions
        for other_ext in SUPPORTED_IMAGE_EXTENSIONS:
            mask_path = os.path.join(dir_path, f"{name}{self.mask_suffix}{other_ext}")
            if os.path.exists(mask_path):
                return mask_path

        # Default to original extension if none found
        return os.path.join(dir_path, f"{name}{self.mask_suffix}{ext}")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]

        try:
            # Load image and mask with proper error handling
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Try to load with OpenCV as fallback
                import cv2
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    raise ValueError(f"Failed to load image at {img_path}")
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_cv)
                
            try:
                mask = Image.open(mask_path).convert("L")  # Grayscale for mask
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                # Try to load with OpenCV as fallback
                import cv2
                mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_cv is None:
                    raise ValueError(f"Failed to load mask at {mask_path}")
                mask = Image.fromarray(mask_cv)
        except Exception as e:
            print(f"Fatal error loading image-mask pair: {e}")
            # Return a simple black image and mask as fallback
            image = Image.new("RGB", (256, 256), color="black")
            mask = Image.new("L", (256, 256), color=0)

        # Apply augmentations: handle albumentations Compose or custom transforms
        applied_joint = False
        if self.transform:
            # Detect albumentations Compose
            try:
                from albumentations.core.composition import Compose
                is_alb = isinstance(self.transform, Compose)
            except ImportError:
                is_alb = False
            if is_alb:
                # Convert PIL to numpy arrays
                img_np = np.array(image) if isinstance(image, Image.Image) else image
                mask_np = np.array(mask) if isinstance(mask, Image.Image) else mask
                augmented = self.transform(image=img_np, mask=mask_np)
                image = augmented['image']
                mask = augmented['mask']
                applied_joint = True
            else:
                # Custom transform: try joint then named args then image-only
                try:
                    image, mask = self.transform(image, mask)
                    applied_joint = True
                except Exception:
                    try:
                        image, mask = self.transform(image=image, mask=mask)
                        applied_joint = True
                    except Exception:
                        image = self.transform(image)
        # Apply mask-only transform if joint not applied
        if not applied_joint:
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = transforms.ToTensor()(mask)

        # Generate prompt for segmentation
        prompt = f"Segment this {self.modality} image. Identify and outline the regions of interest or abnormalities."

        return {
            "image": image,
            "mask": mask,
            "prompt": prompt,
            "task": "segmentation",
            "modality": self.modality,
            "num_classes": 2,  # Binary segmentation for now
        }


class MultiDatasetLoader:
    """Handles loading multiple datasets for foundation model training"""

    def __init__(
        self,
        dataset_configs: List[Dict],
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
        img_size: int = 64,
    ):
        """
        dataset_configs: List of dictionaries with dataset configuration
        Each config should have:
        - 'path': path to dataset
        - 'task': 'classification' or 'segmentation'
        - 'modality': 'MRI', 'CT', 'XRAY', etc.
        - 'limit_samples': number of samples to load (-1 for all)
        - Additional task-specific parameters
        """
        self.dataset_configs = dataset_configs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Store image size for consistent transforms
        self.img_size = img_size
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),  # Use configurable image size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size, self.img_size)
                ),  # Use configurable image size for masks too
                transforms.ToTensor(),
            ]
        )

        self.datasets = []
        self.dataloaders = []
        self._create_datasets()

    def _create_datasets(self):
        """Create datasets and dataloaders from configurations"""
        for config in self.dataset_configs:
            task = config["task"].lower()

            # Get limit_samples_fraction if present
            limit_samples_fraction = config.get("limit_samples_fraction", None)

            # For legacy support - convert old-style limit_samples to fraction if needed
            limit_samples = config.get("limit_samples", -1)

            if task == "classification":
                train_dir = os.path.join(config["path"], "Training")
                test_dir = os.path.join(config["path"], "Testing")

                # Classification augmentations and balancing sampler
                # Use unified image size from loader
                train_transform = ImprovedTransforms.get_training_transforms(self.img_size)
                val_transform = ImprovedTransforms.get_validation_transforms(self.img_size)

                train_dataset = ClassificationDataset(
                    root_dir=train_dir,
                    transform=train_transform,
                    limit_samples=limit_samples,
                    modality=config.get("modality", "MRI"),
                    limit_samples_fraction=limit_samples_fraction,
                )

                # For test, use validation transforms
                test_fraction = limit_samples_fraction * 0.2 if limit_samples_fraction is not None else None

                test_dataset = ClassificationDataset(
                    root_dir=test_dir,
                    transform=val_transform,
                    limit_samples=limit_samples // 5 if limit_samples > 0 else -1,
                    modality=config.get("modality", "MRI"),
                    limit_samples_fraction=test_fraction,
                )

                # Build a weighted sampler for balancing classes
                labels = [label for _, label in train_dataset.samples]
                class_counts = np.bincount(labels)
                class_weights = 1.0 / (class_counts + 1e-8)
                sample_weights = [class_weights[l] for l in labels]
                sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )

            elif task == "segmentation":
                train_dir = os.path.join(config["path"], "Training")
                test_dir = os.path.join(config["path"], "Testing")
                # Segmentation augmentation
                # Use unified image size for segmentation augmentations
                seg_aug = MedicalSegmentationAugmentation(img_size=self.img_size)
                train_transform = seg_aug
                mask_transform = seg_aug
                val_transform = seg_aug.val_transform

                train_dataset = SegmentationDataset(
                    root_dir=train_dir,
                    transform=train_transform,
                    mask_transform=mask_transform,
                    limit_samples=limit_samples,
                    modality=config.get("modality", "MRI"),
                    mask_suffix=config.get("mask_suffix", "_mask"),
                    limit_samples_fraction=limit_samples_fraction,
                )
                # For test, use a fraction of samples if limit_samples_fraction is set
                test_fraction = limit_samples_fraction * 0.2 if limit_samples_fraction is not None else None

                test_dataset = SegmentationDataset(
                    root_dir=test_dir,
                    transform=val_transform,
                    mask_transform=mask_transform,
                    limit_samples=limit_samples // 5 if limit_samples > 0 else -1,
                    modality=config.get("modality", "MRI"),
                    mask_suffix=config.get("mask_suffix", "_mask"),
                    limit_samples_fraction=test_fraction,
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )

            else:
                raise ValueError(f"Unsupported task: {task}")

            # Create dataloaders
            dataset_info = {
                "config": config,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "train_loader": train_loader,
                "test_loader": test_loader,
                "num_classes": getattr(train_dataset, "num_classes", 2),
            }

            self.datasets.append(dataset_info)
            self.dataloaders.append((train_loader, test_loader))

    def _collate_fn(self, batch):
        """Custom collate function to handle dictionary batches"""
        if not batch:
            return {}

        # Get all keys from first item
        keys = batch[0].keys()
        result = {}

        for key in keys:
            if key in ["image", "mask"]:
                # Stack tensors
                result[key] = torch.stack([item[key] for item in batch])
            elif key == "label":
                # Stack labels if they exist
                result[key] = torch.tensor([item[key] for item in batch])
            elif key == "num_classes":
                # Take the first value since all samples in batch should have same num_classes
                result[key] = batch[0][key]
            else:
                # Keep as list for other items
                result[key] = [item[key] for item in batch]

        return result

    def get_dataset_info(self, idx: int) -> Dict:
        """Get information about a specific dataset"""
        return self.datasets[idx]

    def get_dataloader(self, idx: int) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders for a specific dataset"""
        return self.dataloaders[idx]

    def __len__(self):
        return len(self.datasets)
