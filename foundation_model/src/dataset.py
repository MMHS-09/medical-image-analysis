import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple, Optional
import glob


class MedicalImageDataset(Dataset):
    """
    Unified dataset class for both classification and segmentation tasks
    """
    
    def __init__(
        self,
        data_path: str,
        task: str,
        dataset_name: str,
        classes: Optional[List[str]] = None,
        split: str = "train",
        transform=None,
        image_size: Tuple[int, int] = (224, 224),
        max_samples_per_class: Optional[int] = None,
        max_samples_per_dataset: Optional[int] = None,
        data_fraction: float = 1.0
    ):
        self.data_path = data_path
        self.task = task
        self.dataset_name = dataset_name
        self.classes = classes
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.max_samples_per_class = max_samples_per_class
        self.max_samples_per_dataset = max_samples_per_dataset
        self.data_fraction = data_fraction
        
        if task == "classification":
            self.samples = self._load_classification_samples()
        elif task == "segmentation":
            self.samples = self._load_segmentation_samples()
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _load_classification_samples(self) -> List[Dict]:
        """Load classification samples with optional limits"""
        samples = []
        
        # Handle different data path scenarios
        # First, try if data_path is a direct path to the dataset (for finetuning)
        if os.path.exists(os.path.join(self.data_path, self.classes[0] if self.classes else "dummy")):
            dataset_path = self.data_path
        else:
            # Default path construction (for training)
            dataset_path = os.path.join(self.data_path, "classification", self.dataset_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_path, ext)))
                image_files.extend(glob.glob(os.path.join(class_path, ext.upper())))
            
            # Apply data fraction limit
            if self.data_fraction < 1.0:
                num_files = int(len(image_files) * self.data_fraction)
                image_files = image_files[:num_files]
            
            # Apply max samples per class limit
            if self.max_samples_per_class is not None:
                image_files = image_files[:self.max_samples_per_class]
            
            for img_path in image_files:
                samples.append({
                    'image_path': img_path,
                    'label': class_idx,
                    'class_name': class_name
                })
        
        # Shuffle samples to ensure randomness when limiting
        np.random.shuffle(samples)
        
        return samples
    
    def _load_segmentation_samples(self) -> List[Dict]:
        """Load segmentation samples with optional limits"""
        samples = []
        
        # Handle different data path scenarios
        # First, check if data_path contains image files directly (for finetuning)
        test_files = glob.glob(os.path.join(self.data_path, "*.png")) + glob.glob(os.path.join(self.data_path, "*.jpg"))
        if test_files:
            dataset_path = self.data_path
        else:
            # Default path construction (for training)
            dataset_path = os.path.join(self.data_path, "segmentation", self.dataset_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find all image and mask pairs
        image_files = []
        mask_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            # Images (not containing 'mask' in filename)
            all_files = glob.glob(os.path.join(dataset_path, ext))
            all_files.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
            
            for file in all_files:
                if 'mask' in os.path.basename(file).lower():
                    mask_files.append(file)
                elif not file.endswith(':Zone.Identifier'):  # Skip zone identifier files
                    image_files.append(file)
        
        # Match images with masks
        for img_path in image_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Find corresponding mask
            mask_path = None
            for mask_file in mask_files:
                mask_name = os.path.splitext(os.path.basename(mask_file))[0]
                if img_name in mask_name or mask_name.replace('_mask', '') == img_name:
                    mask_path = mask_file
                    break
            
            if mask_path:
                samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path
                })
        
        # Apply data limits
        if self.data_fraction < 1.0:
            num_samples = int(len(samples) * self.data_fraction)
            samples = samples[:num_samples]
        
        if self.max_samples_per_dataset is not None:
            samples = samples[:self.max_samples_per_dataset]
        
        # Shuffle samples to ensure randomness when limiting
        np.random.shuffle(samples)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.task == "classification":
            return self._get_classification_item(sample)
        else:
            return self._get_segmentation_item(sample)
    
    def _get_classification_item(self, sample):
        """Get classification item"""
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = cv2.resize(image, self.image_size)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'image': image,
            'label': label,
            'dataset_name': self.dataset_name,
            'task': self.task
        }
    
    def _get_segmentation_item(self, sample):
        """Get segmentation item"""
        # Load image and mask
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Resize both image and mask to target size first to ensure consistency
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # Ensure mask is long tensor for CrossEntropyLoss
            mask = mask.long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'dataset_name': self.dataset_name,
            'task': self.task
        }


def get_transforms(image_size: Tuple[int, int], augment: bool = True) -> Dict[str, A.Compose]:
    """Get data transforms for training and validation"""
    
    if augment:
        train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    val_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return {'train': train_transform, 'val': val_transform}


def create_dataloaders(
    config: dict,
    task: str,
    dataset_name: str,
    classes: Optional[List[str]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.8,
    data_path: Optional[str] = None
) -> Dict[str, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Use custom data_path if provided, otherwise use config
    if data_path is None:
        data_path = config['paths']['data_root']
    
    image_size = tuple(config['data']['image_size'])
    
    # Get data limiting parameters from config
    max_samples_per_class = config['data'].get('max_samples_per_class', None)
    max_samples_per_dataset = config['data'].get('max_samples_per_dataset', None)
    data_fraction = config['data'].get('data_fraction', 1.0)
    
    transforms = get_transforms(image_size, augment=True)
    
    # Create full dataset to get sample indices
    full_dataset = MedicalImageDataset(
        data_path=data_path,
        task=task,
        dataset_name=dataset_name,
        classes=classes,
        transform=None,  # No transform initially
        image_size=image_size,
        max_samples_per_class=max_samples_per_class,
        max_samples_per_dataset=max_samples_per_dataset,
        data_fraction=data_fraction
    )
    
    # Split dataset indices
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Get train and validation indices
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets with appropriate transforms
    train_dataset = MedicalImageDataset(
        data_path=data_path,
        task=task,
        dataset_name=dataset_name,
        classes=classes,
        transform=transforms['train'],
        image_size=image_size,
        max_samples_per_class=max_samples_per_class,
        max_samples_per_dataset=max_samples_per_dataset,
        data_fraction=data_fraction
    )
    
    val_dataset = MedicalImageDataset(
        data_path=data_path,
        task=task,
        dataset_name=dataset_name,
        classes=classes,
        transform=transforms['val'],
        image_size=image_size,
        max_samples_per_class=max_samples_per_class,
        max_samples_per_dataset=max_samples_per_dataset,
        data_fraction=data_fraction
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {'train': train_loader, 'val': val_loader}


def create_test_dataloader(
    config: dict,
    task: str,
    dataset_name: str,
    classes: Optional[List[str]] = None,
    batch_size: int = 16,
    num_workers: int = 4
) -> DataLoader:
    """Create test dataloader for inference"""
    
    data_path = config['paths']['data_root']
    image_size = tuple(config['data']['image_size'])
    
    # Get data limiting parameters from config (for testing purposes)
    max_samples_per_class = config['data'].get('max_samples_per_class', None)
    max_samples_per_dataset = config['data'].get('max_samples_per_dataset', None)
    data_fraction = config['data'].get('data_fraction', 1.0)
    
    transforms = get_transforms(image_size, augment=False)
    
    test_dataset = MedicalImageDataset(
        data_path=data_path,
        task=task,
        dataset_name=dataset_name,
        classes=classes,
        transform=transforms['val'],
        image_size=image_size,
        max_samples_per_class=max_samples_per_class,
        max_samples_per_dataset=max_samples_per_dataset,
        data_fraction=data_fraction
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader