import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt


class MetricsCalculator:
    """Calculate metrics for both classification and segmentation tasks"""
    
    def __init__(self, task: str, num_classes: int, class_names: Optional[List[str]] = None):
        self.task = task
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.running_loss = 0.0
        self.num_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with new batch"""
        self.running_loss += loss
        self.num_samples += targets.size(0)
        
        if self.task == "classification":
            # Get predicted classes
            pred_classes = predictions.argmax(dim=1)
            self.predictions.extend(pred_classes.cpu().numpy())
            self.targets.extend(targets.cpu().numpy())
        
        elif self.task == "segmentation":
            # Get predicted classes for each pixel
            pred_classes = predictions.argmax(dim=1)
            self.predictions.extend(pred_classes.cpu().numpy().flatten())
            self.targets.extend(targets.cpu().numpy().flatten())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics"""
        if not self.predictions:
            return {"loss": self.running_loss / max(self.num_samples, 1)}
        
        avg_loss = self.running_loss / self.num_samples
        
        if self.task == "classification":
            return self._compute_classification_metrics(avg_loss)
        elif self.task == "segmentation":
            return self._compute_segmentation_metrics(avg_loss)
    
    def _compute_classification_metrics(self, avg_loss: float) -> Dict[str, float]:
        """Compute classification metrics"""
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f"precision_{class_name}"] = precision_per_class[i]
                metrics[f"recall_{class_name}"] = recall_per_class[i]
                metrics[f"f1_{class_name}"] = f1_per_class[i]
        
        return metrics
    
    def _compute_segmentation_metrics(self, avg_loss: float) -> Dict[str, float]:
        """Compute segmentation metrics"""
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # IoU calculation
        iou_scores = []
        dice_scores = []
        hausdorff_distances = []
        assd_scores = []
        
        # Need to reshape for surface distance calculations
        # Assuming we can reconstruct the original image dimensions
        # For now, we'll calculate these metrics on smaller patches or skip if too memory intensive
        calculate_surface_metrics = len(y_true) < 1000000  # Only for smaller images
        
        for class_idx in range(self.num_classes):
            # Binary masks for current class
            true_mask = (y_true == class_idx)
            pred_mask = (y_pred == class_idx)
            
            # IoU
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            iou = intersection / (union + 1e-7)
            iou_scores.append(iou)
            
            # Dice coefficient
            dice = (2.0 * intersection) / (true_mask.sum() + pred_mask.sum() + 1e-7)
            dice_scores.append(dice)
            
            # Advanced surface metrics (only for binary segmentation and manageable sizes)
            if calculate_surface_metrics and self.num_classes == 2 and class_idx == 1:
                try:
                    # Try to estimate image dimensions (assuming square images)
                    img_size = int(np.sqrt(len(y_true)))
                    if img_size * img_size == len(y_true):
                        true_mask_2d = true_mask.reshape(img_size, img_size)
                        pred_mask_2d = pred_mask.reshape(img_size, img_size)
                        
                        hd = hausdorff_distance(pred_mask_2d, true_mask_2d)
                        assd = average_symmetric_surface_distance(pred_mask_2d, true_mask_2d)
                        
                        # Handle infinite values
                        if np.isfinite(hd):
                            hausdorff_distances.append(hd)
                        if np.isfinite(assd):
                            assd_scores.append(assd)
                except:
                    # Skip surface metrics if calculation fails
                    pass
        
        mean_iou = np.mean(iou_scores)
        mean_dice = np.mean(dice_scores)
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice
        }
        
        # Add surface distance metrics if available
        if hausdorff_distances:
            metrics["hausdorff_distance"] = np.mean(hausdorff_distances)
        if assd_scores:
            metrics["avg_symmetric_surface_distance"] = np.mean(assd_scores)
        
        # Per-class IoU and Dice
        for i, class_name in enumerate(self.class_names):
            if i < len(iou_scores):
                metrics[f"iou_{class_name}"] = iou_scores[i]
                metrics[f"dice_{class_name}"] = dice_scores[i]
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix (for classification tasks)"""
        if self.task != "classification" or not self.predictions:
            return None
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Calculate Dice coefficient"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    dice_scores = []
    for class_idx in range(num_classes):
        pred_class = pred[:, class_idx]
        target_class = target_one_hot[:, class_idx]
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2.0 * intersection) / (union + 1e-7)
        dice_scores.append(dice)
    
    return torch.stack(dice_scores).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Calculate IoU score"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    iou_scores = []
    for class_idx in range(num_classes):
        pred_class = pred[:, class_idx]
        target_class = target_one_hot[:, class_idx]
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection
        
        iou = intersection / (union + 1e-7)
        iou_scores.append(iou)
    
    return torch.stack(iou_scores).mean()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate pixel accuracy for segmentation"""
    pred_classes = pred.argmax(dim=1)
    correct = (pred_classes == target).float()
    return correct.mean()


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def create_metrics_calculator(task: str, num_classes: int, class_names: Optional[List[str]] = None) -> MetricsCalculator:
    """Create metrics calculator based on task"""
    return MetricsCalculator(task=task, num_classes=num_classes, class_names=class_names)


def hausdorff_distance(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Calculate Hausdorff distance between two binary masks"""
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return 0.0
    elif pred_mask.sum() == 0 or true_mask.sum() == 0:
        return float('inf')
    
    # Get coordinates of pixels
    pred_coords = np.argwhere(pred_mask)
    true_coords = np.argwhere(true_mask)
    
    if len(pred_coords) == 0 or len(true_coords) == 0:
        return float('inf')
    
    # Calculate directed Hausdorff distances
    d1 = directed_hausdorff(pred_coords, true_coords)[0]
    d2 = directed_hausdorff(true_coords, pred_coords)[0]
    
    return max(d1, d2)


def average_symmetric_surface_distance(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Calculate Average Symmetric Surface Distance (ASSD)"""
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return 0.0
    elif pred_mask.sum() == 0 or true_mask.sum() == 0:
        return float('inf')
    
    # Calculate distance transforms
    pred_dt = distance_transform_edt(~pred_mask.astype(bool))
    true_dt = distance_transform_edt(~true_mask.astype(bool))
    
    # Get surface pixels
    pred_surface = pred_mask & ~distance_transform_edt(pred_mask, return_distances=False, return_indices=False)
    true_surface = true_mask & ~distance_transform_edt(true_mask, return_distances=False, return_indices=False)
    
    if pred_surface.sum() == 0 or true_surface.sum() == 0:
        return float('inf')
    
    # Calculate distances from surfaces
    pred_to_true_dist = pred_dt[pred_surface].mean()
    true_to_pred_dist = true_dt[true_surface].mean()
    
    return (pred_to_true_dist + true_to_pred_dist) / 2.0