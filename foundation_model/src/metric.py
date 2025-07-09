import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
        pixel_accuracy = accuracy_score(y_true, y_pred)
        
        # IoU calculation
        iou_scores = []
        dice_scores = []
        hausdorff_distances_per_class = []
        assd_scores = []
        
        # More permissive surface metrics calculation
        calculate_surface_metrics = len(y_true) < 5000000  # Further increased threshold
        
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
            
            # Enhanced Hausdorff Distance calculation for each class
            if calculate_surface_metrics:
                try:
                    # Try multiple possible image dimensions
                    total_pixels = len(y_true)
                    possible_dims = []
                    
                    # Common image dimensions (expanded list)
                    common_dims = [16, 32, 64, 96, 128, 192, 224, 256, 384, 512, 768, 1024]
                    
                    # Try square dimensions
                    for dim in common_dims:
                        if dim * dim == total_pixels:
                            possible_dims.append((dim, dim))
                    
                    # Try rectangular dimensions
                    for height in common_dims:
                        for width in common_dims:
                            if height * width == total_pixels:
                                possible_dims.append((height, width))
                                if len(possible_dims) >= 5:  # Limit to avoid too many combinations
                                    break
                        if len(possible_dims) >= 5:
                            break
                    
                    # Try factor-based dimensions
                    if not possible_dims:
                        for factor in range(2, int(np.sqrt(total_pixels)) + 1):
                            if total_pixels % factor == 0:
                                width = factor
                                height = total_pixels // factor
                                possible_dims.append((height, width))
                                if len(possible_dims) >= 5:
                                    break
                    
                    # If we still can't determine dimensions, try square root
                    if not possible_dims:
                        img_size = int(np.sqrt(total_pixels))
                        if img_size * img_size == total_pixels:
                            possible_dims.append((img_size, img_size))
                    
                    # Use the first valid dimension
                    if possible_dims:
                        height, width = possible_dims[0]
                        true_mask_2d = true_mask.reshape(height, width)
                        pred_mask_2d = pred_mask.reshape(height, width)
                        
                        # Calculate Hausdorff distance for all classes (including background)
                        hd = hausdorff_distance(pred_mask_2d, true_mask_2d)
                        
                        # Handle infinite and zero values more permissively
                        if np.isfinite(hd):
                            hausdorff_distances_per_class.append(hd)
                            
                            # Calculate ASSD for foreground classes
                            if class_idx > 0:  # Skip background class for ASSD
                                assd = average_symmetric_surface_distance(pred_mask_2d, true_mask_2d)
                                if np.isfinite(assd):
                                    assd_scores.append(assd)
                                
                except Exception as e:
                    # Continue to next class instead of skipping all surface metrics
                    pass
        
        # Calculate mean metrics
        mean_iou = np.mean(iou_scores)
        mean_dice = np.mean(dice_scores)
        
        metrics = {
            "loss": avg_loss,
            "pixel_accuracy": pixel_accuracy,  # Always include pixel accuracy
            "accuracy": pixel_accuracy,  # Keep backward compatibility
            "mean_iou": mean_iou,
            "mean_dice": mean_dice
        }
        
        # Add Hausdorff Distance metrics (more permissive)
        if hausdorff_distances_per_class:
            # Filter out invalid values before calculating statistics
            valid_hausdorff = [h for h in hausdorff_distances_per_class if h >= 0]
            if valid_hausdorff:
                metrics["mean_hausdorff_distance"] = np.mean(valid_hausdorff)
                metrics["max_hausdorff_distance"] = np.max(valid_hausdorff)
                metrics["min_hausdorff_distance"] = np.min(valid_hausdorff)
        
        # Add Average Symmetric Surface Distance
        if assd_scores:
            valid_assd = [a for a in assd_scores if a >= 0]
            if valid_assd:
                metrics["mean_assd"] = np.mean(valid_assd)
                metrics["avg_symmetric_surface_distance"] = np.mean(valid_assd)  # Keep backward compatibility
        
        # Per-class IoU and Dice
        for i, class_name in enumerate(self.class_names):
            if i < len(iou_scores):
                metrics[f"iou_{class_name}"] = iou_scores[i]
                metrics[f"dice_{class_name}"] = dice_scores[i]
        
        # Add per-class Hausdorff distances (for all classes)
        if hausdorff_distances_per_class:
            for i, class_name in enumerate(self.class_names):
                if i < len(hausdorff_distances_per_class) and hausdorff_distances_per_class[i] >= 0:
                    metrics[f"hausdorff_{class_name}"] = hausdorff_distances_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for classification tasks"""
        if self.task != "classification" or not self.predictions:
            return None
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        return confusion_matrix(y_true, y_pred)
    
    def save_confusion_matrix(self, save_path: str, dataset_name: str = "Unknown") -> str:
        """Save confusion matrix as both image and text"""
        if self.task != "classification" or not self.predictions:
            return None
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save image
        img_path = save_path.replace('.txt', '.png')
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as text file
        with open(save_path, 'w') as f:
            f.write(f"Confusion Matrix for {dataset_name}\n")
            f.write("=" * 40 + "\n\n")
            
            # Write class labels
            f.write("Classes: " + ", ".join(self.class_names) + "\n\n")
            
            # Write confusion matrix
            f.write("Confusion Matrix:\n")
            f.write("Rows: Actual, Columns: Predicted\n\n")
            
            # Header
            f.write("Actual\\Predicted")
            for class_name in self.class_names:
                f.write(f"\t{class_name}")
            f.write("\n")
            
            # Matrix rows
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}")
                for j in range(len(self.class_names)):
                    f.write(f"\t{cm[i, j]}")
                f.write("\n")
            
            # Additional statistics
            f.write("\nPer-class Statistics:\n")
            f.write("-" * 20 + "\n")
            
            # Calculate per-class precision, recall, f1
            from sklearn.metrics import classification_report
            report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
            
            for class_name in self.class_names:
                if class_name in report:
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                    f.write(f"  Recall: {report[class_name]['recall']:.4f}\n")
                    f.write(f"  F1-Score: {report[class_name]['f1-score']:.4f}\n")
                    f.write(f"  Support: {report[class_name]['support']}\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {report['accuracy']:.4f}\n")
            f.write(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}\n")
            f.write(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}\n")
            f.write(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Avg Precision: {report['weighted avg']['precision']:.4f}\n")
            f.write(f"Weighted Avg Recall: {report['weighted avg']['recall']:.4f}\n")
            f.write(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
        
        return img_path

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