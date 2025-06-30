import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import json


class Visualizer:
    """Visualization utilities for medical image analysis"""
    
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List[float]], save_name: str = "training_history.png"):
        """Plot training and validation metrics"""
        # Determine if this is classification or segmentation based on available metrics
        is_segmentation = 'train_mean_dice' in history or 'train_mean_iou' in history
        
        if is_segmentation:
            # For segmentation: Loss, Accuracy, Dice, IoU
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Segmentation Training History', fontsize=16)
            
            # Loss plot
            axes[0, 0].plot(history.get('train_loss', []), label='Training Loss', marker='o')
            axes[0, 0].plot(history.get('val_loss', []), label='Validation Loss', marker='s')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy plot
            if 'train_accuracy' in history:
                axes[0, 1].plot(history['train_accuracy'], label='Training Accuracy', marker='o')
                axes[0, 1].plot(history.get('val_accuracy', []), label='Validation Accuracy', marker='s')
                axes[0, 1].set_title('Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Dice Score plot
            if 'train_mean_dice' in history:
                axes[1, 0].plot(history['train_mean_dice'], label='Training Dice', marker='o')
                axes[1, 0].plot(history.get('val_mean_dice', []), label='Validation Dice', marker='s')
                axes[1, 0].set_title('Dice Score')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Dice Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # IoU plot
            if 'train_mean_iou' in history:
                axes[1, 1].plot(history['train_mean_iou'], label='Training IoU', marker='o')
                axes[1, 1].plot(history.get('val_mean_iou', []), label='Validation IoU', marker='s')
                axes[1, 1].set_title('Mean IoU')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('IoU')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        else:
            # For classification: Loss, Accuracy, F1, Precision/Recall
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Classification Training History', fontsize=16)
            
            # Loss plot
            axes[0, 0].plot(history.get('train_loss', []), label='Training Loss', marker='o')
            axes[0, 0].plot(history.get('val_loss', []), label='Validation Loss', marker='s')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy plot
            if 'train_accuracy' in history:
                axes[0, 1].plot(history['train_accuracy'], label='Training Accuracy', marker='o')
                axes[0, 1].plot(history.get('val_accuracy', []), label='Validation Accuracy', marker='s')
                axes[0, 1].set_title('Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # F1 Score plot
            if 'train_f1_score' in history:
                axes[1, 0].plot(history['train_f1_score'], label='Training F1', marker='o')
                axes[1, 0].plot(history.get('val_f1_score', []), label='Validation F1', marker='s')
                axes[1, 0].set_title('F1 Score')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Precision and Recall plot
            if 'train_precision' in history and 'train_recall' in history:
                axes[1, 1].plot(history['train_precision'], label='Training Precision', marker='o')
                axes[1, 1].plot(history.get('val_precision', []), label='Validation Precision', marker='s')
                axes[1, 1].plot(history['train_recall'], label='Training Recall', marker='^', linestyle='--')
                axes[1, 1].plot(history.get('val_recall', []), label='Validation Recall', marker='v', linestyle='--')
                axes[1, 1].set_title('Precision & Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for ax in axes.flat:
            if not ax.has_data():
                ax.remove()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save history to CSV
        csv_path = os.path.join(self.save_dir, "training_history.csv")
        self._save_history_to_csv(history, csv_path)
        
        return save_path
    
    def _save_history_to_csv(self, history: Dict[str, List[float]], file_path: str):
        """Save training history to CSV file"""
        # Convert history dict to DataFrame
        df = pd.DataFrame(history)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
    
    def save_training_history_to_csv(self, history: Dict[str, List[float]], dataset_name: str, task: str):
        """Save training history to CSV file"""
        # Prepare data for CSV
        data = {}
        max_epochs = 0
        
        # Find the maximum number of epochs
        for key, values in history.items():
            if isinstance(values, list) and len(values) > max_epochs:
                max_epochs = len(values)
        
        # Add epoch column
        data['epoch'] = list(range(1, max_epochs + 1))
        
        # Add all metrics
        for key, values in history.items():
            if isinstance(values, list):
                # Pad with None if needed
                padded_values = values + [None] * (max_epochs - len(values))
                data[key] = padded_values
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add metadata columns
        df['dataset'] = dataset_name
        df['task'] = task
        
        # Save to CSV
        csv_filename = f"{task}_{dataset_name}_training_history.csv"
        csv_path = os.path.join(self.save_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def save_all_training_histories_to_csv(self, all_histories: Dict[str, Dict], save_name: str = "all_training_histories.csv"):
        """Save all training histories to a single CSV file"""
        all_data = []
        
        for dataset_key, history in all_histories.items():
            # Parse dataset key (e.g., "classification_brain_mri_nd5" or "segmentation_btcv")
            if dataset_key.startswith('classification_'):
                task = 'classification'
                dataset_name = dataset_key[len('classification_'):]
            elif dataset_key.startswith('segmentation_'):
                task = 'segmentation'
                dataset_name = dataset_key[len('segmentation_'):]
            else:
                task = 'unknown'
                dataset_name = dataset_key
            
            # Get training and validation metrics
            train_metrics = history.get('train_metrics', [])
            val_metrics = history.get('val_metrics', [])
            
            # Process each epoch
            max_epochs = max(len(train_metrics), len(val_metrics))
            
            for epoch in range(max_epochs):
                row = {
                    'dataset': dataset_name,
                    'task': task,
                    'epoch': epoch + 1
                }
                
                # Add training metrics
                if epoch < len(train_metrics):
                    for metric_name, value in train_metrics[epoch].items():
                        row[f'train_{metric_name}'] = value
                
                # Add validation metrics
                if epoch < len(val_metrics):
                    for metric_name, value in val_metrics[epoch].items():
                        row[f'val_{metric_name}'] = value
                
                all_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        csv_path = os.path.join(self.save_dir, save_name)
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def save_final_metrics_summary_to_csv(self, all_results: Dict, save_name: str = "final_metrics_summary.csv"):
        """Save final metrics summary to CSV file"""
        data = []
        
        for dataset_key, metrics in all_results.items():
            # Parse dataset key
            if dataset_key.startswith('classification_'):
                task = 'classification'
                dataset_name = dataset_key[len('classification_'):]
            elif dataset_key.startswith('segmentation_'):
                task = 'segmentation'
                dataset_name = dataset_key[len('segmentation_'):]
            else:
                task = 'unknown'
                dataset_name = dataset_key
            
            row = {
                'dataset': dataset_name,
                'task': task
            }
            
            # Add all metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[metric_name] = value
                else:
                    row[metric_name] = str(value)
            
            data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.save_dir, save_name)
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], save_name: str = "confusion_matrix.png"):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_segmentation_results(self, images: torch.Tensor, masks: torch.Tensor, 
                                predictions: torch.Tensor, save_name: str = "segmentation_results.png",
                                num_samples: int = 4):
        """Plot segmentation results"""
        # Convert tensors to numpy
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Get predicted masks
        pred_masks = np.argmax(predictions, axis=1)
        
        # Plot samples
        num_samples = min(num_samples, len(images))
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original image
            img = images[i].transpose(1, 2, 0)
            # Denormalize if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(masks[i], cmap='tab10')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Predicted mask
            axes[i, 2].imshow(pred_masks[i], cmap='tab10')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_classification_results(self, images: torch.Tensor, labels: torch.Tensor,
                                  predictions: torch.Tensor, class_names: List[str],
                                  save_name: str = "classification_results.png",
                                  num_samples: int = 8):
        """Plot classification results"""
        # Convert tensors to numpy
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Get predicted classes
        pred_classes = np.argmax(predictions, axis=1)
        
        # Plot samples
        num_samples = min(num_samples, len(images))
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(num_samples):
            # Original image
            img = images[i].transpose(1, 2, 0)
            # Denormalize if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img)
            
            # Title with true and predicted labels
            true_label = class_names[labels[i]]
            pred_label = class_names[pred_classes[i]]
            
            color = 'green' if labels[i] == pred_classes[i] else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
            axes[i].axis('off')
        
        # Remove empty subplots
        for i in range(num_samples, len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_class_distribution(self, labels: List[int], class_names: List[str],
                              save_name: str = "class_distribution.png"):
        """Plot class distribution"""
        from collections import Counter
        
        class_counts = Counter(labels)
        counts = [class_counts.get(i, 0) for i in range(len(class_names))]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, counts, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
        
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                              save_name: str = "metrics_comparison.png"):
        """Plot metrics comparison across datasets"""
        # Extract metric names and dataset names
        dataset_names = list(metrics_dict.keys())
        all_metrics = set()
        for dataset_metrics in metrics_dict.values():
            all_metrics.update(dataset_metrics.keys())
        
        # Filter out loss and per-class metrics for cleaner visualization
        main_metrics = [m for m in all_metrics if not any(x in m.lower() for x in ['loss', 'class_', '_0', '_1', '_2', '_3'])]
        
        if not main_metrics:
            return None
        
        # Create subplots
        num_metrics = len(main_metrics)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        for i, metric in enumerate(main_metrics):
            values = [metrics_dict[dataset].get(metric, 0) for dataset in dataset_names]
            
            bars = axes[i].bar(dataset_names, values, color=plt.cm.Set2(np.linspace(0, 1, len(dataset_names))))
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Remove empty subplots
        for i in range(num_metrics, len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_results_summary(self, results: Dict, save_name: str = "results_summary.txt"):
        """Create a text summary of results"""
        save_path = os.path.join(self.save_dir, save_name)
        
        with open(save_path, 'w') as f:
            f.write("Medical Image Analysis - Foundation Model Results\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset_name, metrics in results.items():
                f.write(f"Dataset: {dataset_name}\n")
                f.write("-" * 30 + "\n")
                
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"{metric_name}: {value}\n")
                
                f.write("\n")
        
        return save_path


def denormalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Denormalize image for visualization"""
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    
    # Denormalize
    image = image * std + mean
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay segmentation mask on image"""
    # Ensure image is in correct format
    if image.shape[0] == 3:  # CHW format
        image = image.transpose(1, 2, 0)
    
    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Create colored mask
    colors = plt.cm.Set1(np.linspace(0, 1, mask.max() + 1))
    colored_mask = colors[mask][:, :, :3]  # Remove alpha channel
    
    # Overlay
    overlaid = image * (1 - alpha) + colored_mask * alpha
    
    return overlaid