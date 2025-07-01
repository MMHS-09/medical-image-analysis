import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import json
from typing import Dict, List, Union, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_foundation_model
from src.visualization import Visualizer


class FoundationModelInference:
    """Advanced inference class for the trained foundation model with comprehensive metrics and visualization"""
    
    def __init__(self, model_path: str, config_path: str = None, output_dir: str = "./inference_results"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = self._initialize_model(checkpoint)
        
        # Get image preprocessing transforms
        self.image_size = tuple(self.config.get('data', {}).get('image_size', [224, 224]))
        self.transform = self._get_inference_transform()
        
        # Initialize visualizer
        self.visualizer = Visualizer(str(self.output_dir))
        
        # Initialize metrics storage
        self.all_results = []
        self.metrics_summary = {}
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Available classification datasets: {list(self.model.classification_heads.keys())}")
        print(f"Available segmentation datasets: {list(self.model.segmentation_heads.keys())}")
        print(f"Output directory: {self.output_dir}")
    
    def _initialize_model(self, checkpoint):
        """Initialize model from checkpoint"""
        # Extract model configuration from checkpoint
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print("Found model config in checkpoint")
        else:
            # Reconstruct from training config
            print("Reconstructing model config from training config")
            model_config = {
                'backbone': self.config['model']['backbone'],
                'pretrained': False,
                'dropout': self.config['model']['dropout'],
                'classification_heads': {},
                'segmentation_heads': {}
            }
            
            # Reconstruct heads from original config
            for dataset_config in self.config.get('classification_datasets', []):
                dataset_name = dataset_config['name']
                num_classes = len(dataset_config['classes'])
                model_config['classification_heads'][dataset_name] = num_classes
            
            for dataset_config in self.config.get('segmentation_datasets', []):
                dataset_name = dataset_config['name']
                num_classes = dataset_config['num_classes']
                model_config['segmentation_heads'][dataset_name] = num_classes
        
        # Create model
        model = create_foundation_model(
            backbone=model_config['backbone'],
            pretrained=model_config.get('pretrained', False),
            dropout=model_config.get('dropout', 0.2),
            classification_heads=model_config.get('classification_heads', {}),
            segmentation_heads=model_config.get('segmentation_heads', {})
        ).to(self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _get_inference_transform(self):
        """Get inference transforms"""
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            image = image_path  # Already loaded image
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def load_ground_truth_mask(self, mask_path: str) -> np.ndarray:
        """Load ground truth mask for segmentation"""
        if not os.path.exists(mask_path):
            return None
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        # Resize to match model output size
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary mask (assuming 0=background, >0=foreground)
        mask = (mask > 0).astype(np.uint8)
        
        return mask
    
    def calculate_segmentation_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
        """Calculate comprehensive segmentation metrics"""
        if gt_mask is None:
            return {}
        
        # Flatten masks for calculation
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Calculate metrics
        metrics = {}
        
        # Dice Score
        intersection = np.sum(pred_flat * gt_flat)
        dice = (2. * intersection) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-8)
        metrics['dice_score'] = float(dice)
        
        # IoU (Jaccard Index)
        union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
        iou = intersection / (union + 1e-8)
        metrics['iou'] = float(iou)
        
        # Pixel Accuracy
        correct_pixels = np.sum(pred_flat == gt_flat)
        total_pixels = len(pred_flat)
        pixel_accuracy = correct_pixels / total_pixels
        metrics['pixel_accuracy'] = float(pixel_accuracy)
        
        # Sensitivity (Recall) and Specificity
        tp = intersection
        fp = np.sum(pred_flat) - intersection
        fn = np.sum(gt_flat) - intersection
        tn = total_pixels - tp - fp - fn
        
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        
        metrics['sensitivity'] = float(sensitivity)
        metrics['specificity'] = float(specificity)
        metrics['precision'] = float(precision)
        
        # F1 Score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
        metrics['f1_score'] = float(f1)
        
        return metrics
    
    def calculate_classification_metrics(self, predictions: List[int], ground_truths: List[int], 
                                       class_names: List[str]) -> Dict:
        """Calculate comprehensive classification metrics"""
        if not ground_truths or len(predictions) != len(ground_truths):
            return {}
        
        metrics = {}
        
        # Overall accuracy
        accuracy = accuracy_score(ground_truths, predictions)
        metrics['accuracy'] = float(accuracy)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truths, predictions, average=None, zero_division=0
        )
        
        # Class-wise metrics
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_precision'] = float(precision[i])
            metrics[f'{class_name}_recall'] = float(recall[i])
            metrics[f'{class_name}_f1'] = float(f1[i])
            metrics[f'{class_name}_support'] = int(support[i])
        
        # Averaged metrics
        metrics['macro_precision'] = float(np.mean(precision))
        metrics['macro_recall'] = float(np.mean(recall))
        metrics['macro_f1'] = float(np.mean(f1))
        
        # Weighted metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            ground_truths, predictions, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = float(precision_w)
        metrics['weighted_recall'] = float(recall_w)
        metrics['weighted_f1'] = float(f1_w)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def classify_image(self, image_path: str, dataset_name: str, ground_truth_label: int = None) -> Dict:
        """Classify a single image with comprehensive metrics"""
        if dataset_name not in self.model.classification_heads:
            raise ValueError(f"Classification dataset '{dataset_name}' not found. Available: {list(self.model.classification_heads.keys())}")
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor, task='classification', dataset_name=dataset_name)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get class names
        class_names = self._get_class_names(dataset_name, 'classification')
        
        result = {
            'image_path': image_path,
            'task': 'classification',
            'dataset': dataset_name,
            'predicted_class_idx': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add ground truth information if available
        if ground_truth_label is not None:
            result['ground_truth_idx'] = ground_truth_label
            result['ground_truth_name'] = class_names[ground_truth_label]
            result['correct_prediction'] = (predicted_class == ground_truth_label)
        
        return result
    
    def segment_image(self, image_path: str, dataset_name: str, ground_truth_mask_path: str = None) -> Dict:
        """Segment a single image with comprehensive metrics"""
        if dataset_name not in self.model.segmentation_heads:
            raise ValueError(f"Segmentation dataset '{dataset_name}' not found. Available: {list(self.model.segmentation_heads.keys())}")
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor, task='segmentation', dataset_name=dataset_name)
            probabilities = F.softmax(logits, dim=1)
            predicted_mask = logits.argmax(dim=1)
        
        # Convert to numpy
        predicted_mask_np = predicted_mask[0].cpu().numpy()
        probabilities_np = probabilities[0].cpu().numpy()
        
        result = {
            'image_path': image_path,
            'task': 'segmentation',
            'dataset': dataset_name,
            'predicted_mask': predicted_mask_np,
            'probabilities': probabilities_np,
            'mask_shape': predicted_mask_np.shape,
            'unique_classes': np.unique(predicted_mask_np).tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add ground truth information and metrics if available
        if ground_truth_mask_path:
            gt_mask = self.load_ground_truth_mask(ground_truth_mask_path)
            if gt_mask is not None:
                result['ground_truth_mask_path'] = ground_truth_mask_path
                result['ground_truth_mask'] = gt_mask
                
                # Calculate metrics
                metrics = self.calculate_segmentation_metrics(predicted_mask_np, gt_mask)
                result['metrics'] = metrics
        
        return result
    
    def _get_class_names(self, dataset_name: str, task: str) -> List[str]:
        """Get class names for a dataset"""
        if task == 'classification':
            for ds_config in self.config.get('classification_datasets', []):
                if ds_config['name'] == dataset_name:
                    return ds_config['classes']
            # Fallback
            num_classes = self.model.classification_heads.get(dataset_name, 2)
            return [f"Class_{i}" for i in range(num_classes)]
        else:
            # For segmentation, usually just background and foreground
            num_classes = self.model.segmentation_heads.get(dataset_name, 2)
            return [f"Class_{i}" for i in range(num_classes)]
    
    def predict(self, image_path: str, task: str, dataset_name: str, 
                ground_truth_label: int = None, ground_truth_mask_path: str = None) -> Dict:
        """General prediction method with optional ground truth"""
        if task == 'classification':
            return self.classify_image(image_path, dataset_name, ground_truth_label)
        elif task == 'segmentation':
            return self.segment_image(image_path, dataset_name, ground_truth_mask_path)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'segmentation'")
    
    def predict_dataset(self, data_path: str, task: str, dataset_name: str, 
                       has_ground_truth: bool = True) -> List[Dict]:
        """Predict on an entire dataset with automatic ground truth detection"""
        data_path = Path(data_path)
        results = []
        
        print(f"Processing dataset at: {data_path}")
        print(f"Task: {task}, Dataset: {dataset_name}")
        print(f"Ground truth available: {has_ground_truth}")
        
        if task == 'classification':
            results = self._predict_classification_dataset(data_path, dataset_name, has_ground_truth)
        elif task == 'segmentation':
            results = self._predict_segmentation_dataset(data_path, dataset_name, has_ground_truth)
        
        # Store results
        self.all_results.extend(results)
        
        # Calculate overall metrics
        self._calculate_overall_metrics(results, task, dataset_name)
        
        return results
    
    def _predict_classification_dataset(self, data_path: Path, dataset_name: str, has_ground_truth: bool) -> List[Dict]:
        """Predict on classification dataset"""
        results = []
        class_names = self._get_class_names(dataset_name, 'classification')
        
        if has_ground_truth:
            # Assume data is organized in class folders
            class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                if class_name not in class_names:
                    print(f"Warning: Class '{class_name}' not in expected classes: {class_names}")
                    continue
                    
                ground_truth_idx = class_names.index(class_name)
                image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                
                print(f"Processing class '{class_name}': {len(image_files)} images")
                
                for img_file in image_files:
                    try:
                        result = self.classify_image(str(img_file), dataset_name, ground_truth_idx)
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
                        results.append({
                            'image_path': str(img_file),
                            'error': str(e),
                            'ground_truth_idx': ground_truth_idx,
                            'ground_truth_name': class_name
                        })
        else:
            # Just process all images in the directory
            image_files = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg")) + list(data_path.glob("*.jpeg"))
            
            print(f"Processing {len(image_files)} images without ground truth")
            
            for img_file in image_files:
                try:
                    result = self.classify_image(str(img_file), dataset_name)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    results.append({
                        'image_path': str(img_file),
                        'error': str(e)
                    })
        
        return results
    
    def _predict_segmentation_dataset(self, data_path: Path, dataset_name: str, has_ground_truth: bool) -> List[Dict]:
        """Predict on segmentation dataset"""
        results = []
        
        if has_ground_truth:
            # Assume images and masks are in the same directory with naming convention
            image_files = list(data_path.glob("img*.png"))
            image_files = [f for f in image_files if not f.name.endswith("_mask.png")]
            
            print(f"Processing {len(image_files)} images with ground truth masks")
            
            for img_file in image_files:
                mask_file = img_file.parent / f"{img_file.stem}_mask.png"
                
                try:
                    result = self.segment_image(
                        str(img_file), 
                        dataset_name, 
                        str(mask_file) if mask_file.exists() else None
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    results.append({
                        'image_path': str(img_file),
                        'error': str(e)
                    })
        else:
            # Just process all images
            image_files = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg")) + list(data_path.glob("*.jpeg"))
            
            print(f"Processing {len(image_files)} images without ground truth")
            
            for img_file in image_files:
                try:
                    result = self.segment_image(str(img_file), dataset_name)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    results.append({
                        'image_path': str(img_file),
                        'error': str(e)
                    })
        
        return results
    
    def _calculate_overall_metrics(self, results: List[Dict], task: str, dataset_name: str):
        """Calculate overall metrics for a dataset"""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to calculate metrics")
            return
        
        metrics_key = f"{task}_{dataset_name}"
        
        if task == 'classification':
            # Extract predictions and ground truths
            predictions = []
            ground_truths = []
            
            for result in valid_results:
                if 'ground_truth_idx' in result:
                    predictions.append(result['predicted_class_idx'])
                    ground_truths.append(result['ground_truth_idx'])
            
            if ground_truths:
                class_names = self._get_class_names(dataset_name, 'classification')
                metrics = self.calculate_classification_metrics(predictions, ground_truths, class_names)
                self.metrics_summary[metrics_key] = metrics
                
                print(f"\n📊 Classification Metrics for {dataset_name}:")
                print(f"   Overall Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Macro F1: {metrics['macro_f1']:.4f}")
                print(f"   Weighted F1: {metrics['weighted_f1']:.4f}")
        
        elif task == 'segmentation':
            # Extract segmentation metrics
            all_metrics = []
            
            for result in valid_results:
                if 'metrics' in result:
                    all_metrics.append(result['metrics'])
            
            if all_metrics:
                # Average metrics across all images
                avg_metrics = {}
                for metric_name in all_metrics[0].keys():
                    avg_metrics[f"avg_{metric_name}"] = np.mean([m[metric_name] for m in all_metrics])
                    avg_metrics[f"std_{metric_name}"] = np.std([m[metric_name] for m in all_metrics])
                
                self.metrics_summary[metrics_key] = avg_metrics
                
                print(f"\n📊 Segmentation Metrics for {dataset_name}:")
                print(f"   Avg Dice Score: {avg_metrics['avg_dice_score']:.4f} ± {avg_metrics['std_dice_score']:.4f}")
                print(f"   Avg IoU: {avg_metrics['avg_iou']:.4f} ± {avg_metrics['std_iou']:.4f}")
                print(f"   Avg Pixel Accuracy: {avg_metrics['avg_pixel_accuracy']:.4f} ± {avg_metrics['std_pixel_accuracy']:.4f}")
    
    def visualize_classification_results(self, results: List[Dict], dataset_name: str, max_samples: int = 20):
        """Visualize classification results with comprehensive plots"""
        valid_results = [r for r in results if 'error' not in r][:max_samples]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Create figure with subplots
        n_samples = min(len(valid_results), max_samples)
        cols = 4
        rows = int(np.ceil(n_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(valid_results):
            if i >= max_samples:
                break
                
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Load and display image
            try:
                image = cv2.imread(result['image_path'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image)
                
                # Title with prediction info
                title = f"Pred: {result['predicted_class_name']}\nConf: {result['confidence']:.3f}"
                if 'ground_truth_name' in result:
                    title += f"\nGT: {result['ground_truth_name']}"
                    color = 'green' if result.get('correct_prediction', False) else 'red'
                    ax.set_title(title, color=color, fontsize=10)
                else:
                    ax.set_title(title, fontsize=10)
                
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{result['image_path']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"classification_results_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot confusion matrix if ground truth available
        self._plot_confusion_matrix(results, dataset_name)
        
        # Plot class distribution
        self._plot_class_distribution(results, dataset_name)
    
    def visualize_segmentation_results(self, results: List[Dict], dataset_name: str, max_samples: int = 10):
        """Visualize segmentation results with image, ground truth, and prediction"""
        valid_results = [r for r in results if 'error' not in r and 'predicted_mask' in r][:max_samples]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        for i, result in enumerate(valid_results):
            if i >= max_samples:
                break
            
            # Load original image
            try:
                image = cv2.imread(result['image_path'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)
                
                pred_mask = result['predicted_mask']
                
                # Create figure
                has_gt = 'ground_truth_mask' in result
                n_cols = 3 if has_gt else 2
                fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
                
                # Original image
                axes[0].imshow(image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Ground truth mask (if available)
                if has_gt:
                    gt_mask = result['ground_truth_mask']
                    axes[1].imshow(gt_mask, cmap='gray')
                    axes[1].set_title('Ground Truth Mask')
                    axes[1].axis('off')
                    
                    # Predicted mask
                    axes[2].imshow(pred_mask, cmap='gray')
                    
                    # Add metrics to title if available
                    title = 'Predicted Mask'
                    if 'metrics' in result:
                        metrics = result['metrics']
                        title += f"\nDice: {metrics['dice_score']:.3f}, IoU: {metrics['iou']:.3f}"
                    axes[2].set_title(title)
                    axes[2].axis('off')
                else:
                    # Just predicted mask
                    axes[1].imshow(pred_mask, cmap='gray')
                    axes[1].set_title('Predicted Mask')
                    axes[1].axis('off')
                
                plt.tight_layout()
                
                # Save individual result
                filename = f"segmentation_{dataset_name}_sample_{i+1}.png"
                plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Error visualizing {result['image_path']}: {e}")
        
        # Plot metrics distribution if ground truth available
        self._plot_segmentation_metrics_distribution(results, dataset_name)
    
    def _plot_confusion_matrix(self, results: List[Dict], dataset_name: str):
        """Plot confusion matrix for classification results"""
        valid_results = [r for r in results if 'error' not in r and 'ground_truth_idx' in r]
        
        if not valid_results:
            return
        
        predictions = [r['predicted_class_idx'] for r in valid_results]
        ground_truths = [r['ground_truth_idx'] for r in valid_results]
        class_names = self._get_class_names(dataset_name, 'classification')
        
        cm = confusion_matrix(ground_truths, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"confusion_matrix_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_class_distribution(self, results: List[Dict], dataset_name: str):
        """Plot class distribution for predictions"""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return
        
        predictions = [r['predicted_class_name'] for r in valid_results]
        class_names = self._get_class_names(dataset_name, 'classification')
        
        # Count predictions
        pred_counts = {name: predictions.count(name) for name in class_names}
        
        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(pred_counts.keys(), pred_counts.values())
        plt.title(f'Predicted Class Distribution - {dataset_name}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add counts on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"class_distribution_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_segmentation_metrics_distribution(self, results: List[Dict], dataset_name: str):
        """Plot distribution of segmentation metrics"""
        valid_results = [r for r in results if 'error' not in r and 'metrics' in r]
        
        if not valid_results:
            return
        
        metrics_data = [r['metrics'] for r in valid_results]
        
        # Extract metric values
        dice_scores = [m['dice_score'] for m in metrics_data]
        iou_scores = [m['iou'] for m in metrics_data]
        pixel_accuracies = [m['pixel_accuracy'] for m in metrics_data]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Dice scores
        axes[0].hist(dice_scores, bins=20, alpha=0.7, color='blue')
        axes[0].set_title(f'Dice Score Distribution\nMean: {np.mean(dice_scores):.3f} ± {np.std(dice_scores):.3f}')
        axes[0].set_xlabel('Dice Score')
        axes[0].set_ylabel('Frequency')
        
        # IoU scores
        axes[1].hist(iou_scores, bins=20, alpha=0.7, color='green')
        axes[1].set_title(f'IoU Distribution\nMean: {np.mean(iou_scores):.3f} ± {np.std(iou_scores):.3f}')
        axes[1].set_xlabel('IoU')
        axes[1].set_ylabel('Frequency')
        
        # Pixel accuracies
        axes[2].hist(pixel_accuracies, bins=20, alpha=0.7, color='red')
        axes[2].set_title(f'Pixel Accuracy Distribution\nMean: {np.mean(pixel_accuracies):.3f} ± {np.std(pixel_accuracies):.3f}')
        axes[2].set_xlabel('Pixel Accuracy')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"segmentation_metrics_distribution_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save all results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_results_{timestamp}.json"
        
        save_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to: {save_path}")
        return save_path
    
    def save_metrics_summary(self, filename: str = None):
        """Save metrics summary to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_summary_{timestamp}.json"
        
        save_path = self.output_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics_summary, f, indent=2, default=str)
        
        print(f"Metrics summary saved to: {save_path}")
        return save_path
    
    def generate_report(self, results: List[Dict], task: str, dataset_name: str):
        """Generate a comprehensive report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_filename = f"inference_report_{task}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = self.output_dir / report_filename
        
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        with open(report_path, 'w') as f:
            f.write("FOUNDATION MODEL INFERENCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Task: {task}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples processed: {len(results)}\n")
            f.write(f"Successful predictions: {len(valid_results)}\n")
            f.write(f"Errors: {len(error_results)}\n")
            f.write(f"Success rate: {len(valid_results)/len(results)*100:.2f}%\n\n")
            
            if task == 'classification':
                # Classification specific metrics
                has_gt = any('ground_truth_idx' in r for r in valid_results)
                if has_gt:
                    correct_predictions = sum(1 for r in valid_results if r.get('correct_prediction', False))
                    total_with_gt = sum(1 for r in valid_results if 'ground_truth_idx' in r)
                    accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
                    
                    f.write("CLASSIFICATION METRICS\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Overall Accuracy: {accuracy:.4f}\n")
                    
                    if f"{task}_{dataset_name}" in self.metrics_summary:
                        metrics = self.metrics_summary[f"{task}_{dataset_name}"]
                        f.write(f"Macro Precision: {metrics.get('macro_precision', 'N/A'):.4f}\n")
                        f.write(f"Macro Recall: {metrics.get('macro_recall', 'N/A'):.4f}\n")
                        f.write(f"Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}\n")
                        f.write(f"Weighted F1: {metrics.get('weighted_f1', 'N/A'):.4f}\n\n")
                
                # Prediction distribution
                pred_distribution = {}
                for result in valid_results:
                    pred_class = result['predicted_class_name']
                    pred_distribution[pred_class] = pred_distribution.get(pred_class, 0) + 1
                
                f.write("PREDICTION DISTRIBUTION\n")
                f.write("-" * 25 + "\n")
                for class_name, count in sorted(pred_distribution.items()):
                    percentage = count / len(valid_results) * 100
                    f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            elif task == 'segmentation':
                # Segmentation specific metrics
                has_metrics = any('metrics' in r for r in valid_results)
                if has_metrics:
                    if f"{task}_{dataset_name}" in self.metrics_summary:
                        metrics = self.metrics_summary[f"{task}_{dataset_name}"]
                        f.write("SEGMENTATION METRICS\n")
                        f.write("-" * 25 + "\n")
                        f.write(f"Average Dice Score: {metrics.get('avg_dice_score', 'N/A'):.4f} ± {metrics.get('std_dice_score', 'N/A'):.4f}\n")
                        f.write(f"Average IoU: {metrics.get('avg_iou', 'N/A'):.4f} ± {metrics.get('std_iou', 'N/A'):.4f}\n")
                        f.write(f"Average Pixel Accuracy: {metrics.get('avg_pixel_accuracy', 'N/A'):.4f} ± {metrics.get('std_pixel_accuracy', 'N/A'):.4f}\n")
                        f.write(f"Average Precision: {metrics.get('avg_precision', 'N/A'):.4f} ± {metrics.get('std_precision', 'N/A'):.4f}\n")
                        f.write(f"Average Recall: {metrics.get('avg_sensitivity', 'N/A'):.4f} ± {metrics.get('std_sensitivity', 'N/A'):.4f}\n")
                        f.write(f"Average F1: {metrics.get('avg_f1_score', 'N/A'):.4f} ± {metrics.get('std_f1_score', 'N/A'):.4f}\n\n")
            
            # Error analysis
            if error_results:
                f.write("ERROR ANALYSIS\n")
                f.write("-" * 15 + "\n")
                for i, error_result in enumerate(error_results, 1):
                    f.write(f"{i}. {error_result['image_path']}: {error_result['error']}\n")
                f.write("\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 15 + "\n")
            f.write(f"- Results JSON: inference_results_*.json\n")
            f.write(f"- Metrics Summary: metrics_summary_*.json\n")
            f.write(f"- Visualizations: {task}_results_{dataset_name}.png\n")
            if task == 'classification':
                f.write(f"- Confusion Matrix: confusion_matrix_{dataset_name}.png\n")
                f.write(f"- Class Distribution: class_distribution_{dataset_name}.png\n")
            else:
                f.write(f"- Individual Samples: segmentation_{dataset_name}_sample_*.png\n")
                f.write(f"- Metrics Distribution: segmentation_metrics_distribution_{dataset_name}.png\n")
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Advanced Foundation Model Inference with Comprehensive Metrics')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset or single image')
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'segmentation'], help='Task type')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name for task head')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory for results')
    parser.add_argument('--has_ground_truth', action='store_true', help='Whether ground truth is available')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--generate_report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--max_samples_viz', type=int, default=20, help='Maximum samples to visualize')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Foundation Model Inference")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)
    
    # Initialize inference
    inference = FoundationModelInference(args.model_path, args.config_path, args.output_dir)
    
    # Check if single image or dataset
    data_path = Path(args.data_path)
    
    if data_path.is_file():
        # Single image prediction
        print("Processing single image...")
        if args.task == 'classification':
            result = inference.classify_image(str(data_path), args.dataset_name)
        else:
            result = inference.segment_image(str(data_path), args.dataset_name)
        
        results = [result]
        
        # Print result
        print(f"\n📊 Prediction Result:")
        if args.task == 'classification':
            print(f"Predicted Class: {result['predicted_class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"Mask Shape: {result['mask_shape']}")
            print(f"Unique Classes: {result['unique_classes']}")
    
    elif data_path.is_dir():
        # Dataset prediction
        print("Processing dataset...")
        results = inference.predict_dataset(str(data_path), args.task, args.dataset_name, args.has_ground_truth)
        
        print(f"\n📊 Dataset Processing Complete:")
        print(f"Total samples: {len(results)}")
        print(f"Successful: {len([r for r in results if 'error' not in r])}")
        print(f"Errors: {len([r for r in results if 'error' in r])}")
    
    else:
        raise ValueError(f"Invalid data path: {args.data_path}")
    
    # Save results
    results_file = inference.save_results(results)
    metrics_file = inference.save_metrics_summary()
    
    # Generate visualizations
    if args.visualize:
        print(f"\n🎨 Generating Visualizations...")
        if args.task == 'classification':
            inference.visualize_classification_results(results, args.dataset_name, args.max_samples_viz)
        else:
            inference.visualize_segmentation_results(results, args.dataset_name, args.max_samples_viz)
    
    # Generate comprehensive report
    if args.generate_report:
        print(f"\n📝 Generating Comprehensive Report...")
        report_file = inference.generate_report(results, args.task, args.dataset_name)
    
    print(f"\n✅ Inference Complete!")
    print(f"📁 All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
