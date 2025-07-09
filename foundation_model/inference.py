import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import json
import logging
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_foundation_model, load_model_weights_safely

# Setup logger
logger = logging.getLogger(__name__)


class FoundationModelInference:
    """Foundation model inference class with config-based settings"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        self.inference_config = self.config.get('inference', {})
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get settings from config
        self.model_path = self.inference_config.get('model_path')
        self.dataset_path = Path(self.inference_config.get('dataset_path', './data'))
        self.dataset_name = self.inference_config.get('dataset_name')
        self.task = self.inference_config.get('task', 'segmentation')
        self.output_dir = Path(self.inference_config.get('output_dir', './inference_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(f"Model path not found: {self.model_path}")
        if not self.dataset_name:
            raise ValueError("Dataset name not specified in config")
        
        # Load model
        print(f"Loading model from: {self.model_path}")
        self.checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Get preprocessing settings
        self.image_size = tuple(self.inference_config.get('image_size', self.config.get('data', {}).get('image_size', [64, 64])))
        self.transform = self._get_transforms()
        
        # Storage for results
        self.results = []
        self.metrics = {}
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Task: {self.task}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_model(self):
        """Initialize model from checkpoint"""
        # Get model config from checkpoint
        if 'model_config' in self.checkpoint:
            model_config = self.checkpoint['model_config']
            print("Found model config in checkpoint")
        else:
            # Reconstruct from config
            print("Reconstructing model config from training config")
            model_config = {
                'backbone': self.config['model']['backbone'],
                'pretrained': False,
                'dropout': self.config['model']['dropout'],
                'classification_heads': {},
                'segmentation_heads': {}
            }
            
            # Reconstruct heads from config
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
        
        # Load weights
        missing_keys, unexpected_keys = load_model_weights_safely(model, self.checkpoint, logger)
        
        model.eval()
        
        # Store available datasets
        self.available_datasets = {
            'classification': list(model_config.get('classification_heads', {}).keys()),
            'segmentation': list(model_config.get('segmentation_heads', {}).keys())
        }
        
        print(f"Available {self.task} datasets: {self.available_datasets[self.task]}")
        
        # Validate dataset name
        if self.dataset_name not in self.available_datasets[self.task]:
            available = self.available_datasets[self.task]
            raise ValueError(f"Dataset '{self.dataset_name}' not found. Available: {available}")
        
        return model
    
    def _get_transforms(self):
        """Get preprocessing transforms"""
        normalization = self.inference_config.get('normalization', {})
        mean = normalization.get('mean', [0.485, 0.456, 0.406])
        std = normalization.get('std', [0.229, 0.224, 0.225])
        
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def load_ground_truth_mask(self, mask_path: str) -> Optional[np.ndarray]:
        """Load ground truth mask for segmentation"""
        if not os.path.exists(mask_path):
            return None
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        # Resize to match model output
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary (0 and 1)
        mask = (mask > 0).astype(np.uint8)
        
        return mask
    
    def calculate_segmentation_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
        """Calculate segmentation metrics"""
        if gt_mask is None:
            return {}
        
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Calculate metrics
        intersection = np.sum(pred_flat * gt_flat)
        
        # Dice Score
        dice = (2. * intersection) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-8)
        
        # IoU
        union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
        iou = intersection / (union + 1e-8)
        
        # Pixel Accuracy
        correct_pixels = np.sum(pred_flat == gt_flat)
        pixel_accuracy = correct_pixels / len(pred_flat)
        
        # Sensitivity and Specificity
        tp = intersection
        fp = np.sum(pred_flat) - intersection
        fn = np.sum(gt_flat) - intersection
        tn = len(pred_flat) - tp - fp - fn
        
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
        
        return {
            'dice_score': float(dice),
            'iou': float(iou),
            'pixel_accuracy': float(pixel_accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1)
        }
    
    def classify_image(self, image_path: str, ground_truth_label: int = None) -> Dict:
        """Classify a single image"""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor, task='classification', dataset_name=self.dataset_name)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        result = {
            'image_path': str(image_path),
            'predicted_class_idx': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().numpy().tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        if ground_truth_label is not None:
            result['ground_truth_idx'] = ground_truth_label
            result['correct_prediction'] = (predicted_class == ground_truth_label)
        
        return result
    
    def segment_image(self, image_path: str, ground_truth_mask_path: str = None) -> Dict:
        """Segment a single image"""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor, task='segmentation', dataset_name=self.dataset_name)
            probabilities = F.softmax(logits, dim=1)
            predicted_mask = logits.argmax(dim=1)
        
        # Convert to numpy
        predicted_mask_np = predicted_mask[0].cpu().numpy()
        probabilities_np = probabilities[0].cpu().numpy()
        
        result = {
            'image_path': str(image_path),
            'predicted_mask': predicted_mask_np,
            'probabilities': probabilities_np,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add ground truth and metrics if available
        if ground_truth_mask_path:
            gt_mask = self.load_ground_truth_mask(ground_truth_mask_path)
            if gt_mask is not None:
                result['ground_truth_mask'] = gt_mask
                result['metrics'] = self.calculate_segmentation_metrics(predicted_mask_np, gt_mask)
        
        return result
    
    def run_inference(self):
        """Run inference on the configured dataset"""
        print(f"\n🚀 Starting inference on dataset: {self.dataset_name}")
        print(f"Task: {self.task}")
        
        if self.task == 'classification':
            self._run_classification_inference()
        elif self.task == 'segmentation':
            self._run_segmentation_inference()
        
        # Calculate overall metrics
        self._calculate_overall_metrics()
        
        # Save results
        self._save_results()
        
        print(f"\n✅ Inference completed!")
        print(f"Total samples: {len(self.results)}")
        print(f"Results saved to: {self.output_dir}")
    
    def _run_classification_inference(self):
        """Run classification inference"""
        # Look for class folders
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        
        if not class_dirs:
            # No class folders, process all images directly
            image_files = list(self.dataset_path.glob("*.png")) + list(self.dataset_path.glob("*.jpg"))
            print(f"Processing {len(image_files)} images without ground truth")
            
            for img_file in image_files:
                try:
                    result = self.classify_image(str(img_file))
                    self.results.append(result)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        else:
            # Process class folders
            print(f"Found {len(class_dirs)} classes")
            
            for class_idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
                
                print(f"Processing class '{class_name}': {len(image_files)} images")
                
                for img_file in image_files:
                    try:
                        result = self.classify_image(str(img_file), class_idx)
                        result['ground_truth_class_name'] = class_name
                        self.results.append(result)
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
    
    def _run_segmentation_inference(self):
        """Run segmentation inference"""
        # Look for image files
        image_files = list(self.dataset_path.glob("*.png")) + list(self.dataset_path.glob("*.jpg"))
        image_files = [f for f in image_files if not f.name.endswith("_mask.png")]
        
        print(f"Processing {len(image_files)} images")
        
        for img_file in image_files:
            # Look for corresponding mask file
            mask_file = img_file.parent / f"{img_file.stem}_mask.png"
            if not mask_file.exists():
                # Try alternative naming conventions
                mask_file = img_file.parent / f"{img_file.stem}_gt.png"
                if not mask_file.exists():
                    mask_file = None
            
            try:
                result = self.segment_image(
                    str(img_file), 
                    str(mask_file) if mask_file else None
                )
                self.results.append(result)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    def _calculate_overall_metrics(self):
        """Calculate overall metrics"""
        valid_results = [r for r in self.results if 'error' not in r]
        
        if self.task == 'classification':
            # Classification metrics
            predictions = []
            ground_truths = []
            
            for result in valid_results:
                if 'ground_truth_idx' in result:
                    predictions.append(result['predicted_class_idx'])
                    ground_truths.append(result['ground_truth_idx'])
            
            if ground_truths:
                accuracy = accuracy_score(ground_truths, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    ground_truths, predictions, average='weighted', zero_division=0
                )
                
                self.metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'num_samples': len(ground_truths)
                }
                
                print(f"\n📊 Classification Metrics:")
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
                print(f"   F1 Score: {f1:.4f}")
        
        elif self.task == 'segmentation':
            # Segmentation metrics
            all_metrics = [r['metrics'] for r in valid_results if 'metrics' in r]
            
            if all_metrics:
                avg_metrics = {}
                for metric_name in all_metrics[0].keys():
                    values = [m[metric_name] for m in all_metrics]
                    avg_metrics[f"avg_{metric_name}"] = np.mean(values)
                    avg_metrics[f"std_{metric_name}"] = np.std(values)
                
                avg_metrics['num_samples'] = len(all_metrics)
                self.metrics = avg_metrics
                
                print(f"\n📊 Segmentation Metrics:")
                print(f"   Avg Dice Score: {avg_metrics['avg_dice_score']:.4f} ± {avg_metrics['std_dice_score']:.4f}")
                print(f"   Avg IoU: {avg_metrics['avg_iou']:.4f} ± {avg_metrics['std_iou']:.4f}")
                print(f"   Avg Pixel Accuracy: {avg_metrics['avg_pixel_accuracy']:.4f} ± {avg_metrics['std_pixel_accuracy']:.4f}")
    
    def visualize_results(self):
        """Generate visualizations based on config settings"""
        if not self.inference_config.get('visualize', False):
            return
        
        print(f"\n🎨 Generating visualizations...")
        max_samples = self.inference_config.get('max_samples_viz', 10)
        
        if self.task == 'classification':
            self._visualize_classification_results(max_samples)
        elif self.task == 'segmentation':
            self._visualize_segmentation_results(max_samples)
    
    def _visualize_classification_results(self, max_samples: int):
        """Visualize classification results"""
        valid_results = [r for r in self.results if 'error' not in r][:max_samples]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Create sample grid
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
                title = f"Pred: Class {result['predicted_class_idx']}\nConf: {result['confidence']:.3f}"
                if 'ground_truth_idx' in result:
                    title += f"\nGT: Class {result['ground_truth_idx']}"
                    color = 'green' if result.get('correct_prediction', False) else 'red'
                    ax.set_title(title, color=color, fontsize=10)
                else:
                    ax.set_title(title, fontsize=10)
                
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save as both PNG and PDF
        plt.savefig(self.output_dir / f"classification_results_{self.dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"classification_results_{self.dataset_name}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrix if we have ground truth
        self._plot_confusion_matrix(valid_results)
    
    def _visualize_segmentation_results(self, max_samples: int):
        """Create comprehensive PDF with segmentation results in 4-column layout"""
        valid_results = [r for r in self.results if 'error' not in r and 'predicted_mask' in r][:max_samples]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Create PDF with 4-column layout: Original, GT Mask, Pred Mask, Combined Overlay
        pdf_path = self.output_dir / f"segmentation_results_{self.dataset_name}.pdf"
        
        with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
            # Create figure with proper sizing for 4 columns
            fig, axes = plt.subplots(max_samples, 4, figsize=(16, 4 * max_samples))
            
            # Handle single row case
            if max_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i, result in enumerate(valid_results):
                if i >= max_samples:
                    break
                
                try:
                    # Load original image
                    image = cv2.imread(result['image_path'])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.image_size)
                    
                    pred_mask = result['predicted_mask']
                    has_gt = 'ground_truth_mask' in result
                    
                    # Column 1: Original Image
                    axes[i, 0].imshow(image)
                    axes[i, 0].set_title(f'Sample {i+1}: Original Image')
                    axes[i, 0].axis('off')
                    
                    if has_gt:
                        gt_mask = result['ground_truth_mask']
                        
                        # Column 2: Ground Truth Mask
                        axes[i, 1].imshow(gt_mask, cmap='gray')
                        axes[i, 1].set_title('Ground Truth Mask')
                        axes[i, 1].axis('off')
                        
                        # Column 3: Predicted Mask
                        axes[i, 2].imshow(pred_mask, cmap='gray')
                        title = 'Predicted Mask'
                        if 'metrics' in result:
                            metrics = result['metrics']
                            title += f"\nDice: {metrics['dice_score']:.3f}"
                        axes[i, 2].set_title(title)
                        axes[i, 2].axis('off')
                        
                        # Column 4: Combined Overlay (GT=Green, Pred=Red, Overlap=Yellow)
                        combined_overlay = self._create_combined_mask_overlay(image, gt_mask, pred_mask)
                        axes[i, 3].imshow(combined_overlay)
                        axes[i, 3].set_title('Combined Overlay\n(GT=Green, Pred=Red, Overlap=Yellow)')
                        axes[i, 3].axis('off')
                        
                    else:
                        # No ground truth available
                        axes[i, 1].text(0.5, 0.5, 'No Ground Truth\nAvailable', 
                                       ha='center', va='center', transform=axes[i, 1].transAxes, 
                                       fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
                        axes[i, 1].axis('off')
                        
                        # Column 3: Predicted Mask
                        axes[i, 2].imshow(pred_mask, cmap='gray')
                        axes[i, 2].set_title('Predicted Mask')
                        axes[i, 2].axis('off')
                        
                        # Column 4: Prediction Overlay
                        pred_overlay = self._create_mask_overlay(image, pred_mask, color=[255, 0, 0])
                        axes[i, 3].imshow(pred_overlay)
                        axes[i, 3].set_title('Prediction Overlay (Red)')
                        axes[i, 3].axis('off')
                
                except Exception as e:
                    print(f"Error visualizing {result['image_path']}: {e}")
                    # Fill row with error message
                    for col in range(4):
                        axes[i, col].text(0.5, 0.5, f'Error loading\n{os.path.basename(result["image_path"])}', 
                                         ha='center', va='center', transform=axes[i, col].transAxes,
                                         fontsize=10, bbox=dict(boxstyle="round", facecolor='lightcoral'))
                        axes[i, col].axis('off')
            
            # Fill empty rows if we have fewer samples than max_samples
            for i in range(len(valid_results), max_samples):
                for col in range(4):
                    axes[i, col].axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Add metrics distribution page if we have ground truth
            self._add_metrics_to_pdf(pdf, valid_results)
        
        print(f"Segmentation visualization saved to: {pdf_path}")
    
    def _add_metrics_to_pdf(self, pdf, valid_results: List[Dict]):
        """Add metrics distribution plots to PDF"""
        valid_metrics = [r for r in valid_results if 'metrics' in r]
        
        if not valid_metrics:
            return
        
        metrics_data = [r['metrics'] for r in valid_metrics]
        
        # Extract metric values
        dice_scores = [m['dice_score'] for m in metrics_data]
        iou_scores = [m['iou'] for m in metrics_data]
        pixel_accuracies = [m['pixel_accuracy'] for m in metrics_data]
        
        # Create metrics visualization page
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Dice Score Distribution
        axes[0, 0].hist(dice_scores, bins=min(10, len(dice_scores)), alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title(f'Dice Score Distribution\nMean: {np.mean(dice_scores):.3f} ± {np.std(dice_scores):.3f}')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU Distribution
        axes[0, 1].hist(iou_scores, bins=min(10, len(iou_scores)), alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title(f'IoU Distribution\nMean: {np.mean(iou_scores):.3f} ± {np.std(iou_scores):.3f}')
        axes[0, 1].set_xlabel('IoU')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Pixel Accuracy Distribution
        axes[1, 0].hist(pixel_accuracies, bins=min(10, len(pixel_accuracies)), alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title(f'Pixel Accuracy Distribution\nMean: {np.mean(pixel_accuracies):.3f} ± {np.std(pixel_accuracies):.3f}')
        axes[1, 0].set_xlabel('Pixel Accuracy')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics table
        axes[1, 1].axis('off')
        summary_text = f"""
        SEGMENTATION METRICS SUMMARY
        
        Dataset: {self.dataset_name}
        Number of samples: {len(valid_metrics)}
        
        Dice Score:
        • Mean: {np.mean(dice_scores):.4f}
        • Std:  {np.std(dice_scores):.4f}
        • Min:  {np.min(dice_scores):.4f}
        • Max:  {np.max(dice_scores):.4f}
        
        IoU Score:
        • Mean: {np.mean(iou_scores):.4f}
        • Std:  {np.std(iou_scores):.4f}
        • Min:  {np.min(iou_scores):.4f}
        • Max:  {np.max(iou_scores):.4f}
        
        Pixel Accuracy:
        • Mean: {np.mean(pixel_accuracies):.4f}
        • Std:  {np.std(pixel_accuracies):.4f}
        • Min:  {np.min(pixel_accuracies):.4f}
        • Max:  {np.max(pixel_accuracies):.4f}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_mask_overlay(self, image: np.ndarray, mask: np.ndarray, color: list, alpha: float = 0.5) -> np.ndarray:
        """Create overlay of mask on image"""
        overlay = image.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend with original image
        result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        return result
    
    def _create_combined_mask_overlay(self, image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Create combined overlay showing GT, prediction, and overlap"""
        overlay = image.copy()
        colored_overlay = np.zeros_like(image)
        
        # GT only (Green)
        gt_only = (gt_mask > 0) & (pred_mask == 0)
        colored_overlay[gt_only] = [0, 255, 0]
        
        # Prediction only (Red)
        pred_only = (pred_mask > 0) & (gt_mask == 0)
        colored_overlay[pred_only] = [255, 0, 0]
        
        # Overlap (Yellow)
        overlap = (gt_mask > 0) & (pred_mask > 0)
        colored_overlay[overlap] = [255, 255, 0]
        
        # Blend with original image
        result = cv2.addWeighted(overlay, 1 - alpha, colored_overlay, alpha, 0)
        return result
    
    def _plot_confusion_matrix(self, results: List[Dict]):
        """Plot confusion matrix for classification and save to PDF"""
        valid_results = [r for r in results if 'ground_truth_idx' in r]
        
        if not valid_results:
            return
        
        predictions = [r['predicted_class_idx'] for r in valid_results]
        ground_truths = [r['ground_truth_idx'] for r in valid_results]
        
        cm = confusion_matrix(ground_truths, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save as both PNG and PDF
        plt.savefig(self.output_dir / f"confusion_matrix_{self.dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"confusion_matrix_{self.dataset_name}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save results and metrics"""
        if not self.inference_config.get('save_detailed_results', True):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"results_{self.dataset_name}_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save metrics summary
        metrics_file = self.output_dir / f"metrics_{self.dataset_name}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Generate text report if requested
        if self.inference_config.get('generate_report', True):
            self._generate_report(timestamp)
        
        print(f"Results saved to: {results_file}")
        print(f"Metrics saved to: {metrics_file}")
    
    def _generate_report(self, timestamp: str):
        """Generate text report"""
        report_file = self.output_dir / f"report_{self.dataset_name}_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("FOUNDATION MODEL INFERENCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Dataset Path: {self.dataset_path}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples processed: {len(self.results)}\n")
            f.write(f"Available datasets in model: {self.available_datasets[self.task]}\n\n")
            
            if self.metrics:
                f.write("METRICS\n")
                f.write("-" * 20 + "\n")
                for key, value in self.metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        
        print(f"Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Foundation Model Inference (Config-based)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, help='Override model path from config')
    parser.add_argument('--dataset_path', type=str, help='Override dataset path from config')
    parser.add_argument('--dataset_name', type=str, help='Override dataset name from config')
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], help='Override task from config')
    parser.add_argument('--output_dir', type=str, help='Override output directory from config')
    parser.add_argument('--no_visualize', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Foundation Model Inference")
    print(f"Config file: {args.config}")
    print("-" * 50)
    
    try:
        # Initialize inference
        inference = FoundationModelInference(args.config)
        
        # Override config settings with command line arguments if provided
        if args.model_path:
            inference.model_path = args.model_path
        if args.dataset_path:
            inference.dataset_path = Path(args.dataset_path)
        if args.dataset_name:
            inference.dataset_name = args.dataset_name
        if args.task:
            inference.task = args.task
        if args.output_dir:
            inference.output_dir = Path(args.output_dir)
            inference.output_dir.mkdir(parents=True, exist_ok=True)
        if args.no_visualize:
            inference.inference_config['visualize'] = False
        
        # Run inference
        inference.run_inference()
        
        # Generate visualizations
        inference.visualize_results()
        
        print(f"\n✅ Inference Complete!")
        print(f"📁 All outputs saved to: {inference.output_dir}")
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
