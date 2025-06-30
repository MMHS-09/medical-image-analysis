#!/usr/bin/env python3
"""
Evaluation script for the trained foundation model
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_foundation_model
from src.dataset import create_test_dataloader
from src.metric import create_metrics_calculator
from src.visualization import Visualizer
from inference import FoundationModelInference


class ModelEvaluator:
    """Evaluate trained foundation model on test datasets"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.inference = FoundationModelInference(model_path, config_path)
        self.model = self.inference.model
        self.config = self.inference.config
        
        # Initialize visualizer
        self.visualizer = Visualizer("./evaluation_results")
        
        print(f"Model loaded for evaluation on {self.device}")
    
    def evaluate_classification_dataset(self, dataset_name: str, classes: list):
        """Evaluate on classification dataset"""
        print(f"\n🔍 Evaluating classification dataset: {dataset_name}")
        
        # Create test dataloader
        test_loader = create_test_dataloader(
            config=self.config,
            task='classification',
            dataset_name=dataset_name,
            classes=classes,
            batch_size=16
        )
        
        # Initialize metrics calculator
        metrics_calculator = create_metrics_calculator(
            task='classification',
            num_classes=len(classes),
            class_names=classes
        )
        
        # Evaluation loop
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, task='classification', dataset_name=dataset_name)
                
                # Store predictions and labels
                predictions = torch.softmax(outputs, dim=1)
                pred_classes = outputs.argmax(dim=1)
                
                all_predictions.extend(pred_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update metrics (using dummy loss for compatibility)
                metrics_calculator.update(outputs, labels, 0.0)
        
        # Compute final metrics
        metrics = metrics_calculator.compute_metrics()
        
        # Create visualizations
        confusion_matrix_path = self.visualizer.plot_confusion_matrix(
            y_true=np.array(all_labels),
            y_pred=np.array(all_predictions),
            class_names=classes,
            save_name=f"confusion_matrix_{dataset_name}.png"
        )
        
        # Sample predictions visualization
        sample_images, sample_labels = next(iter(test_loader))
        sample_images = sample_images[:8]  # Take first 8 samples
        sample_labels = sample_labels[:8]
        
        with torch.no_grad():
            sample_outputs = self.model(
                sample_images.to(self.device), 
                task='classification', 
                dataset_name=dataset_name
            )
        
        sample_viz_path = self.visualizer.plot_classification_results(
            images=sample_images,
            labels=sample_labels,
            predictions=sample_outputs,
            class_names=classes,
            save_name=f"classification_samples_{dataset_name}.png"
        )
        
        print(f"✅ Classification evaluation completed for {dataset_name}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
        print(f"  Confusion Matrix: {confusion_matrix_path}")
        print(f"  Sample Predictions: {sample_viz_path}")
        
        return metrics
    
    def evaluate_segmentation_dataset(self, dataset_name: str, num_classes: int):
        """Evaluate on segmentation dataset"""
        print(f"\n🔍 Evaluating segmentation dataset: {dataset_name}")
        
        # Create test dataloader
        test_loader = create_test_dataloader(
            config=self.config,
            task='segmentation',
            dataset_name=dataset_name,
            classes=None,
            batch_size=8  # Smaller batch size for segmentation
        )
        
        # Initialize metrics calculator
        class_names = [f"Class_{i}" for i in range(num_classes)]
        metrics_calculator = create_metrics_calculator(
            task='segmentation',
            num_classes=num_classes,
            class_names=class_names
        )
        
        # Evaluation loop
        self.model.eval()
        sample_batch = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, task='segmentation', dataset_name=dataset_name)
                
                # Store first batch for visualization
                if batch_idx == 0:
                    sample_batch = {
                        'images': images[:4].cpu(),
                        'masks': masks[:4].cpu(),
                        'outputs': outputs[:4].cpu()
                    }
                
                # Update metrics (using dummy loss for compatibility)
                metrics_calculator.update(outputs, masks, 0.0)
        
        # Compute final metrics
        metrics = metrics_calculator.compute_metrics()
        
        # Create visualization
        if sample_batch is not None:
            sample_viz_path = self.visualizer.plot_segmentation_results(
                images=sample_batch['images'],
                masks=sample_batch['masks'],
                predictions=sample_batch['outputs'],
                save_name=f"segmentation_samples_{dataset_name}.png"
            )
        
        print(f"✅ Segmentation evaluation completed for {dataset_name}")
        print(f"  Mean IoU: {metrics.get('mean_iou', 0):.4f}")
        print(f"  Mean Dice: {metrics.get('mean_dice', 0):.4f}")
        print(f"  Pixel Accuracy: {metrics.get('accuracy', 0):.4f}")
        if sample_batch is not None:
            print(f"  Sample Predictions: {sample_viz_path}")
        
        return metrics
    
    def evaluate_all_datasets(self):
        """Evaluate on all datasets"""
        print("🚀 Starting comprehensive evaluation...")
        
        all_results = {}
        
        # Evaluate classification datasets
        print("\n" + "="*60)
        print("CLASSIFICATION EVALUATION")
        print("="*60)
        
        for dataset_config in self.config.get('classification_datasets', []):
            dataset_name = dataset_config['name']
            classes = dataset_config['classes']
            
            try:
                metrics = self.evaluate_classification_dataset(dataset_name, classes)
                all_results[f"classification_{dataset_name}"] = metrics
            except Exception as e:
                print(f"❌ Error evaluating {dataset_name}: {str(e)}")
                all_results[f"classification_{dataset_name}"] = {"error": str(e)}
        
        # Evaluate segmentation datasets
        print("\n" + "="*60)
        print("SEGMENTATION EVALUATION")
        print("="*60)
        
        for dataset_config in self.config.get('segmentation_datasets', []):
            dataset_name = dataset_config['name']
            num_classes = dataset_config['num_classes']
            
            try:
                metrics = self.evaluate_segmentation_dataset(dataset_name, num_classes)
                all_results[f"segmentation_{dataset_name}"] = metrics
            except Exception as e:
                print(f"❌ Error evaluating {dataset_name}: {str(e)}")
                all_results[f"segmentation_{dataset_name}"] = {"error": str(e)}
        
        # Create final summary
        self.create_evaluation_summary(all_results)
        
        print("\n🎉 Comprehensive evaluation completed!")
        print(f"Results saved to: ./evaluation_results/")
        
        return all_results
    
    def create_evaluation_summary(self, results: dict):
        """Create evaluation summary"""
        # Save detailed results
        results_path = "./evaluation_results/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create metrics comparison plot
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            self.visualizer.plot_metrics_comparison(
                valid_results,
                save_name="evaluation_metrics_comparison.png"
            )
        
        # Create text summary
        summary_path = "./evaluation_results/evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("FOUNDATION MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Classification results
            f.write("CLASSIFICATION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, metrics in results.items():
                if key.startswith('classification_') and 'error' not in metrics:
                    dataset_name = key.replace('classification_', '')
                    f.write(f"\n{dataset_name}:\n")
                    f.write(f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"  F1 Score: {metrics.get('f1_score', 0):.4f}\n")
                    f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
            
            # Segmentation results
            f.write("\n\nSEGMENTATION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, metrics in results.items():
                if key.startswith('segmentation_') and 'error' not in metrics:
                    dataset_name = key.replace('segmentation_', '')
                    f.write(f"\n{dataset_name}:\n")
                    f.write(f"  Mean IoU: {metrics.get('mean_iou', 0):.4f}\n")
                    f.write(f"  Mean Dice: {metrics.get('mean_dice', 0):.4f}\n")
                    f.write(f"  Pixel Accuracy: {metrics.get('accuracy', 0):.4f}\n")
            
            # Errors
            errors = {k: v for k, v in results.items() if 'error' in v}
            if errors:
                f.write("\n\nERRORS:\n")
                f.write("-" * 30 + "\n")
                for key, error_info in errors.items():
                    f.write(f"{key}: {error_info['error']}\n")
        
        print(f"📄 Evaluation summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Foundation Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    parser.add_argument('--dataset_name', type=str, help='Specific dataset to evaluate (optional)')
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], 
                       help='Specific task to evaluate (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.config_path)
    
    if args.dataset_name and args.task:
        # Evaluate specific dataset
        if args.task == 'classification':
            # Find dataset config
            dataset_config = None
            for ds_config in evaluator.config.get('classification_datasets', []):
                if ds_config['name'] == args.dataset_name:
                    dataset_config = ds_config
                    break
            
            if dataset_config:
                evaluator.evaluate_classification_dataset(
                    args.dataset_name, 
                    dataset_config['classes']
                )
            else:
                print(f"❌ Classification dataset '{args.dataset_name}' not found in config")
        
        else:  # segmentation
            # Find dataset config
            dataset_config = None
            for ds_config in evaluator.config.get('segmentation_datasets', []):
                if ds_config['name'] == args.dataset_name:
                    dataset_config = ds_config
                    break
            
            if dataset_config:
                evaluator.evaluate_segmentation_dataset(
                    args.dataset_name, 
                    dataset_config['num_classes']
                )
            else:
                print(f"❌ Segmentation dataset '{args.dataset_name}' not found in config")
    
    else:
        # Evaluate all datasets
        evaluator.evaluate_all_datasets()


if __name__ == "__main__":
    main()
