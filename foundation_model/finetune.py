import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import argparse
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_foundation_model, load_model_weights_safely
from src.dataset import create_dataloaders
from src.loss import get_loss_function
from src.metric import create_metrics_calculator, EarlyStopping
from src.visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FoundationModelFineTuner:
    """Foundation Model Fine-tuner for Medical Image Analysis"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['results_dir'], exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.config)
        
        # Load foundation model
        self.load_foundation_model()
        
        # Training history
        self.finetuning_history = {}
    
    def load_foundation_model(self):
        """Load the pre-trained foundation model"""
        foundation_model_path = self.config['finetuning']['foundation_model_path']
        
        if not os.path.exists(foundation_model_path):
            raise FileNotFoundError(f"Foundation model not found at {foundation_model_path}")
        
        logger.info(f"Loading foundation model from {foundation_model_path}")
        
        # Load model state (set weights_only=False for compatibility with older checkpoints)
        checkpoint = torch.load(foundation_model_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration from checkpoint
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            logger.info("Found model config in checkpoint")
        else:
            # If no model config, we need to reconstruct it from the original training config
            logger.warning("No model config found in checkpoint, reconstructing from training config")
            model_config = {
                'backbone': self.config['model']['backbone'],
                'pretrained': False,  # Already trained
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
        
        # Create model with the exact same configuration as the saved model
        self.model = create_foundation_model(
            backbone=model_config['backbone'],
            pretrained=model_config.get('pretrained', False),
            dropout=model_config.get('dropout', 0.2),
            classification_heads=model_config.get('classification_heads', {}),
            segmentation_heads=model_config.get('segmentation_heads', {})
        ).to(self.device)
        
        # Load model weights with proper error handling
        suppress_warnings = self.config.get('logging', {}).get('suppress_model_loading_warnings', False)
        missing_keys, unexpected_keys = load_model_weights_safely(self.model, checkpoint, logger, suppress_warnings)
        
        logger.info(f"Foundation model loaded successfully")
        logger.info(f"Loaded classification heads: {list(self.model.classification_heads.keys())}")
        logger.info(f"Loaded segmentation heads: {list(self.model.segmentation_heads.keys())}")
        logger.info("Foundation model is ready for finetuning on new tasks")
    
    def add_task_head(self, task: str, dataset_name: str, classes: list = None, num_classes: int = None):
        """Add a new task-specific head to the foundation model"""
        if task == 'classification':
            if classes is None:
                raise ValueError("Classes must be provided for classification task")
            num_classes = len(classes)
            self.model.add_classification_head(dataset_name, num_classes, self.config['model']['dropout'])
            logger.info(f"Added classification head for {dataset_name} with {num_classes} classes")
        
        elif task == 'segmentation':
            if num_classes is None:
                raise ValueError("num_classes must be provided for segmentation task")
            self.model.add_segmentation_head(dataset_name, num_classes, self.config['model']['dropout'])
            logger.info(f"Added segmentation head for {dataset_name} with {num_classes} classes")
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def finetune_all_tasks(self):
        """Fine-tune the foundation model on all specified tasks"""
        all_results = {}
        
        for task_config in self.config['finetuning']['tasks']:
            task = task_config['task']
            dataset_name = task_config['dataset_name']
            
            # Add task-specific head
            if task == 'classification':
                classes = task_config['classes']
                self.add_task_head(task, dataset_name, classes=classes)
                results = self.finetune_task(task, dataset_name, classes=classes, task_config=task_config)
            else:  # segmentation
                num_classes = task_config['num_classes']
                self.add_task_head(task, dataset_name, num_classes=num_classes)
                results = self.finetune_task(task, dataset_name, num_classes=num_classes, task_config=task_config)
            
            all_results[f"{task}_{dataset_name}"] = results
            
            # Save intermediate model
            self.save_finetuned_model(task_config)
        
        # Create final results summary
        self.create_final_summary(all_results)
        
        logger.info("Fine-tuning completed!")
        
        return all_results
    
    def finetune_task(self, task: str, dataset_name: str, classes: list = None, num_classes: int = None, task_config: dict = None):
        """Fine-tune on a specific task"""
        print(f"\n{'='*60}")
        print(f"🔧 FINE-TUNING: {task.upper()} - {dataset_name}")
        print(f"{'='*60}")
        
        if task == 'classification':
            print(f"📊 Classes: {classes}")
            print(f"🎯 Target Metric: Accuracy")
        else:
            print(f"🎯 Target Metrics: Dice Score, IoU")
        
        print(f"📦 Batch Size: {self.config['training']['batch_size']}")
        print(f"🔄 Max Epochs: {self.config['finetuning']['num_epochs']}")
        print(f"🖼️  Image Size: {self.config['data']['image_size']}")
        print(f"📂 Data Path: {task_config['data_path']}")
        print("")
        
        logger.info(f"Fine-tuning {task} task on {dataset_name} dataset")
        
        # Create dataloaders with custom data path
        dataloaders = create_dataloaders(
            config=self.config,
            task=task,
            dataset_name=dataset_name,
            classes=classes,
            batch_size=self.config['training']['batch_size'],
            data_path=task_config['data_path']  # Use custom data path
        )
        
        # Show dataset size info
        train_size = len(dataloaders['train'].dataset)
        val_size = len(dataloaders['val'].dataset)
        total_size = train_size + val_size
        
        print(f"📊 Dataset Size:")
        print(f"  Total samples: {total_size}")
        print(f"  Training: {train_size} samples")
        print(f"  Validation: {val_size} samples")
        print("")
        
        # Setup loss function
        loss_fn = get_loss_function(task=task, num_classes=num_classes)
        
        # Setup optimizer with different strategies
        if self.config['finetuning']['freeze_backbone']:
            # Freeze backbone, only train task-specific head
            for name, param in self.model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
                elif f"{task}_layers.{dataset_name}" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            logger.info("Backbone frozen, only training task-specific head")
        else:
            # Fine-tune entire model with lower learning rate for backbone
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'backbone' in name:
                    backbone_params.append(param)
                elif f"{task}_layers.{dataset_name}" in name:
                    head_params.append(param)
                else:
                    param.requires_grad = False  # Freeze other task heads
            
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config['finetuning']['learning_rate'] * 0.1},  # Lower LR for backbone
                {'params': head_params, 'lr': self.config['finetuning']['learning_rate']}  # Higher LR for new head
            ], weight_decay=1e-4)
            
            logger.info("Fine-tuning entire model with differential learning rates")
        
        if self.config['finetuning']['freeze_backbone']:
            optimizer = optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config['finetuning']['learning_rate'],
                weight_decay=1e-4
            )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Setup metrics calculator
        class_names = classes if classes else [f"Class_{i}" for i in range(num_classes)]
        metrics_calculator = create_metrics_calculator(task, num_classes, class_names)
        
        # Setup early stopping (if enabled)
        early_stopping = None
        if self.config['finetuning'].get('enable_early_stopping', True):
            early_stopping = EarlyStopping(
                patience=self.config['finetuning']['early_stopping_patience'],
                mode='min'
            )
            logger.info(f"Early stopping enabled with patience: {self.config['finetuning']['early_stopping_patience']}")
        else:
            logger.info("Early stopping disabled - finetuning will run for full number of epochs")
        
        # Training loop
        best_val_loss = float('inf')
        task_history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}
        
        for epoch in range(self.config['finetuning']['num_epochs']):
            # Training phase
            train_metrics = self.train_epoch(
                model=self.model,
                dataloader=dataloaders['train'],
                loss_fn=loss_fn,
                optimizer=optimizer,
                metrics_calculator=metrics_calculator,
                task=task,
                dataset_name=dataset_name
            )
            
            # Validation phase
            val_metrics = self.validate_epoch(
                model=self.model,
                dataloader=dataloaders['val'],
                loss_fn=loss_fn,
                metrics_calculator=metrics_calculator,
                task=task,
                dataset_name=dataset_name
            )
            
            # Update history
            task_history['train_loss'].append(train_metrics['loss'])
            task_history['val_loss'].append(val_metrics['loss'])
            task_history['train_metrics'].append(train_metrics)
            task_history['val_metrics'].append(val_metrics)
            
            # Scheduler step
            old_lr = scheduler.optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['loss'])
            new_lr = scheduler.optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.config['finetuning']['num_epochs']}")
            
            if task == 'classification':
                train_acc = train_metrics.get('accuracy', 0.0)
                val_acc = val_metrics.get('accuracy', 0.0)
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
                
                print(f"Epoch {epoch+1}/{self.config['finetuning']['num_epochs']} - "
                      f"Loss: {val_metrics['loss']:.4f}, Accuracy: {val_acc:.4f}")
                
            else:  # segmentation
                train_dice = train_metrics.get('mean_dice', 0.0)
                val_dice = val_metrics.get('mean_dice', 0.0)
                train_iou = train_metrics.get('mean_iou', 0.0)
                val_iou = val_metrics.get('mean_iou', 0.0)
                
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
                logger.info(f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")
                
                print(f"Epoch {epoch+1}/{self.config['finetuning']['num_epochs']} - "
                      f"Loss: {val_metrics['loss']:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, task, dataset_name, val_metrics['loss'], task_config)
            
            # Early stopping check (if enabled)
            if early_stopping is not None and early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config['finetuning']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch, task, dataset_name, val_metrics['loss'], task_config, is_best=False)
        
        # Store history
        self.finetuning_history[f"{task}_{dataset_name}"] = task_history
        
        # Plot training history
        plot_history = self._prepare_history_for_plotting(task_history)
        self.visualizer.plot_training_history(
            plot_history,
            save_name=f"{task}_{dataset_name}_finetuning_history.png"
        )
        
        # Save training history to CSV
        csv_path = self.visualizer.save_training_history_to_csv(
            plot_history, 
            dataset_name, 
            task,
            prefix="finetuning_"
        )
        logger.info(f"Fine-tuning history saved to CSV: {csv_path}")
        
        # Print final results summary
        final_metrics = task_history['val_metrics'][-1]
        print(f"\n🎉 FINE-TUNING COMPLETED: {task.upper()} - {dataset_name}")
        print(f"{'='*60}")
        print(f"📈 Final Results:")
        print(f"  Loss: {final_metrics['loss']:.4f}")
        
        if task == 'classification':
            print(f"  Accuracy: {final_metrics.get('accuracy', 0):.4f}")
            print(f"  F1 Score: {final_metrics.get('f1_score', 0):.4f}")
        else:
            print(f"  Dice Score: {final_metrics.get('mean_dice', 0):.4f}")
            print(f"  IoU Score: {final_metrics.get('mean_iou', 0):.4f}")
        
        print(f"{'='*60}\n")
        
        # Save results to task-specific directory
        self.save_task_results(task_config, final_metrics, task_history)
        
        return task_history['val_metrics'][-1]
    
    def train_epoch(self, model, dataloader, loss_fn, optimizer, metrics_calculator, task, dataset_name):
        """Train for one epoch"""
        model.train()
        metrics_calculator.reset()
        
        pbar = tqdm(dataloader, desc=f"Training")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            
            if task == 'classification':
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = model(images, task=task, dataset_name=dataset_name)
                loss = loss_fn(outputs, labels)
                
                # Update metrics
                metrics_calculator.update(outputs, labels, loss.item())
                
            else:  # segmentation
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = model(images, task=task, dataset_name=dataset_name)
                
                loss = loss_fn(outputs, masks)
                
                # Update metrics
                metrics_calculator.update(outputs, masks, loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            current_metrics = metrics_calculator.compute_metrics()
            if task == 'classification':
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_metrics.get('accuracy', 0):.3f}"
                })
            else:  # segmentation
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{current_metrics.get('mean_dice', 0):.3f}"
                })
        
        return metrics_calculator.compute_metrics()
    
    def validate_epoch(self, model, dataloader, loss_fn, metrics_calculator, task, dataset_name):
        """Validate for one epoch"""
        model.eval()
        metrics_calculator.reset()
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validation")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                
                if task == 'classification':
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    outputs = model(images, task=task, dataset_name=dataset_name)
                    loss = loss_fn(outputs, labels)
                    
                    # Update metrics
                    metrics_calculator.update(outputs, labels, loss.item())
                    
                else:  # segmentation
                    masks = batch['mask'].to(self.device)
                    
                    # Forward pass
                    outputs = model(images, task=task, dataset_name=dataset_name)
                    loss = loss_fn(outputs, masks)
                    
                    # Update metrics
                    metrics_calculator.update(outputs, masks, loss.item())
                
                # Update progress bar with current loss
                current_metrics = metrics_calculator.compute_metrics()
                if task == 'classification':
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{current_metrics.get('accuracy', 0):.3f}"
                    })
                else:  # segmentation
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'dice': f"{current_metrics.get('mean_dice', 0):.3f}"
                    })
        
        return metrics_calculator.compute_metrics()
    
    def save_checkpoint(self, epoch, task, dataset_name, val_loss, task_config, is_best=True):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'task': task,
            'dataset_name': dataset_name,
            'model_config': {
                'backbone': self.model.backbone_name,
                'classification_heads': self.model.classification_heads,
                'segmentation_heads': self.model.segmentation_heads,
                'dropout': self.config['model']['dropout']
            }
        }
        
        if is_best:
            # Save to task-specific path
            save_path = task_config['model_save_path']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
            logger.info(f"Best model saved to {save_path}")
        else:
            # Save checkpoint with epoch info
            save_path = os.path.join(
                self.config['paths']['models_dir'],
                f"checkpoint_finetuned_{task}_{dataset_name}_epoch_{epoch+1}.pth"
            )
            torch.save(checkpoint, save_path)
    
    def save_finetuned_model(self, task_config):
        """Save the final fine-tuned model"""
        model_path = task_config['model_save_path']
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'backbone': self.model.backbone_name,
                'classification_heads': self.model.classification_heads,
                'segmentation_heads': self.model.segmentation_heads,
                'dropout': self.config['model']['dropout']
            },
            'task_config': task_config,
            'training_config': self.config['finetuning']
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Fine-tuned model saved to {model_path}")
    
    def save_task_results(self, task_config, final_metrics, history):
        """Save task-specific results"""
        results_path = task_config['results_save_path']
        os.makedirs(results_path, exist_ok=True)
        
        # Save final metrics
        metrics_file = os.path.join(results_path, 'final_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save training history
        history_file = os.path.join(results_path, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def _prepare_history_for_plotting(self, history):
        """Prepare history data for plotting"""
        plot_history = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        }
        
        # Add task-specific metrics
        if 'accuracy' in history['train_metrics'][0]:
            plot_history['train_accuracy'] = [m.get('accuracy', 0) for m in history['train_metrics']]
            plot_history['val_accuracy'] = [m.get('accuracy', 0) for m in history['val_metrics']]
        
        if 'mean_dice' in history['train_metrics'][0]:
            plot_history['train_dice'] = [m.get('mean_dice', 0) for m in history['train_metrics']]
            plot_history['val_dice'] = [m.get('mean_dice', 0) for m in history['val_metrics']]
            plot_history['train_iou'] = [m.get('mean_iou', 0) for m in history['train_metrics']]
            plot_history['val_iou'] = [m.get('mean_iou', 0) for m in history['val_metrics']]
        
        return plot_history
    
    def create_final_summary(self, all_results):
        """Create final summary of all fine-tuning results"""
        summary = {
            'finetuning_timestamp': datetime.now().isoformat(),
            'foundation_model_path': self.config['finetuning']['foundation_model_path'],
            'tasks_completed': len(all_results),
            'results': {}
        }
        
        for task_dataset, metrics in all_results.items():
            summary['results'][task_dataset] = {
                'final_loss': metrics['loss'],
                'final_metrics': metrics
            }
        
        # Save summary
        summary_path = os.path.join(self.config['paths']['results_dir'], 'finetuning_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Final summary saved to {summary_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("🎉 FINE-TUNING SUMMARY")
        print(f"{'='*60}")
        print(f"Foundation Model: {self.config['finetuning']['foundation_model_path']}")
        print(f"Tasks Completed: {len(all_results)}")
        print(f"Timestamp: {summary['finetuning_timestamp']}")
        print("")
        
        for task_dataset, metrics in all_results.items():
            print(f"📊 {task_dataset}:")
            print(f"  Final Loss: {metrics['loss']:.4f}")
            if 'accuracy' in metrics:
                print(f"  Final Accuracy: {metrics['accuracy']:.4f}")
            if 'mean_dice' in metrics:
                print(f"  Final Dice: {metrics['mean_dice']:.4f}")
                print(f"  Final IoU: {metrics['mean_iou']:.4f}")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Foundation Model for Medical Image Analysis")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--task', type=str, help='Specific task to fine-tune (optional)')
    parser.add_argument('--dataset', type=str, help='Specific dataset to fine-tune on (optional)')
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    finetuner = FoundationModelFineTuner(args.config)
    
    if args.task and args.dataset:
        # Fine-tune specific task
        task_config = None
        for config in finetuner.config['finetuning']['tasks']:
            if config['task'] == args.task and config['dataset_name'] == args.dataset:
                task_config = config
                break
        
        if task_config is None:
            logger.error(f"Task {args.task} with dataset {args.dataset} not found in configuration")
            return
        
        # Add task head and fine-tune
        if args.task == 'classification':
            finetuner.add_task_head(args.task, args.dataset, classes=task_config['classes'])
            finetuner.finetune_task(args.task, args.dataset, classes=task_config['classes'], task_config=task_config)
        else:
            finetuner.add_task_head(args.task, args.dataset, num_classes=task_config['num_classes'])
            finetuner.finetune_task(args.task, args.dataset, num_classes=task_config['num_classes'], task_config=task_config)
    
    else:
        # Fine-tune all tasks
        finetuner.finetune_all_tasks()


if __name__ == "__main__":
    main()
