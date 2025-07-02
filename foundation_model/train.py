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

from src.model import create_foundation_model
from src.dataset import create_dataloaders
from src.loss import get_loss_function
from src.metric import create_metrics_calculator, EarlyStopping
from src.visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FoundationModelTrainer:
    """Foundation Model Trainer for Medical Image Analysis"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['results_dir'], exist_ok=True)
        
        # Initialize model
        self.model = create_foundation_model(self.config)
        self.model.to(self.device)
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.config['paths']['results_dir'])
        
        # Training history
        self.training_history = {}
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_sequential(self):
        """Train the model sequentially on all datasets"""
        logger.info("Starting sequential training on all datasets")
        
        all_results = {}
        
        # Train classification datasets
        for dataset_config in self.config['classification_datasets']:
            dataset_name = dataset_config['name']
            classes = dataset_config['classes']
            
            logger.info(f"Training on classification dataset: {dataset_name}")
            
            results = self.train_dataset(
                task='classification',
                dataset_name=dataset_name,
                classes=classes,
                num_classes=len(classes)
            )
            
            all_results[f"classification_{dataset_name}"] = results
        
        # Train segmentation datasets
        for dataset_config in self.config['segmentation_datasets']:
            dataset_name = dataset_config['name']
            num_classes = dataset_config['num_classes']
            
            logger.info(f"Training on segmentation dataset: {dataset_name}")
            
            results = self.train_dataset(
                task='segmentation',
                dataset_name=dataset_name,
                classes=None,
                num_classes=num_classes
            )
            
            all_results[f"segmentation_{dataset_name}"] = results
        
        # Save final model
        self.save_final_model()
        
        # Create final results summary
        self.create_final_summary(all_results)
        
        logger.info("Sequential training completed!")
        
        return all_results
    
    def train_dataset(self, task: str, dataset_name: str, classes: list = None, num_classes: int = None):
        """Train on a specific dataset"""
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING: {task.upper()} - {dataset_name}")
        print(f"{'='*60}")
        
        if task == 'classification':
            print(f"📊 Classes: {classes}")
            print(f"🎯 Target Metric: Accuracy")
        else:
            print(f"🎯 Target Metrics: Dice Score, IoU")
        
        print(f"📦 Batch Size: {self.config['training']['batch_size']}")
        print(f"🔄 Max Epochs: {self.config['training']['num_epochs']}")
        print(f"🖼️  Image Size: {self.config['data']['image_size']}")
        
        # Show data limiting info
        data_fraction = self.config['data'].get('data_fraction', 1.0)
        max_samples_per_class = self.config['data'].get('max_samples_per_class', None)
        max_samples_per_dataset = self.config['data'].get('max_samples_per_dataset', None)
        
        if data_fraction < 1.0:
            print(f"📊 Data Fraction: {data_fraction:.2%} (testing mode)")
        if max_samples_per_class:
            print(f"🔢 Max Samples per Class: {max_samples_per_class}")
        if max_samples_per_dataset:
            print(f"🔢 Max Samples per Dataset: {max_samples_per_dataset}")
        
        print("")
        
        logger.info(f"Training {task} task on {dataset_name} dataset")
        
        # Create dataloaders
        dataloaders = create_dataloaders(
            config=self.config,
            task=task,
            dataset_name=dataset_name,
            classes=classes,
            batch_size=self.config['training']['batch_size']
        )
        
        # Show dataset size info and data limiting info
        train_size = len(dataloaders['train'].dataset)
        val_size = len(dataloaders['val'].dataset)
        total_size = train_size + val_size
        
        print(f"📊 Dataset Size:")
        print(f"  Total samples: {total_size}")
        print(f"  Training: {train_size} samples")
        print(f"  Validation: {val_size} samples")
        
        # Show data limiting configuration
        data_config = self.config['data']
        print(f"📉 Data Limiting Configuration:")
        print(f"  Data fraction: {data_config.get('data_fraction', 1.0):.1%}")
        if task == 'classification':
            max_per_class = data_config.get('max_samples_per_class', 'unlimited')
            print(f"  Max samples per class: {max_per_class}")
        else:
            max_per_dataset = data_config.get('max_samples_per_dataset', 'unlimited')
            print(f"  Max samples per dataset: {max_per_dataset}")
        print("")
        
        # Setup loss function
        loss_fn = get_loss_function(task=task, num_classes=num_classes)
        
        # Setup optimizer (only train the specific head + shared backbone)
        if task == 'classification':
            # Freeze other heads
            for name, param in self.model.named_parameters():
                if 'segmentation_layers' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:  # segmentation
            # Freeze other heads
            for name, param in self.model.named_parameters():
                if 'classification_layers' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-4
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Setup metrics calculator
        class_names = classes if classes else [f"Class_{i}" for i in range(num_classes)]
        metrics_calculator = create_metrics_calculator(task, num_classes, class_names)
        
        # Setup early stopping (if enabled)
        early_stopping = None
        if self.config['training'].get('enable_early_stopping', True):
            early_stopping = EarlyStopping(
                patience=self.config['training']['early_stopping_patience'],
                mode='min'
            )
            logger.info(f"Early stopping enabled with patience: {self.config['training']['early_stopping_patience']}")
        else:
            logger.info("Early stopping disabled - training will run for full number of epochs")
        
        # Training loop
        best_val_loss = float('inf')
        dataset_history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}
        
        for epoch in range(self.config['training']['num_epochs']):
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
            dataset_history['train_loss'].append(train_metrics['loss'])
            dataset_history['val_loss'].append(val_metrics['loss'])
            dataset_history['train_metrics'].append(train_metrics)
            dataset_history['val_metrics'].append(val_metrics)
            
            # Scheduler step - track learning rate changes
            old_lr = scheduler.optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['loss'])
            new_lr = scheduler.optimizer.param_groups[0]['lr']
            
            # Log learning rate changes manually (replaces deprecated verbose=True)
            if old_lr != new_lr:
                logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            if task == 'classification':
                train_acc = train_metrics.get('accuracy', 0.0)
                val_acc = val_metrics.get('accuracy', 0.0)
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
                
                # Also print to console for immediate feedback
                print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']} - "
                      f"Loss: {val_metrics['loss']:.4f}, Accuracy: {val_acc:.4f}")
                
            else:  # segmentation
                train_dice = train_metrics.get('mean_dice', 0.0)
                val_dice = val_metrics.get('mean_dice', 0.0)
                train_iou = train_metrics.get('mean_iou', 0.0)
                val_iou = val_metrics.get('mean_iou', 0.0)
                
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
                logger.info(f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")
                
                # Also print to console for immediate feedback
                print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']} - "
                      f"Loss: {val_metrics['loss']:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, task, dataset_name, val_metrics['loss'])
            
            # Early stopping check (if enabled)
            if early_stopping is not None and early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch, task, dataset_name, val_metrics['loss'], is_best=False)
        
        # Store history
        self.training_history[f"{task}_{dataset_name}"] = dataset_history
        
        # Plot training history
        plot_history = self._prepare_history_for_plotting(dataset_history)
        self.visualizer.plot_training_history(
            plot_history,
            save_name=f"{task}_{dataset_name}_training_history.png"
        )
        
        # Save training history to CSV
        csv_path = self.visualizer.save_training_history_to_csv(
            plot_history, 
            dataset_name, 
            task
        )
        logger.info(f"Training history saved to CSV: {csv_path}")
        
        # Print final results summary
        final_metrics = dataset_history['val_metrics'][-1]
        print(f"\n🎉 TRAINING COMPLETED: {task.upper()} - {dataset_name}")
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
        
        # Return final validation metrics
        return dataset_history['val_metrics'][-1]
    
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
                
                # Update progress bar with current metrics
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
    
    def save_checkpoint(self, epoch, task, dataset_name, val_loss, is_best=True):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'task': task,
            'dataset_name': dataset_name,
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = os.path.join(
                self.config['paths']['models_dir'],
                f"best_{task}_{dataset_name}.pth"
            )
        else:
            checkpoint_path = os.path.join(
                self.config['paths']['models_dir'],
                f"checkpoint_{task}_{dataset_name}_epoch_{epoch+1}.pth"
            )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save the final foundation model"""
        model_path = os.path.join(self.config['paths']['models_dir'], "foundation_model_final.pth")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, model_path)
        
        logger.info(f"Final model saved: {model_path}")
    
    def create_final_summary(self, all_results):
        """Create final results summary"""
        # Save results as JSON
        results_path = os.path.join(self.config['paths']['results_dir'], "all_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create text summary
        self.visualizer.create_results_summary(all_results)
        
        # Create metrics comparison plot
        self.visualizer.plot_metrics_comparison(all_results)
        
        # Save all training histories to CSV
        all_histories_csv = self.visualizer.save_all_training_histories_to_csv(
            self.training_history,
            save_name="all_training_histories.csv"
        )
        logger.info(f"All training histories saved to CSV: {all_histories_csv}")
        
        # Save final metrics summary to CSV  
        final_metrics_csv = self.visualizer.save_final_metrics_summary_to_csv(
            all_results,
            save_name="final_metrics_summary.csv"
        )
        logger.info(f"Final metrics summary saved to CSV: {final_metrics_csv}")
        
        logger.info("Final summary created!")
    
    def _prepare_history_for_plotting(self, history):
        """Prepare history for plotting"""
        plot_history = {}
        
        # Extract metrics from train and val metrics
        if history['train_metrics']:
            for key in history['train_metrics'][0].keys():
                plot_history[f'train_{key}'] = [m[key] for m in history['train_metrics']]
                
        if history['val_metrics']:
            for key in history['val_metrics'][0].keys():
                plot_history[f'val_{key}'] = [m[key] for m in history['val_metrics']]
        
        return plot_history


def main():
    parser = argparse.ArgumentParser(description='Train Foundation Model for Medical Image Analysis')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FoundationModelTrainer(args.config)
    
    # Start training
    results = trainer.train_sequential()
    
    print("Training completed successfully!")
    print(f"Results saved to: {trainer.config['paths']['results_dir']}")


if __name__ == "__main__":
    main()