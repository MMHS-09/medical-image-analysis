import os
import re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend to avoid Qt plugin issues
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

class Visualizer:
    """
    Class for visualization functions related to deep learning models,
    particularly for segmentation and classification tasks.
    """
    
    def __init__(self, save_dir, logger=None):
        """
        Initialize the visualizer
        
        Args:
            save_dir: Directory to save visualizations
            logger: Logger instance for logging messages
        """
        self.save_dir = Path(save_dir)
        self.logger = logger
        
        # Create visualization directory
        self.vis_dir = self.save_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figures directory
        self.fig_dir = self.save_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, dataset_id, history, task_type):
        """Generate and save plots of training and validation metrics"""
        
        # Common plotting code for loss
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{dataset_id} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.fig_dir / f"{dataset_id}_loss.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        if task_type == 'classification':
            # Plot accuracy
            plt.figure(figsize=(12, 6))
            plt.plot(history['train_accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{dataset_id} - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.fig_dir / f"{dataset_id}_accuracy.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Optional: Plot other metrics like F1 if available
            if 'val_f1_macro' in history:
                plt.figure(figsize=(12, 6))
                plt.plot(history['val_f1_macro'], label='F1 Macro')
                plt.plot(history['val_precision_macro'], label='Precision Macro')
                plt.plot(history['val_recall_macro'], label='Recall Macro')
                plt.title(f'{dataset_id} - Validation Metrics')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.fig_dir / f"{dataset_id}_metrics.pdf", dpi=300, bbox_inches='tight')
                plt.close()
                
        elif task_type == 'segmentation':
            # Plot Dice coefficient
            plt.figure(figsize=(12, 6))
            plt.plot(history['train_dice'], label='Training Dice')
            plt.plot(history['val_dice'], label='Validation Dice')
            plt.title(f'{dataset_id} - Dice Coefficient')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.fig_dir / f"{dataset_id}_dice.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot IoU
            if 'val_iou' in history:
                plt.figure(figsize=(12, 6))
                plt.plot(history['train_iou'], label='Training IoU')
                plt.plot(history['val_iou'], label='Validation IoU')
                plt.title(f'{dataset_id} - IoU')
                plt.xlabel('Epoch')
                plt.ylabel('IoU')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.fig_dir / f"{dataset_id}_iou.pdf", dpi=300, bbox_inches='tight')
                plt.close()
    
    def visualize_segmentation_results(self, dataset_id, model, dataloader, device):
        """Generate and save visualizations of segmentation predictions"""
        # Create visualization directory
        vis_dir = self.vis_dir / dataset_id
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get a small batch of samples for visualization
        num_samples = min(8, len(dataloader.dataset))
        samples_to_visualize = []
        
        with torch.no_grad():
            # Get one batch
            for batch in dataloader:
                # Move data to device
                batch['image'] = batch['image'].to(device)
                batch['mask'] = batch['mask'].to(device)
                
                # Add task info to batch if it doesn't exist
                if 'task' not in batch:
                    batch['task'] = ['segmentation'] * batch['image'].size(0)
                
                # Forward pass
                outputs = model(batch)
                seg_output = outputs['segmentation']
                
                # Convert outputs to binary masks
                if seg_output.shape[1] > 1:  # Multi-class segmentation
                    preds = torch.argmax(seg_output, dim=1, keepdim=True)
                else:  # Binary segmentation
                    preds = (torch.sigmoid(seg_output) > 0.5).float()
                
                # Prepare samples for visualization
                for i in range(min(batch['image'].size(0), num_samples)):
                    # Get image, mask, and prediction
                    image = batch['image'][i].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                    
                    # Normalize image for visualization
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    
                    # Convert to RGB if grayscale
                    if image.shape[2] == 1:
                        image = np.repeat(image, 3, axis=2)
                    
                    # Get mask and prediction
                    mask = batch['mask'][i].squeeze().cpu().numpy()
                    pred = preds[i].squeeze().cpu().numpy()
                    
                    # Store for visualization
                    samples_to_visualize.append((image, mask, pred))
                
                break  # Only need one batch
        
        # Create visualization plots
        for idx, (image, mask, pred) in enumerate(samples_to_visualize):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth mask
            axes[1].imshow(image)
            # Ensure mask is properly displayed by using binary values
            mask_binary = (mask > 0).astype(np.float32)
            mask_overlay = np.ma.masked_where(mask_binary == 0, mask_binary)
            axes[1].imshow(mask_overlay, alpha=0.5, cmap='cool')
            
            # Add contour of the mask
            if np.any(mask_binary):  # Only if mask is not empty
                mask_edges = ndimage.binary_dilation(mask_binary) & ~ndimage.binary_erosion(mask_binary)
                y_mask, x_mask = np.where(mask_edges)
                if len(y_mask) > 0:  # Only if we have edges
                    axes[1].scatter(x_mask, y_mask, c='white', s=0.5, alpha=0.8)
                    
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Predicted mask overlay using contour fill and boundary
            axes[2].imshow(image)
            pred_binary = (pred > 0).astype(np.float32)
            if np.any(pred_binary):  # Only if prediction is not empty
                # Filled red overlay for mask
                axes[2].contourf(pred_binary, levels=[0.5, 1], colors='red', alpha=0.3)
                # Yellow boundary contour
                axes[2].contour(pred_binary, levels=[0.5], colors='yellow', linewidths=1)
            
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_dir / f"sample_{idx}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
