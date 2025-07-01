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
from typing import Dict, List, Union, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_foundation_model
from src.visualization import Visualizer


class FoundationModelInference:
    """Inference class for the trained foundation model"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = create_foundation_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get image preprocessing transforms
        self.image_size = tuple(self.config.get('data', {}).get('image_size', [224, 224]))
        self.transform = self._get_inference_transform()
        
        # Initialize visualizer
        self.visualizer = Visualizer("./inference_results")
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Available classification datasets: {list(self.model.classification_heads.keys())}")
        print(f"Available segmentation datasets: {list(self.model.segmentation_heads.keys())}")
    
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
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def classify_image(self, image_path: str, dataset_name: str) -> Dict:
        """Classify a single image"""
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
        dataset_config = None
        for ds_config in self.config.get('classification_datasets', []):
            if ds_config['name'] == dataset_name:
                dataset_config = ds_config
                break
        
        class_names = dataset_config['classes'] if dataset_config else [f"Class_{i}" for i in range(logits.size(1))]
        
        result = {
            'task': 'classification',
            'dataset': dataset_name,
            'predicted_class_idx': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])
            }
        }
        
        return result
    
    def segment_image(self, image_path: str, dataset_name: str) -> Dict:
        """Segment a single image"""
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
            'task': 'segmentation',
            'dataset': dataset_name,
            'predicted_mask': predicted_mask_np,
            'probabilities': probabilities_np,
            'mask_shape': predicted_mask_np.shape,
            'unique_classes': np.unique(predicted_mask_np).tolist()
        }
        
        return result
    
    def predict(self, image_path: str, task: str, dataset_name: str) -> Dict:
        """General prediction method"""
        if task == 'classification':
            return self.classify_image(image_path, dataset_name)
        elif task == 'segmentation':
            return self.segment_image(image_path, dataset_name)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'segmentation'")
    
    def batch_predict(self, image_paths: List[str], task: str, dataset_name: str) -> List[Dict]:
        """Predict on multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, task, dataset_name)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_classification_result(self, image_path: str, result: Dict, save_path: str = None):
        """Visualize classification result"""
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Show prediction probabilities
        class_names = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())
        
        bars = ax2.bar(class_names, probabilities)
        ax2.set_title(f'Prediction: {result["predicted_class_name"]} ({result["confidence"]:.3f})')
        ax2.set_ylabel('Probability')
        ax2.tick_params(axis='x', rotation=45)
        
        # Highlight predicted class
        max_idx = probabilities.index(max(probabilities))
        bars[max_idx].set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_segmentation_result(self, image_path: str, result: Dict, save_path: str = None):
        """Visualize segmentation result"""
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to match mask size
        mask = result['predicted_mask']
        image_resized = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        
        # Create visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_resized)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(mask, cmap='tab10')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = self._overlay_mask_on_image(image_resized, mask)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _overlay_mask_on_image(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay mask on image"""
        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0
        
        # Create colored mask
        import matplotlib.pyplot as plt
        colors = plt.cm.Set1(np.linspace(0, 1, mask.max() + 1))
        colored_mask = colors[mask][:, :, :3]
        
        # Overlay
        overlaid = image * (1 - alpha) + colored_mask * alpha
        
        return overlaid
    
    def save_results(self, results: List[Dict], save_path: str):
        """Save results to file"""
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Foundation Model Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'segmentation'], help='Task type')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name for task head')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--save_path', type=str, help='Path to save results')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = FoundationModelInference(args.model_path, args.config_path)
    
    # Make prediction
    result = inference.predict(args.image_path, args.task, args.dataset_name)
    
    # Print result
    print(f"\nPrediction Result:")
    print(f"Task: {result['task']}")
    print(f"Dataset: {result['dataset']}")
    
    if args.task == 'classification':
        print(f"Predicted Class: {result['predicted_class_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    
    elif args.task == 'segmentation':
        print(f"Mask Shape: {result['mask_shape']}")
        print(f"Unique Classes: {result['unique_classes']}")
    
    # Visualize if requested
    if args.visualize:
        if args.task == 'classification':
            inference.visualize_classification_result(args.image_path, result)
        else:
            inference.visualize_segmentation_result(args.image_path, result)
    
    # Save results if requested
    if args.save_path:
        inference.save_results([result], args.save_path)


if __name__ == "__main__":
    main()
