import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from typing import Dict, Optional, Union


class FoundationModel(nn.Module):
    """
    Foundation Model for Medical Image Analysis
    Supports both classification and segmentation tasks
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet-b3",
        pretrained: bool = True,
        dropout: float = 0.2,
        classification_heads: Optional[Dict[str, int]] = None,
        segmentation_heads: Optional[Dict[str, int]] = None,
    ):
        super(FoundationModel, self).__init__()
        
        self.backbone_name = backbone
        self.classification_heads = classification_heads or {}
        self.segmentation_heads = segmentation_heads or {}
        
        # Initialize backbone encoder
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        
        # Get feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        # Classification heads
        self.classification_layers = nn.ModuleDict()
        for dataset_name, num_classes in self.classification_heads.items():
            self.classification_layers[dataset_name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dims[-1], 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
        
        # Segmentation heads (using UNet decoder)
        self.segmentation_layers = nn.ModuleDict()
        for dataset_name, num_classes in self.segmentation_heads.items():
            self.segmentation_layers[dataset_name] = UNetDecoder(
                encoder_channels=self.feature_dims,
                decoder_channels=(256, 128, 64, 32, 16),
                num_classes=num_classes,
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor, task: str, dataset_name: str):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            task: 'classification' or 'segmentation'
            dataset_name: Name of the dataset to use appropriate head
        """
        # Store input size for segmentation
        input_size = x.shape[-2:]
        
        # Extract features
        features = self.backbone(x)
        
        if task == "classification":
            if dataset_name not in self.classification_layers:
                raise ValueError(f"Classification head for {dataset_name} not found")
            return self.classification_layers[dataset_name](features[-1])
        
        elif task == "segmentation":
            if dataset_name not in self.segmentation_layers:
                raise ValueError(f"Segmentation head for {dataset_name} not found")
            return self.segmentation_layers[dataset_name](features, input_size)
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def add_classification_head(self, dataset_name: str, num_classes: int, dropout: float = 0.2):
        """Add a new classification head for a dataset"""
        self.classification_heads[dataset_name] = num_classes
        self.classification_layers[dataset_name] = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def add_segmentation_head(self, dataset_name: str, num_classes: int, dropout: float = 0.2):
        """Add a new segmentation head for a dataset"""
        self.segmentation_heads[dataset_name] = num_classes
        self.segmentation_layers[dataset_name] = UNetDecoder(
            encoder_channels=self.feature_dims,
            decoder_channels=(256, 128, 64, 32, 16),
            num_classes=num_classes,
            dropout=dropout
        )


class UNetDecoder(nn.Module):
    """UNet-style decoder for segmentation"""
    
    def __init__(
        self,
        encoder_channels,
        decoder_channels=(256, 128, 64, 32, 16),
        num_classes=1,
        dropout=0.2
    ):
        super().__init__()
        
        # Reverse encoder channels for decoder
        encoder_channels = encoder_channels[::-1]
        
        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i, (enc_ch, dec_ch) in enumerate(zip(encoder_channels, decoder_channels)):
            if i == 0:
                # First decoder block (bottleneck)
                self.decoder_blocks.append(
                    DecoderBlock(enc_ch, dec_ch, dropout=dropout)
                )
            else:
                # Skip connections from encoder
                self.decoder_blocks.append(
                    DecoderBlock(enc_ch + decoder_channels[i-1], dec_ch, dropout=dropout)
                )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
    
    def forward(self, features, input_size):
        # Reverse features (from deepest to shallowest)
        features = features[::-1]
        
        x = features[0]  # Start with deepest features
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i == 0:
                x = decoder_block(x)
            else:
                # Upsample and concatenate with skip connection
                x = F.interpolate(x, size=features[i].shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, features[i]], dim=1)
                x = decoder_block(x)
        
        # Final upsampling to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return self.final_conv(x)


class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


def create_foundation_model(config: dict) -> FoundationModel:
    """Create foundation model based on configuration"""
    
    # Prepare classification heads
    classification_heads = {}
    for dataset_config in config.get('classification_datasets', []):
        dataset_name = dataset_config['name']
        num_classes = len(dataset_config['classes'])
        classification_heads[dataset_name] = num_classes
    
    # Prepare segmentation heads
    segmentation_heads = {}
    for dataset_config in config.get('segmentation_datasets', []):
        dataset_name = dataset_config['name']
        num_classes = dataset_config['num_classes']
        segmentation_heads[dataset_name] = num_classes
    
    model = FoundationModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        classification_heads=classification_heads,
        segmentation_heads=segmentation_heads
    )
    
    return model