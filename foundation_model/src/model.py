import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from typing import Dict, Optional, Union
import math


# CBAM Implementation
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
    
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# Multi-Head Attention for Classification
class MultiHeadAttentionClassifier(nn.Module):
    def __init__(self, feature_dim, num_heads=8, num_classes=1, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # CBAM attention module
        self.cbam = CBAM(feature_dim)
        
        # Multi-head self-attention
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Apply CBAM attention
        x = self.cbam(x)
        
        # Prepare for multi-head attention
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        # Multi-head self-attention
        Q = self.query(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, H * W, C)
        
        # Output projection
        attended = self.output_projection(attended)
        
        # Reshape back to spatial dimensions
        attended = attended.permute(0, 2, 1).view(B, C, H, W)
        
        # Add residual connection
        x = x + attended
        
        # Classification
        return self.classifier(x)


# Enhanced UNet Decoder with CBAM
class EnhancedUNetDecoder(nn.Module):
    """Enhanced UNet-style decoder with CBAM attention for segmentation"""
    
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
        
        # Create decoder blocks with CBAM
        self.decoder_blocks = nn.ModuleList()
        self.cbam_modules = nn.ModuleList()
        
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
            
            # Add CBAM for each decoder block
            self.cbam_modules.append(CBAM(dec_ch))
        
        # Final classification layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
    
    def forward(self, features, input_size):
        # Reverse features (from deepest to shallowest)
        features = features[::-1]
        
        x = features[0]  # Start with deepest features
        
        for i, (decoder_block, cbam) in enumerate(zip(self.decoder_blocks, self.cbam_modules)):
            if i == 0:
                x = decoder_block(x)
            else:
                # Upsample and concatenate with skip connection
                x = F.interpolate(x, size=features[i].shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, features[i]], dim=1)
                x = decoder_block(x)
            
            # Apply CBAM attention
            x = cbam(x)
        
        # Final upsampling to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return self.final_conv(x)


class FoundationModel(nn.Module):
    """
    Foundation Model for Medical Image Analysis
    Supports both classification and segmentation tasks
    
    Enhanced version includes:
    - CBAM (Convolutional Block Attention Module) for spatial and channel attention
    - Multi-head self-attention for classification tasks
    - Enhanced UNet decoder with CBAM for segmentation tasks
    - Backward compatibility with simple heads (use_attention=False)
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet-b3",
        pretrained: bool = True,
        dropout: float = 0.2,
        classification_heads: Optional[Dict[str, int]] = None,
        segmentation_heads: Optional[Dict[str, int]] = None,
        use_attention: bool = True,
        num_attention_heads: int = 8,
    ):
        """
        Initialize the Foundation Model
        
        Args:
            backbone: Name of the backbone model (e.g., "efficientnet-b3")
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for regularization
            classification_heads: Dict mapping dataset names to number of classes
            segmentation_heads: Dict mapping dataset names to number of classes
            use_attention: Whether to use CBAM and multi-head attention (default: True)
            num_attention_heads: Number of attention heads for multi-head attention (default: 8)
        """
        super(FoundationModel, self).__init__()
        
        self.backbone_name = backbone
        self.classification_heads = classification_heads or {}
        self.segmentation_heads = segmentation_heads or {}
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        
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
        
        # Classification heads with or without attention
        self.classification_layers = nn.ModuleDict()
        for dataset_name, num_classes in self.classification_heads.items():
            if self.use_attention:
                self.classification_layers[dataset_name] = MultiHeadAttentionClassifier(
                    feature_dim=self.feature_dims[-1],
                    num_heads=self.num_attention_heads,
                    num_classes=num_classes,
                    dropout=dropout
                )
            else:
                # Original simple classification head
                self.classification_layers[dataset_name] = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(self.feature_dims[-1], 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, num_classes)
                )
        
        # Segmentation heads with or without attention
        self.segmentation_layers = nn.ModuleDict()
        for dataset_name, num_classes in self.segmentation_heads.items():
            if self.use_attention:
                self.segmentation_layers[dataset_name] = EnhancedUNetDecoder(
                    encoder_channels=self.feature_dims,
                    decoder_channels=(256, 128, 64, 32, 16),
                    num_classes=num_classes,
                    dropout=dropout
                )
            else:
                # Original UNet decoder
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
        
        if self.use_attention:
            self.classification_layers[dataset_name] = MultiHeadAttentionClassifier(
                feature_dim=self.feature_dims[-1],
                num_heads=self.num_attention_heads,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            # Original simple classification head
            self.classification_layers[dataset_name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dims[-1], 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
        
        # Move the new head to the same device as the model
        device = next(self.parameters()).device
        self.classification_layers[dataset_name] = self.classification_layers[dataset_name].to(device)
    
    def add_segmentation_head(self, dataset_name: str, num_classes: int, dropout: float = 0.2):
        """Add a new segmentation head for a dataset"""
        self.segmentation_heads[dataset_name] = num_classes
        
        if self.use_attention:
            self.segmentation_layers[dataset_name] = EnhancedUNetDecoder(
                encoder_channels=self.feature_dims,
                decoder_channels=(256, 128, 64, 32, 16),
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            # Original UNet decoder
            self.segmentation_layers[dataset_name] = UNetDecoder(
                encoder_channels=self.feature_dims,
                decoder_channels=(256, 128, 64, 32, 16),
                num_classes=num_classes,
                dropout=dropout
            )
        
        # Move the new head to the same device as the model
        device = next(self.parameters()).device
        self.segmentation_layers[dataset_name] = self.segmentation_layers[dataset_name].to(device)


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


def create_foundation_model(
    config: dict = None,
    backbone: str = None,
    pretrained: bool = None,
    dropout: float = None,
    classification_heads: dict = None,
    segmentation_heads: dict = None,
    use_attention: bool = None,
    num_attention_heads: int = None
) -> FoundationModel:
    """Create foundation model based on configuration or direct parameters"""
    
    if config is not None:
        # Use configuration-based creation
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
            segmentation_heads=segmentation_heads,
            use_attention=config['model'].get('use_attention', True),
            num_attention_heads=config['model'].get('num_attention_heads', 8)
        )
    else:
        # Use direct parameters
        model = FoundationModel(
            backbone=backbone or "efficientnet-b3",
            pretrained=pretrained if pretrained is not None else True,
            dropout=dropout or 0.2,
            classification_heads=classification_heads or {},
            segmentation_heads=segmentation_heads or {},
            use_attention=use_attention if use_attention is not None else True,
            num_attention_heads=num_attention_heads or 8
        )
    
    return model


def load_model_weights_safely(model, checkpoint, logger=None, suppress_warnings=False):
    """
    Safely load model weights with proper error handling and logging
    
    Args:
        model: PyTorch model to load weights into
        checkpoint: Checkpoint dictionary or state dict
        logger: Logger instance for logging (optional)
        suppress_warnings: Whether to suppress expected model loading warnings (default: False)
    
    Returns:
        tuple: (missing_keys, unexpected_keys) from load_state_dict
    """
    import logging
    
    # Use module logger if no logger provided
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Determine if checkpoint contains model_state_dict key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict with strict=False to handle mismatched keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Filter out common pretrained model keys that are expected to be missing/unexpected
    expected_unexpected_keys = [
        'classifier.weight', 'classifier.bias',  # Original classifier from pretrained model
        'conv_head.weight', 'conv_head.bias',    # EfficientNet head
        'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked'  # Final BN layer
    ]
    
    # Filter unexpected keys to only show truly unexpected ones
    truly_unexpected = [key for key in unexpected_keys if not any(expected in key for expected in expected_unexpected_keys)]
    
    # Log information about key mismatches
    if not suppress_warnings:
        if missing_keys:
            logger.warning(f"Missing keys when loading model (these will be randomly initialized): {missing_keys}")
        
        if truly_unexpected:
            logger.warning(f"Truly unexpected keys found in checkpoint: {truly_unexpected}")
        
        if unexpected_keys and not truly_unexpected:
            logger.info("Found expected pretrained model keys that were replaced with task-specific layers (this is normal)")
    else:
        # Only log truly concerning issues even when suppressing warnings
        if missing_keys:
            logger.debug(f"Missing keys when loading model: {missing_keys}")
        if truly_unexpected:
            logger.warning(f"Truly unexpected keys found in checkpoint: {truly_unexpected}")
    
    return missing_keys, unexpected_keys