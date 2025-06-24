#!/usr/bin/env python3
"""
Pretrained Medical Image Segmentation Models

This module implements lightweight pretrained segmentation models for medical images,
leveraging popular architectures like Swin-UNETR and UNET.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional, Dict
import torchvision


class PretrainedSegmentationDecoder(nn.Module):
    """
    Lightweight pretrained segmentation decoder for medical images.
    
    Uses a pretrained encoder (ResNet) and a lightweight decoder structure
    for medical image segmentation.
    """
    
    def __init__(self, encoder_dims: List[int], num_classes: int = 2, img_size: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Use pretrained ResNet18 or ResNet34 as encoder base (fast and lightweight)
        # We'll initialize our own weights based on these 
        self.pretrained_encoder = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Fusion layer to combine ViT features with ResNet architecture
        self.fusion_layers = nn.ModuleList()
        self.fusion_norms = nn.ModuleList()
        
        # Get standard channels from ResNet
        resnet_channels = [64, 128, 256, 512]
        
        for i, dim in enumerate(encoder_dims):
            # Create fusion layer that maps ViT features to ResNet channels
            target_channels = resnet_channels[min(i, len(resnet_channels)-1)]
            self.fusion_layers.append(nn.Conv2d(dim, target_channels, kernel_size=1))
            self.fusion_norms.append(nn.BatchNorm2d(target_channels))
        
        # Decoder layers that increase resolution and reduce channels
        self.decoder_blocks = nn.ModuleList()
        
        # Current channels based on encoder (assuming ViT-like structure)
        current_channels = resnet_channels[-1]
        
        # Create decoder blocks - we'll use 4 upsampling steps
        decoder_channels = [256, 128, 64, 32]
        
        for channels in decoder_channels:
            block = nn.Sequential(
                # Upsample with transposed conv
                nn.ConvTranspose2d(current_channels, channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # Refine with conv
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(block)
            current_channels = channels
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        
        # Initialize with better weights for segmentation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for segmentation"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img: torch.Tensor, encoder_features: List[torch.Tensor]):
        # 1) Run the image through the ResNet backbone and collect its 4 stages
        res_feats = []
        x = self.pretrained_encoder.conv1(img)
        x = self.pretrained_encoder.bn1(x)
        x = self.pretrained_encoder.relu(x)
        x = self.pretrained_encoder.maxpool(x)
        x = self.pretrained_encoder.layer1(x); res_feats.append(x)  # 1/4
        x = self.pretrained_encoder.layer2(x); res_feats.append(x)  # 1/8
        x = self.pretrained_encoder.layer3(x); res_feats.append(x)  # 1/16
        x = self.pretrained_encoder.layer4(x); res_feats.append(x)  # 1/32

        # 2) Convert your ViT tokens into the same 4 spatial maps
        vit_feats = []
        for i, (feat, fusion, norm) in enumerate(zip(encoder_features, self.fusion_layers, self.fusion_norms)):
            # unpack [B, N, C] → [B, C, H, W]
            if feat.dim() == 3:
                B, N, C = feat.shape
                H = W = int(N**0.5)
                feat = feat.permute(0,2,1).reshape(B,C,H,W)
            v = norm(fusion(feat))
            # Resize ViT feature to match ResNet feature resolution
            target_h, target_w = res_feats[i].shape[2], res_feats[i].shape[3]
            if v.shape[2:] != (target_h, target_w):
                v = F.interpolate(v, size=(target_h, target_w), mode='bilinear', align_corners=False)
            vit_feats.append(v)

        # 3) Reverse lists so lowest-res (512 channels) is first, then fuse
        res_feats = res_feats[::-1]
        vit_feats = vit_feats[::-1]
        fused = [r + v for r, v in zip(res_feats, vit_feats)]

        # 4) Continue as before with your decoder blocks
        x = fused[0]
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            if i+1 < len(fused):
                skip = fused[i+1]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                x = x + skip

        seg_logits = self.seg_head(x)
        return seg_logits, []


class UNet2Plus(nn.Module):
    """
    Implementation of UNet++ (UNet 2+), a nested UNet architecture designed
    for medical image segmentation with improved performance.
    
    This model is specifically designed to work with medical images and 
    addresses the challenges in semantic segmentation with dense skip
    connections.
    """
    
    def __init__(self, in_channels=3, num_classes=2, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        # Initial encoder (downsampling) path
        filters = [32, 64, 128, 256, 512]
        
        # Encoder blocks
        self.encoder1 = self._conv_block(in_channels, filters[0])
        self.encoder2 = self._conv_block(filters[0], filters[1])
        self.encoder3 = self._conv_block(filters[1], filters[2])
        self.encoder4 = self._conv_block(filters[2], filters[3])
        self.encoder5 = self._conv_block(filters[3], filters[4])
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Nested decoder (upsampling) path
        # First level
        self.up1_1 = self._up_conv(filters[1], filters[0])
        self.decoder1_1 = self._conv_block(filters[0] + filters[0], filters[0])
        
        # Second level
        self.up2_1 = self._up_conv(filters[2], filters[1])
        self.decoder2_1 = self._conv_block(filters[1] + filters[1], filters[1])
        self.up1_2 = self._up_conv(filters[1], filters[0])
        self.decoder1_2 = self._conv_block(filters[0] + filters[0] + filters[0], filters[0])
        
        # Third level
        self.up3_1 = self._up_conv(filters[3], filters[2])
        self.decoder3_1 = self._conv_block(filters[2] + filters[2], filters[2])
        self.up2_2 = self._up_conv(filters[2], filters[1])
        self.decoder2_2 = self._conv_block(filters[1] + filters[1] + filters[1], filters[1])
        self.up1_3 = self._up_conv(filters[1], filters[0])
        self.decoder1_3 = self._conv_block(filters[0] + filters[0] + filters[0] + filters[0], filters[0])
        
        # Fourth level
        self.up4_1 = self._up_conv(filters[4], filters[3])
        self.decoder4_1 = self._conv_block(filters[3] + filters[3], filters[3])
        self.up3_2 = self._up_conv(filters[3], filters[2])
        self.decoder3_2 = self._conv_block(filters[2] + filters[2] + filters[2], filters[2])
        self.up2_3 = self._up_conv(filters[2], filters[1])
        self.decoder2_3 = self._conv_block(filters[1] + filters[1] + filters[1] + filters[1], filters[1])
        self.up1_4 = self._up_conv(filters[1], filters[0])
        self.decoder1_4 = self._conv_block(filters[0] + filters[0] + filters[0] + filters[0] + filters[0], filters[0])
        
        # Output segmentation layers
        if self.deep_supervision:
            self.out1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.out2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.out3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.out4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.out = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward_unet2plus(self, x):
        """Forward pass with UNet++ architecture"""
        # Encoder path
        x1 = self.encoder1(x)
        
        x2 = self.pool(x1)
        x2 = self.encoder2(x2)
        
        x3 = self.pool(x2)
        x3 = self.encoder3(x3)
        
        x4 = self.pool(x3)
        x4 = self.encoder4(x4)
        
        x5 = self.pool(x4)
        x5 = self.encoder5(x5)
        
        # First level
        up1_1 = self.up1_1(x2)
        combine1_1 = torch.cat([up1_1, x1], dim=1)
        decoder1_1 = self.decoder1_1(combine1_1)
        
        # Second level
        up2_1 = self.up2_1(x3)
        combine2_1 = torch.cat([up2_1, x2], dim=1)
        decoder2_1 = self.decoder2_1(combine2_1)
        
        up1_2 = self.up1_2(decoder2_1)
        combine1_2 = torch.cat([up1_2, x1, decoder1_1], dim=1)
        decoder1_2 = self.decoder1_2(combine1_2)
        
        # Third level
        up3_1 = self.up3_1(x4)
        combine3_1 = torch.cat([up3_1, x3], dim=1)
        decoder3_1 = self.decoder3_1(combine3_1)
        
        up2_2 = self.up2_2(decoder3_1)
        combine2_2 = torch.cat([up2_2, x2, decoder2_1], dim=1)
        decoder2_2 = self.decoder2_2(combine2_2)
        
        up1_3 = self.up1_3(decoder2_2)
        combine1_3 = torch.cat([up1_3, x1, decoder1_1, decoder1_2], dim=1)
        decoder1_3 = self.decoder1_3(combine1_3)
        
        # Fourth level
        up4_1 = self.up4_1(x5)
        combine4_1 = torch.cat([up4_1, x4], dim=1)
        decoder4_1 = self.decoder4_1(combine4_1)
        
        up3_2 = self.up3_2(decoder4_1)
        combine3_2 = torch.cat([up3_2, x3, decoder3_1], dim=1)
        decoder3_2 = self.decoder3_2(combine3_2)
        
        up2_3 = self.up2_3(decoder3_2)
        combine2_3 = torch.cat([up2_3, x2, decoder2_1, decoder2_2], dim=1)
        decoder2_3 = self.decoder2_3(combine2_3)
        
        up1_4 = self.up1_4(decoder2_3)
        combine1_4 = torch.cat([up1_4, x1, decoder1_1, decoder1_2, decoder1_3], dim=1)
        decoder1_4 = self.decoder1_4(combine1_4)
        
        # Output
        if self.deep_supervision:
            out1 = self.out1(decoder1_1)
            out2 = self.out2(decoder1_2)
            out3 = self.out3(decoder1_3)
            out4 = self.out4(decoder1_4)
            return (out1 + out2 + out3 + out4) / 4
        else:
            return self.out(decoder1_4)

    def init_from_encoder_features(self, encoder_features):
        """Initialize UNet++ from encoder features (ViT-based)"""
        # Adapt encoder_features to new shape if needed
        adapted_features = []
        for feat in encoder_features:
            # Convert [B, N, C] to [B, C, H, W] if needed
            if feat.dim() == 3:
                B, N, C = feat.shape
                H = W = int(N**0.5)
                feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
            adapted_features.append(feat)
        
        # Use the adapted features to initialize UNet++
        if adapted_features:
            # Just use the first feature to determine input size
            x = F.interpolate(
                adapted_features[0], 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            )
            return self.forward_unet2plus(x)
        else:
            return None

    def forward(self, encoder_features):
        """Forward pass adapting to the encoder features format"""
        # First converted to a standard tensor if needed
        if isinstance(encoder_features, list) and len(encoder_features) > 0:
            # Get first feature and resize to input size
            feat = encoder_features[0]
            if feat.dim() == 3:  # [B, N, C] format
                B, N, C = feat.shape
                H = W = int(N**0.5)
                x = feat.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                x = feat
                
            # Resize to expected input size
            x = F.interpolate(
                x, size=(self.img_size, self.img_size), 
                mode='bilinear', align_corners=False
            )
            
            # Forward through UNet++
            seg_logits = self.forward_unet2plus(x)
            return seg_logits, []  # Return logits and empty list for compatibility
        else:
            # Handle error case
            return None, []


class PretrainedModels:
    """Factory class to create various pretrained segmentation models"""
    
    @staticmethod
    def create_model(model_type: str, encoder_dims: list, num_classes: int = 2, img_size: int = 128):
        """
        Create a pretrained segmentation model
        
        Args:
            model_type: Type of model to create ('unet2plus', 'pretrained', or 'swin_unetr')
            encoder_dims: List of encoder dimensions
            num_classes: Number of output classes
            img_size: Input image size
            
        Returns:
            Initialized segmentation model
        """
        if model_type == 'unet2plus':
            model = UNet2Plus(in_channels=encoder_dims[0], num_classes=num_classes)
            model.img_size = img_size
            return model
        elif model_type == 'swin_unetr':
            try:
                # Import MONAI SwinUNETR (3D). Using type ignore to suppress lint if monai missing
                from monai.networks.nets.swin_unetr import SwinUNETR  # type: ignore
            except ImportError:
                raise ImportError("MONAI is required for swin_unetr backbone. Install via 'pip install monai'")
            # Wrap 3D model for 2D use by stacking singleton depth
            class SwinUNETRWrapper(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = SwinUNETR(
                        img_size=(img_size, img_size, img_size),
                        in_chans=encoder_dims[0],
                        out_chans=num_classes,
                        feature_size=48,
                        drop_rate=0.0
                    )
                    self.img_size = img_size
                def forward(self, x, encoder_features=None):
                    # Expand 2D to 3D by adding depth dim
                    x3d = x.unsqueeze(2)  # [B,C,1,H,W]
                    seg3d = self.model(x3d)
                    # Remove depth dim
                    seg2d = seg3d.squeeze(2)
                    return seg2d, []
            return SwinUNETRWrapper()
        else:
            return PretrainedSegmentationDecoder(encoder_dims, num_classes, img_size)
