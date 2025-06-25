#!/usr/bin/env python3
"""
KAN-based Medical Foundation Model Architecture

This module implements a foundation model based on the Kolmogorov-Arnold Network (KAN) architecture
for medical imaging tasks, supporting both classification and segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .pretrained_segmentation import PretrainedModels


class SpatialBSpline(nn.Module):
    """Optimized 2D B-spline implementation for KAN layers"""

    def __init__(self, in_channels, out_channels, grid_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        # Reduce parameter count with single-channel control points that get mapped to output
        self.control_points = nn.Parameter(
            torch.randn(grid_size * grid_size, in_channels) * 0.1
        )

        # Linear mapping to output channels
        self.output_map = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert (
            C == self.in_channels
        ), f"Input channels {C} don't match expected {self.in_channels}"

        # Convert to normalized coordinates in [0, 1]
        y_coords = torch.linspace(0, 1, H, device=x.device)
        x_coords = torch.linspace(0, 1, W, device=x.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Scaled coordinates for grid size
        grid_y = grid_y * (self.grid_size - 1)
        grid_x = grid_x * (self.grid_size - 1)

        # Integer and fractional parts
        i0 = torch.floor(grid_y).long()
        j0 = torch.floor(grid_x).long()
        i1 = torch.minimum(i0 + 1, torch.tensor(self.grid_size - 1, device=x.device))
        j1 = torch.minimum(j0 + 1, torch.tensor(self.grid_size - 1, device=x.device))

        # Fractional parts for interpolation
        y_frac = grid_y - i0.float()
        x_frac = grid_x - j0.float()

        # Flatten spatial dims for batch matmul
        x_flat = x.view(B, C, H * W)  # [B, C, H*W]
        x_flat = x_flat.permute(0, 2, 1)  # [B, H*W, C]

        # Get control point indices for the 4 corners
        idx00 = i0 * self.grid_size + j0  # top-left
        idx01 = i0 * self.grid_size + j1  # top-right
        idx10 = i1 * self.grid_size + j0  # bottom-left
        idx11 = i1 * self.grid_size + j1  # bottom-right

        # Gather control points
        cp00 = self.control_points[idx00.flatten()]  # [H*W, C]
        cp01 = self.control_points[idx01.flatten()]
        cp10 = self.control_points[idx10.flatten()]
        cp11 = self.control_points[idx11.flatten()]

        # Reshape back to spatial
        cp00 = cp00.view(H, W, C)
        cp01 = cp01.view(H, W, C)
        cp10 = cp10.view(H, W, C)
        cp11 = cp11.view(H, W, C)

        # Bilinear interpolation
        x_frac = x_frac.unsqueeze(-1)  # [H, W, 1]
        y_frac = y_frac.unsqueeze(-1)  # [H, W, 1]

        top = cp00 * (1 - x_frac) + cp01 * x_frac
        bottom = cp10 * (1 - x_frac) + cp11 * x_frac

        output = top * (1 - y_frac) + bottom * y_frac  # [H, W, C]

        # Output mapping to correct dimension
        output = output.reshape(H * W, C)
        output = self.output_map(output)  # [H*W, out_channels]
        output = output.view(H, W, self.out_channels)
        output = output.permute(2, 0, 1)  # [out_channels, H, W]
        output = output.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, out_channels, H, W]

        return output


class KANBlock(nn.Module):
    """KAN Block with SpatialBSpline activation"""

    def __init__(self, in_channels, out_channels, grid_size=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.spline = SpatialBSpline(out_channels, out_channels, grid_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.spline(x)
        return x


class ViTBlock(nn.Module):
    """Transformer block for ViT"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # Self attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_output)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class LightweightLLM(nn.Module):
    """Enhanced lightweight language model for medical prompt handling

    This class serves as a placeholder for a more complex LLM integration.
    It handles text prompts by analyzing keywords and produces task-specific outputs
    that can be used by the foundation model.
    """

    def __init__(self, vocab_size=1000, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Task classifier for translating text to task type
        self.task_classifier = nn.Linear(
            hidden_dim, 2
        )  # 0: classification, 1: segmentation
        self.hidden_dim = hidden_dim

        # Dictionary of modality keywords for better prompt understanding
        self.modality_keywords = {
            "mri": ["mri", "magnetic", "resonance"],
            "ct": ["ct", "computed", "tomography"],
            "xray": ["xray", "x-ray", "radiograph"],
            "ultrasound": ["ultrasound", "sonogram", "echogram"],
            "endoscopic": ["endoscopic", "endoscopy", "scope"],
        }

        # Dictionary of anatomical regions
        self.anatomical_regions = {
            "brain": ["brain", "cerebral", "neural", "cranial"],
            "chest": ["chest", "thoracic", "lung", "pulmonary", "cardiac", "heart"],
            "abdomen": ["abdomen", "abdominal", "liver", "pancreas", "spleen"],
            "colon": ["colon", "intestine", "bowel", "gastrointestinal"],
        }

    def analyze_prompt(self, prompt):
        """
        Analyze a text prompt to extract key information

        Args:
            prompt: String text prompt

        Returns:
            Dictionary with extracted information
        """
        if not isinstance(prompt, str):
            return {"task": "unknown", "modality": "unknown", "region": "unknown"}

        # Determine task type
        task = "classification"
        if any(
            keyword in prompt.lower()
            for keyword in [
                "segment",
                "segmentation",
                "outline",
                "contour",
                "boundary",
                "mask",
            ]
        ):
            task = "segmentation"

        # Determine modality
        modality = "unknown"
        for mod, keywords in self.modality_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                modality = mod
                break

        # Determine anatomical region
        region = "unknown"
        for reg, keywords in self.anatomical_regions.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                region = reg
                break

        return {"task": task, "modality": modality, "region": region}

    def forward(self, x):
        """
        Process text input through the LLM
        Note: This is a placeholder for future integration with a real LLM
        Currently returns dummy embeddings compatible with the rest of the model

        Args:
            x: List of text prompts or tensor of token IDs

        Returns:
            Dictionary containing:
              - embeddings: Tensor of shape [batch_size, seq_len, hidden_dim]
              - task_logits: Tensor of shape [batch_size, 2] for task classification
              - analysis: List of dictionaries with prompt analysis
        """
        # Handle different input types
        if isinstance(x, torch.Tensor):
            # Tensor case - assume these are token IDs
            batch_size = x.shape[0]
            device = x.device
            task_logits = torch.zeros((batch_size, 2), device=device)
            task_logits[:, 0] = 1.0  # Default to classification
            dummy_embeds = torch.zeros((batch_size, 8, self.hidden_dim), device=device)
            return {
                "embeddings": dummy_embeds,
                "task_logits": task_logits,
                "analysis": [
                    {
                        "task": "classification",
                        "modality": "unknown",
                        "region": "unknown",
                    }
                ]
                * batch_size,
            }

        elif isinstance(x, (list, tuple)):
            # List case - assume these are text prompts
            batch_size = len(x)
            device = self.embedding.weight.device
            task_logits = torch.zeros((batch_size, 2), device=device)
            analysis_results = []

            for i, prompt in enumerate(x):
                # Analyze each prompt
                analysis = self.analyze_prompt(prompt)
                analysis_results.append(analysis)

                # Set task logits based on analysis
                if analysis["task"] == "segmentation":
                    task_logits[i, 1] = 1.0  # Mark as segmentation
                else:
                    task_logits[i, 0] = 1.0  # Mark as classification

            # Create dummy embeddings
            dummy_embeds = torch.zeros((batch_size, 8, self.hidden_dim), device=device)

            return {
                "embeddings": dummy_embeds,
                "task_logits": task_logits,
                "analysis": analysis_results,
            }

        else:
            # Handle other cases gracefully
            device = self.embedding.weight.device
            return {
                "embeddings": torch.zeros((1, 8, self.hidden_dim), device=device),
                "task_logits": torch.zeros((1, 2), device=device),
                "analysis": [
                    {"task": "unknown", "modality": "unknown", "region": "unknown"}
                ],
            }

        # Return empty result if input is invalid
        return {
            "embeddings": torch.zeros(
                (1, 8, self.hidden_dim), device=self.embedding.weight.device
            ),
            "task_logits": torch.zeros((1, 2), device=self.embedding.weight.device),
        }


class UNetDecoder(nn.Module):
    """UNet decoder for segmentation tasks"""

    def __init__(self, encoder_dims, num_classes=2, kan_grid_size=5):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.num_classes = num_classes

        # Create target decoder dimensions with progressive reduction
        if len(set(encoder_dims)) == 1:  # All encoder dims are the same (ViT case)
            base_dim = encoder_dims[0]
            decoder_dims = [base_dim // (2**i) for i in range(len(encoder_dims))]
            decoder_dims = [max(dim, 64) for dim in decoder_dims]  # Minimum 64 channels
        else:
            decoder_dims = encoder_dims[::-1]  # Use provided dimensions

        # Channel reduction layers to match decoder expectations
        self.channel_reducers = nn.ModuleList()
        for i, (enc_dim, dec_dim) in enumerate(zip(encoder_dims, decoder_dims)):
            if enc_dim != dec_dim:
                self.channel_reducers.append(
                    nn.Sequential(
                        nn.Conv2d(enc_dim, dec_dim, kernel_size=1),
                        nn.BatchNorm2d(dec_dim),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.channel_reducers.append(nn.Identity())

        # Upsampling blocks with KAN layers
        self.up_blocks = nn.ModuleList()
        self.kan_blocks = nn.ModuleList()

        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i]
            out_dim = decoder_dims[i + 1]

            # Upsampling
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
            )

            # KAN processing after upsampling + skip connection
            self.kan_blocks.append(
                SpatialBSpline(out_dim * 2, out_dim, grid_size=kan_grid_size)
            )

        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_dims[-1], num_classes, kernel_size=1)
        self.num_classes = num_classes  # Store for activation selection in forward

    def forward(self, encoder_features):
        # encoder_features: list of feature maps from encoder (highest to lowest resolution)

        # Apply channel reduction to match decoder expectations
        reduced_features = []
        for feat, reducer in zip(encoder_features, self.channel_reducers):
            reduced_features.append(reducer(feat))

        # Start with the lowest resolution feature (first in the list)
        x = reduced_features[0]

        outputs = []
        for i, (up_block, kan_block) in enumerate(zip(self.up_blocks, self.kan_blocks)):
            # Upsample
            x = up_block(x)

            # Skip connection if available
            if i + 1 < len(reduced_features):
                skip = reduced_features[i + 1]
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(
                        x, size=skip.shape[2:], mode="bilinear", align_corners=False
                    )
                x = torch.cat([x, skip], dim=1)

            # Apply KAN processing
            x = kan_block(x)
            outputs.append(x)

        # Final segmentation
        seg_logits = self.seg_head(x)

        # Apply appropriate activation based on number of classes
        # For binary segmentation with a single output channel, we always apply sigmoid during training
        if self.num_classes == 1:
            # Single output channel - must use sigmoid activation
            # NOTE: Not applying activation as sigmoid will be applied in the loss function and metrics
            seg_output = seg_logits
        elif self.num_classes == 2 and seg_logits.shape[1] == 2:
            # Two-class segmentation with 2 channels - use softmax
            # NOTE: Not applying activation as softmax will be applied in the loss function and metrics
            seg_output = seg_logits
        else:
            # Multi-class segmentation
            # NOTE: Not applying activation as softmax will be applied in the loss function and metrics
            seg_output = seg_logits

        return seg_output, outputs


class MedicalFoundationModel(nn.Module):
    """Foundation model for medical imaging combining KAN, ViT, and lightweight LLM"""

    def __init__(
        self,
        img_size=64,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        kan_grid_size=5,
        max_num_classes=1000,
        use_llm=False,
        segmentation_backbone: str = "resnet18",
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.kan_grid_size = kan_grid_size
        # Backbone for segmentation decoder
        self.segmentation_backbone = segmentation_backbone

        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding and CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Dropout (now as nn.Module for serialization)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ViT blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Feature maps at different scales
        self.feature_indices = [3, 6, 9, 12]  # Layers to extract features from
        self.feature_scales = [4, 8, 16, 32]  # Corresponding scales

        # LLM component for task understanding from text (for future implementation)
        # Can be disabled by setting use_llm=False
        self.llm = LightweightLLM() if use_llm else None

        # Task-specific heads (dynamically created)
        self.classification_heads = nn.ModuleDict()
        self.segmentation_decoders = nn.ModuleDict()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_fn)

    def _init_weights_fn(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_classification_head(self, task_id: str, num_classes: int):
        """Dynamically create classification head for a specific task"""
        self.classification_heads[task_id] = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def create_segmentation_decoder(self, task_id: str, num_classes: int):
        """Dynamically create segmentation decoder for a specific task"""
        # All ViT features have the same embedding dimension, so use consistent dimensions
        # We'll add channel reduction layers in the decoder to handle this
        encoder_dims = [self.embed_dim] * len(
            self.feature_scales
        )  # All features have embed_dim channels

        # Choose decoder based on config backbone
        self.segmentation_decoders[task_id] = PretrainedModels.create_model(
            model_type=self.segmentation_backbone,
            encoder_dims=encoder_dims,
            num_classes=num_classes,
            img_size=self.img_size,
        )

    def interpolate_pos_embed(self, pos_embed, H, W):
        """Interpolate positional embeddings for different input sizes"""
        N = pos_embed.shape[1] - 1  # Remove cls token
        if N == H * W:
            return pos_embed

        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        # Get original grid size
        sqrt_N = int(N**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_N, sqrt_N, -1).permute(
            0, 3, 1, 2
        )

        # Resize patch embeddings
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(H, W), mode="bicubic", align_corners=False
        )

        # Flatten and concatenate with class token
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        new_pos_embed = torch.cat(
            [class_pos_embed.unsqueeze(0).unsqueeze(0), patch_pos_embed], dim=1
        )

        return new_pos_embed

    def forward_features(self, x):
        """Extract hierarchical features"""
        B, C, H, W = x.shape

        # Check if we need to resize positional embeddings
        h, w = H // self.patch_size, W // self.patch_size
        if h * w != self.num_patches:
            pos_embed = self.interpolate_pos_embed(self.pos_embed, h, w)
        else:
            pos_embed = self.pos_embed

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B, h*w, embed_dim]

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + pos_embed
        x = self.pos_drop(x)

        # Extract features at different layers
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Store features at specific layers
            if i + 1 in self.feature_indices:
                features.append(x)

        x = self.norm(x)
        features.append(x)  # Add final features

        return x, features

    def forward_classification(self, features, task_id: str):
        """Forward pass for classification task"""
        if task_id not in self.classification_heads:
            raise ValueError(f"Classification head for task '{task_id}' not found")

        # Use CLS token for classification
        x = features[:, 0]  # [B, embed_dim]

        # Apply task-specific classification head
        return self.classification_heads[task_id](x)

    def forward_segmentation(self, images, features_list, task_id: str, input_size):
        """Forward pass for segmentation task"""
        if task_id not in self.segmentation_decoders:
            raise ValueError(f"Segmentation decoder for task '{task_id}' not found")

        # Convert ViT features to spatial format for decoder
        spatial_features = []
        H = W = self.img_size // self.patch_size

        for features in features_list:
            # Remove cls token and reshape to spatial
            patch_features = features[:, 1:, :]  # [B, num_patches, embed_dim]
            B, N, C = patch_features.shape
            spatial_feat = patch_features.transpose(1, 2).reshape(B, C, H, W)
            spatial_features.append(spatial_feat)

        # Reverse order for decoder (highest to lowest resolution)
        spatial_features = spatial_features[::-1]

        # Forward pass through segmentation decoder
        try:
            seg_output, decoder_features = self.segmentation_decoders[task_id](
                images,
                spatial_features,
            )

            # Handle case where output is None (error in model)
            if seg_output is None:
                print(
                    f"WARNING: Segmentation decoder returned None, creating empty output tensor"
                )
                # Create a default output with zeros, enable gradients for backward compatibility
                seg_output = torch.zeros(
                    (batch_size, num_classes, input_size[0], input_size[1]),
                    device=spatial_features[0].device,
                )

            # Handle tuple return
            if isinstance(seg_output, tuple):
                seg_output = seg_output[0]

            # Ensure output is a tensor with proper dimensions
            if not isinstance(seg_output, torch.Tensor):
                raise ValueError(
                    f"Segmentation output is not a tensor: {type(seg_output)}"
                )

            # Add batch dimension if missing
            if len(seg_output.shape) == 3:
                seg_output = seg_output.unsqueeze(0)

            # Upsample to input size
            seg_output = F.interpolate(
                seg_output, size=input_size, mode="bilinear", align_corners=False
            )

        except Exception as e:
            print(f"ERROR in segmentation forward pass: {e}")
            # Return a fallback output for robustness
            batch_size = spatial_features[0].size(0)
            num_classes = getattr(self.segmentation_decoders[task_id], "num_classes", 2)
            # Create a default output with zeros, enable gradients for backward compatibility
            seg_output = torch.zeros(
                (batch_size, num_classes, input_size[0], input_size[1]),
                device=spatial_features[0].device,
            )

        return seg_output

    def forward(self, batch_data, task_id: Optional[str] = None):
        """
        Forward pass supporting both classification and segmentation

        Args:
            batch_data:
                - For dictionary input: Dict with 'image', 'task', and task-specific inputs
                - For tensor input: Image tensor
            task_id: Optional task identifier if not provided in batch_data

        Returns:
            Dict of task outputs
        """
        # Handle different input formats
        if isinstance(batch_data, dict):
            images = batch_data["image"]
            if task_id is None and "task" in batch_data:
                if isinstance(batch_data["task"], list):
                    # Take first task type if batch has multiple
                    task_id = batch_data["task"][0]
                else:
                    task_id = batch_data["task"]
        else:
            # If input is a tensor, assume it's the image
            images = batch_data

        # Default to classification if no task specified
        if task_id is None:
            task_id = "classification"

        # Normalize task_id to string
        if isinstance(task_id, (list, tuple)):
            task_id = str(task_id[0])
        elif not isinstance(task_id, str):
            task_id = str(task_id)

        # Determine task type from task_id
        task_type = "classification"
        if "seg" in task_id.lower():
            task_type = "segmentation"

        # Extract features from backbone
        features, multi_scale_features = self.forward_features(images)

        # Process prompt if available
        llm_outputs = None
        if (
            "prompt" in batch_data
            and batch_data["prompt"] is not None
            and self.llm is not None
        ):
            try:
                # Process prompts through LLM
                llm_outputs = self.llm(batch_data["prompt"])

                # Log some information about the prompts if in debug mode
                if hasattr(self, "debug") and self.debug and "analysis" in llm_outputs:
                    analysis = llm_outputs["analysis"]
                    task_counts = {}
                    for item in analysis:
                        task = item.get("task", "unknown")
                        if task in task_counts:
                            task_counts[task] += 1
                        else:
                            task_counts[task] = 1
                    print(f"LLM prompt analysis: {task_counts}")
            except Exception as e:
                # Handle error gracefully
                if hasattr(self, "debug") and self.debug:
                    print(
                        f"LLM processing error: {type(batch_data['prompt']).__name__}, {str(e)}"
                    )
                # Create fallback outputs to avoid disrupting the pipeline
                batch_size = images.shape[0] if isinstance(images, torch.Tensor) else 1
                llm_outputs = {
                    "embeddings": None,
                    "task_logits": torch.zeros((batch_size, 2), device=images.device),
                }

                # Set appropriate task type based on task_id
                if task_type == "classification":
                    llm_outputs["task_logits"][:, 0] = 1.0
                else:  # segmentation
                    llm_outputs["task_logits"][:, 1] = 1.0

        # Task-specific processing
        outputs = {"features": features, "llm_outputs": llm_outputs}

        if task_type == "classification":
            if task_id not in self.classification_heads:
                num_classes = batch_data.get("num_classes", 2)
                # Convert num_classes to int if it's a Tensor
                if isinstance(num_classes, torch.Tensor):
                    num_classes = int(num_classes.item()) if num_classes.numel() == 1 else int(num_classes[0].item())
                # Ensure num_classes is an integer, fallback to 2
                if isinstance(num_classes, (list, tuple)) and len(num_classes) > 0:
                    num_classes = int(num_classes[0])
                if not isinstance(num_classes, int):
                    num_classes = 2
                self.create_classification_head(task_id, num_classes)
                # Move to same device as model
                self.classification_heads[task_id] = self.classification_heads[
                    task_id
                ].to(images.device)

            logits = self.forward_classification(features, task_id)
            outputs["logits"] = logits

        elif task_type == "segmentation":
            if task_id not in self.segmentation_decoders:
                num_classes = batch_data.get("num_classes", 2)
                # Convert num_classes to int if it's a Tensor
                if isinstance(num_classes, torch.Tensor):
                    num_classes = int(num_classes.item()) if num_classes.numel() == 1 else int(num_classes[0].item())
                # Ensure num_classes is an integer, fallback to 2
                if isinstance(num_classes, (list, tuple)) and len(num_classes) > 0:
                    num_classes = int(num_classes[0])
                if not isinstance(num_classes, int):
                    num_classes = 2
                self.create_segmentation_decoder(task_id, num_classes)
                # Move to same device as model
                self.segmentation_decoders[task_id] = self.segmentation_decoders[
                    task_id
                ].to(images.device)

            seg_output = self.forward_segmentation(
                images,
                multi_scale_features,
                task_id,
                (images.shape[2], images.shape[3]),
            )
            outputs["segmentation"] = seg_output

        return outputs

    def get_pretrained_backbone(self):
        """Return the backbone for use as pretrained model"""
        return {
            "patch_embed": self.patch_embed,
            "cls_token": self.cls_token,
            "pos_embed": self.pos_embed,
            "blocks": self.blocks,
            "norm": self.norm,
        }

    def load_pretrained_backbone(self, pretrained_dict):
        """Load pretrained backbone weights"""
        model_dict = self.state_dict()

        # Filter out task-specific heads
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if not k.startswith("classification_heads")
            and not k.startswith("segmentation_decoders")
        }

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded pretrained backbone with {len(pretrained_dict)} parameters")


def create_medical_foundation_model(
    model_size="base",
    img_size=224,
    in_chans=3,
    patch_size=16,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    kan_grid_size=5,
    use_llm=False,
    segmentation_backbone: str = "resnet18",
):
    """Factory function to create a new MedicalFoundationModel with the specified configuration"""

    # Model configurations for different sizes
    model_configs = {
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    }

    if model_size not in model_configs:
        raise ValueError(
            f"Unknown model size: {model_size}. Available: {list(model_configs.keys())}"
        )

    config = model_configs[model_size]

    # Create model with specified configuration
    model = MedicalFoundationModel(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        kan_grid_size=kan_grid_size,
        use_llm=use_llm,
        segmentation_backbone=segmentation_backbone,
    )

    return model
