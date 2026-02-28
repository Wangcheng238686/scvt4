"""
SAM2 Backbone Wrapper for Cross-View Transformer

This module wraps SAM2 Vision Encoder to be compatible with CVTEncoder,
enabling shared backbone between satellite and drone branches.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

SAM2_AVAILABLE = False
try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    SAM2_AVAILABLE = True
except ImportError:
    pass


def _load_sam2_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> dict:
    """Load SAM2 checkpoint from file."""
    from mmengine.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


class SAM2BackboneWrapper(nn.Module):
    """
    SAM2 Backbone Wrapper for Cross-View Transformer
    
    Wraps SAM2 Vision Encoder to output multi-scale features compatible with CVTEncoder.
    SAM2 outputs:
        - vision_features: (B, 256, H/16, W/16)
        - backbone_fpn: [(B, 256, H/4, W/4), (B, 256, H/8, W/8), 
                         (B, 256, H/16, W/16), (B, 256, H/32, W/32)]
    
    Cross-View expects:
        - Multi-scale feature list
        - output_shapes attribute
    """
    
    def __init__(
        self,
        sam2_encoder: nn.Module,
        input_size: int = 512,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.sam2_encoder = sam2_encoder
        self.input_size = input_size
        self.use_checkpoint = use_checkpoint
        
        feat_size = input_size // 16
        self.output_shapes = [
            (1, 256, feat_size * 4, feat_size * 4),
            (1, 256, feat_size * 2, feat_size * 2),
            (1, 256, feat_size, feat_size),
            (1, 256, feat_size // 2, feat_size // 2),
        ]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            features: Multi-scale feature list
                [(B, 256, H/4, W/4), (B, 256, H/8, W/8), 
                 (B, 256, H/16, W/16), (B, 256, H/32, W/32)]
        """
        if self.use_checkpoint and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self.sam2_encoder, x, use_reentrant=False
            )
        else:
            features = self.sam2_encoder(x)
        
        if isinstance(features, dict):
            vision_features = features.get("vision_features")
            backbone_fpn = features.get("backbone_fpn")
            
            if backbone_fpn is not None and len(backbone_fpn) >= 4:
                return list(backbone_fpn)[:4]
            elif backbone_fpn is not None and len(backbone_fpn) > 0:
                return self._create_pyramid_from_fpn(backbone_fpn, vision_features)
            elif vision_features is not None:
                return self._create_pyramid(vision_features)
            else:
                raise ValueError("SAM2 encoder returned empty features")
        elif isinstance(features, (list, tuple)):
            if len(features) == 2 and isinstance(features[1], (list, tuple)):
                backbone_fpn = features[1]
                if len(backbone_fpn) >= 4:
                    return list(backbone_fpn)[:4]
                return self._create_pyramid_from_fpn(backbone_fpn, features[0])
            return self._create_pyramid(features[0])
        else:
            return self._create_pyramid(features)
    
    def _create_pyramid(self, base_feat: torch.Tensor) -> List[torch.Tensor]:
        """Create feature pyramid from base feature."""
        B, C, H, W = base_feat.shape
        features = [base_feat]
        
        for _ in range(3):
            features.append(F.max_pool2d(features[-1], kernel_size=2, stride=2))
        
        upsampled = F.interpolate(features[0], scale_factor=4, mode='bilinear', align_corners=False)
        features[0] = upsampled
        
        upsampled2 = F.interpolate(features[0], scale_factor=0.5, mode='bilinear', align_corners=False)
        features.insert(1, upsampled2)
        
        return features[:4]
    
    def _create_pyramid_from_fpn(
        self, 
        fpn_features: List[torch.Tensor],
        vision_features: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Create complete pyramid from partial FPN features."""
        features = list(fpn_features)
        
        while len(features) < 4:
            features.append(F.max_pool2d(features[-1], kernel_size=2, stride=2))
        
        return features[:4]


class SAM2BackboneForCVT(nn.Module):
    """
    SAM2 Backbone specifically designed for CVTEncoder
    
    This wrapper ensures SAM2 output is compatible with CVTEncoder's expectations:
    - Returns list of multi-scale features
    - Provides output_shapes attribute with correct dimensions
    - Handles gradient checkpointing
    """
    
    def __init__(
        self,
        sam2_encoder: nn.Module,
        input_size: int = 512,
        feature_dim: int = 256,
        num_levels: int = 4,
    ):
        super().__init__()
        self.sam2_encoder = sam2_encoder
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        
        feat_size = input_size // 16
        self.output_shapes = [
            (1, feature_dim, feat_size * 4, feat_size * 4),
            (1, feature_dim, feat_size * 2, feat_size * 2),
            (1, feature_dim, feat_size, feat_size),
            (1, feature_dim, feat_size // 2, feat_size // 2),
        ][:num_levels]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features for CVTEncoder
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            List of feature tensors at different scales
        """
        features = self.sam2_encoder(x)
        
        if isinstance(features, dict):
            vision_features = features.get("vision_features")
            backbone_fpn = features.get("backbone_fpn")
            
            if backbone_fpn is not None and len(backbone_fpn) >= self.num_levels:
                return [f.float() for f in list(backbone_fpn)[:self.num_levels]]
            elif backbone_fpn is not None:
                return self._complete_pyramid(list(backbone_fpn), vision_features)
            else:
                return self._build_pyramid(vision_features)
        
        elif isinstance(features, (list, tuple)):
            if len(features) == 2 and isinstance(features[1], (list, tuple)):
                fpn = list(features[1])
                if len(fpn) >= self.num_levels:
                    return [f.float() for f in fpn[:self.num_levels]]
                return self._complete_pyramid(fpn, features[0])
            return self._build_pyramid(features[0])
        
        else:
            return self._build_pyramid(features)
    
    def _build_pyramid(self, base_feat: torch.Tensor) -> List[torch.Tensor]:
        """Build complete feature pyramid from base feature."""
        base_feat = base_feat.float()
        B, C, H, W = base_feat.shape
        
        features = []
        
        for i in range(self.num_levels):
            scale = 2 ** (self.num_levels - 1 - i)
            target_h = H * scale
            target_w = W * scale
            
            if i == 0:
                feat = F.interpolate(base_feat, size=(target_h, target_w), 
                                    mode='bilinear', align_corners=False)
            elif i < self.num_levels - 1:
                prev_feat = features[-1]
                feat = F.max_pool2d(prev_feat, kernel_size=2, stride=2)
            else:
                feat = F.max_pool2d(features[-1], kernel_size=2, stride=2)
            
            features.append(feat)
        
        return features
    
    def _complete_pyramid(
        self, 
        fpn_features: List[torch.Tensor],
        vision_features: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Complete partial FPN to full pyramid."""
        features = [f.float() for f in fpn_features]
        
        while len(features) < self.num_levels:
            features.append(F.max_pool2d(features[-1], kernel_size=2, stride=2))
        
        return features[:self.num_levels]


class StandaloneSAM2Backbone(nn.Module):
    """
    Standalone SAM2 Backbone that creates its own encoder instance.
    
    This is used when share_sam2_backbone=False but use_sam2_backbone=True,
    meaning the drone branch should have its own independent SAM2 encoder.
    
    Unlike SAM2BackboneForCVT which wraps a shared encoder, this class
    creates and loads its own SAM2 encoder from checkpoint.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        input_size: int = 512,
        feature_dim: int = 256,
        num_levels: int = 4,
        use_checkpoint: bool = False,
        freeze: bool = False,
    ):
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Please install it via:\n"
                "pip install git+https://github.com/facebookresearch/sam2.git"
            )
        
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint
        self.freeze = freeze
        
        self.encoder = self._build_and_load_encoder(checkpoint_path)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        feat_size = input_size // 16
        self.output_shapes = [
            (1, feature_dim, feat_size * 4, feat_size * 4),
            (1, feature_dim, feat_size * 2, feat_size * 2),
            (1, feature_dim, feat_size, feat_size),
            (1, feature_dim, feat_size // 2, feat_size // 2),
        ][:num_levels]
    
    def _build_and_load_encoder(self, checkpoint_path: str) -> nn.Module:
        """Build SAM2 encoder architecture and load weights."""
        import sys
        sys.path.insert(0, '/home/wangcheng/project/sam2-main')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
        
        trunk = Hiera(
            embed_dim=112,
            num_heads=2,
            drop_path_rate=0.1,
            q_pool=3,
            q_stride=(2, 2),
            stages=(2, 3, 16, 3),
            dim_mul=2.0,
            head_mul=2.0,
            window_spec=(8, 4, 14, 7),
            global_att_blocks=(12, 16, 20),
            return_interm_layers=True,
        )
        
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            temperature=10000,
        )
        
        neck = FpnNeck(
            position_encoding=position_encoding,
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )
        
        image_encoder = ImageEncoder(
            trunk=trunk,
            neck=neck,
            scalp=0,
        )
        
        checkpoint = _load_sam2_checkpoint(checkpoint_path)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("image_encoder."):
                new_k = k[14:]
                new_state_dict[new_k] = v
        
        msg = image_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"[StandaloneSAM2Backbone] Loaded from {checkpoint_path}. Missing keys: {len(msg.missing_keys)}")
        
        return image_encoder
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features for CVTEncoder.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            List of feature tensors at different scales
        """
        if self.use_checkpoint and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self.encoder, x, use_reentrant=False
            )
        else:
            features = self.encoder(x)
        
        if isinstance(features, dict):
            vision_features = features.get("vision_features")
            backbone_fpn = features.get("backbone_fpn")
            
            if backbone_fpn is not None and len(backbone_fpn) >= self.num_levels:
                return [f.float() for f in list(backbone_fpn)[:self.num_levels]]
            elif backbone_fpn is not None:
                return self._complete_pyramid(list(backbone_fpn), vision_features)
            else:
                return self._build_pyramid(vision_features)
        
        elif isinstance(features, (list, tuple)):
            if len(features) == 2 and isinstance(features[1], (list, tuple)):
                fpn = list(features[1])
                if len(fpn) >= self.num_levels:
                    return [f.float() for f in fpn[:self.num_levels]]
                return self._complete_pyramid(fpn, features[0])
            return self._build_pyramid(features[0])
        
        else:
            return self._build_pyramid(features)
    
    def _build_pyramid(self, base_feat: torch.Tensor) -> List[torch.Tensor]:
        """Build complete feature pyramid from base feature."""
        base_feat = base_feat.float()
        B, C, H, W = base_feat.shape
        
        features = []
        
        for i in range(self.num_levels):
            scale = 2 ** (self.num_levels - 1 - i)
            target_h = H * scale
            target_w = W * scale
            
            if i == 0:
                feat = F.interpolate(base_feat, size=(target_h, target_w), 
                                    mode='bilinear', align_corners=False)
            elif i < self.num_levels - 1:
                prev_feat = features[-1]
                feat = F.max_pool2d(prev_feat, kernel_size=2, stride=2)
            else:
                feat = F.max_pool2d(features[-1], kernel_size=2, stride=2)
            
            features.append(feat)
        
        return features
    
    def _complete_pyramid(
        self, 
        fpn_features: List[torch.Tensor],
        vision_features: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Complete partial FPN to full pyramid."""
        features = [f.float() for f in fpn_features]
        
        while len(features) < self.num_levels:
            features.append(F.max_pool2d(features[-1], kernel_size=2, stride=2))
        
        return features[:self.num_levels]
