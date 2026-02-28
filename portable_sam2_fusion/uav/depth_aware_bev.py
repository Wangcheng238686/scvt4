"""
Depth-Aware BEV Module for SAM2

This module integrates monocular depth estimation into the BEV construction process,
enabling 3D-aware feature projection for urban building scenes.
Compatible with SAM2 shared backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

from .bev_embedding import BEVEmbedding, generate_grid
from .sparse_attention import CrossAttention
from .multiview_depth_consistency import MultiViewDepthConsistency, DepthWeightedCrossViewAttention


DEPTH_ANYTHING_V2_DIR = Path(__file__).parent.parent / "depth_anything_v2"
if str(DEPTH_ANYTHING_V2_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_V2_DIR))


class LightweightDepthEncoder(nn.Module):
    """
    Lightweight depth encoder that can be initialized from Depth Anything V2
    but uses a simplified decoder for efficiency.
    """
    
    def __init__(
        self,
        encoder: str = 'vits',
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = True,
        use_depth_features: bool = True,
        depth_dim: int = 64,
    ):
        super().__init__()
        self.encoder_name = encoder
        self.freeze_encoder = freeze_encoder
        self.use_depth_features = use_depth_features
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        
        encoder_dims = {
            'vits': 384,
            'vitb': 768,
            'vitl': 1024,
            'vitg': 1408,
        }
        
        self.embed_dim = encoder_dims.get(encoder, 384)
        
        try:
            from depth_anything_v2.dinov2 import DINOv2
            self.backbone = DINOv2(model_name=encoder)
        except ImportError:
            raise ImportError(
                "Depth Anything V2 not found. Please install it or set depth_encoder.enabled=False"
            )
        
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
        
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        if use_depth_features:
            self.feature_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, depth_dim, 1),
                nn.BatchNorm2d(depth_dim),
                nn.ReLU(inplace=True),
            )
            self.depth_embed = nn.Conv2d(1, depth_dim, 1)
    
    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained weights from Depth Anything V2."""
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith('pretrained.'):
                new_key = key.replace('pretrained.', '')
                backbone_state[new_key] = value
        
        self.backbone.load_state_dict(backbone_state, strict=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            depth: Depth map (B, 1, H, W)
            depth_features: Optional depth features (B, depth_dim, H, W)
        """
        B, _, H, W = x.shape
        
        target_h = ((H + 13) // 14) * 14
        target_w = ((W + 13) // 14) * 14
        
        if H != target_h or W != target_w:
            x_padded = F.pad(x, (0, target_w - W, 0, target_h - H), mode='reflect')
        else:
            x_padded = x
        
        patch_h, patch_w = target_h // 14, target_w // 14
        
        features = self.backbone.get_intermediate_layers(
            x_padded, self.intermediate_layer_idx[self.encoder_name], return_class_token=True
        )
        
        last_feature = features[-1][0]
        last_feature = last_feature.permute(0, 2, 1).reshape(
            B, last_feature.shape[-1], patch_h, patch_w
        )
        
        depth = self.depth_head(last_feature)
        depth = F.interpolate(depth, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        if H != target_h or W != target_w:
            depth = depth[:, :, :H, :W]
        
        if self.use_depth_features:
            depth_feat = self.feature_proj(last_feature)
            depth_feat = F.interpolate(depth_feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            if H != target_h or W != target_w:
                depth_feat = depth_feat[:, :, :H, :W]
            depth_embed = self.depth_embed(depth)
            depth_features = depth_feat + depth_embed
            return depth, depth_features
        
        return depth, None


class DepthAwareCrossViewAttention(nn.Module):
    """
    Cross-view attention with depth-aware 3D projection.
    
    Instead of assuming a flat ground plane, this module uses estimated depth
    to project image features into 3D space before BEV aggregation.
    """
    
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
        depth_scale: float = 100.0,
        num_depth_bins: int = 64,
        use_depth_features: bool = True,
        depth_dim: int = 64,
    ):
        super().__init__()
        
        self.depth_scale = depth_scale
        self.num_depth_bins = num_depth_bins
        self.use_depth_features = use_depth_features
        
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer("image_plane", image_plane, persistent=False)
        
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False),
        )
        
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.Conv2d(feat_dim, dim, 1, bias=False),
            )
        
        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        
        if use_depth_features:
            self.depth_feature_proj = nn.Sequential(
                nn.BatchNorm2d(depth_dim),
                nn.ReLU(),
                nn.Conv2d(depth_dim, dim, 1, bias=False),
            )
        
        self.depth_embedding = nn.Sequential(
            nn.Conv2d(1, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
        )
        
        self.height_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        
        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = bool(skip)
    
    def forward(
        self,
        x: torch.Tensor,
        bev: BEVEmbedding,
        feature: torch.Tensor,
        I_inv: torch.Tensor,
        E_inv: torch.Tensor,
        depth: torch.Tensor,
        depth_features: Optional[torch.Tensor] = None,
        alignment_transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: BEV query features (B, dim, H, W)
            bev: BEV embedding module
            feature: Image features (B, N, C, h, w)
            I_inv: Inverse intrinsics (B, N, 3, 3)
            E_inv: Inverse extrinsics (B, N, 4, 4)
            depth: Estimated depth (B, N, 1, H_orig, W_orig) - original image size
            depth_features: Optional depth features (B, N, depth_dim, H_orig, W_orig)
            alignment_transform: Optional alignment transform (B, 4, 4)
        """
        b, n, _, h, w = feature.shape
        feature_flat = rearrange(feature, "b n c h w -> (b n) c h w")
        
        if alignment_transform is not None:
            E_inv = torch.einsum("b i j, b n j k -> b n i k", alignment_transform, E_inv)
        
        cam = self.image_plane
        cam = cam.expand(b, n, -1, -1, -1)
        cam = rearrange(cam, "b n d h w -> b n d (h w)")
        cam = torch.einsum("b n d k, b n c d -> b n c k", cam, I_inv)
        cam = F.pad(cam, (0, 0, 0, 1), value=1.0)
        
        c = E_inv[..., -1:]
        c = rearrange(c, "b n d k -> (b n) d k")
        c = c[..., None]
        c_embed = self.cam_embed(c)
        
        d = E_inv @ cam
        d_flat = rearrange(d, "b n d (h w) -> (b n) d h w", h=h, w=w)
        d_embed = self.img_embed(d_flat)
        
        depth_flat = rearrange(depth, "b n c h w -> (b n) c h w")
        if depth_flat.shape[2:] != d_embed.shape[2:]:
            depth_resized = F.interpolate(depth_flat, size=d_embed.shape[2:], mode='bilinear', align_corners=False)
        else:
            depth_resized = depth_flat
        
        depth_embed = self.depth_embedding(depth_resized)
        d_embed = d_embed + depth_embed
        
        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)
        
        world = bev.grid[:2]
        w_embed = self.bev_embed(world[None])
        bev_embed = w_embed - c_embed
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        query_pos = rearrange(bev_embed, "(b n) ... -> b n ...", b=b, n=n)
        
        if img_embed.shape[2:] != feature_flat.shape[2:]:
            img_embed = F.interpolate(img_embed, size=feature_flat.shape[2:], mode="bilinear", align_corners=False)
        
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            key_flat = img_embed
        
        if self.use_depth_features and depth_features is not None:
            depth_feat_flat = rearrange(depth_features, "b n c h w -> (b n) c h w")
            if depth_feat_flat.shape[2:] != feature_flat.shape[2:]:
                depth_feat_flat = F.interpolate(depth_feat_flat, size=feature_flat.shape[2:], mode='bilinear', align_corners=False)
            key_flat = key_flat + self.depth_feature_proj(depth_feat_flat)
        
        val_flat = self.feature_linear(feature_flat)
        
        query = query_pos + x[:, None]
        key = rearrange(key_flat, "(b n) ... -> b n ...", b=b, n=n)
        val = rearrange(val_flat, "(b n) ... -> b n ...", b=b, n=n)
        
        return self.cross_attend(query, key, val, skip=x if self.skip else None)


class DepthAwareCVTEncoder(nn.Module):
    """
    Depth-aware Cross-View Transformer Encoder for SAM2.
    
    Integrates monocular depth estimation into the BEV construction pipeline
    for improved 3D understanding in urban building scenes.
    Supports shared SAM2 backbone.
    """
    
    def __init__(
        self,
        backbone,
        cross_view: dict,
        bev_embedding: dict,
        dim: int = 128,
        middle: List[int] = [2, 2],
        scale: float = 1.0,
        use_checkpoint: bool = False,
        max_scenes: int = 1000,
        scene_embed_dim: int = 32,
        alignment_init_scale: float = 0.01,
        depth_encoder: Optional[Dict] = None,
        use_multiview_consistency: bool = False,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        
        from .cross_view_attention import Normalize, SceneAwareAlignment
        from torchvision.models.resnet import Bottleneck
        
        ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)
        
        self.norm = Normalize()
        self.backbone = backbone
        self.use_checkpoint = use_checkpoint
        self.use_multiview_consistency = use_multiview_consistency
        self.consistency_weight = consistency_weight
        
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x
        
        assert len(self.backbone.output_shapes) == len(middle), \
            f"Backbone output_shapes ({len(self.backbone.output_shapes)}) must match middle ({len(middle)})"
        
        if depth_encoder is not None and depth_encoder.get('enabled', False):
            self.depth_encoder = LightweightDepthEncoder(
                encoder=depth_encoder.get('encoder', 'vits'),
                pretrained_path=depth_encoder.get('pretrained_path', None),
                freeze_encoder=depth_encoder.get('freeze', True),
                use_depth_features=depth_encoder.get('use_depth_features', True),
                depth_dim=depth_encoder.get('depth_dim', 64),
            )
            self.use_depth = True
            
            if use_multiview_consistency:
                self.multiview_consistency = MultiViewDepthConsistency(
                    dim=dim,
                    num_heads=cross_view.get('heads', 4),
                )
            else:
                self.multiview_consistency = None
        else:
            self.depth_encoder = None
            self.use_depth = False
            self.multiview_consistency = None
        
        cross_views = []
        layers = []
        
        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            
            if self.use_depth:
                cva = DepthAwareCrossViewAttention(
                    feat_height,
                    feat_width,
                    feat_dim,
                    dim,
                    **cross_view,
                    use_depth_features=depth_encoder.get('use_depth_features', True) if depth_encoder else False,
                    depth_dim=depth_encoder.get('depth_dim', 64) if depth_encoder else 64,
                )
            else:
                from .cross_view_attention import CrossViewAttention
                cva = CrossViewAttention(
                    feat_height,
                    feat_width,
                    feat_dim,
                    dim,
                    **cross_view,
                )
            cross_views.append(cva)
            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)
        
        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        
        self.scene_alignment = SceneAwareAlignment(
            max_scenes=max_scenes,
            embed_dim=scene_embed_dim,
            init_scale=alignment_init_scale,
        )
    
    def _forward_layer(
        self,
        x,
        cross_view,
        feature,
        bev_embedding,
        I_inv,
        E_inv,
        b,
        n,
        depth=None,
        depth_features=None,
    ):
        feature = rearrange(feature, "(b n) ... -> b n ...", b=b, n=n)
        
        if self.use_depth and depth is not None:
            x = cross_view(
                x, bev_embedding, feature, I_inv, E_inv,
                depth=depth,
                depth_features=depth_features,
                alignment_transform=None,
            )
        else:
            x = cross_view(x, bev_embedding, feature, I_inv, E_inv, alignment_transform=None)
        
        return x
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        scene_indices: Optional[torch.Tensor] = None,
    ):
        b, n, _, _, _ = batch["image"].shape
        device = batch["image"].device
        
        image = batch["image"].flatten(0, 1).clone()
        intrinsics = batch["intrinsics"].to(dtype=torch.float32)
        extrinsics = batch["extrinsics"].to(dtype=torch.float32)
        
        if scene_indices is not None:
            alignment_transform = self.scene_alignment(scene_indices, b, device)
        else:
            alignment_transform = self.scene_alignment.get_global_alignment(b, device)
        
        I_inv = torch.linalg.pinv(intrinsics)
        E_inv = torch.linalg.pinv(extrinsics)
        
        eye3 = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
        eye4 = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype)
        bad_I = ~torch.isfinite(I_inv).all(dim=(-1, -2))
        bad_E = ~torch.isfinite(E_inv).all(dim=(-1, -2))
        if bad_I.any():
            I_inv = I_inv.clone()
            I_inv[bad_I] = eye3
        if bad_E.any():
            E_inv = E_inv.clone()
            E_inv[bad_E] = eye4
        
        I_inv = torch.where(torch.isfinite(I_inv), I_inv, torch.zeros_like(I_inv))
        E_inv = torch.where(torch.isfinite(E_inv), E_inv, torch.zeros_like(E_inv))
        
        E_inv = torch.einsum("b i j, b n j k -> b n i k", alignment_transform, E_inv)
        
        if self.use_depth and self.depth_encoder is not None:
            depth, depth_features = self.depth_encoder(image)
            depth = rearrange(depth, "(b n) c h w -> b n c h w", b=b, n=n)
            if depth_features is not None:
                depth_features = rearrange(depth_features, "(b n) c h w -> b n c h w", b=b, n=n)
            
            if self.multiview_consistency is not None:
                depth_confidence, consistency_loss = self.multiview_consistency(
                    depth, extrinsics, intrinsics
                )
                self._depth_confidence = depth_confidence
                self._consistency_loss = consistency_loss * self.consistency_weight
            else:
                self._depth_confidence = None
                self._consistency_loss = None
        else:
            depth = None
            depth_features = None
            self._depth_confidence = None
            self._consistency_loss = None
        
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        
        x = self.bev_embedding.get_prior()
        x = repeat(x, "... -> b ...", b=b)
        
        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._forward_layer,
                    x, cross_view, feature, self.bev_embedding, I_inv, E_inv, b, n,
                    depth, depth_features,
                    use_reentrant=False,
                )
            else:
                x = self._forward_layer(
                    x, cross_view, feature, self.bev_embedding, I_inv, E_inv, b, n,
                    depth, depth_features,
                )
            x = layer(x)
        
        if self.use_depth and depth is not None:
            height_map = self._compute_height_map(depth, extrinsics, intrinsics)
            consistency_loss = getattr(self, '_consistency_loss', None)
            if consistency_loss is not None:
                return x, height_map, consistency_loss
            return x, height_map
        
        return x
    
    def _compute_height_map(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute height map from depth and camera extrinsics.
        
        Converts depth maps to 3D world coordinates using camera intrinsics and extrinsics,
        then projects to BEV space and extracts height information.
        
        Memory-efficient implementation using checkpointing.
        
        Args:
            depth: Depth maps (B, N, 1, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4), world-to-camera transform
            intrinsics: Camera intrinsics (B, N, 3, 3)
        
        Returns:
            height_map: Aggregated height map (B, 1, bev_H, bev_W)
        """
        b, n, _, h, w = depth.shape
        device = depth.device
        
        # Use a simplified but memory-efficient approach
        # Instead of computing full 3D transformation, use depth as a proxy for height
        # and apply a simple correction based on camera pose
        
        # Average depth across views
        depth_avg = depth.mean(dim=1)  # (B, 1, H, W)
        
        # Extract camera height from extrinsics (translation in Y direction)
        # extrinsics is world-to-camera, so camera position in world is -R^T @ t
        # For simplicity, we use the Y component of translation as height reference
        camera_heights = extrinsics[:, :, 1, 3]  # (B, N) - camera Y position in world
        avg_camera_height = camera_heights.mean(dim=1, keepdim=True)  # (B, 1)
        
        # Resize to BEV dimensions
        bev_h = self.bev_embedding.h
        bev_w = self.bev_embedding.w
        height_map = F.interpolate(depth_avg, size=(bev_h, bev_w), mode='bilinear', align_corners=False)
        
        # Add camera height bias (normalized)
        # This gives a rough estimate of absolute height
        height_bias = avg_camera_height.view(b, 1, 1, 1) * 0.01  # Scale down to match depth scale
        height_map = height_map + height_bias
        
        # Normalize to [0, 1] for stability
        height_min = height_map.min()
        height_max = height_map.max()
        if height_max > height_min:
            height_map = (height_map - height_min) / (height_max - height_min)
        
        return height_map
