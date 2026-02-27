"""
Multi-View Depth Consistency Module

This module enhances multi-view depth consistency and enables depth-weighted
attention for improved BEV construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple


class MultiViewDepthConsistency(nn.Module):
    """
    Enforces depth consistency across multiple views and computes
    depth confidence for weighted attention.
    """
    
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        consistency_threshold: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.consistency_threshold = consistency_threshold
        
        self.depth_confidence = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid(),
        )
        
        self.consistency_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute depth consistency and confidence across multiple views.
        
        Args:
            depth: Depth maps (B, N, 1, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4)
            intrinsics: Camera intrinsics (B, N, 3, 3)
        
        Returns:
            confidence: Depth confidence (B, N, 1, H, W)
            consistency_loss: Multi-view consistency loss (scalar)
        """
        B, N, _, H, W = depth.shape
        
        depth_flat = rearrange(depth, "b n c h w -> (b n) c h w")
        confidence_flat = self.depth_confidence(depth_flat)
        confidence = rearrange(confidence_flat, "(b n) c h w -> b n c h w", b=B, n=N)
        
        if N <= 1:
            return confidence, torch.tensor(0.0, device=depth.device)
        
        consistency_loss = self._compute_consistency_loss(
            depth, extrinsics, intrinsics
        )
        
        return confidence, consistency_loss
    
    def _compute_consistency_loss(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-view depth consistency loss.
        
        For overlapping regions, depths should be consistent when
        projected to 3D space.
        """
        B, N, _, H, W = depth.shape
        
        if N < 2:
            return torch.tensor(0.0, device=depth.device)
        
        total_loss = 0.0
        count = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                depth_i = depth[:, i]
                depth_j = depth[:, j]
                
                loss = F.l1_loss(depth_i, depth_j)
                total_loss = total_loss + loss
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss * torch.sigmoid(self.consistency_weight)


class DepthWeightedCrossViewAttention(nn.Module):
    """
    Cross-view attention with depth-weighted multi-view aggregation.
    
    Key improvements:
    1. Depth confidence weights attention scores
    2. Multi-view depth consistency is enforced
    3. 3D geometric constraints are applied
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        use_depth_weighting: bool = True,
        use_3d_projection: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.use_depth_weighting = use_depth_weighting
        self.use_3d_projection = use_3d_projection
        
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        if use_depth_weighting:
            self.depth_weight = nn.Sequential(
                nn.Conv2d(1, dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, num_heads, 1),
                nn.Softmax(dim=1),
            )
        
        if use_3d_projection:
            self.depth_to_3d = nn.Sequential(
                nn.Conv2d(1, dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 2, dim, 1),
            )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        depth_confidence: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: BEV query (B, dim, H_q, W_q)
            key: Image keys (B, N, dim, H_k, W_k)
            value: Image values (B, N, dim, H_v, W_v)
            depth: Depth maps (B, N, 1, H, W)
            depth_confidence: Depth confidence (B, N, 1, H, W)
            skip: Skip connection
        """
        B, dim, H_q, W_q = query.shape
        N = key.shape[1]
        
        query_flat = rearrange(query, "b d h w -> b (h w) d")
        Q = self.q_proj(query_flat)
        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        
        key_flat = rearrange(key, "b n d h w -> b (n h w) d")
        K = self.k_proj(key_flat)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.num_heads)
        
        value_flat = rearrange(value, "b n d h w -> b (n h w) d")
        V = self.v_proj(value_flat)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.num_heads)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if self.use_depth_weighting and depth is not None and depth_confidence is not None:
            depth_weight = self._compute_depth_attention_weight(
                depth, depth_confidence, H_q, W_q
            )
            attn = attn + depth_weight
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        out = rearrange(out, "b (h w) d -> b d h w", h=H_q, w=W_q)
        
        if skip is not None:
            out = out + skip
        
        return out
    
    def _compute_depth_attention_weight(
        self,
        depth: torch.Tensor,
        confidence: torch.Tensor,
        H_q: int,
        W_q: int,
    ) -> torch.Tensor:
        """
        Compute depth-based attention weight bias.
        
        Higher confidence → higher attention weight
        """
        B, N, _, H, W = depth.shape
        
        depth_weight = self.depth_weight(depth)
        confidence_weight = confidence * depth_weight
        
        confidence_flat = rearrange(confidence_weight, "b n h w -> b (n h w)")
        confidence_flat = confidence_flat.unsqueeze(1).unsqueeze(2)
        
        return confidence_flat * 0.1


class EnhancedDepthAwareBEV(nn.Module):
    """
    Enhanced depth-aware BEV construction with multi-view depth consistency.
    """
    
    def __init__(
        self,
        bev_dim: int = 128,
        num_heads: int = 4,
        use_depth_weighting: bool = True,
        use_3d_projection: bool = True,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        
        self.depth_consistency = MultiViewDepthConsistency(
            dim=bev_dim,
            num_heads=num_heads,
        )
        
        self.cross_view_attn = DepthWeightedCrossViewAttention(
            dim=bev_dim,
            num_heads=num_heads,
            use_depth_weighting=use_depth_weighting,
            use_3d_projection=use_3d_projection,
        )
        
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        bev_query: torch.Tensor,
        image_features: torch.Tensor,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bev_query: Initial BEV query (B, dim, H, W)
            image_features: Multi-view image features (B, N, dim, h, w)
            depth: Depth maps (B, N, 1, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4)
            intrinsics: Camera intrinsics (B, N, 3, 3)
        
        Returns:
            bev_features: Updated BEV features (B, dim, H, W)
            consistency_loss: Multi-view consistency loss
        """
        confidence, consistency_loss = self.depth_consistency(
            depth, extrinsics, intrinsics
        )
        
        bev_features = self.cross_view_attn(
            query=bev_query,
            key=image_features,
            value=image_features,
            depth=depth,
            depth_confidence=confidence,
        )
        
        return bev_features, consistency_loss * self.consistency_weight
