"""
Multi-View Depth Consistency Module (Improved)

This module enhances multi-view depth consistency with proper 3D geometric constraints
and enables depth-weighted attention for improved BEV construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple


class MultiViewDepthConsistency(nn.Module):
    """
    Enforces depth consistency across multiple views with 3D geometric constraints
    and computes depth confidence for weighted attention.
    """
    
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        consistency_threshold: float = 0.1,
        min_overlap_ratio: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.consistency_threshold = consistency_threshold
        self.min_overlap_ratio = min_overlap_ratio
        
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
            extrinsics: Camera extrinsics (B, N, 4, 4), world-to-camera transform
            intrinsics: Camera intrinsics (B, N, 3, 3)
        
        Returns:
            confidence: Depth confidence (B, N, 1, H, W)
            consistency_loss: Multi-view consistency loss with 3D geometric constraints
        """
        B, N, _, H, W = depth.shape
        
        depth_flat = rearrange(depth, "b n c h w -> (b n) c h w")
        confidence_flat = self.depth_confidence(depth_flat)
        confidence = rearrange(confidence_flat, "(b n) c h w -> b n c h w", b=B, n=N)
        
        if N <= 1:
            return confidence, torch.tensor(0.0, device=depth.device)
        
        consistency_loss = self._compute_3d_consistency_loss(
            depth, extrinsics, intrinsics
        )
        
        return confidence, consistency_loss
    
    def _backproject_to_3d(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backproject depth map to 3D points in camera coordinate system.
        
        Args:
            depth: Depth map (B, 1, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)
        
        Returns:
            points_3d: 3D points in camera coordinates (B, 3, H, W)
        """
        B, _, H, W = depth.shape
        device = depth.device
        
        # Create pixel coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        # (H, W) -> (1, H, W) -> (B, H, W)
        x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
        y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
        
        # Get intrinsics parameters
        fx = intrinsics[:, 0, 0].view(B, 1, 1)  # (B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)
        
        # Backproject to 3D camera coordinates
        # X = (x - cx) * Z / fx
        # Y = (y - cy) * Z / fy
        # Z = depth
        Z = depth.squeeze(1)  # (B, H, W)
        X = (x_coords - cx) * Z / fx
        Y = (y_coords - cy) * Z / fy
        
        points_3d = torch.stack([X, Y, Z], dim=1)  # (B, 3, H, W)
        return points_3d
    
    def _transform_to_world(
        self,
        points_cam: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform 3D points from camera coordinates to world coordinates.
        
        Args:
            points_cam: 3D points in camera coordinates (B, 3, H, W)
            extrinsics: Camera extrinsics (B, 4, 4), world-to-camera transform
        
        Returns:
            points_world: 3D points in world coordinates (B, 3, H, W)
        """
        B, _, H, W = points_cam.shape
        
        # Convert to homogeneous coordinates
        ones = torch.ones(B, 1, H, W, device=points_cam.device, dtype=points_cam.dtype)
        points_cam_homo = torch.cat([points_cam, ones], dim=1)  # (B, 4, H, W)
        
        # Reshape for matrix multiplication
        points_cam_flat = rearrange(points_cam_homo, "b c h w -> b c (h w)")  # (B, 4, H*W)
        
        # Transform to world coordinates: P_world = E^{-1} @ P_cam
        # Since extrinsics is world-to-camera, we need its inverse
        extrinsics_inv = torch.linalg.inv(extrinsics)  # (B, 4, 4)
        points_world_flat = torch.bmm(extrinsics_inv, points_cam_flat)  # (B, 4, H*W)
        
        # Convert back to non-homogeneous and reshape
        points_world = points_world_flat[:, :3, :]  # (B, 3, H*W)
        points_world = rearrange(points_world, "b c (h w) -> b c h w", h=H, w=W)
        
        return points_world
    
    def _project_to_camera(
        self,
        points_world: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        target_H: int,
        target_W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project 3D world points to target camera image plane.
        
        Args:
            points_world: 3D points in world coordinates (B, 3, H, W)
            extrinsics: Target camera extrinsics (B, 4, 4)
            intrinsics: Target camera intrinsics (B, 3, 3)
            target_H: Target image height
            target_W: Target image width
        
        Returns:
            projected_depth: Projected depth in target camera (B, 1, H, W)
            valid_mask: Valid projection mask (B, 1, H, W)
            x_coords, y_coords: Projected pixel coordinates
        """
        B, _, H, W = points_world.shape
        device = points_world.device
        
        # Convert to homogeneous coordinates
        ones = torch.ones(B, 1, H, W, device=device, dtype=points_world.dtype)
        points_world_homo = torch.cat([points_world, ones], dim=1)  # (B, 4, H, W)
        points_world_flat = rearrange(points_world_homo, "b c h w -> b c (h w)")  # (B, 4, H*W)
        
        # Transform to target camera coordinates
        points_cam_flat = torch.bmm(extrinsics, points_world_flat)  # (B, 4, H*W)
        points_cam = rearrange(points_cam_flat[:, :3, :], "b c (h w) -> b c h w", h=H, w=W)
        
        # Extract Z (depth in target camera)
        Z_target = points_cam[:, 2:3, :, :]  # (B, 1, H, W)
        
        # Project to image plane
        X_cam = points_cam[:, 0:1, :, :]  # (B, 1, H, W)
        Y_cam = points_cam[:, 1:2, :, :]  # (B, 1, H, W)
        
        fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)
        
        x_proj = (X_cam * fx / (Z_target + 1e-6)) + cx
        y_proj = (Y_cam * fy / (Z_target + 1e-6)) + cy
        
        # Check validity: points in front of camera and within image bounds
        valid_depth = (Z_target > 0.1) & (Z_target < 1000.0)
        valid_x = (x_proj >= 0) & (x_proj < target_W)
        valid_y = (y_proj >= 0) & (y_proj < target_H)
        valid_mask = valid_depth & valid_x & valid_y
        
        return Z_target, valid_mask.float(), x_proj, y_proj
    
    def _compute_3d_consistency_loss(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-view depth consistency loss with proper 3D geometric constraints.
        
        For overlapping regions, depths should be consistent when projected to 3D space
        and reprojected to other views.
        
        Args:
            depth: Depth maps (B, N, 1, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4)
            intrinsics: Camera intrinsics (B, N, 3, 3)
        
        Returns:
            consistency_loss: 3D geometric consistency loss
        """
        B, N, _, H, W = depth.shape
        
        if N < 2:
            return torch.tensor(0.0, device=depth.device)
        
        total_loss = 0.0
        count = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                # Get depth and camera parameters for view i and j
                depth_i = depth[:, i]  # (B, 1, H, W)
                depth_j = depth[:, j]  # (B, 1, H, W)
                extrinsics_i = extrinsics[:, i]  # (B, 4, 4)
                extrinsics_j = extrinsics[:, j]  # (B, 4, 4)
                intrinsics_i = intrinsics[:, i]  # (B, 3, 3)
                intrinsics_j = intrinsics[:, j]  # (B, 3, 3)
                
                # Step 1: Backproject view i depth to 3D camera coordinates
                points_3d_cam_i = self._backproject_to_3d(depth_i, intrinsics_i)
                
                # Step 2: Transform to world coordinates
                points_3d_world = self._transform_to_world(points_3d_cam_i, extrinsics_i)
                
                # Step 3: Project view i 3D points to view j camera
                depth_i_in_j, valid_mask_ij, x_proj_ij, y_proj_ij = self._project_to_camera(
                    points_3d_world, extrinsics_j, intrinsics_j, H, W
                )
                
                # Step 4: Sample depth_j at projected coordinates
                # Normalize coordinates to [-1, 1] for grid_sample
                x_norm = 2.0 * x_proj_ij / (W - 1) - 1.0
                y_norm = 2.0 * y_proj_ij / (H - 1) - 1.0
                grid = torch.cat([x_norm, y_norm], dim=1).permute(0, 2, 3, 1)  # (B, H, W, 2)
                
                depth_j_sampled = F.grid_sample(
                    depth_j, grid, mode='bilinear', padding_mode='zeros', align_corners=True
                )  # (B, 1, H, W)
                
                # Step 5: Compute consistency loss only for valid overlapping regions
                if valid_mask_ij.sum() > self.min_overlap_ratio * B * H * W:
                    # Use L1 loss with Huber loss for robustness
                    diff = torch.abs(depth_i_in_j - depth_j_sampled)
                    
                    # Huber loss: less sensitive to outliers
                    delta = 0.1
                    huber_loss = torch.where(
                        diff < delta,
                        0.5 * diff ** 2,
                        delta * (diff - 0.5 * delta)
                    )
                    
                    # Apply valid mask and normalize
                    masked_loss = (huber_loss * valid_mask_ij).sum() / (valid_mask_ij.sum() + 1e-6)
                    total_loss = total_loss + masked_loss
                    count += 1
                
                # Also do the reverse: project j to i for symmetry
                points_3d_cam_j = self._backproject_to_3d(depth_j, intrinsics_j)
                points_3d_world_j = self._transform_to_world(points_3d_cam_j, extrinsics_j)
                depth_j_in_i, valid_mask_ji, x_proj_ji, y_proj_ji = self._project_to_camera(
                    points_3d_world_j, extrinsics_i, intrinsics_i, H, W
                )
                
                x_norm_ji = 2.0 * x_proj_ji / (W - 1) - 1.0
                y_norm_ji = 2.0 * y_proj_ji / (H - 1) - 1.0
                grid_ji = torch.cat([x_norm_ji, y_norm_ji], dim=1).permute(0, 2, 3, 1)
                depth_i_sampled = F.grid_sample(
                    depth_i, grid_ji, mode='bilinear', padding_mode='zeros', align_corners=True
                )
                
                if valid_mask_ji.sum() > self.min_overlap_ratio * B * H * W:
                    diff_ji = torch.abs(depth_j_in_i - depth_i_sampled)
                    huber_loss_ji = torch.where(
                        diff_ji < delta,
                        0.5 * diff_ji ** 2,
                        delta * (diff_ji - 0.5 * delta)
                    )
                    masked_loss_ji = (huber_loss_ji * valid_mask_ji).sum() / (valid_mask_ji.sum() + 1e-6)
                    total_loss = total_loss + masked_loss_ji
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
        
        Higher confidence -> higher attention weight
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
