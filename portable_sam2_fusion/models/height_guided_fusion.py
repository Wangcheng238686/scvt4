"""
Height-Guided Spatial Fusion Module

This module enhances the BEV-satellite feature fusion by using height information
derived from depth estimation to guide the attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HeightGuidedSpatialFusion(nn.Module):
    """
    Height-guided spatial fusion that uses estimated height/depth information
    to weight BEV features during satellite-BEV fusion.
    
    Key idea: Building regions (higher height) should receive more attention
    in the fusion process for better building instance segmentation.
    
    Height Loss Design:
    - Variance loss: Encourages height distribution to have clear separation
                     between buildings (high) and ground (low)
    - Sparsity loss: Encourages most regions to have low height (ground)
                     while allowing few regions to have high height (buildings)
    - Smoothness loss: Encourages spatial smoothness in height predictions
    """
    
    def __init__(
        self,
        bev_dim: int,
        sat_dim: int,
        num_heads: int = 4,
        temperature: float = 0.1,
        downsample_factor: int = 1,
        max_attn_size: int = 64,
        use_checkpoint: bool = False,
        height_dim: int = 64,
        use_height_gate: bool = True,
        height_threshold: float = 0.3,
        use_variance_loss: bool = True,
        use_sparsity_loss: bool = True,
        use_smoothness_loss: bool = True,
        variance_weight: float = 0.1,
        sparsity_weight: float = 0.05,
        smoothness_weight: float = 0.01,
    ):
        super().__init__()
        
        self.bev_dim = bev_dim
        self.sat_dim = sat_dim
        self.temperature = temperature
        self.downsample_factor = downsample_factor
        self.max_attn_size = max_attn_size
        self.use_checkpoint = use_checkpoint
        self.use_height_gate = use_height_gate
        self.height_threshold = height_threshold
        
        # Height loss configuration
        self.use_variance_loss = use_variance_loss
        self.use_sparsity_loss = use_sparsity_loss
        self.use_smoothness_loss = use_smoothness_loss
        self.variance_weight = variance_weight
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
        
        self.bev_proj = nn.Sequential(
            nn.Conv2d(bev_dim, sat_dim, 1, bias=False),
            nn.BatchNorm2d(sat_dim),
            nn.ReLU(inplace=True),
        )
        
        self.sat_proj = nn.Sequential(
            nn.Conv2d(sat_dim, sat_dim, 1, bias=False),
            nn.BatchNorm2d(sat_dim),
            nn.ReLU(inplace=True),
        )
        
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, height_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(height_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(height_dim, height_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(height_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(height_dim, 1, 1),
            nn.Sigmoid(),
        )
        
        if use_height_gate:
            self.height_gate = nn.Sequential(
                nn.Conv2d(1 + sat_dim, sat_dim, 1, bias=False),
                nn.BatchNorm2d(sat_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(sat_dim, sat_dim, 1),
                nn.Sigmoid(),
            )
        
        self.spatial_attn = nn.MultiheadAttention(
            sat_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(sat_dim, sat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(sat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(sat_dim, sat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(sat_dim),
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))
        self.height_weight = nn.Parameter(torch.tensor(0.5))
    
    def _compute_downsample_size(self, H: int, W: int) -> Tuple[int, int]:
        if self.downsample_factor > 1:
            new_h = max(1, H // self.downsample_factor)
            new_w = max(1, W // self.downsample_factor)
            return new_h, new_w
        if self.max_attn_size > 0 and (H > self.max_attn_size or W > self.max_attn_size):
            scale = max(H / self.max_attn_size, W / self.max_attn_size)
            new_h = max(1, int(H / scale))
            new_w = max(1, int(W / scale))
            return new_h, new_w
        return H, W
    
    def _forward_core(
        self,
        sat_feat: torch.Tensor,
        bev_feat: torch.Tensor,
        height_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = sat_feat.shape
        orig_size = (H, W)
        
        attn_h, attn_w = self._compute_downsample_size(H, W)
        
        bev_proj = self.bev_proj(bev_feat)
        bev_up = F.interpolate(bev_proj, size=orig_size, mode='bilinear', align_corners=False)
        
        sat_proj = self.sat_proj(sat_feat)
        
        if height_map is not None:
            if height_map.dim() == 3:
                height_map = height_map.unsqueeze(1)
            height_attn = self.height_encoder(height_map)
            if height_attn.shape[2:] != orig_size:
                height_attn = F.interpolate(height_attn, size=orig_size, mode='bilinear', align_corners=False)
            
            building_mask = (height_attn > self.height_threshold).float()
            
            if self.use_height_gate:
                gate_input = torch.cat([height_attn, bev_up], dim=1)
                height_gate = self.height_gate(gate_input)
                bev_weighted = bev_up * (1.0 + torch.sigmoid(self.height_weight) * height_gate)
            else:
                bev_weighted = bev_up * (1.0 + height_attn)
        else:
            height_attn = None
            building_mask = None
            bev_weighted = bev_up
        
        if (attn_h, attn_w) != orig_size:
            sat_for_attn = F.interpolate(sat_proj, size=(attn_h, attn_w), mode='bilinear', align_corners=False)
            bev_for_attn = F.interpolate(bev_weighted, size=(attn_h, attn_w), mode='bilinear', align_corners=False)
            if height_attn is not None:
                height_for_attn = F.interpolate(height_attn, size=(attn_h, attn_w), mode='bilinear', align_corners=False)
            else:
                height_for_attn = None
        else:
            sat_for_attn = sat_proj
            bev_for_attn = bev_weighted
            height_for_attn = height_attn
        
        sat_flat = sat_for_attn.flatten(2).transpose(1, 2)
        bev_flat = bev_for_attn.flatten(2).transpose(1, 2)
        
        attn_out, attn_weights = self.spatial_attn(sat_flat, bev_flat, bev_flat)
        attn_out = attn_out.transpose(1, 2).view(B, C, attn_h, attn_w)
        
        if (attn_h, attn_w) != orig_size:
            attn_out = F.interpolate(attn_out, size=orig_size, mode='bilinear', align_corners=False)
        
        sat_norm = F.normalize(sat_flat, dim=-1)
        bev_norm = F.normalize(bev_flat, dim=-1)
        similarity = torch.bmm(sat_norm, bev_norm.transpose(1, 2))
        similarity_map = similarity.mean(dim=1).view(B, 1, attn_h, attn_w)
        similarity_map = F.interpolate(similarity_map, size=orig_size, mode='bilinear', align_corners=False)
        similarity_map = torch.sigmoid(similarity_map / self.temperature)
        
        alpha = torch.sigmoid(self.fusion_weight)
        fused = alpha * attn_out + (1 - alpha) * sat_feat
        
        output = sat_feat + self.refine(fused - sat_feat)
        
        align_loss = 1 - similarity.mean()
        
        # Compute height losses with physical meaning
        height_loss_dict = self._compute_height_losses(height_attn, building_mask)
        height_loss = height_loss_dict.get('loss_height_total', torch.tensor(0.0, device=sat_feat.device))
        
        return output, align_loss, height_loss, height_loss_dict
    
    def _compute_height_losses(
        self,
        height_attn: Optional[torch.Tensor],
        building_mask: Optional[torch.Tensor],
    ) -> dict:
        """
        Compute height-related losses with physical meaning.
        
        Args:
            height_attn: Height attention map (B, 1, H, W) in range [0, 1]
            building_mask: Binary mask indicating potential building regions
        
        Returns:
            loss_dict: Dictionary containing all height loss components
        """
        loss_dict = {}
        
        if height_attn is None:
            loss_dict['loss_height_total'] = torch.tensor(0.0, device=height_attn.device if height_attn is not None else 'cpu')
            return loss_dict
        
        device = height_attn.device
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. Variance Loss: Encourage clear separation between high and low heights
        if self.use_variance_loss:
            height_mean = height_attn.mean()
            height_variance = ((height_attn - height_mean) ** 2).mean()
            variance_loss = -height_variance  # Negative because we maximize variance
            weighted_variance_loss = self.variance_weight * variance_loss
            total_loss = total_loss + weighted_variance_loss
            loss_dict['loss_height_variance'] = weighted_variance_loss
        
        # 2. Sparsity Loss: Encourage sparse high-height regions
        if self.use_sparsity_loss:
            sparsity_loss = height_attn.mean()
            high_value_penalty = F.relu(height_attn - 0.9).mean()
            sparsity_loss = sparsity_loss + 0.1 * high_value_penalty
            weighted_sparsity_loss = self.sparsity_weight * sparsity_loss
            total_loss = total_loss + weighted_sparsity_loss
            loss_dict['loss_height_sparsity'] = weighted_sparsity_loss
        
        # 3. Smoothness Loss: Encourage spatial smoothness
        if self.use_smoothness_loss:
            grad_x = height_attn[:, :, :, 1:] - height_attn[:, :, :, :-1]
            grad_y = height_attn[:, :, 1:, :] - height_attn[:, :, :-1, :]
            smoothness_loss = (grad_x.abs().mean() + grad_y.abs().mean()) / 2.0
            weighted_smoothness_loss = self.smoothness_weight * smoothness_loss
            total_loss = total_loss + weighted_smoothness_loss
            loss_dict['loss_height_smoothness'] = weighted_smoothness_loss
        
        # 4. Building Contrast Loss
        if building_mask is not None and building_mask.sum() > 0:
            building_height = (height_attn * building_mask).sum() / (building_mask.sum() + 1e-6)
            non_building_mask = 1.0 - building_mask
            if non_building_mask.sum() > 0:
                non_building_height = (height_attn * non_building_mask).sum() / (non_building_mask.sum() + 1e-6)
            else:
                non_building_height = torch.tensor(0.0, device=device)
            contrast_loss = F.relu(0.5 - (building_height - non_building_height))
            total_loss = total_loss + 0.1 * contrast_loss
            loss_dict['loss_height_contrast'] = contrast_loss
        
        loss_dict['loss_height_total'] = total_loss
        return loss_dict
    
    def forward(
        self,
        sat_feat: torch.Tensor,
        bev_feat: torch.Tensor,
        height_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_core,
                sat_feat,
                bev_feat,
                height_map,
                use_reentrant=False,
            )
        return self._forward_core(sat_feat, bev_feat, height_map)


class MultiLevelHeightGuidedFusion(nn.Module):
    """
    Multi-level height-guided spatial fusion for multi-scale feature pyramids.
    """
    
    def __init__(
        self,
        bev_dim: int,
        level_channels: Tuple[int, ...],
        num_heads: int = 4,
        temperature: float = 0.1,
        downsample_factor: int = 1,
        max_attn_size: int = 64,
        use_checkpoint: bool = False,
        height_dim: int = 64,
        use_height_gate: bool = True,
        height_loss_weight: float = 0.01,
        use_variance_loss: bool = True,
        use_sparsity_loss: bool = True,
        use_smoothness_loss: bool = True,
        variance_weight: float = 0.1,
        sparsity_weight: float = 0.05,
        smoothness_weight: float = 0.01,
    ):
        super().__init__()
        
        self.level_channels = tuple(int(c) for c in level_channels)
        self.downsample_factor = downsample_factor
        self.max_attn_size = max_attn_size
        self.height_loss_weight = height_loss_weight
        
        self.spatial_fusions = nn.ModuleList([
            HeightGuidedSpatialFusion(
                bev_dim=bev_dim,
                sat_dim=int(c),
                num_heads=num_heads,
                temperature=temperature,
                downsample_factor=downsample_factor,
                max_attn_size=max_attn_size,
                use_checkpoint=use_checkpoint,
                height_dim=height_dim,
                use_height_gate=use_height_gate,
                use_variance_loss=use_variance_loss,
                use_sparsity_loss=use_sparsity_loss,
                use_smoothness_loss=use_smoothness_loss,
                variance_weight=variance_weight,
                sparsity_weight=sparsity_weight,
                smoothness_weight=smoothness_weight,
            )
            for c in self.level_channels
        ])
        
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(level_channels[i]), int(level_channels[i-1]), 1, bias=False),
                nn.BatchNorm2d(int(level_channels[i-1])),
                nn.ReLU(inplace=True),
            )
            for i in range(len(level_channels) - 1, 0, -1)
        ])
    
    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
        bev_feat: torch.Tensor,
        height_map: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, dict]:
        guided_feats = []
        total_align_loss = 0.0
        total_height_loss = 0.0
        
        # Collect detailed height losses from all levels
        all_height_loss_dicts = []
        
        for feat, fusion in zip(feats, self.spatial_fusions):
            guided_feat, align_loss, height_loss, height_loss_dict = fusion(feat, bev_feat, height_map)
            guided_feats.append(guided_feat)
            total_align_loss = total_align_loss + align_loss
            total_height_loss = total_height_loss + height_loss
            all_height_loss_dicts.append(height_loss_dict)
        
        for i in range(len(guided_feats) - 1, 0, -1):
            upsampled = F.interpolate(
                guided_feats[i],
                size=guided_feats[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            guided_feats[i-1] = guided_feats[i-1] + self.td_convs[len(self.td_convs) - i](upsampled)
        
        avg_align_loss = total_align_loss / len(feats)
        avg_height_loss = total_height_loss / len(feats)
        
        # Aggregate detailed height losses across all levels
        # Only aggregate keys that exist in all dictionaries
        aggregated_height_loss_dict = {}
        if all_height_loss_dicts:
            all_keys = set(all_height_loss_dicts[0].keys())
            for d in all_height_loss_dicts[1:]:
                all_keys &= set(d.keys())
            
            for key in all_keys:
                aggregated_height_loss_dict[key] = sum(d.get(key, 0.0) for d in all_height_loss_dicts) / len(feats)
        
        return tuple(guided_feats), avg_align_loss, avg_height_loss, aggregated_height_loss_dict
