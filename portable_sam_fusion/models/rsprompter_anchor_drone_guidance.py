from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.structures import SampleList
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS

from portable_sam_fusion.rsprompter.models import RSPrompterAnchor
from portable_sam_fusion.uav import build_cvt_encoder
from .losses import CrossViewContrastiveLoss, FeatureConsistencyLoss, GeometricConsistencyLoss, SpatialSmoothnessLoss
from .height_guided_fusion import MultiLevelHeightGuidedFusion


@dataclass
class GuidanceConfig:
    bev_dim: int = 128
    level_channels: Tuple[int, ...] = (256, 256, 256, 256, 256)
    gate_image_embeddings: bool = False


class ContrastiveSpatialFusion(nn.Module):
    def __init__(
        self,
        bev_dim: int,
        sat_dim: int,
        num_heads: int = 4,
        temperature: float = 0.1,
        use_deformable: bool = False,
        downsample_factor: int = 1,
        max_attn_size: int = 64,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.bev_dim = bev_dim
        self.sat_dim = sat_dim
        self.temperature = temperature
        self.downsample_factor = downsample_factor
        self.max_attn_size = max_attn_size
        self.use_checkpoint = use_checkpoint
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = sat_feat.shape
        orig_size = (H, W)
        
        attn_h, attn_w = self._compute_downsample_size(H, W)
        
        bev_proj = self.bev_proj(bev_feat)
        bev_up = F.interpolate(bev_proj, size=orig_size, mode='bilinear', align_corners=False)
        
        sat_proj = self.sat_proj(sat_feat)
        
        if (attn_h, attn_w) != orig_size:
            sat_for_attn = F.interpolate(sat_proj, size=(attn_h, attn_w), mode='bilinear', align_corners=False)
            bev_for_attn = F.interpolate(bev_up, size=(attn_h, attn_w), mode='bilinear', align_corners=False)
        else:
            sat_for_attn = sat_proj
            bev_for_attn = bev_up
        
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
        
        return output, align_loss
    
    def forward(
        self,
        sat_feat: torch.Tensor,
        bev_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_core,
                sat_feat,
                bev_feat,
                use_reentrant=False,
            )
        return self._forward_core(sat_feat, bev_feat)


class MultiLevelSpatialFusion(nn.Module):
    def __init__(
        self,
        bev_dim: int,
        level_channels: Tuple[int, ...],
        num_heads: int = 4,
        temperature: float = 0.1,
        downsample_factor: int = 1,
        max_attn_size: int = 64,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.level_channels = tuple(int(c) for c in level_channels)
        self.downsample_factor = downsample_factor
        self.max_attn_size = max_attn_size
        
        self.spatial_fusions = nn.ModuleList([
            ContrastiveSpatialFusion(
                bev_dim=bev_dim,
                sat_dim=int(c),
                num_heads=num_heads,
                temperature=temperature,
                downsample_factor=downsample_factor,
                max_attn_size=max_attn_size,
                use_checkpoint=use_checkpoint,
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
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        guided_feats = []
        total_align_loss = 0.0
        
        for feat, fusion in zip(feats, self.spatial_fusions):
            guided_feat, align_loss = fusion(feat, bev_feat)
            guided_feats.append(guided_feat)
            total_align_loss = total_align_loss + align_loss
        
        for i in range(len(guided_feats) - 1, 0, -1):
            upsampled = F.interpolate(
                guided_feats[i],
                size=guided_feats[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            guided_feats[i-1] = guided_feats[i-1] + self.td_convs[len(self.td_convs) - i](upsampled)
        
        avg_align_loss = total_align_loss / len(feats)
        
        return tuple(guided_feats), avg_align_loss


class DroneSemanticGuidance(nn.Module):
    def __init__(self, bev_dim: int, level_channels: Tuple[int, ...], embed_channels: Optional[int] = None, gate_scale: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.level_channels = tuple(int(c) for c in level_channels)
        self.gate_scale = gate_scale

        # Residual gate: learns a small offset to modulate satellite features
        # Output centered around 0 via Tanh, scaled by gate_scale so that
        # the default behaviour is near-identity (x * (1 + 0) = x).
        self.level_gates = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(bev_dim + int(c), int(c)),
                nn.ReLU(inplace=True),
                nn.Linear(int(c), int(c)),
                nn.Tanh(),
            ) for c in self.level_channels]
        )
        self.embed_gate = None
        if embed_channels is not None:
            self.embed_gate = nn.Sequential(
                nn.Linear(bev_dim + int(embed_channels), int(embed_channels)),
                nn.ReLU(inplace=True),
                nn.Linear(int(embed_channels), int(embed_channels)),
                nn.Tanh(),
            )

    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
        drone_bev: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Optional[torch.Tensor]]:
        context = self.pool(drone_bev).flatten(1)

        guided: List[torch.Tensor] = []
        for x, gate_mlp in zip(feats, self.level_gates):
            x_context = self.pool(x).flatten(1)
            combined_context = torch.cat([context, x_context], dim=1)

            # Residual gate: x * (1 + scale * tanh(mlp(context)))
            # When gate_offset ≈ 0, output ≈ x (identity pass-through)
            gate_offset = gate_mlp(combined_context).unsqueeze(-1).unsqueeze(-1)
            guided.append(x * (1.0 + self.gate_scale * gate_offset))

        guided_emb = None
        if image_embeddings is not None and self.embed_gate is not None:
            emb_context = self.pool(image_embeddings).flatten(1)
            combined_context = torch.cat([context, emb_context], dim=1)
            e_gate = self.embed_gate(combined_context).unsqueeze(-1).unsqueeze(-1)
            guided_emb = image_embeddings * (1.0 + self.gate_scale * e_gate)

        return tuple(guided), guided_emb


@MODELS.register_module()
@MMENGINE_MODELS.register_module()
class RSPrompterAnchorDroneGuidance(RSPrompterAnchor):
    def __init__(
        self,
        shared_image_embedding: Dict,
        drone_branch: Optional[Dict] = None,
        guidance: Optional[Dict] = None,
        decoder_freeze: bool = True,
        enable_drone_branch: bool = True,
        cross_view_loss: Optional[Dict] = None,
        max_scenes: int = 1000,
        use_spatial_fusion: bool = False,
        spatial_fusion_cfg: Optional[Dict] = None,
        use_height_guided_fusion: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            shared_image_embedding=shared_image_embedding,
            decoder_freeze=decoder_freeze,
            *args,
            **kwargs,
        )

        self.enable_drone_branch = enable_drone_branch
        self.use_spatial_fusion = use_spatial_fusion
        self.use_height_guided_fusion = use_height_guided_fusion

        if self.enable_drone_branch and drone_branch is not None:
            self.drone_encoder = build_cvt_encoder(drone_branch, max_scenes=max_scenes)
            self.freeze_drone = bool(drone_branch.get("freeze", False))
            if self.freeze_drone:
                self.drone_encoder.requires_grad_(False)
                self.drone_encoder.eval()

            cfg = guidance or {}
            bev_dim = int(cfg.get("bev_dim", int(drone_branch.get("dim", 128))))
            level_channels = tuple(cfg.get("level_channels", [256, 256, 256, 256, 256]))
            gate_image_embeddings = bool(cfg.get("gate_image_embeddings", False))
            gate_scale = float(cfg.get("gate_scale", 0.1))
            
            if use_height_guided_fusion:
                spatial_cfg = spatial_fusion_cfg or {}
                num_heads = int(spatial_cfg.get("num_heads", 4))
                temperature = float(spatial_cfg.get("temperature", 0.1))
                align_loss_weight = float(spatial_cfg.get("align_loss_weight", 0.1))
                downsample_factor = int(spatial_cfg.get("downsample_factor", 1))
                max_attn_size = int(spatial_cfg.get("max_attn_size", 64))
                use_checkpoint = bool(spatial_cfg.get("use_checkpoint", False))
                height_dim = int(spatial_cfg.get("height_dim", 64))
                use_height_gate = bool(spatial_cfg.get("use_height_gate", True))
                height_loss_weight = float(spatial_cfg.get("height_loss_weight", 0.01))
                
                self.guidance = MultiLevelHeightGuidedFusion(
                    bev_dim=bev_dim,
                    level_channels=level_channels,
                    num_heads=num_heads,
                    temperature=temperature,
                    downsample_factor=downsample_factor,
                    max_attn_size=max_attn_size,
                    use_checkpoint=use_checkpoint,
                    height_dim=height_dim,
                    use_height_gate=use_height_gate,
                    height_loss_weight=height_loss_weight,
                )
                self.spatial_align_loss_weight = align_loss_weight
                self.height_loss_weight = height_loss_weight
                self.gate_image_embeddings = False
            elif use_spatial_fusion:
                spatial_cfg = spatial_fusion_cfg or {}
                num_heads = int(spatial_cfg.get("num_heads", 4))
                temperature = float(spatial_cfg.get("temperature", 0.1))
                align_loss_weight = float(spatial_cfg.get("align_loss_weight", 0.1))
                downsample_factor = int(spatial_cfg.get("downsample_factor", 1))
                max_attn_size = int(spatial_cfg.get("max_attn_size", 64))
                use_checkpoint = bool(spatial_cfg.get("use_checkpoint", False))
                
                self.guidance = MultiLevelSpatialFusion(
                    bev_dim=bev_dim,
                    level_channels=level_channels,
                    num_heads=num_heads,
                    temperature=temperature,
                    downsample_factor=downsample_factor,
                    max_attn_size=max_attn_size,
                    use_checkpoint=use_checkpoint,
                )
                self.spatial_align_loss_weight = align_loss_weight
                self.gate_image_embeddings = False
            else:
                embed_channels = 256 if gate_image_embeddings else None
                self.guidance = DroneSemanticGuidance(
                    bev_dim=bev_dim,
                    level_channels=level_channels,
                    embed_channels=embed_channels,
                    gate_scale=gate_scale,
                )
                self.gate_image_embeddings = gate_image_embeddings
                self.spatial_align_loss_weight = 0.0
            
            if cross_view_loss is not None:
                self.use_cross_view_loss = True
                sat_dim = int(level_channels[0]) if level_channels else 256
                self.contrastive_loss = CrossViewContrastiveLoss(
                    bev_dim=bev_dim,
                    sat_dim=sat_dim,
                    temperature=float(cross_view_loss.get("temperature", 0.07))
                )
                self.contrastive_weight = float(cross_view_loss.get("contrastive_weight", 0.1))
                hidden_dim = int(cross_view_loss.get("hidden_dim", 256))
                self.consistency_loss = FeatureConsistencyLoss(
                    bev_dim=bev_dim,
                    sat_dim=sat_dim,
                    hidden_dim=hidden_dim,
                )
                self.consistency_weight = float(cross_view_loss.get("consistency_weight", 0.1))
                
                geometric_weight = float(cross_view_loss.get("geometric_weight", 0.01))
                if geometric_weight > 0:
                    self.geometric_loss = GeometricConsistencyLoss(
                        translation_weight=float(cross_view_loss.get("geometric_trans_weight", 1.0)),
                        rotation_weight=float(cross_view_loss.get("geometric_rot_weight", 1.0)),
                    )
                    self.geometric_weight = geometric_weight
                else:
                    self.geometric_loss = None
                    self.geometric_weight = 0.0
                
                smoothness_weight = float(cross_view_loss.get("smoothness_weight", 0.01))
                if smoothness_weight > 0:
                    self.smoothness_loss = SpatialSmoothnessLoss()
                    self.smoothness_weight = smoothness_weight
                else:
                    self.smoothness_loss = None
                    self.smoothness_weight = 0.0
            else:
                self.use_cross_view_loss = False
                self.contrastive_loss = None
                self.consistency_loss = None
                self.geometric_loss = None
                self.geometric_weight = 0.0
                self.smoothness_loss = None
                self.smoothness_weight = 0.0
        else:
            self.drone_encoder = None
            self.guidance = None
            self.freeze_drone = False
            self.gate_image_embeddings = False
            self.use_cross_view_loss = False
            self.contrastive_loss = None
            self.consistency_loss = None
            self.spatial_align_loss_weight = 0.0
            self.geometric_loss = None
            self.geometric_weight = 0.0
            self.smoothness_loss = None
            self.smoothness_weight = 0.0

    def extract_feat(
        self, 
        batch_inputs: Union[torch.Tensor, Dict],
        scene_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        if not isinstance(batch_inputs, dict):
            x, image_embeddings, image_positional_embeddings = super().extract_feat(batch_inputs)
            self._drone_bev = None
            self._spatial_align_loss = None
            return x, image_embeddings, image_positional_embeddings

        sat = batch_inputs["sat"]

        if not self.enable_drone_branch or self.drone_encoder is None:
            x, image_embeddings, image_positional_embeddings = super().extract_feat(sat)
            self._drone_bev = None
            self._spatial_align_loss = None
            return x, image_embeddings, image_positional_embeddings

        drone_images = batch_inputs["drone_images"]
        intrinsics = batch_inputs["intrinsics"]
        extrinsics = batch_inputs["extrinsics"]

        drone_batch = {
            "image": drone_images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

        if self.freeze_drone:
            with torch.no_grad():
                encoder_output = self.drone_encoder(drone_batch, scene_indices=scene_indices)
        else:
            encoder_output = self.drone_encoder(drone_batch, scene_indices=scene_indices)
        
        if isinstance(encoder_output, tuple):
            drone_bev = encoder_output[0]
            height_map = encoder_output[1] if len(encoder_output) > 1 else None
            consistency_loss = encoder_output[2] if len(encoder_output) > 2 else None
        else:
            drone_bev = encoder_output
            height_map = None
            consistency_loss = None
        
        self._drone_bev = drone_bev
        self._height_map = height_map
        self._consistency_loss = consistency_loss

        x, image_embeddings, image_positional_embeddings = super().extract_feat(sat)

        if self.guidance is not None:
            height_map = getattr(self, "_height_map", None)
            
            if self.use_height_guided_fusion:
                guided_feats, align_loss, height_loss = self.guidance(
                    feats=x, bev_feat=drone_bev, height_map=height_map
                )
                self._spatial_align_loss = align_loss
                self._height_loss = height_loss
                return guided_feats, image_embeddings, image_positional_embeddings
            elif self.use_spatial_fusion:
                guided_feats, align_loss = self.guidance(feats=x, bev_feat=drone_bev)
                self._spatial_align_loss = align_loss
                self._height_loss = None
                return guided_feats, image_embeddings, image_positional_embeddings
            else:
                guided_feats, guided_emb = self.guidance(
                    feats=x,
                    drone_bev=drone_bev,
                    image_embeddings=image_embeddings if self.gate_image_embeddings else None,
                )
                self._spatial_align_loss = None
                self._height_loss = None

                if guided_emb is not None:
                    image_embeddings = guided_emb

                return guided_feats, image_embeddings, image_positional_embeddings

        self._spatial_align_loss = None
        self._height_loss = None
        return x, image_embeddings, image_positional_embeddings

    def loss(
        self, 
        batch_inputs: Union[torch.Tensor, Dict], 
        batch_data_samples: SampleList,
        scene_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs, scene_indices=scene_indices)
        
        if not self.use_cross_view_loss:
            return self._compute_standard_loss(x, image_embeddings, image_positional_embeddings, batch_data_samples)
        
        return self._compute_loss_with_cross_view(x, image_embeddings, image_positional_embeddings, batch_data_samples)
    
    def _compute_standard_loss(
        self,
        x,
        image_embeddings,
        image_positional_embeddings,
        batch_data_samples: SampleList,
    ) -> dict:
        losses = dict()

        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(x, rpn_data_samples, proposal_cfg=proposal_cfg)
        keys = rpn_losses.keys()
        for key in list(keys):
            if "loss" in key and "rpn" not in key:
                rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(
            x,
            rpn_results_list,
            batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        losses.update(roi_losses)
        
        spatial_align_loss = getattr(self, "_spatial_align_loss", None)
        if spatial_align_loss is not None and self.spatial_align_loss_weight > 0:
            losses["loss_spatial_align"] = spatial_align_loss * self.spatial_align_loss_weight
        
        height_loss = getattr(self, "_height_loss", None)
        if height_loss is not None and hasattr(self, "height_loss_weight") and self.height_loss_weight > 0:
            losses["loss_height"] = height_loss
        
        consistency_loss = getattr(self, "_consistency_loss", None)
        if consistency_loss is not None:
            losses["loss_depth_consistency"] = consistency_loss
        
        return losses
    
    def _compute_loss_with_cross_view(
        self,
        x,
        image_embeddings,
        image_positional_embeddings,
        batch_data_samples: SampleList,
    ) -> dict:
        losses = dict()

        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(x, rpn_data_samples, proposal_cfg=proposal_cfg)
        keys = rpn_losses.keys()
        for key in list(keys):
            if "loss" in key and "rpn" not in key:
                rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(
            x,
            rpn_results_list,
            batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        losses.update(roi_losses)
        
        drone_bev = getattr(self, "_drone_bev", None)
        if drone_bev is not None:
            sat_feat = x[0] if isinstance(x, tuple) else x
            if self.contrastive_loss is not None:
                contrastive_loss = self.contrastive_loss(drone_bev, sat_feat)
                losses["loss_contrastive"] = contrastive_loss * self.contrastive_weight
            if self.consistency_loss is not None:
                consistency_loss = self.consistency_loss(sat_feat, drone_bev)
                losses["loss_consistency"] = consistency_loss * self.consistency_weight
            
            if self.geometric_loss is not None and hasattr(self.drone_encoder, 'scene_alignment'):
                alignment_matrix = self.drone_encoder.scene_alignment.global_alignment.unsqueeze(0)
                alignment_matrix = alignment_matrix.expand(drone_bev.shape[0], -1, -1)
                geo_loss = self.geometric_loss(alignment_matrix)
                losses["loss_geometric"] = geo_loss * self.geometric_weight
            
            if self.smoothness_loss is not None:
                smoothness_loss = self.smoothness_loss(drone_bev)
                losses["loss_smoothness"] = smoothness_loss * self.smoothness_weight
        
        spatial_align_loss = getattr(self, "_spatial_align_loss", None)
        if spatial_align_loss is not None and self.spatial_align_loss_weight > 0:
            losses["loss_spatial_align"] = spatial_align_loss * self.spatial_align_loss_weight
        
        return losses

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_drone", False) and self.drone_encoder is not None:
            self.drone_encoder.eval()
        return self
