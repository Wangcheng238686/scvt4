from __future__ import annotations

import os
from typing import Dict, Optional

from .model_factory import create_resnet_backbone
from .cross_view_attention import CVTEncoder
from .depth_aware_bev import DepthAwareCVTEncoder
from .sam2_backbone import SAM2BackboneWrapper, SAM2BackboneForCVT, StandaloneSAM2Backbone


def build_cvt_encoder(
    drone_cfg: Dict, 
    max_scenes: Optional[int] = None,
    shared_sam2_encoder: Optional[object] = None,
    sam2_ckpt_path: Optional[str] = None,
) -> CVTEncoder:
    """
    Build Cross-View Transformer encoder for drone branch.
    
    Args:
        drone_cfg: Configuration dictionary for drone branch
        max_scenes: Maximum number of scenes for scene alignment
        shared_sam2_encoder: Optional shared SAM2 encoder from satellite branch
            If provided and use_sam2_backbone=True, will use this encoder (shared weights)
        sam2_ckpt_path: Path to SAM2 checkpoint for standalone SAM2 backbone
            Used when share_sam2_backbone=False but use_sam2_backbone=True
    
    Returns:
        CVTEncoder or DepthAwareCVTEncoder
    """
    use_sam2_backbone = drone_cfg.get("use_sam2_backbone", False)
    
    if use_sam2_backbone:
        if shared_sam2_encoder is not None:
            backbone = SAM2BackboneForCVT(
                sam2_encoder=shared_sam2_encoder,
                input_size=drone_cfg.get("input_size", 512),
                feature_dim=256,
                num_levels=4,
            )
        else:
            ckpt_path = sam2_ckpt_path or drone_cfg.get("sam2_ckpt_path")
            if ckpt_path is None:
                ckpt_path = os.environ.get("SAM2_CKPT")
            if ckpt_path is None:
                raise ValueError(
                    "SAM2 checkpoint path required for standalone SAM2 backbone. "
                    "Set sam2_ckpt_path in drone_branch config or SAM2_CKPT environment variable."
                )
            
            backbone = StandaloneSAM2Backbone(
                checkpoint_path=ckpt_path,
                input_size=drone_cfg.get("input_size", 512),
                feature_dim=256,
                num_levels=4,
                use_checkpoint=drone_cfg.get("use_checkpoint", False),
                freeze=drone_cfg.get("freeze_sam2", False),
            )
    else:
        backbone = create_resnet_backbone(pretrained=drone_cfg.get("pretrained", True))
    
    cross_view = drone_cfg["cross_view"]
    bev_embedding = drone_cfg["bev_embedding"]
    dim = int(drone_cfg.get("dim", 128))
    middle = drone_cfg.get("middle", [2, 2, 2, 2])
    scale = float(drone_cfg.get("scale", 1.0))
    use_checkpoint = bool(drone_cfg.get("use_checkpoint", False))
    
    scene_embed_cfg = drone_cfg.get("scene_alignment", {})
    max_scenes = max_scenes or int(scene_embed_cfg.get("max_scenes", 1000))
    scene_embed_dim = int(scene_embed_cfg.get("embed_dim", 32))
    alignment_init_scale = float(scene_embed_cfg.get("init_scale", 0.01))
    
    depth_encoder_cfg = drone_cfg.get("depth_encoder", None)
    use_depth_aware = depth_encoder_cfg is not None and depth_encoder_cfg.get("enabled", False)
    
    use_multiview_consistency = False
    consistency_weight = 0.1
    if depth_encoder_cfg is not None:
        use_multiview_consistency = bool(depth_encoder_cfg.get("use_multiview_consistency", False))
        consistency_weight = float(depth_encoder_cfg.get("consistency_weight", 0.1))
    
    if use_depth_aware:
        return DepthAwareCVTEncoder(
            backbone=backbone,
            cross_view=cross_view,
            bev_embedding=bev_embedding,
            dim=dim,
            middle=middle,
            scale=scale,
            use_checkpoint=use_checkpoint,
            max_scenes=max_scenes,
            scene_embed_dim=scene_embed_dim,
            alignment_init_scale=alignment_init_scale,
            depth_encoder=depth_encoder_cfg,
            use_multiview_consistency=use_multiview_consistency,
            consistency_weight=consistency_weight,
        )
    else:
        return CVTEncoder(
            backbone=backbone,
            cross_view=cross_view,
            bev_embedding=bev_embedding,
            dim=dim,
            middle=middle,
            scale=scale,
            use_checkpoint=use_checkpoint,
            max_scenes=max_scenes,
            scene_embed_dim=scene_embed_dim,
            alignment_init_scale=alignment_init_scale,
        )
