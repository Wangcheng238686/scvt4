from __future__ import annotations

from typing import Dict, Optional

from .model_factory import create_resnet_backbone
from .cross_view_attention import CVTEncoder


def build_cvt_encoder(drone_cfg: Dict, max_scenes: Optional[int] = None) -> CVTEncoder:
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

