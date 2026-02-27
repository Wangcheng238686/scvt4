"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

import os
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.models import MaskRCNN, StandardRoIHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList, OptConfigType


SAM2_AVAILABLE = False
try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    SAM2_AVAILABLE = True
except ImportError:
    pass


def _load_sam2_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict:
    """Load SAM2 checkpoint from file."""
    from mmengine.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)
    # SAM2 checkpoint format is {"model": {...}}
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


@MODELS.register_module()
class RSSAM2PositionalEmbedding(BaseModule):
    """SAM2 uses a different positional embedding scheme."""

    def __init__(
        self,
        image_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 256,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size, self.grid_size, embed_dim)
        )

        class PositionalEmbeddingWrapper:
            def __init__(self, pos_embed):
                self.positional_embedding = pos_embed

        self.shared_image_embedding = PositionalEmbeddingWrapper(self.pos_embed)

    def get_dense_pe(self) -> Tensor:
        """Returns positional encoding in SAM2 format: [1, C, H, W]"""
        pos = self.pos_embed
        if pos.shape[-1] == 256:
            pos = pos.permute(0, 3, 1, 2)
        return pos

    def forward(self, x: Tensor) -> Tensor:
        return self.get_dense_pe()


@MODELS.register_module()
class RSSAM2VisionEncoder(BaseModule):
    """SAM2 Vision Encoder adapter using Hiera backbone.
    直接从checkpoint加载整个SAM2模型，然后提取encoder
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
        use_gradient_checkpointing: bool = False,
        freeze_backbone: bool = True,
        freeze_backbone_stages: int = -1,
    ):
        super().__init__(init_cfg=init_cfg)
        self.checkpoint_path = checkpoint_path
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_stages = freeze_backbone_stages
        self.encoder = None
        self.encoder_out_channels = 256

        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Please install it via:\n"
                "pip install git+https://github.com/facebookresearch/sam2.git"
            )

        if init_cfg is not None and checkpoint_path is None:
            checkpoint_path = init_cfg.get("checkpoint")

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_sam2_encoder(checkpoint_path)
        else:
            raise ValueError(
                f"SAM2 checkpoint not found at: {checkpoint_path}. "
                "Please download SAM2 checkpoint first."
            )

        if freeze_backbone:
            self._freeze_params()

    def _load_sam2_encoder(self, checkpoint_path: str):
        """Load SAM2 encoder from checkpoint by directly building the model."""
        import sys
        sys.path.insert(0, '/home/wangcheng/project/sam2-main')

        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 is not properly installed.")

        try:
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
            if is_main_process():
                print(f"Loaded SAM2 VisionEncoder. Missing keys: {len(msg.missing_keys)}")

            self.encoder = image_encoder
            self.encoder_out_channels = 256

        except Exception as e:
            if is_main_process():
                print(f"Failed to load SAM2 encoder: {e}")
            raise

    def _freeze_params(self):
        """Freeze encoder parameters with optional partial unfreezing."""
        if self.encoder is None:
            return
        
        if self.freeze_backbone_stages < 0:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            for name, param in self.encoder.named_parameters():
                stage_idx = -1
                for i in range(4):
                    if f"stages.{i}" in name or f"backbone_fpn.{i}" in name:
                        stage_idx = i
                        break
                
                if stage_idx >= 0 and stage_idx >= self.freeze_backbone_stages:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            self.encoder.train()

    def init_weights(self):
        if is_main_process():
            print("SAM2 Vision Encoder initialized")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """Forward pass returning intermediate features."""
        if self.encoder is None:
            raise RuntimeError("SAM2 encoder not loaded")

        if self.use_gradient_checkpointing and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self.encoder, x, use_reentrant=False
            )
        else:
            features = self.encoder(x)

        if isinstance(features, dict):
            vision_features = features["vision_features"]
            backbone_fpn = features.get("backbone_fpn")
            if backbone_fpn is not None:
                return vision_features, tuple(backbone_fpn)
            else:
                return vision_features, (vision_features,)
        elif isinstance(features, (list, tuple)):
            return features[0], features
        else:
            return features, (features,)


class RSSAM2MaskDecoderWrapper(BaseModule):
    """Wrapper class to make RSSAM2MaskDecoder compatible with existing RSPrompterAnchorMaskHead."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_high_res_features: bool = False,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.decoder = None
        self.use_high_res_features = use_high_res_features

        if init_cfg is not None and checkpoint_path is None:
            checkpoint_path = init_cfg.get("checkpoint")

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_sam2_decoder(checkpoint_path)

    def _load_sam2_decoder(self, checkpoint_path: str):
        """Load SAM2 decoder from checkpoint."""
        try:
            import sys
            sys.path.insert(0, '/home/wangcheng/project/sam2-main')

            from sam2.modeling.sam.transformer import TwoWayTransformer

            transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                num_heads=8,
                mlp_dim=2048,
                activation=nn.GELU,
                attention_downsample_rate=2,
            )

            mask_decoder = MaskDecoder(
                transformer_dim=256,
                transformer=transformer,
                num_multimask_outputs=3,
                activation=nn.GELU,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                use_high_res_features=self.use_high_res_features,
                iou_prediction_use_sigmoid=True,
                pred_obj_scores=True,
                pred_obj_scores_mlp=True,
            )

            checkpoint = _load_sam2_checkpoint(checkpoint_path)
            decoder_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("sam_mask_decoder."):
                    new_k = k[17:]
                    decoder_state_dict[new_k] = v

            print(f'Decoder state dict keys: {len(decoder_state_dict)}')
            msg = mask_decoder.load_state_dict(decoder_state_dict, strict=False)
            if is_main_process():
                print(f"Loaded SAM2 MaskDecoder. Missing keys: {len(msg.missing_keys)}")
                if msg.missing_keys:
                    print(f"  Missing keys: {msg.missing_keys[:10]}...")
                if self.use_high_res_features:
                    print(f"  use_high_res_features: True")
                    print(f"  Has conv_s0: {hasattr(mask_decoder, 'conv_s0')}")
                    print(f"  Has conv_s1: {hasattr(mask_decoder, 'conv_s1')}")

            self.decoder = mask_decoder
        except Exception as e:
            if is_main_process():
                print(f"Failed to load SAM2 decoder: {e}")
            self.decoder = None

    def forward(
        self,
        image_embeddings: Tensor,
        image_positional_embeddings: Optional[Tensor] = None,
        sparse_prompt_embeddings: Optional[Tensor] = None,
        dense_prompt_embeddings: Optional[Tensor] = None,
        multimask_output: bool = False,
        repeat_image: bool = False,
        attention_similarity: Optional[Tensor] = None,
        target_embedding: Optional[Tensor] = None,
        output_attentions: bool = False,
        high_res_features: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward pass compatible with original SAM mask decoder interface."""
        if self.decoder is None:
            batch_size = image_embeddings.shape[0]
            h = w = image_embeddings.shape[2]
            low_res_masks = torch.zeros(batch_size, 1, h, w, device=image_embeddings.device)
            iou_predictions = torch.ones(batch_size, 1, device=image_embeddings.device)
            return low_res_masks, iou_predictions, None

        batch_size = image_embeddings.shape[0]

        if sparse_prompt_embeddings is None:
            sparse_prompt_embeddings = torch.zeros(
                batch_size, 1, 1, 256, device=image_embeddings.device
            )

        if dense_prompt_embeddings is None:
            h = w = image_embeddings.shape[2]
            dense_prompt_embeddings = torch.zeros(
                batch_size, 1, h, w, device=image_embeddings.device
            )

        if image_positional_embeddings is not None:
            pe = image_positional_embeddings
            if pe.dim() == 3:
                pe = pe.unsqueeze(0)
            if pe.dim() == 4 and pe.shape[0] != batch_size:
                pe = pe.expand(batch_size, -1, -1, -1)
            
            target_h, target_w = image_embeddings.shape[2], image_embeddings.shape[3]
            if pe.shape[2:] != (target_h, target_w):
                pe = torch.nn.functional.interpolate(
                    pe, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
            
            image_positional_embeddings_pe = pe[0:1, :, :, :].contiguous()
            image_positional_embeddings = pe

        try:
            processed_high_res = None
            if high_res_features is not None and self.use_high_res_features:
                if hasattr(self.decoder, 'conv_s0') and hasattr(self.decoder, 'conv_s1'):
                    feat_s0, feat_s1 = high_res_features
                    processed_high_res = [
                        self.decoder.conv_s0(feat_s0),
                        self.decoder.conv_s1(feat_s1),
                    ]
            
            low_res_masks, iou_predictions, _, _ = self.decoder(
                image_embeddings,
                image_positional_embeddings_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                multimask_output,
                repeat_image,
                processed_high_res,
            )
            attn_weights = None
        except Exception as e:
            import traceback
            if is_main_process() and (not hasattr(self, '_warn_count') or self._warn_count < 10):
                if not hasattr(self, '_warn_count'):
                    self._warn_count = 0
                self._warn_count += 1
                print(f"Warning: SAM2 decoder forward failed (count={self._warn_count}): {e}")
                print(f"  image_embeddings.shape: {image_embeddings.shape}")
                print(f"  image_positional_embeddings_pe.shape: {image_positional_embeddings_pe.shape}")
                print(f"  sparse_prompt_embeddings.shape: {sparse_prompt_embeddings.shape}")
                print(f"  dense_prompt_embeddings.shape: {dense_prompt_embeddings.shape}")
                if high_res_features is not None:
                    print(f"  high_res_features: {[f.shape for f in high_res_features]}")
                if processed_high_res is not None:
                    print(f"  processed_high_res: {[f.shape for f in processed_high_res]}")
            h = w = image_embeddings.shape[2]
            low_res_masks = torch.zeros(batch_size, 1, h, w, device=image_embeddings.device)
            iou_predictions = torch.ones(batch_size, 1, device=image_embeddings.device)
            attn_weights = None

        return low_res_masks, iou_predictions, attn_weights


@MODELS.register_module()
class RSFeatureAggregatorSAM2(BaseModule):
    """Feature aggregator adapted for SAM2's output dimensions."""

    in_channels_dict = {
        "tiny": [192, 96, 48, 24],
        "small": [384, 192, 96, 48],
        "base": [768, 384, 192, 96],
        "large": [1024, 512, 256, 128],
        "base_plus": [896, 448, 224, 112],
        "sam2_hiera_base": [256, 256, 256, 256],
    }

    def __init__(
        self,
        in_channels: str = "sam2_hiera_base",
        hidden_channels: int = 64,
        out_channels: int = 256,
        select_layers: Optional[List[int]] = None,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        if "sam2" in in_channels.lower():
            model_size = in_channels
        else:
            model_size = "base_plus" if "base" in in_channels else "large" if "large" in in_channels else "small" if "small" in in_channels else "tiny"
        self.in_channels = self.in_channels_dict.get(model_size, self.in_channels_dict["sam2_hiera_base"])
        self.select_layers = select_layers or [0, 1, 2]

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            if i_layer < len(self.in_channels):
                in_ch = self.in_channels[i_layer]
            else:
                in_ch = self.in_channels[-1]

            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs) -> Tuple[Tensor, ...]:
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], tuple):
            inputs = inputs[1]
        
        if not inputs:
            raise ValueError("Empty inputs to neck")

        inputs = tuple(x.float() for x in inputs)

        if is_main_process() and not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[DEBUG] RSFeatureAggregatorSAM2 inputs: {len(inputs)} tensors")
            for i, t in enumerate(inputs):
                print(f"  [{i}] shape={t.shape}, dtype={t.dtype}")

        if len(inputs) == 1:
            x = inputs[0]
            x = self.fusion_conv(x)
            feat_64 = x
            feat_32 = torch.nn.functional.max_pool2d(feat_64, kernel_size=2, stride=2)
            feat_16 = torch.nn.functional.max_pool2d(feat_32, kernel_size=2, stride=2)
            feat_8 = torch.nn.functional.max_pool2d(feat_16, kernel_size=2, stride=2)
            feat_4 = torch.nn.functional.max_pool2d(feat_8, kernel_size=2, stride=2)
            # return features in order of increasing stride (4, 8, 16, 32, 64)
            return (feat_64, feat_32, feat_16, feat_8, feat_4)

        while len(inputs) < len(self.in_channels):
            inputs = inputs + (inputs[-1],)
        
        inputs = [einops.rearrange(x, "b h w c -> b c h w") if x.dim() == 3 else x for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            if i_layer < len(inputs):
                x = inputs[i_layer]
                if x.dim() == 3:
                    x = x.permute(0, 2, 1)
                features.append(self.downconvs[idx](x))

        if not features:
            raise ValueError("No features to process")

        # Find largest feature map size
        target_h, target_w = 0, 0
        for feat in features:
            if feat.shape[2] > target_h:
                target_h = feat.shape[2]
                target_w = feat.shape[3]
        target_size = (target_h, target_w)

        aligned_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = torch.nn.functional.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            else:
                feat = feat
            aligned_features.append(feat)
        
        x = aligned_features[0]
        for idx, hidden_conv in enumerate(self.hidden_convs[1:], 1):
            if idx < len(aligned_features):
                x = x + aligned_features[idx]
            residual = hidden_conv(x)
            x = x + residual
        x = self.fusion_conv(x)
        
        feat_64 = x
        feat_32 = torch.nn.functional.max_pool2d(feat_64, kernel_size=2, stride=2)
        feat_16 = torch.nn.functional.max_pool2d(feat_32, kernel_size=2, stride=2)
        feat_8 = torch.nn.functional.max_pool2d(feat_16, kernel_size=2, stride=2)
        feat_4 = torch.nn.functional.max_pool2d(feat_8, kernel_size=2, stride=2)
        
        # return features in order of increasing stride (4, 8, 16, 32, 64)
        return (feat_64, feat_32, feat_16, feat_8, feat_4)


@MODELS.register_module()
class RSPrompterAnchorMaskHeadSAM2(FCNMaskHead, BaseModule):
    """SAM2-compatible Mask Head for RSPrompter."""

    def __init__(
        self,
        sam2_mask_decoder,
        in_channels,
        roi_feat_size=14,
        per_pointset_point=5,
        with_sincos=True,
        multimask_output=False,
        attention_similarity=None,
        target_embedding=None,
        output_attentions=None,
        class_agnostic=False,
        loss_mask: ConfigType = dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        init_cfg=None,
        *args,
        **kwargs,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = RSSAM2MaskDecoderWrapper(**sam2_mask_decoder)

        self.no_mask_embed = nn.Parameter(torch.zeros(1, 256, 1, 1))

        num_sincos = 2 if with_sincos else 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels * roi_feat_size**2 // 4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point),
        )

        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(self, x, image_embeddings, image_positional_embeddings, roi_img_ids=None, high_res_features=None):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embedings = self.point_emb(x)
        point_embedings = einops.rearrange(point_embedings, "b (n c) -> b n c", n=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        sparse_embeddings = point_embedings
        num_roi_per_image = torch.bincount(roi_img_ids.long())
        num_roi_per_image = torch.cat(
            [
                num_roi_per_image,
                torch.zeros(
                    img_bs - len(num_roi_per_image),
                    device=num_roi_per_image.device,
                    dtype=num_roi_per_image.dtype,
                ),
            ]
        )

        dense_embeddings = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
            roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
        )
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        
        if image_positional_embeddings.dim() == 3:
            image_positional_embeddings = image_positional_embeddings.unsqueeze(0)
        if image_positional_embeddings.dim() == 4:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        else:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        
        if high_res_features is not None:
            high_res_features_expanded = [
                feat.repeat_interleave(num_roi_per_image, dim=0)
                for feat in high_res_features
            ]
        else:
            high_res_features_expanded = None
        
        low_res_masks, iou_predictions, _ = self.mask_decoder(
            image_embeddings,
            image_positional_embeddings,
            sparse_embeddings,
            dense_embeddings,
            self.multimask_output,
            False,
            high_res_features=high_res_features_expanded,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        return low_res_masks, iou_predictions

    def get_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> dict:
        mask_targets = self.get_targets(sampling_results=sampling_results, batch_gt_instances=batch_gt_instances, rcnn_train_cfg=rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode="bilinear", align_corners=False)

        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
        loss["loss_mask"] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def _predict_by_feat_single(
        self,
        mask_preds: Tensor,
        bboxes: Tensor,
        labels: Tensor,
        img_meta: dict,
        rcnn_test_cfg: ConfigDict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> Tensor:
        import numpy as np

        _ = labels
        scale_factor = bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))
        img_h, img_w = img_meta["ori_shape"][:2]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(
            mask_preds,
            size=img_meta["batch_input_shape"],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        scale_factor_w, scale_factor_h = img_meta["scale_factor"]
        ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
        im_mask = im_mask[:, : int(ori_rescaled_size[0]), : int(ori_rescaled_size[1])]

        h, w = img_meta["ori_shape"]
        im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask
