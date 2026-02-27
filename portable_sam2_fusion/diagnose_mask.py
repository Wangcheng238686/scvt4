#!/usr/bin/env python
"""诊断 Mask Decoder 输出质量"""

import os
import sys
import torch
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def diagnose_mask_decoder():
    print("=" * 60)
    print("Mask Decoder 诊断")
    print("=" * 60)

    checkpoint_path = os.path.join(project_root, "checkpoints/sam2.1_hiera_base_plus.pt")
    model_path = os.path.join(project_root, "checkpoints/rsprompter-sam2-v11/best_model.pth")

    from rsprompter.models_sam2 import (
        RSSAM2VisionEncoder,
        RSSAM2MaskDecoderWrapper,
        RSFeatureAggregatorSAM2,
        RSPrompterAnchorMaskHeadSAM2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("\n" + "-" * 60)
    print("1. 构建 Backbone 和 Neck")
    print("-" * 60)

    backbone = RSSAM2VisionEncoder(checkpoint_path=checkpoint_path)
    backbone.to(device)
    backbone.eval()

    neck = RSFeatureAggregatorSAM2(
        in_channels="sam2_hiera_base",
        out_channels=256,
        hidden_channels=32,
        select_layers=[0, 1, 2, 3],
    )
    neck.to(device)
    neck.eval()

    print("\n" + "-" * 60)
    print("2. 构建 Mask Head 并加载权重")
    print("-" * 60)

    mask_head = RSPrompterAnchorMaskHeadSAM2(
        sam2_mask_decoder=dict(checkpoint_path=checkpoint_path),
        in_channels=256,
        roi_feat_size=14,
        per_pointset_point=4,
        with_sincos=False,
        multimask_output=False,
        class_agnostic=True,
    )
    mask_head.to(device)
    mask_head.eval()

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    mask_head_keys = {k: v for k, v in state_dict.items() if k.startswith("roi_head.mask_head.")}
    if mask_head_keys:
        new_state_dict = {}
        for k, v in mask_head_keys.items():
            new_k = k.replace("roi_head.mask_head.", "")
            new_state_dict[new_k] = v
        mask_head.load_state_dict(new_state_dict, strict=False)
        print(f"加载了 {len(new_state_dict)} 个 mask head 参数")

    if hasattr(mask_head, 'point_emb'):
        print(f"\npoint_emb 权重统计:")
        for name, param in mask_head.point_emb.named_parameters():
            print(f"  {name}: shape={param.shape}, mean={param.data.mean():.4f}, std={param.data.std():.4f}")

    print("\n" + "-" * 60)
    print("3. 测试前向传播")
    print("-" * 60)

    size = 512
    dummy_input = torch.randn(1, 3, size, size).to(device)

    with torch.no_grad():
        vision_outputs = backbone(dummy_input)
        image_embeddings = vision_outputs[0]
        vision_hidden_states = vision_outputs[1]

        print(f"image_embeddings shape: {image_embeddings.shape}")
        print(f"  mean: {image_embeddings.mean():.4f}, std: {image_embeddings.std():.4f}")

        neck_outputs = neck(vision_hidden_states)
        print(f"\nNeck outputs:")
        for i, feat in enumerate(neck_outputs):
            print(f"  [{i}] shape: {feat.shape}, mean: {feat.mean():.4f}, std: {feat.std():.4f}")

        pos_embed = torch.zeros(1, 256, 32, 32, device=device)
        if pos_embed.shape[2:] != image_embeddings.shape[2:]:
            pos_embed_resized = torch.nn.functional.interpolate(
                pos_embed, size=image_embeddings.shape[2:], mode='bilinear', align_corners=False
            )
        else:
            pos_embed_resized = pos_embed
        print(f"\nPosition embedding: {pos_embed.shape} -> {pos_embed_resized.shape}")

        dummy_roi_feat = torch.randn(2, 256, 14, 14).to(device)
        roi_img_ids = torch.tensor([0, 0]).to(device)

        low_res_masks, iou_pred = mask_head(
            dummy_roi_feat,
            image_embeddings,
            pos_embed_resized,
            roi_img_ids,
        )

        print(f"\nMask head 输出:")
        print(f"  low_res_masks shape: {low_res_masks.shape}")
        print(f"  low_res_masks range: [{low_res_masks.min():.4f}, {low_res_masks.max():.4f}]")
        print(f"  low_res_masks mean: {low_res_masks.mean():.4f}, std: {low_res_masks.std():.4f}")
        print(f"  iou_pred shape: {iou_pred.shape}")
        print(f"  iou_pred range: [{iou_pred.min():.4f}, {iou_pred.max():.4f}]")

        mask_probs = torch.sigmoid(low_res_masks)
        print(f"\n  sigmoid 后的 mask 概率:")
        print(f"    range: [{mask_probs.min():.4f}, {mask_probs.max():.4f}]")
        print(f"    mean: {mask_probs.mean():.4f}")

        binary_mask = mask_probs > 0.5
        print(f"    > 0.5 的比例: {binary_mask.float().mean():.4f}")

    print("\n" + "-" * 60)
    print("4. 检查 mask decoder 内部状态")
    print("-" * 60)

    decoder = mask_head.mask_decoder.decoder
    print(f"Decoder type: {type(decoder).__name__}")
    print(f"use_high_res_features: {getattr(decoder, 'use_high_res_features', 'N/A')}")

    if hasattr(decoder, 'output_upscaling'):
        print(f"\noutput_upscaling 结构:")
        for i, layer in enumerate(decoder.output_upscaling):
            print(f"  [{i}] {type(layer).__name__}")

    print("\n" + "-" * 60)
    print("5. 模拟完整的 mask 预测流程")
    print("-" * 60)

    with torch.no_grad():
        mask_logits = low_res_masks[0, 0]
        mask_probs = torch.sigmoid(mask_logits)

        print(f"原始 mask logits (64x64):")
        print(f"  range: [{mask_logits.min():.4f}, {mask_logits.max():.4f}]")

        mask_upsampled = torch.nn.functional.interpolate(
            mask_probs.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        print(f"\n上采样后 (512x512):")
        print(f"  range: [{mask_upsampled.min():.4f}, {mask_upsampled.max():.4f}]")
        print(f"  mean: {mask_upsampled.mean():.4f}")

        binary_mask_512 = mask_upsampled > 0.4
        print(f"  > 0.4 的比例: {binary_mask_512.float().mean():.4f}")

    print("\n" + "-" * 60)
    print("6. 对比原始 SAM2 Mask Decoder")
    print("-" * 60)

    mask_decoder_standalone = RSSAM2MaskDecoderWrapper(checkpoint_path=checkpoint_path)
    mask_decoder_standalone.to(device)
    mask_decoder_standalone.eval()

    with torch.no_grad():
        batch_size = 1
        h, w = image_embeddings.shape[2], image_embeddings.shape[3]

        sparse_prompt_embeddings = torch.zeros(batch_size, 4, 256, device=device)
        dense_prompt_embeddings = torch.zeros(batch_size, 256, h, w, device=device)
        image_pe = torch.zeros(1, 256, h, w, device=device)

        standalone_masks, standalone_iou, _ = mask_decoder_standalone(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output=False,
            repeat_image=False,
        )

        print(f"独立 Mask Decoder 输出:")
        print(f"  masks shape: {standalone_masks.shape}")
        print(f"  masks range: [{standalone_masks.min():.4f}, {standalone_masks.max():.4f}]")
        print(f"  iou range: [{standalone_iou.min():.4f}, {standalone_iou.max():.4f}]")

        standalone_probs = torch.sigmoid(standalone_masks)
        print(f"  sigmoid 后 range: [{standalone_probs.min():.4f}, {standalone_probs.max():.4f}]")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    diagnose_mask_decoder()
