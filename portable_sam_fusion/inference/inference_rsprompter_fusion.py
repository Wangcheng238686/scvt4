#!/usr/bin/env python
"""RSPrompter Fusion inference script with COCO-style evaluation metrics.

Metric computation follows migrated_project/tools/test.py which uses
mmengine Runner + CocoMetric (pycocotools.COCOeval).
"""

import argparse
import json
import logging
import os
import os.path as osp
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.config import Config
from mmengine.structures import InstanceData

# Add project root to sys.path
project_root = osp.abspath(osp.join(osp.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Register components (same as train script)
import mmdet.models
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor

if "DetDataPreprocessor" not in MMENGINE_MODELS:
    MMENGINE_MODELS.register_module(name="DetDataPreprocessor", module=DetDataPreprocessor)

import portable_sam_fusion.rsprompter
import portable_sam_fusion.models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# COCO-style evaluation helpers
# ---------------------------------------------------------------------------
def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask (H, W) to COCO RLE format."""
    from pycocotools import mask as mask_util

    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    out = bbox.copy()
    out[..., 2] = bbox[..., 2] - bbox[..., 0]
    out[..., 3] = bbox[..., 3] - bbox[..., 1]
    return out


def build_coco_gt_and_dt(
    all_gt: List[dict], all_dt: List[dict], img_metas_list: List[dict]
) -> Tuple:
    """Build pycocotools COCO objects for GT and detections.

    Args:
        all_gt: list of dicts per image, each with 'bboxes', 'labels', 'masks'
                (numpy arrays; bboxes in xyxy, masks in H×W bool).
        all_dt: list of dicts per image, each with 'bboxes', 'labels', 'scores',
                'masks' (numpy arrays).
        img_metas_list: list of img_meta dicts with at least 'img_shape'.

    Returns:
        (coco_gt, coco_dt) pycocotools COCO objects.
    """
    from pycocotools.coco import COCO

    images = []
    annotations = []
    predictions = []
    ann_id = 1

    for img_id, (gt, dt, meta) in enumerate(
        zip(all_gt, all_dt, img_metas_list), start=1
    ):
        h, w = meta["img_shape"][:2]
        images.append({"id": img_id, "height": int(h), "width": int(w)})

        # GT annotations
        gt_bboxes = gt["bboxes"]  # (N, 4) xyxy
        gt_labels = gt["labels"]  # (N,)
        gt_masks = gt["masks"]  # (N, H, W) or BitmapMasks

        if hasattr(gt_masks, "masks"):
            gt_masks = gt_masks.masks  # BitmapMasks → np array

        num_gt = len(gt_labels)
        for i in range(num_gt):
            bbox_xywh = _xyxy_to_xywh(gt_bboxes[i]).tolist()
            area = float(bbox_xywh[2] * bbox_xywh[3])
            seg_rle = _mask_to_rle(gt_masks[i])
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "area": area,
                    "segmentation": seg_rle,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        # Predictions
        dt_bboxes = dt["bboxes"]  # (M, 4) xyxy
        dt_scores = dt["scores"]  # (M,)
        dt_masks = dt["masks"]  # (M, H, W)

        if hasattr(dt_masks, "masks"):
            dt_masks = dt_masks.masks

        num_dt = len(dt_scores)
        for i in range(num_dt):
            bbox_xywh = _xyxy_to_xywh(dt_bboxes[i]).tolist()
            seg_rle = _mask_to_rle(dt_masks[i])
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "score": float(dt_scores[i]),
                    "segmentation": seg_rle,
                }
            )

    # Build COCO GT
    gt_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "building"}],
    }
    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()

    # Build COCO DT
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": images, "annotations": [], "categories": gt_dataset["categories"]}
        coco_dt.createIndex()

    return coco_gt, coco_dt


def run_coco_eval(coco_gt, coco_dt, iou_type: str = "bbox") -> OrderedDict:
    """Run COCOeval and return metrics dict.

    Args:
        iou_type: 'bbox' or 'segm'
    """
    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l",
        "AR@100", "AR@300", "AR@1000", "AR_s", "AR_m", "AR_l",
    ]
    results = OrderedDict()
    prefix = "bbox" if iou_type == "bbox" else "segm"
    for name, val in zip(metric_names, coco_eval.stats):
        results[f"{prefix}/{name}"] = float(val)
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_comparison(img_path, pred_instances, gt_instances, output_path, score_thr=0.3):
    """Visualize original image, ground truth, and predictions side-by-side."""
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        return
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    target_size = (1024, 1024)
    img = cv2.resize(img_orig, target_size, interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(img)
    axes[0].set_title("Original Image (1024x1024)")
    axes[0].axis("off")

    def draw_instances(ax, instances, title, color):
        vis_img = img.copy()
        if instances is None or len(instances) == 0:
            ax.imshow(vis_img)
            ax.set_title(f"{title} (0 items)")
            ax.axis("off")
            return

        bboxes = instances.bboxes
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()

        scores = None
        if hasattr(instances, "scores"):
            scores = instances.scores
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

        masks = None
        if hasattr(instances, "masks"):
            masks = instances.masks
            if hasattr(masks, "masks"):
                masks = masks.masks
            elif hasattr(masks, "to_ndarray"):
                masks = masks.to_ndarray()
            elif isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()

        if scores is not None:
            valid = scores > score_thr
            if not valid.any():
                ax.imshow(vis_img)
                ax.set_title(f"{title} (0 items)")
                ax.axis("off")
                return
            bboxes = bboxes[valid]
            if masks is not None:
                masks = masks[valid]
            scores = scores[valid]

        count = len(bboxes)
        for i in range(count):
            x1, y1, x2, y2 = bboxes[i].astype(int)
            if masks is not None:
                mask = masks[i]
                if mask.shape[:2] != vis_img.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (vis_img.shape[1], vis_img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                mask_colored = np.zeros_like(vis_img)
                mask_colored[mask > 0] = color
                vis_img = cv2.addWeighted(vis_img, 1, mask_colored, 0.5, 0)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            if scores is not None:
                cv2.putText(vis_img, f"{scores[i]:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        ax.imshow(vis_img)
        ax.set_title(f"{title} ({count} items)")
        ax.axis("off")

    draw_instances(axes[1], gt_instances, "Ground Truth", (0, 255, 0))
    draw_instances(axes[2], pred_instances, "Prediction", (255, 0, 0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _prepare_data_samples(batch: Dict, device: str) -> List[DetDataSample]:
    """Prepare DetDataSample list from batch, mirroring the training script."""
    imgs = batch["imgs"]
    img_metas = batch["img_metas"]
    gt_bboxes_list = [b.to(device) for b in batch["gt_bboxes"]]
    gt_labels_list = [l.to(device) for l in batch["gt_labels"]]
    gt_masks_list = batch["gt_masks"]

    data_samples: List[DetDataSample] = []
    for i in range(len(imgs)):
        ds = DetDataSample()
        ds.set_metainfo(img_metas[i])
        gt_instances = InstanceData()
        num_inst = len(gt_labels_list[i])
        if num_inst > 0:
            valid_mask = gt_labels_list[i] >= 0
            num_valid = int(valid_mask.sum().item())
            if num_valid > 0:
                valid_indices = valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                gt_instances.bboxes = gt_bboxes_list[i][valid_indices]
                gt_instances.labels = gt_labels_list[i][valid_indices]
                masks = gt_masks_list[i][valid_indices.cpu().numpy()]
                gt_instances.masks = BitmapMasks(masks, *img_metas[i]["img_shape"])
            else:
                gt_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                gt_instances.labels = torch.zeros((0,), dtype=torch.int64, device=device)
                gt_instances.masks = BitmapMasks(
                    np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                    *img_metas[i]["img_shape"],
                )
        else:
            gt_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            gt_instances.labels = torch.zeros((0,), dtype=torch.int64, device=device)
            gt_instances.masks = BitmapMasks(
                np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                *img_metas[i]["img_shape"],
            )
        ds.gt_instances = gt_instances
        data_samples.append(ds.to(device))
    return data_samples


def _extract_instances_numpy(
    instances, img_shape: Tuple[int, int]
) -> dict:
    """Extract bboxes, labels, scores, masks from InstanceData to numpy."""
    bboxes = instances.bboxes
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    labels = instances.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    scores = None
    if hasattr(instances, "scores"):
        scores = instances.scores
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

    masks = None
    if hasattr(instances, "masks"):
        masks = instances.masks
        if hasattr(masks, "masks"):
            masks = masks.masks  # BitmapMasks → ndarray
        elif hasattr(masks, "to_ndarray"):
            masks = masks.to_ndarray()
        elif isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

    # Ensure masks shape matches img_shape
    h, w = img_shape[:2]
    if masks is not None and len(masks) > 0 and masks.shape[1:] != (h, w):
        resized = []
        for m in masks:
            resized.append(
                cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            )
        masks = np.stack(resized, axis=0)

    if masks is None or len(masks) == 0:
        masks = np.zeros((0, h, w), dtype=np.uint8)

    out = {"bboxes": bboxes, "labels": labels, "masks": masks}
    if scores is not None:
        out["scores"] = scores
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RSPrompter Inference Script")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--work-dir", help="directory to save evaluation metrics")
    parser.add_argument("--show-dir", help="directory to save visualizations")
    parser.add_argument("--show-score-thr", type=float, default=0.3,
                        help="score threshold for visualization")
    parser.add_argument("--max-num", type=int, default=100, help="max images to visualize")
    parser.add_argument("--gpu-id", type=int, default=0, help="gpu id")
    parser.add_argument("--data-root", help="satellite test data root")
    parser.add_argument("--drone-data-root", help="drone test data root")
    parser.add_argument("--use-drone", action="store_true", default=False,
                        help="use drone guidance (auto-detected from model config if not specified)")
    parser.add_argument("--no-drone", action="store_true",
                        help="force disable drone guidance even if model supports it")
    parser.add_argument("--random-sample", action="store_true",
                        help="randomly sample drone images")
    parser.add_argument("--num-views", type=int, default=4, help="number of drone views")
    parser.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--drone-image-size", type=int, nargs=2, default=[512, 512])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # ---- Build model ----
    model = MODELS.build(cfg.model)
    model.to(device)
    model_base = model.module if hasattr(model, "module") else model
    data_preprocessor = model_base.data_preprocessor

    # ---- Load checkpoint ----
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model_base.load_state_dict(checkpoint["model"], strict=False)
    elif "state_dict" in checkpoint:
        model_base.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model_base.load_state_dict(checkpoint, strict=False)
    logger.info("Loaded checkpoint from %s", args.checkpoint)
    model.eval()

    # ---- Create dataloader ----
    from portable_sam_fusion.data.satellite_drone_dataset import (
        SatelliteDroneDataset, rtmdet_drone_collate_fn,
    )
    from portable_sam_fusion.data.satellite_dataset import SatelliteInstanceDataset
    from portable_sam_fusion.data.loader import rtmdet_collate_fn
    from torch.utils.data import DataLoader

    sat_data_root = (args.data_root or
                     "/home/wangcheng/data/unversity-big-after-without-negative/university-s-test")
    drone_data_root = (args.drone_data_root or
                       "/home/wangcheng/data/unversity-big-after-without-negative/university-d-test-selected-5")
    image_size = tuple(args.image_size)
    drone_image_size = tuple(args.drone_image_size)

    enable_drone_branch = getattr(model_base, "enable_drone_branch", False)

    if args.no_drone:
        use_drone = False
        logger.info("Drone guidance force disabled via --no-drone flag")
    elif args.use_drone:
        if not enable_drone_branch:
            logger.warning(
                "--use-drone specified but model has enable_drone_branch=False. "
                "Will run in satellite-only mode."
            )
            use_drone = False
        else:
            use_drone = True
    else:
        use_drone = enable_drone_branch
        if enable_drone_branch:
            logger.info("Auto-detected enable_drone_branch=True, using drone guidance")
        else:
            logger.info("Auto-detected enable_drone_branch=False, running satellite-only mode")

    if use_drone:
        test_dataset = SatelliteDroneDataset(
            satellite_data_root=sat_data_root,
            drone_data_root=drone_data_root,
            scene_ids=None,
            image_size=image_size,
            drone_image_size=drone_image_size,
            num_sample_images=args.num_views,
            random_sample=args.random_sample,
            normalize_drone=True,
        )
        collate_fn = rtmdet_drone_collate_fn
    else:
        test_dataset = SatelliteInstanceDataset(
            satellite_data_root=sat_data_root,
            scene_ids=None,
            image_size=image_size,
            val_ratio=0.0,
            is_val=False,
        )
        collate_fn = rtmdet_collate_fn

    dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=True, drop_last=False,
    )
    logger.info("Test dataset: %d samples", len(test_dataset))

    # ---- Prepare output dirs ----
    if args.show_dir:
        os.makedirs(args.show_dir, exist_ok=True)
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)

    # ---- Inference loop ----
    all_gt: List[dict] = []
    all_dt: List[dict] = []
    all_img_metas: List[dict] = []
    visualized_count = 0
    # Loss meters
    total_loss = 0.0
    loss_meter: Dict[str, float] = {}
    num_loss_batches = 0

    print("========================================")
    print(f"开始推理: {len(test_dataset)} 个样本")
    print(f"配置: {args.config}")
    print(f"权重: {args.checkpoint}")
    print(f"卫星数据: {sat_data_root}")
    if use_drone:
        print(f"无人机数据: {drone_data_root}")
        print(f"视角数: {args.num_views}")
    print("========================================")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
        with torch.no_grad():
            imgs = batch["imgs"].to(device)
            img_metas = batch["img_metas"]

            # ---- Prepare data_samples ----
            data_samples = _prepare_data_samples(batch, device)

            batch_has_valid_gt = any(
                ds.gt_instances.labels.numel() > 0 and (ds.gt_instances.labels >= 0).any()
                for ds in data_samples
            )

            # ---- Data preprocessing ----
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=False
            )
            imgs_processed = processed["inputs"]
            data_samples = processed["data_samples"]

            # ---- Build model inputs ----
            if use_drone and "drone_images" in batch:
                model_inputs = {
                    "sat": imgs_processed,
                    "drone_images": batch["drone_images"].to(device),
                    "intrinsics": batch["intrinsics"].to(device),
                    "extrinsics": batch["extrinsics"].to(device),
                }
            else:
                model_inputs = imgs_processed

            # ---- Run prediction (same as runner.test → model.predict) ----
            outputs = model_base.predict(model_inputs, data_samples, rescale=False)

            # ---- Collect GT and DT for COCO evaluation ----
            for j, output in enumerate(outputs):
                meta = img_metas[j]
                img_shape = meta["img_shape"]

                # GT
                gt_inst = output.gt_instances
                gt_np = _extract_instances_numpy(gt_inst, img_shape)
                all_gt.append(gt_np)

                # Predictions
                pred_inst = output.pred_instances
                pred_np = _extract_instances_numpy(pred_inst, img_shape)
                all_dt.append(pred_np)

                all_img_metas.append(meta)

            # ---- Compute loss (additional metric) ----
            if batch_has_valid_gt:
                loss_dict = model_base.loss(model_inputs, data_samples)
                loss_parts = []
                for k, v in loss_dict.items():
                    if "loss" not in k.lower():
                        continue
                    if isinstance(v, torch.Tensor):
                        loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(x for x in v if isinstance(x, torch.Tensor))
                loss = sum(loss_parts) if loss_parts else torch.tensor(0.0)
                total_loss += float(loss.detach().item())
                num_loss_batches += 1
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_meter[k] = loss_meter.get(k, 0.0) + float(v.detach().item())
                    elif isinstance(v, (list, tuple)):
                        val = sum(float(x.detach().item()) for x in v if isinstance(x, torch.Tensor))
                        loss_meter[k] = loss_meter.get(k, 0.0) + val

            # ---- Visualization ----
            if args.show_dir and visualized_count < args.max_num:
                for j, output in enumerate(outputs):
                    if visualized_count >= args.max_num:
                        break
                    img_path = img_metas[j].get("img_path", img_metas[j].get("filename", ""))
                    if not img_path:
                        continue
                    # GT for visualization (from raw batch, un-preprocessed)
                    gt_vis = None
                    if "gt_bboxes" in batch and j < len(batch["gt_bboxes"]):
                        gt_vis = InstanceData()
                        curr_labels = batch["gt_labels"][j]
                        valid = curr_labels >= 0 if isinstance(curr_labels, torch.Tensor) else np.array(curr_labels) >= 0
                        gt_vis.bboxes = batch["gt_bboxes"][j][valid]
                        gt_vis.labels = batch["gt_labels"][j][valid]
                        if "gt_masks" in batch:
                            idx = valid.cpu().numpy() if isinstance(valid, torch.Tensor) else valid
                            gt_vis.masks = batch["gt_masks"][j][idx]
                    out_file = osp.join(args.show_dir, osp.basename(img_path))
                    visualize_comparison(
                        img_path, output.pred_instances, gt_vis, out_file,
                        score_thr=args.show_score_thr,
                    )
                    visualized_count += 1

    # ---- COCO-style evaluation ----
    print("\n========================================")
    print("计算 COCO 评估指标 ...")
    print("========================================")

    metrics: Dict[str, float] = {}
    total_gt = sum(len(g["labels"]) for g in all_gt)
    total_dt = sum(len(d["scores"]) for d in all_dt if "scores" in d)
    print(f"  总 GT 实例数: {total_gt}")
    print(f"  总预测实例数: {total_dt}")

    if total_gt > 0:
        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, all_img_metas)

        print("\n--- BBox 评估 ---")
        bbox_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="bbox")
        metrics.update(bbox_metrics)

        print("\n--- Mask (Segm) 评估 ---")
        segm_metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        metrics.update(segm_metrics)
    else:
        print("  WARNING: No GT instances found. Skipping COCO evaluation.")

    # ---- Loss metrics ----
    print("\n--- Loss 指标 ---")
    if num_loss_batches > 0:
        avg_loss = total_loss / num_loss_batches
        metrics["loss/avg_total"] = avg_loss
        print(f"  avg_total_loss: {avg_loss:.6f}  ({num_loss_batches} batches)")
        for k, v in loss_meter.items():
            avg_k = v / num_loss_batches
            metrics[f"loss/{k}"] = avg_k
        detail = ", ".join(f"{k}={v / num_loss_batches:.4f}" for k, v in loss_meter.items())
        print(f"  详细: {detail}")
    else:
        print("  WARNING: No batches with valid GT for loss computation.")

    # ---- Summary ----
    print("\n========================================")
    print("评估结果汇总:")
    print("========================================")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("========================================")

    if args.show_dir:
        print(f"可视化结果: {args.show_dir} ({visualized_count} images)")
    if args.work_dir:
        with open(osp.join(args.work_dir, "eval_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"评估指标已保存: {osp.join(args.work_dir, 'eval_results.json')}")


if __name__ == "__main__":
    main()
