#!/usr/bin/env python

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("portable_sam_fusion")


def _init_distributed() -> Dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    launched_by_torchrun = os.environ.get("LOCAL_RANK", None) is not None
    should_init = (world_size > 1) or launched_by_torchrun
    if should_init and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    distributed = dist.is_initialized()
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "distributed": int(distributed),
    }


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _build_optimizer(
    model: torch.nn.Module,
    lr: float,
    sat_backbone_lr_mult: float,
    sat_other_lr_mult: float,
    drone_lr_mult: float,
    scene_align_lr_mult: float = 2.0,
) -> optim.Optimizer:
    params_sat_backbone = []
    params_sat_other = []
    params_drone = []
    params_scene_align = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "scene_alignment" in name:
            params_scene_align.append(param)
        elif name.startswith("drone_encoder.") or name.startswith("guidance."):
            params_drone.append(param)
        elif name.startswith("contrastive_loss.") or name.startswith("consistency_loss."):
            params_drone.append(param)
        elif name.startswith("backbone.") or name.startswith("shared_image_embedding."):
            params_sat_backbone.append(param)
        else:
            params_sat_other.append(param)

    param_groups = []
    if params_sat_backbone:
        param_groups.append(
            {
                "params": params_sat_backbone,
                "lr": lr * sat_backbone_lr_mult,
                "name": "sat_backbone",
            }
        )
    if params_sat_other:
        param_groups.append(
            {
                "params": params_sat_other,
                "lr": lr * sat_other_lr_mult,
                "name": "sat_other",
            }
        )
    if params_drone:
        param_groups.append(
            {"params": params_drone, "lr": lr * drone_lr_mult, "name": "drone_guidance"}
        )
    if params_scene_align:
        param_groups.append(
            {"params": params_scene_align, "lr": lr * drone_lr_mult * scene_align_lr_mult, "name": "scene_alignment"}
        )

    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check freezing / requires_grad."
        )

    return optim.AdamW(param_groups, lr=lr, weight_decay=0.05, eps=1e-6)


def _set_norm_eval(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(
            m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.GroupNorm)
        ):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_metrics: Dict,
    history: Dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metrics": best_metrics,
        "history": history,
    }
    torch.save(ckpt, str(path))


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def main():
    parser = argparse.ArgumentParser(
        description="portable_sam_fusion: RSPrompter(SAM) + Drone semantic guidance"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--data-root", type=str, default="/data/wangcheng/dataset/university-test/S"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="/tmp/portable_sam_fusion_ckpts"
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--use-drone", action="store_true", default=False)
    parser.add_argument(
        "--drone-data-root",
        type=str,
        default="/data/wangcheng/dataset/university-test/D",
    )
    parser.add_argument("--num-views", type=int, default=15)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--drone-image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument(
        "--random-sample", action="store_true", help="Randomly sample drone images"
    )
    parser.add_argument(
        "--normalize-drone",
        action="store_true",
        default=True,
        help="Normalize drone images",
    )

    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD")
    parser.add_argument("--fsdp-min-num-params", type=int, default=1000000)

    parser.add_argument("--freeze-bn", action="store_true")

    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-every-n-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=15)

    parser.add_argument("--sat-backbone-lr-mult", type=float, default=0.5)
    parser.add_argument("--sat-other-lr-mult", type=float, default=1.0)
    parser.add_argument("--drone-lr-mult", type=float, default=0.5)
    parser.add_argument("--scene-align-lr-mult", type=float, default=2.0, help="Learning rate multiplier for scene alignment parameters")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=1000, help="Maximum number of scenes for scene-specific alignment")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    import mmdet.models
    from mmengine.config import Config
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.registry import MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor
        )

    import portable_sam2_fusion.rsprompter
    import portable_sam2_fusion.models

    dist_info = _init_distributed()
    distributed = bool(dist_info.get("distributed", 0))
    rank = int(dist_info["rank"])

    if torch.cuda.is_available():
        device = (
            f"cuda:{dist_info['local_rank']}" if distributed else f"cuda:{args.gpu_id}"
        )
    else:
        device = "cpu"

    is_main = _is_main_process(rank)
    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if hasattr(cfg.model, 'max_scenes'):
        cfg.model.max_scenes = args.max_scenes
    elif args.use_drone:
        cfg.model.max_scenes = args.max_scenes
    model = MODELS.build(cfg.model)
    model.to(device)

    model_for_preproc = model.module if hasattr(model, "module") else model
    data_preprocessor = model_for_preproc.data_preprocessor

    enable_drone_branch = getattr(model_for_preproc, "enable_drone_branch", False)
    if args.use_drone and not enable_drone_branch:
        logger.warning(
            "--use-drone is set but model has enable_drone_branch=False. "
            "Disabling drone data loading."
        )
        args.use_drone = False
    if enable_drone_branch and not args.use_drone:
        logger.info(
            "Model has enable_drone_branch=True but --use-drone not set. "
            "Will use drone branch if drone data is available."
        )

    if args.freeze_bn:
        _set_norm_eval(model)

    if distributed and not args.use_fsdp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[int(device.split(":")[1])]
            if device.startswith("cuda")
            else None,
        )

    if distributed and args.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        sharding = getattr(
            ShardingStrategy, args.fsdp_sharding, ShardingStrategy.FULL_SHARD
        )
        ignored_modules = None
        model_base = model.module if hasattr(model, "module") else model
        if (
            getattr(model_base, "freeze_drone", False)
            and hasattr(model_base, "drone_encoder")
            and model_base.drone_encoder is not None
        ):
            model_base.drone_encoder.requires_grad_(False)
            model_base.drone_encoder.eval()
            ignored_modules = [model_base.drone_encoder]

        model = FSDP(
            model,
            sharding_strategy=sharding,
            device_id=int(device.split(":")[1]) if device.startswith("cuda") else None,
            ignored_modules=ignored_modules,
            use_orig_params=True,
            min_num_params=int(args.fsdp_min_num_params),
        )

    from portable_sam2_fusion.data import create_train_loader

    train_loader, val_loader, train_dataset = (
        create_train_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            mosaic_prob=0.0,
            rotate_prob=0.0,
            scale_prob=0.0,
            flip_prob=0.5,
            vflip_prob=0.0,
            crop_prob=0.0,
            color_jitter_prob=0.2,
            hue_prob=0.2,
            sharpness_prob=0.2,
            gaussian_noise_prob=0.3,
            gaussian_noise_std=0.02,
            random_erasing_prob=0.2,
            random_erasing_scale=(0.02, 0.15),
            random_erasing_ratio=(0.3, 3.3),
            image_size=tuple(args.image_size),
            use_drone=args.use_drone,
            drone_data_root=args.drone_data_root,
            num_views=args.num_views,
            drone_image_size=tuple(args.drone_image_size),
            distributed=distributed,
            rank=rank,
            world_size=int(dist_info.get("world_size", 1)),
            val_ratio=args.val_ratio,
            val_batch_size=args.val_batch_size,
            random_sample=args.random_sample,
            normalize_drone=args.normalize_drone,
        )
    )

    num_batches_per_epoch = len(train_loader)

    if is_main:
        logger.info("Dataset size: %d", len(train_dataset))

    optimizer = _build_optimizer(
        model.module if hasattr(model, "module") else model,
        lr=args.lr,
        sat_backbone_lr_mult=args.sat_backbone_lr_mult,
        sat_other_lr_mult=args.sat_other_lr_mult,
        drone_lr_mult=args.drone_lr_mult,
        scene_align_lr_mult=args.scene_align_lr_mult,
    )
    warmup_iters = min(50, num_batches_per_epoch)
    cosine_t_max = max(1, args.epochs * num_batches_per_epoch - warmup_iters)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=args.lr * 0.001
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=warmup_iters
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters],
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    early_counter = 0
    history = {"epochs": [], "train_losses": [], "val_losses": [], "learning_rates": []}

    if args.resume_from:
        ckpt = _load_checkpoint(Path(args.resume_from), model, optimizer, scheduler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = ckpt.get("best_metrics", {}) or {}
        best_val_loss = float(best.get("best_val_loss", best_val_loss))
        best_train_loss = float(best.get("best_train_loss", best_train_loss))
        history = ckpt.get("history", history) or history
        if is_main:
            logger.info(
                "Resumed from %s (start_epoch=%d)",
                args.resume_from,
                start_epoch,
            )

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        model.train()
        if args.freeze_bn:
            _set_norm_eval(model)

        total_loss = 0.0
        loss_meter = {}
        num_batches = 0
        grad_accum = args.grad_accum_steps
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["imgs"].to(device)
            img_metas = batch["img_metas"]
            gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
            gt_labels = [l.to(device) for l in batch["gt_labels"]]
            gt_masks = batch["gt_masks"]

            # Prepare data_samples and move to device
            data_samples: List[DetDataSample] = []
            for i in range(len(imgs)):
                ds = DetDataSample()
                ds.set_metainfo(img_metas[i])
                gt_instances = InstanceData()
                num_inst = len(gt_labels[i])
                if num_inst > 0:
                    valid_mask = gt_labels[i] >= 0
                    num_valid = int(valid_mask.sum().item())
                    if num_valid > 0:
                        valid_indices = (
                            valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                        )
                        gt_instances.bboxes = gt_bboxes[i][valid_indices]
                        gt_instances.labels = gt_labels[i][valid_indices]
                        masks = gt_masks[i][valid_indices.cpu().numpy()]
                        gt_instances.masks = BitmapMasks(
                            masks, *img_metas[i]["img_shape"]
                        )
                    else:
                        gt_instances.bboxes = torch.zeros(
                            (0, 4), dtype=torch.float32, device=device
                        )
                        gt_instances.labels = torch.zeros(
                            (0,), dtype=torch.int64, device=device
                        )
                        gt_instances.masks = BitmapMasks(
                            np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                            *img_metas[i]["img_shape"],
                        )
                else:
                    gt_instances.bboxes = torch.zeros(
                        (0, 4), dtype=torch.float32, device=device
                    )
                    gt_instances.labels = torch.zeros(
                        (0,), dtype=torch.int64, device=device
                    )
                    gt_instances.masks = BitmapMasks(
                        np.zeros((0, *img_metas[i]["img_shape"]), dtype=np.uint8),
                        *img_metas[i]["img_shape"],
                    )
                ds.gt_instances = gt_instances
                data_samples.append(ds.to(device))

            batch_has_valid_gt = False
            for ds in data_samples:
                labels = getattr(ds.gt_instances, "labels", torch.tensor([]))
                if labels.numel() > 0 and (labels >= 0).any():
                    batch_has_valid_gt = True
                    break
            if not batch_has_valid_gt:
                if is_main:
                    logger.warning(
                        "Skip batch with no valid GT instances (epoch=%d)", epoch + 1
                    )
                continue

            # Synchronize data preprocessing for both images and ground truth
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=True
            )
            imgs = processed["inputs"]
            data_samples = processed["data_samples"]

            if args.use_drone and "drone_images" in batch:
                model_inputs = {
                    "sat": imgs,
                    "drone_images": batch["drone_images"].to(device),
                    "intrinsics": batch["intrinsics"].to(device),
                    "extrinsics": batch["extrinsics"].to(device),
                }
                scene_indices = batch.get("scene_indices")
                if scene_indices is not None:
                    scene_indices = scene_indices.to(device)
            else:
                model_inputs = imgs
                scene_indices = None

            model_for_loss = model.module if hasattr(model, "module") else model

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss_dict = model_for_loss.loss(model_inputs, data_samples, scene_indices=scene_indices)
                loss_parts = []
                for k, v in loss_dict.items():
                    if "loss" not in k:
                        continue
                    if isinstance(v, torch.Tensor):
                        loss_parts.append(v)
                    elif isinstance(v, (list, tuple)):
                        loss_parts.extend(x for x in v if isinstance(x, torch.Tensor))
                loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=device)
                loss_for_backward = loss / grad_accum

            # Record detailed losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    val = v.detach().item()
                    loss_meter[k] = loss_meter.get(k, 0.0) + val
                elif isinstance(v, list):
                    # Handle case where loss might be a list of tensors (though rare in final output dict)
                    val = sum(
                        x.detach().item() for x in v if isinstance(x, torch.Tensor)
                    )
                    loss_meter[k] = loss_meter.get(k, 0.0) + val

            if not torch.isfinite(loss):
                if is_main:
                    logger.error("Non-finite loss detected (epoch=%d).", epoch + 1)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss_for_backward).backward()

            # Step optimizer every grad_accum batches or at the last batch
            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(
                train_loader
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item())
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)

        # Collect LRs for all groups
        lr_groups = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            lr_groups[name] = group["lr"]

        if is_main:
            loss_str = []
            for k, v in loss_meter.items():
                avg_k = v / max(1, num_batches)
                loss_str.append(f"{k}={avg_k:.4f}")
            logger.info(f"Epoch {epoch + 1} detailed losses: {', '.join(loss_str)}")

        val_loss = None
        if val_loader is not None and (epoch + 1) % args.val_every_n_epochs == 0:
            model.eval()
            if args.freeze_bn:
                _set_norm_eval(model)
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["imgs"].to(device)
                    img_metas = batch["img_metas"]
                    gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
                    gt_labels = [l.to(device) for l in batch["gt_labels"]]
                    gt_masks = batch["gt_masks"]

                    # Prepare data_samples and move to device
                    data_samples = []
                    for i in range(len(imgs)):
                        ds = DetDataSample()
                        ds.set_metainfo(img_metas[i])
                        gt_instances = InstanceData()
                        num_inst = len(gt_labels[i])
                        if num_inst > 0:
                            valid_mask = gt_labels[i] >= 0
                            num_valid = int(valid_mask.sum().item())
                            if num_valid > 0:
                                valid_indices = (
                                    valid_mask.nonzero().squeeze(-1)[:num_valid].cpu()
                                )
                                gt_instances.bboxes = gt_bboxes[i][valid_indices]
                                gt_instances.labels = gt_labels[i][valid_indices]
                                masks = gt_masks[i][valid_indices.cpu().numpy()]
                                gt_instances.masks = BitmapMasks(
                                    masks, *img_metas[i]["img_shape"]
                                )
                            else:
                                gt_instances.bboxes = torch.zeros(
                                    (0, 4), dtype=torch.float32, device=device
                                )
                                gt_instances.labels = torch.zeros(
                                    (0,), dtype=torch.int64, device=device
                                )
                                gt_instances.masks = BitmapMasks(
                                    np.zeros(
                                        (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                    ),
                                    *img_metas[i]["img_shape"],
                                )
                        else:
                            gt_instances.bboxes = torch.zeros(
                                (0, 4), dtype=torch.float32, device=device
                            )
                            gt_instances.labels = torch.zeros(
                                (0,), dtype=torch.int64, device=device
                            )
                            gt_instances.masks = BitmapMasks(
                                np.zeros(
                                    (0, *img_metas[i]["img_shape"]), dtype=np.uint8
                                ),
                                *img_metas[i]["img_shape"],
                            )
                        ds.gt_instances = gt_instances
                        data_samples.append(ds.to(device))

                    # Skip validation batch if all images have no valid GT
                    val_batch_has_valid_gt = False
                    for ds in data_samples:
                        labels = getattr(ds.gt_instances, "labels", torch.tensor([]))
                        if labels.numel() > 0 and (labels >= 0).any():
                            val_batch_has_valid_gt = True
                            break
                    if not val_batch_has_valid_gt:
                        if is_main:
                            logger.debug(
                                "Skip val batch with no valid GT instances (epoch=%d)", epoch + 1
                            )
                        continue

                    # Synchronize data preprocessing for both images and ground truth
                    processed = data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    imgs = processed["inputs"]
                    data_samples = processed["data_samples"]

                    if args.use_drone and "drone_images" in batch:
                        model_inputs = {
                            "sat": imgs,
                            "drone_images": batch["drone_images"].to(device),
                            "intrinsics": batch["intrinsics"].to(device),
                            "extrinsics": batch["extrinsics"].to(device),
                        }
                        scene_indices = batch.get("scene_indices")
                        if scene_indices is not None:
                            scene_indices = scene_indices.to(device)
                    else:
                        model_inputs = imgs
                        scene_indices = None

                    model_for_loss = model.module if hasattr(model, "module") else model
                    loss_dict = model_for_loss.loss(model_inputs, data_samples, scene_indices=scene_indices)

                    val_loss_parts = []
                    for k, v in loss_dict.items():
                        if "loss" not in k.lower():
                            continue
                        if isinstance(v, torch.Tensor):
                            val_loss_parts.append(v)
                        elif isinstance(v, (list, tuple)):
                            val_loss_parts.extend(x for x in v if isinstance(x, torch.Tensor))
                    loss = sum(val_loss_parts) if val_loss_parts else torch.tensor(0.0, device=device)
                    total_val_loss += float(loss.detach().item())
                    val_batches += 1
            val_loss = total_val_loss / max(1, val_batches)

        if is_main:
            history["epochs"].append(epoch)
            history["train_losses"].append(avg_loss)
            history["val_losses"].append(val_loss)
            history["learning_rates"].append(lr_groups)

            best_metrics = {
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
            }

            is_best = False
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_metrics["best_val_loss"] = best_val_loss
                    early_counter = 0
                    is_best = True
                else:
                    early_counter += 1
            else:
                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    best_metrics["best_train_loss"] = best_train_loss
                    is_best = True

            _save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metrics,
                history,
            )
            if is_best:
                _save_checkpoint(
                    checkpoint_dir / "best_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metrics,
                    history,
                )

            if val_loss is not None:
                lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
                logger.info(
                    "Epoch %d/%d | train=%.6f val=%.6f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    val_loss,
                    lr_str,
                )
                if args.early_stopping_patience > 0 and early_counter >= int(
                    args.early_stopping_patience
                ):
                    logger.info(
                        "Early stopping triggered (patience=%d)",
                        int(args.early_stopping_patience),
                    )
                    break
            else:
                lr_str = " ".join([f"{k}={v:.2e}" for k, v in lr_groups.items()])
                logger.info(
                    "Epoch %d/%d | train=%.6f | %s",
                    epoch + 1,
                    args.epochs,
                    avg_loss,
                    lr_str,
                )

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
