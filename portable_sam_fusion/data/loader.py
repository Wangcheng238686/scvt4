import logging
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from portable_sam_fusion.data.satellite_dataset import SatelliteInstanceDataset
from portable_sam_fusion.data.satellite_drone_dataset import (
    SatelliteDroneDataset,
    rtmdet_drone_collate_fn,
    build_scene_id_mapping,
)


logger = logging.getLogger(__name__)


def rtmdet_collate_fn(batch):
    import numpy as np
    import torch
    import torch.nn.functional as F

    imgs = []
    img_metas = []
    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    max_instances = max(len(item["gt_labels"]) for item in batch)

    for item in batch:
        imgs.append(item["img"])
        img_metas.append(item["img_metas"])

        bboxes = item["gt_bboxes"]
        labels = item["gt_labels"]
        masks = item["gt_masks"]

        num_inst = len(labels)
        if num_inst < max_instances:
            pad_size = max_instances - num_inst
            bboxes = F.pad(bboxes, (0, 0, 0, pad_size))
            labels = F.pad(labels, (0, pad_size), value=-1)
            masks = np.concatenate(
                [
                    masks,
                    np.zeros(
                        (pad_size, *item["img_metas"]["img_shape"]), dtype=np.uint8
                    ),
                ],
                axis=0,
            )

        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
        gt_masks.append(masks)

    return {
        "imgs": torch.stack(imgs),
        "img_metas": img_metas,
        "gt_bboxes": gt_bboxes,
        "gt_labels": gt_labels,
        "gt_masks": gt_masks,
    }


def create_train_loader(
    data_root: str,
    batch_size: int,
    mosaic_prob: float = 0.0,
    rotate_prob: float = 0.0,
    scale_prob: float = 0.0,
    flip_prob: float = 0.0,
    vflip_prob: float = 0.0,
    crop_prob: float = 0.0,
    color_jitter_prob: float = 0.0,
    hue_prob: float = 0.0,
    sharpness_prob: float = 0.0,
    image_size: Tuple[int, int] = (512, 512),
    use_drone: bool = False,
    drone_data_root: str = "",
    num_views: int = 15,
    drone_image_size: Tuple[int, int] = (512, 512),
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    val_ratio: float = 0.0,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    normalize_drone: bool = True,
    random_sample: bool = True,
):
    if use_drone:
        scene_id_to_index = build_scene_id_mapping(data_root, drone_data_root)
        logger.info("Built scene ID mapping: %d scenes", len(scene_id_to_index))
        
        train_dataset = SatelliteDroneDataset(
            satellite_data_root=data_root,
            drone_data_root=drone_data_root,
            scene_ids=None,
            image_size=image_size,
            drone_image_size=drone_image_size,
            num_sample_images=num_views,
            random_sample=random_sample,
            normalize_drone=normalize_drone,
            val_ratio=val_ratio,
            is_val=False,
            flip_prob=flip_prob,
            scene_id_to_index=scene_id_to_index,
        )
        collate_fn = rtmdet_drone_collate_fn
        if val_ratio > 0:
            val_dataset = SatelliteDroneDataset(
                satellite_data_root=data_root,
                drone_data_root=drone_data_root,
                scene_ids=None,
                image_size=image_size,
                drone_image_size=drone_image_size,
                num_sample_images=num_views,
                random_sample=False,
                normalize_drone=normalize_drone,
                val_ratio=val_ratio,
                is_val=True,
                scene_id_to_index=scene_id_to_index,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size if val_batch_size else batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        else:
            val_loader = None
    else:
        train_dataset = SatelliteInstanceDataset(
            satellite_data_root=data_root,
            scene_ids=None,
            image_size=image_size,
            val_ratio=val_ratio,
            is_val=False,
        )
        collate_fn = rtmdet_collate_fn

        val_loader = None
        if val_ratio > 0:
            val_dataset = SatelliteInstanceDataset(
                satellite_data_root=data_root,
                scene_ids=None,
                image_size=image_size,
                val_ratio=val_ratio,
                is_val=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size or batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
                persistent_workers=num_workers > 0,
            )
            logger.info("创建验证集加载器: %d 样本", len(val_dataset))

    sampler = None
    shuffle = True
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, train_dataset
