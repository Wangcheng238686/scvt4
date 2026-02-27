import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class SatelliteInstanceDataset(Dataset):
    def __init__(
        self,
        satellite_data_root: str,
        scene_ids: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (512, 512),
        val_ratio: float = 0.0,
        is_val: bool = False,
        seed: int = 42,
    ):
        self.satellite_data_root = Path(satellite_data_root)
        self.image_size = tuple(image_size)
        self.val_ratio = float(val_ratio)
        self.is_val = bool(is_val)
        self.seed = int(seed)

        self.scene_data = self._load_scenes(scene_ids)
        split_name = "验证" if self.is_val else "训练"
        logger.info("加载 %d 个卫星场景 (%s集)", len(self.scene_data), split_name)

    def _load_scenes(self, scene_ids: Optional[List[str]]) -> List[Dict]:
        scene_data: List[Dict] = []
        scene_to_image_path: Dict[str, Path] = {}

        if scene_ids is None:
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            image_files: List[Path] = []
            for ext in image_extensions:
                image_files.extend(list(self.satellite_data_root.glob(ext)))
            for f in image_files:
                scene_to_image_path[f.stem] = f
            scene_ids = sorted(list(scene_to_image_path.keys()))

        sorted_scene_ids = sorted(scene_ids)

        if self.val_ratio > 0:
            val_scenes = []
            train_scenes = []
            for scene_id in sorted_scene_ids:
                hash_val = int(hashlib.md5(f"{scene_id}_{self.seed}".encode()).hexdigest(), 16)
                if (hash_val % 100) < int(self.val_ratio * 100):
                    val_scenes.append(scene_id)
                else:
                    train_scenes.append(scene_id)
            selected_ids = val_scenes if self.is_val else train_scenes
            scene_ids_to_load = selected_ids
        else:
            scene_ids_to_load = sorted_scene_ids

        for scene_id in scene_ids_to_load:
            if scene_id in scene_to_image_path:
                image_path = scene_to_image_path[scene_id]
            else:
                found = False
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                    p = self.satellite_data_root / f"{scene_id}{ext}"
                    if p.exists():
                        image_path = p
                        found = True
                        break
                if not found:
                    continue

            annotation_path = self.satellite_data_root / f"{scene_id}.json"
            if not annotation_path.exists():
                continue
            try:
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotation = json.load(f)
            except Exception:
                continue

            scene_data.append({"scene_id": scene_id, "image_path": image_path, "annotation": annotation})

        return scene_data

    def __len__(self) -> int:
        return len(self.scene_data)

    def __getitem__(self, idx: int) -> Dict:
        scene_info = self.scene_data[idx]
        scene_id = scene_info["scene_id"]
        image_path = scene_info["image_path"]
        annotation = scene_info["annotation"]

        img = cv2.imread(str(image_path))
        orig_h, orig_w = img.shape[:2]

        instances = self._parse_instances(annotation, orig_h, orig_w)

        scale_w, scale_h = 1.0, 1.0
        if (orig_w, orig_h) != self.image_size:
            scale_h = self.image_size[1] / orig_h
            scale_w = self.image_size[0] / orig_w
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)

            for inst in instances:
                x1, y1, x2, y2 = inst["bbox"]
                inst["bbox"] = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]
                inst["mask"] = cv2.resize(inst["mask"].astype(np.uint8), self.image_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        if len(instances) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = np.zeros((0, *self.image_size), dtype=np.uint8)
        else:
            bboxes = torch.tensor([inst["bbox"] for inst in instances], dtype=torch.float32)
            labels = torch.tensor([inst["label"] for inst in instances], dtype=torch.int64)
            masks = np.stack([inst["mask"] for inst in instances], axis=0)

        img_metas = {
            "img_shape": self.image_size,
            "ori_shape": (orig_h, orig_w),
            "pad_shape": self.image_size,
            "batch_input_shape": self.image_size,
            "scale_factor": (scale_w, scale_h),
            "filename": str(image_path),
            "scene_id": scene_id,
        }

        return {
            "img": img_tensor,
            "img_metas": img_metas,
            "gt_bboxes": bboxes,
            "gt_labels": labels,
            "gt_masks": masks,
            "scene_id": scene_id,
        }

    def _parse_instances(self, annotation: Dict, img_h: int, img_w: int) -> List[Dict]:
        shapes = annotation.get("shapes", [])
        instances = []

        group_polygons: Dict[int, List[np.ndarray]] = {}

        for shape in shapes:
            if shape.get("shape_type") != "polygon" or "points" not in shape:
                continue
            points = shape["points"]
            if len(points) < 3:
                continue

            group_id = shape.get("group_id", None)
            if group_id is None:
                group_id = len(group_polygons)
            if group_id not in group_polygons:
                group_polygons[group_id] = []
            group_polygons[group_id].append(np.array(points, dtype=np.int32))

        for polygons in group_polygons.values():
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for polygon in polygons:
                cv2.fillPoly(mask, [polygon], 1)
            if mask.sum() == 0:
                continue

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y1, y2 = y_indices[0], y_indices[-1] + 1
            x1, x2 = x_indices[0], x_indices[-1] + 1

            y1 = max(0, min(int(y1), img_h - 1))
            y2 = max(1, min(int(y2), img_h))
            x1 = max(0, min(int(x1), img_w - 1))
            x2 = max(1, min(int(x2), img_w))

            if y2 <= y1 or x2 <= x1:
                continue

            instances.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "label": 0, "mask": mask})

        return instances

