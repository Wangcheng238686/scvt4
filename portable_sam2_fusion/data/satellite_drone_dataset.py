import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


logger = logging.getLogger(__name__)


def build_scene_id_mapping(satellite_data_root: str, drone_data_root: str) -> Dict[str, int]:
    satellite_data_root = Path(satellite_data_root)
    drone_data_root = Path(drone_data_root)
    
    scene_ids = set()
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for ext in image_extensions:
        for p in satellite_data_root.glob(ext):
            scene_id = p.stem
            drone_dir = drone_data_root / scene_id
            cam_path = drone_dir / "colmap_result" / "camera_params.json"
            ann_path = satellite_data_root / f"{scene_id}.json"
            if drone_dir.exists() and cam_path.exists() and ann_path.exists():
                scene_ids.add(scene_id)
    
    sorted_scene_ids = sorted(list(scene_ids))
    return {sid: i for i, sid in enumerate(sorted_scene_ids)}


class SatelliteDroneDataset(Dataset):
    def __init__(
        self,
        satellite_data_root: str,
        drone_data_root: str,
        scene_ids: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (512, 512),
        drone_image_size: Tuple[int, int] = (512, 512),
        num_sample_images: int = 15,
        random_sample: bool = True,
        normalize_drone: bool = True,
        val_ratio: float = 0.0,
        is_val: bool = False,
        seed: int = 42,
        flip_prob: float = 0.0,
        scene_id_to_index: Optional[Dict[str, int]] = None,
        gaussian_noise_prob: float = 0.0,
        gaussian_noise_std: float = 0.02,
        random_erasing_prob: float = 0.0,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.2),
        random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
    ):
        self.satellite_data_root = Path(satellite_data_root)
        self.drone_data_root = Path(drone_data_root)
        self.scene_ids = scene_ids
        self.image_size = tuple(image_size)
        self.drone_image_size = tuple(drone_image_size)
        self.num_sample_images = int(num_sample_images)
        self.random_sample = bool(random_sample)
        self.normalize_drone = bool(normalize_drone)
        self.val_ratio = float(val_ratio)
        self.is_val = bool(is_val)
        self.seed = int(seed)
        self.flip_prob = float(flip_prob)
        self.scene_id_to_index = scene_id_to_index or {}
        self.gaussian_noise_prob = float(gaussian_noise_prob)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.random_erasing_prob = float(random_erasing_prob)
        self.random_erasing_scale = tuple(random_erasing_scale)
        self.random_erasing_ratio = tuple(random_erasing_ratio)

        self._drone_transform = self._build_drone_transform()
        self.scene_data = self._load_scene_data()

        split_name = "验证" if self.is_val else "训练"
        logger.info(
            "SatelliteDroneDataset initialized (%s集): %d scenes, views=%d, sat_size=%s, drone_size=%s",
            split_name,
            len(self.scene_data),
            self.num_sample_images,
            self.image_size,
            self.drone_image_size,
        )

    def _build_drone_transform(self) -> T.Compose:
        transforms: List = [T.Resize(self.drone_image_size), T.ToTensor()]
        if self.normalize_drone:
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return T.Compose(transforms)

    @staticmethod
    def _build_image_index(drone_dir: Path) -> Dict[str, Path]:
        image_index: Dict[str, Path] = {}
        if not drone_dir.exists():
            return image_index
        for p in drone_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image_index[p.stem.lower()] = p
        return image_index

    def _load_scene_data(self) -> List[Dict]:
        items: List[Dict] = []
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        sat_imgs: List[Path] = []
        for ext in image_extensions:
            sat_imgs.extend(list(self.satellite_data_root.glob(ext)))
        sat_imgs = sorted(list(set(sat_imgs)), key=lambda p: p.stem)

        if self.scene_ids is not None:
            allowed = set(self.scene_ids)
            sat_imgs = [p for p in sat_imgs if p.stem in allowed]

        sorted_sat_imgs = sorted(sat_imgs, key=lambda p: p.stem)

        if self.val_ratio > 0:
            val_imgs = []
            train_imgs = []
            for img_path in sorted_sat_imgs:
                scene_id = img_path.stem
                hash_val = int(hashlib.md5(f"{scene_id}_{self.seed}".encode()).hexdigest(), 16)
                if (hash_val % 100) < int(self.val_ratio * 100):
                    val_imgs.append(img_path)
                else:
                    train_imgs.append(img_path)
            sat_imgs_to_load = val_imgs if self.is_val else train_imgs
        else:
            sat_imgs_to_load = sorted_sat_imgs

        for image_path in sat_imgs_to_load:
            scene_id = image_path.stem
            ann_path = self.satellite_data_root / f"{scene_id}.json"
            drone_dir = self.drone_data_root / scene_id
            cam_path = drone_dir / "colmap_result" / "camera_params.json"

            if not ann_path.exists():
                continue
            if not drone_dir.exists() or not cam_path.exists():
                continue

            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    annotation = json.load(f)
            except Exception:
                continue

            try:
                with open(cam_path, "r", encoding="utf-8") as f:
                    camera_params = json.load(f)
            except Exception:
                continue

            image_index = self._build_image_index(drone_dir)
            items.append(
                {
                    "scene_id": scene_id,
                    "image_path": image_path,
                    "annotation": annotation,
                    "drone_dir": drone_dir,
                    "camera_params": camera_params,
                    "drone_image_index": image_index,
                }
            )

        return items

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

        if (orig_w, orig_h) != self.image_size:
            scale_w = self.image_size[0] / orig_w
            scale_h = self.image_size[1] / orig_h
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            for inst in instances:
                x1, y1, x2, y2 = inst["bbox"]
                inst["bbox"] = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]
                inst["mask"] = cv2.resize(inst["mask"].astype(np.uint8), self.image_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:
            scale_w, scale_h = 1.0, 1.0

        do_flip = self.flip_prob > 0 and np.random.rand() < self.flip_prob
        if do_flip:
            img = img[:, ::-1].copy()
            w = self.image_size[0]
            for inst in instances:
                x1, y1, x2, y2 = inst["bbox"]
                inst["bbox"] = [w - x2, y1, w - x1, y2]
                inst["mask"] = inst["mask"][:, ::-1].copy()

        if self.gaussian_noise_prob > 0 and np.random.rand() < self.gaussian_noise_prob:
            noise = np.random.randn(*img.shape) * (self.gaussian_noise_std * 255)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if self.random_erasing_prob > 0 and np.random.rand() < self.random_erasing_prob:
            img, instances = self._apply_random_erasing(img, instances)

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

        drone_images, intrinsics, extrinsics = self._load_drone_data(scene_info, do_flip=do_flip)

        scene_index = self.scene_id_to_index.get(scene_id, 0)

        return {
            "img": img_tensor,
            "img_metas": img_metas,
            "gt_bboxes": bboxes,
            "gt_labels": labels,
            "gt_masks": masks,
            "drone_images": drone_images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "scene_id": scene_id,
            "scene_index": scene_index,
        }

    def _load_drone_data(self, scene_info: Dict, do_flip: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drone_dir: Path = scene_info["drone_dir"]
        camera_params: Dict = scene_info["camera_params"]
        image_index: Dict[str, Path] = scene_info.get("drone_image_index", {})

        all_image_names = camera_params.get("image_names", [])
        if len(all_image_names) == 0:
            return self._dummy_drone()

        total_images = len(all_image_names)
        if self.random_sample:
            if total_images >= self.num_sample_images:
                sample_indices = np.random.choice(total_images, size=self.num_sample_images, replace=False)
            else:
                sample_indices = np.random.choice(total_images, size=self.num_sample_images, replace=True)
        else:
            if total_images >= self.num_sample_images:
                sample_indices = np.linspace(0, total_images - 1, self.num_sample_images, dtype=int)
            else:
                sample_indices = np.arange(total_images)
            if len(sample_indices) < self.num_sample_images:
                pad = np.random.choice(sample_indices if len(sample_indices) > 0 else [0], size=self.num_sample_images - len(sample_indices), replace=True)
                sample_indices = np.concatenate([sample_indices, pad], axis=0)

        images_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []

        intr_all = camera_params.get("intrinsics", [])
        ext_all = camera_params.get("extrinsics", [])

        for idx in sample_indices:
            image_name = all_image_names[int(idx)]
            stem = image_name.rsplit(".", 1)[0].lower()
            image_path = image_index.get(stem)
            if image_path is None:
                candidate = drone_dir / image_name
                if candidate.exists():
                    image_path = candidate

            if image_path is None:
                images_list.append(torch.zeros(3, *self.drone_image_size))
                intrinsics_list.append(torch.eye(3))
                extrinsics_list.append(torch.eye(4))
                continue

            img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = img.size
            
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            img_tensor = self._drone_transform(img)
            images_list.append(img_tensor)

            if int(idx) < len(intr_all):
                intr = torch.tensor(intr_all[int(idx)], dtype=torch.float32)
                scale_x = self.drone_image_size[0] / orig_w
                scale_y = self.drone_image_size[1] / orig_h
                intr[0, 0] *= scale_x
                intr[1, 1] *= scale_y
                intr[0, 2] *= scale_x
                intr[1, 2] *= scale_y
                
                if do_flip:
                    intr[0, 2] = self.drone_image_size[0] - intr[0, 2]
            else:
                intr = torch.eye(3, dtype=torch.float32)
            
            if int(idx) < len(ext_all):
                ext = torch.tensor(ext_all[int(idx)], dtype=torch.float32)
                
                if do_flip:
                    ext[0, 0] = -ext[0, 0]
                    ext[0, 3] = -ext[0, 3]
            else:
                ext = torch.eye(4, dtype=torch.float32)
            intrinsics_list.append(intr)
            extrinsics_list.append(ext)

        drone_images = torch.stack(images_list, dim=0)
        intrinsics = torch.stack(intrinsics_list, dim=0)
        extrinsics = torch.stack(extrinsics_list, dim=0)

        return drone_images, intrinsics, extrinsics

    def _dummy_drone(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drone_images = torch.zeros(self.num_sample_images, 3, *self.drone_image_size, dtype=torch.float32)
        intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(self.num_sample_images, 1, 1)
        extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(self.num_sample_images, 1, 1)
        return drone_images, intrinsics, extrinsics

    def _apply_random_erasing(
        self, 
        img: np.ndarray, 
        instances: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict]]:
        h, w = img.shape[:2]
        img_area = h * w
        
        for _ in range(10):
            target_area = img_area * np.random.uniform(self.random_erasing_scale[0], self.random_erasing_scale[1])
            aspect_ratio = np.random.uniform(self.random_erasing_ratio[0], self.random_erasing_ratio[1])
            
            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if erase_h >= h or erase_w >= w:
                continue
            
            y1 = np.random.randint(0, h - erase_h)
            x1 = np.random.randint(0, w - erase_w)
            y2 = y1 + erase_h
            x2 = x1 + erase_w
            
            erase_mask = np.zeros((h, w), dtype=np.uint8)
            erase_mask[y1:y2, x1:x2] = 1
            
            overlap_ratio = 0.0
            for inst in instances:
                inst_mask = inst["mask"]
                intersection = np.logical_and(inst_mask, erase_mask).sum()
                inst_area = inst_mask.sum()
                if inst_area > 0:
                    overlap_ratio = max(overlap_ratio, intersection / inst_area)
            
            if overlap_ratio < 0.3:
                img[y1:y2, x1:x2] = np.random.randint(0, 256, (erase_h, erase_w, 3), dtype=np.uint8)
                
                for inst in instances:
                    inst["mask"][y1:y2, x1:x2] = 0
                    
                    x1_i, y1_i, x2_i, y2_i = inst["bbox"]
                    if x1_i >= x2 and x2_i <= x1:
                        continue
                    if y1_i >= y2 and y2_i <= y1:
                        continue
                    
                    new_mask = inst["mask"]
                    rows = np.any(new_mask, axis=1)
                    cols = np.any(new_mask, axis=0)
                    if rows.any() and cols.any():
                        y_indices = np.where(rows)[0]
                        x_indices = np.where(cols)[0]
                        inst["bbox"] = [
                            float(x_indices[0]),
                            float(y_indices[0]),
                            float(x_indices[-1] + 1),
                            float(y_indices[-1] + 1),
                        ]
                
                break
        
        return img, instances

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


def rtmdet_drone_collate_fn(batch: List[Dict]) -> Dict:
    imgs = []
    img_metas = []
    gt_bboxes = []
    gt_labels = []
    gt_masks = []
    drone_images = []
    intrinsics = []
    extrinsics = []
    scene_ids = []
    scene_indices = []

    max_instances = max(len(item["gt_labels"]) for item in batch)

    for item in batch:
        imgs.append(item["img"])
        img_metas.append(item["img_metas"])
        scene_ids.append(item.get("scene_id", ""))
        scene_indices.append(item.get("scene_index", 0))

        bboxes = item["gt_bboxes"]
        labels = item["gt_labels"]
        masks = item["gt_masks"]

        num_inst = len(labels)
        if num_inst < max_instances:
            pad_size = max_instances - num_inst
            bboxes = F.pad(bboxes, (0, 0, 0, pad_size))
            labels = F.pad(labels, (0, pad_size), value=-1)
            masks = np.concatenate(
                [masks, np.zeros((pad_size, *item["img_metas"]["img_shape"]), dtype=np.uint8)],
                axis=0,
            )

        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
        gt_masks.append(masks)

        drone_images.append(item["drone_images"])
        intrinsics.append(item["intrinsics"])
        extrinsics.append(item["extrinsics"])

    return {
        "imgs": torch.stack(imgs, dim=0),
        "img_metas": img_metas,
        "gt_bboxes": gt_bboxes,
        "gt_labels": gt_labels,
        "gt_masks": gt_masks,
        "drone_images": torch.stack(drone_images, dim=0),
        "intrinsics": torch.stack(intrinsics, dim=0),
        "extrinsics": torch.stack(extrinsics, dim=0),
        "scene_ids": scene_ids,
        "scene_indices": torch.tensor(scene_indices, dtype=torch.long),
    }

