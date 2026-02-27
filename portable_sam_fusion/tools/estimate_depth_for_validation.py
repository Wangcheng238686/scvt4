"""
无人机图像深度/高度估计验证脚本
使用 Depth Anything V2 官方实现
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"


def get_depth_model(device="cuda"):
    """加载 Depth Anything V2 模型"""
    weights_path = PRETRAINED_DIR / "depth_anything_v2_vits.pth"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"未找到权重: {weights_path}")
    
    print(f"加载模型: {weights_path}")
    
    repo_dir = PROJECT_ROOT / "depth_anything_v2"
    
    try:
        sys.path.insert(0, str(repo_dir))
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("  模型加载成功!")
        return model
        
    except Exception as e:
        print(f"  官方实现加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def estimate_depth(model, image_path, device="cuda"):
    """估计深度"""
    import cv2
    
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    depth = model.infer_image(image_bgr)
    
    return depth, image


def depth_to_height_map(depth, max_height=50.0, method="inverse"):
    """将相对深度转换为高度图"""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-7)
    
    if method == "inverse":
        height_map = (1 - depth_norm) * max_height
    else:
        height_map = depth_norm * max_height
    
    return height_map


def detect_building_boundaries(height_map):
    """检测建筑物边界"""
    from scipy import ndimage
    
    sobel_x = ndimage.sobel(height_map, axis=1)
    sobel_y = ndimage.sobel(height_map, axis=0)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    
    boundary = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-7)
    return boundary, gradient


def visualize_results(image, depth, height_map, boundary, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("无人机图像", fontsize=12)
    axes[0, 0].axis("off")
    
    im1 = axes[0, 1].imshow(depth, cmap="plasma")
    axes[0, 1].set_title("深度图 (相对值)", fontsize=12)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(height_map, cmap="terrain")
    axes[0, 2].set_title("高度图 (米)", fontsize=12)
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    im3 = axes[1, 0].imshow(boundary, cmap="hot")
    axes[1, 0].set_title("建筑物边界检测\n(高度梯度)", fontsize=12)
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(boundary, cmap="hot", alpha=0.5)
    axes[1, 1].set_title("边界叠加", fontsize=12)
    axes[1, 1].axis("off")
    
    height_threshold = height_map < np.percentile(height_map, 30)
    axes[1, 2].imshow(height_threshold, cmap="gray")
    axes[1, 2].set_title("建筑物区域\n(高度阈值)", fontsize=12)
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  保存: {save_path}")
    
    plt.close()
    return height_threshold


def process_scene(scene_dir, output_dir, model, device="cuda", max_images=None):
    """处理单个场景"""
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(scene_dir.glob(f"*{ext}")))
    
    image_files = sorted(set(image_files), key=lambda x: x.name)
    
    if len(image_files) == 0:
        print(f"未找到图像: {scene_dir}")
        return None
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n处理场景: {scene_dir.name}")
    print(f"图像数量: {len(image_files)}")
    
    all_heights = []
    all_boundaries = []
    all_depths = []
    
    for i, image_path in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] {image_path.name}", end="")
        
        try:
            depth, image = estimate_depth(model, str(image_path), device)
            height_map = depth_to_height_map(depth)
            boundary, gradient = detect_building_boundaries(height_map)
            
            save_path = output_dir / f"{image_path.stem}_vis.png"
            height_threshold = visualize_results(image, depth, height_map, boundary, str(save_path))
            
            np.save(output_dir / f"{image_path.stem}_depth.npy", depth)
            np.save(output_dir / f"{image_path.stem}_height.npy", height_map)
            np.save(output_dir / f"{image_path.stem}_boundary.npy", boundary)
            
            all_heights.append(height_map)
            all_boundaries.append(boundary)
            all_depths.append(depth)
            print(" ✓")
            
        except Exception as e:
            print(f" ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_heights) > 0:
        avg_height = np.mean(all_heights, axis=0)
        avg_boundary = np.mean(all_boundaries, axis=0)
        avg_depth = np.mean(all_depths, axis=0)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = axes[0].imshow(avg_depth, cmap="plasma")
        axes[0].set_title("平均深度图 (多视角融合)", fontsize=12)
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        im2 = axes[1].imshow(avg_height, cmap="terrain")
        axes[1].set_title("平均高度图 (多视角融合)", fontsize=12)
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        im3 = axes[2].imshow(avg_boundary, cmap="hot")
        axes[2].set_title("平均边界图 (多视角融合)", fontsize=12)
        axes[2].axis("off")
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_dir / "scene_fusion.png", dpi=150, bbox_inches="tight")
        print(f"\n场景融合结果: {output_dir / 'scene_fusion.png'}")
        plt.close()
        
        np.save(output_dir / "scene_avg_height.npy", avg_height)
        np.save(output_dir / "scene_avg_boundary.npy", avg_boundary)
        np.save(output_dir / "scene_avg_depth.npy", avg_depth)
    
    return all_heights


def count_images_in_dir(directory):
    """统计目录中的图像数量"""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    count = 0
    for ext in image_extensions:
        count += len(list(directory.glob(ext)))
    return count


def list_available_scenes(drone_data_root):
    """列出所有可用场景"""
    drone_data_root = Path(drone_data_root)
    
    if not drone_data_root.exists():
        print(f"数据目录不存在: {drone_data_root}")
        return []
    
    scenes = []
    for item in sorted(drone_data_root.iterdir()):
        if item.is_dir():
            cam_params = item / "colmap_result" / "camera_params.json"
            if cam_params.exists():
                image_count = count_images_in_dir(item)
                scenes.append({
                    "scene_id": item.name,
                    "path": item,
                    "image_count": image_count
                })
    
    return scenes


def main():
    parser = argparse.ArgumentParser(description="无人机图像深度/高度估计验证")
    parser.add_argument("--drone-data-root", type=str, 
                        default="/home/wangcheng/data/unversity-big-after-without-negative/university-d-train-selected-5",
                        help="无人机数据根目录")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="指定场景ID")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--max-images", type=int, default=None,
                        help="每个场景最多处理的图像数")
    parser.add_argument("--list-scenes", action="store_true",
                        help="仅列出所有可用场景")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = "cpu"
    
    print("=" * 60)
    print("无人机图像深度/高度估计验证")
    print("=" * 60)
    
    scenes = list_available_scenes(args.drone_data_root)
    
    if args.list_scenes or args.scene_id is None:
        print(f"\n找到 {len(scenes)} 个场景:")
        print("-" * 60)
        for i, s in enumerate(scenes[:20]):
            print(f"  {i+1}. {s['scene_id']}: {s['image_count']} 张图像")
        if len(scenes) > 20:
            print(f"  ... 还有 {len(scenes) - 20} 个场景")
        print("-" * 60)
        print(f"\n使用方法:")
        print(f"  python tools/estimate_depth_for_validation.py --scene-id <场景ID>")
        print(f"  python tools/estimate_depth_for_validation.py --scene-id <场景ID> --max-images 5")
        return
    
    target_scene = None
    for s in scenes:
        if s["scene_id"] == args.scene_id:
            target_scene = s
            break
    
    if target_scene is None:
        print(f"未找到场景: {args.scene_id}")
        return
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "depth_output" / args.scene_id
    
    try:
        model = get_depth_model(args.device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    process_scene(
        target_scene["path"],
        output_dir,
        model,
        args.device,
        args.max_images
    )
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
