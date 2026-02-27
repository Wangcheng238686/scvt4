#!/usr/bin/env python3
"""
分析训练日志文件，提取loss数据并生成可视化分析
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log(log_path):
    """解析日志文件，提取每个epoch的loss数据"""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"日志文件不存在: {log_path}")
        return None
    
    epochs = []
    loss_data = {}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 匹配detailed losses的行
    pattern = re.compile(r'Epoch (\d+) detailed losses: (.*)')
    total_loss_pattern = re.compile(r'Epoch (\d+)/\d+ \| train=([\d.]+)')
    
    for line in lines:
        # 解析detailed losses
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss_str = match.group(2)
            
            if epoch not in loss_data:
                loss_data[epoch] = {}
            
            # 解析各个loss项
            loss_items = loss_str.split(', ')
            for item in loss_items:
                if '=' in item:
                    key, value = item.split('=')
                    try:
                        loss_data[epoch][key] = float(value)
                    except ValueError:
                        pass
        
        # 解析total loss
        total_match = total_loss_pattern.search(line)
        if total_match:
            epoch = int(total_match.group(1))
            total_loss = float(total_match.group(2))
            if epoch not in loss_data:
                loss_data[epoch] = {}
            loss_data[epoch]['total_loss'] = total_loss
    
    # 按epoch排序
    sorted_epochs = sorted(loss_data.keys())
    return {e: loss_data[e] for e in sorted_epochs}

def analyze_loss_trends(loss_data):
    """分析loss下降趋势"""
    epochs = list(loss_data.keys())
    
    # 收集所有loss类型
    all_loss_types = set()
    for data in loss_data.values():
        all_loss_types.update(data.keys())
    all_loss_types = sorted(list(all_loss_types))
    
    print("=" * 80)
    print("训练日志分析报告")
    print("=" * 80)
    print(f"\n训练轮数: {len(epochs)} 轮 (Epoch {min(epochs)} - {max(epochs)})")
    
    # 总loss分析
    if 'total_loss' in all_loss_types:
        total_losses = [loss_data[e]['total_loss'] for e in epochs]
        first_total = total_losses[0]
        last_total = total_losses[-1]
        reduction = (first_total - last_total) / first_total * 100
        
        print(f"\n总Loss分析:")
        print(f"  初始总Loss: {first_total:.6f}")
        print(f"  最终总Loss: {last_total:.6f}")
        print(f"  总下降幅度: {reduction:.2f}%")
    
    # 各loss项分析
    print(f"\n各Loss项详细分析:")
    print("-" * 80)
    
    for loss_type in all_loss_types:
        if loss_type == 'total_loss' or loss_type == 'acc':
            continue
        
        losses = []
        valid_epochs = []
        for e in epochs:
            if loss_type in loss_data[e]:
                losses.append(loss_data[e][loss_type])
                valid_epochs.append(e)
        
        if len(losses) >= 2:
            first = losses[0]
            last = losses[-1]
            min_val = min(losses)
            max_val = max(losses)
            
            if first > 0:
                reduction = (first - last) / first * 100
            else:
                reduction = 0.0
            
            print(f"\n{loss_type}:")
            print(f"  初始值: {first:.6f}")
            print(f"  最终值: {last:.6f}")
            print(f"  最小值: {min_val:.6f} (Epoch {valid_epochs[losses.index(min_val)]})")
            print(f"  最大值: {max_val:.6f} (Epoch {valid_epochs[losses.index(max_val)]})")
            print(f"  下降幅度: {reduction:.2f}%")
    
    # 准确率分析
    if 'acc' in all_loss_types:
        accs = [loss_data[e]['acc'] for e in epochs if 'acc' in loss_data[e]]
        if len(accs) >= 2:
            print(f"\n准确率分析:")
            print(f"  初始准确率: {accs[0]:.2f}%")
            print(f"  最终准确率: {accs[-1]:.2f}%")
            print(f"  提升幅度: {accs[-1] - accs[0]:.2f}%")
    
    print("\n" + "=" * 80)
    print("Loss权重优化建议")
    print("=" * 80)
    
    # 分析各loss的相对大小
    print("\n1. Loss相对大小分析（最终Epoch）:")
    last_epoch = max(epochs)
    last_data = loss_data[last_epoch]
    
    detection_losses = ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'loss_mask']
    auxiliary_losses = ['loss_contrastive', 'loss_consistency', 'loss_spatial_align']
    
    det_total = 0
    for loss in detection_losses:
        if loss in last_data:
            det_total += last_data[loss]
    
    aux_total = 0
    for loss in auxiliary_losses:
        if loss in last_data:
            aux_total += last_data[loss]
    
    print(f"  检测任务总Loss: {det_total:.6f}")
    print(f"  辅助任务总Loss: {aux_total:.6f}")
    print(f"  比例 (辅助/检测): {aux_total/det_total*100:.2f}%" if det_total > 0 else "")
    
    print("\n2. 具体建议:")
    
    # 对比loss下降速度
    if 'loss_contrastive' in all_loss_types:
        contrastive_losses = [loss_data[e]['loss_contrastive'] for e in epochs if 'loss_contrastive' in loss_data[e]]
        if len(contrastive_losses) >= 2:
            contrastive_reduction = (contrastive_losses[0] - contrastive_losses[-1]) / contrastive_losses[0] * 100 if contrastive_losses[0] > 0 else 0
            print(f"   - loss_contrastive: 下降 {contrastive_reduction:.1f}%")
            if contrastive_losses[-1] < 0.005:
                print(f"     → 当前值很小 ({contrastive_losses[-1]:.4f})，可以考虑增大权重")
    
    if 'loss_consistency' in all_loss_types:
        consistency_losses = [loss_data[e]['loss_consistency'] for e in epochs if 'loss_consistency' in loss_data[e]]
        if len(consistency_losses) >= 2:
            consistency_reduction = (consistency_losses[0] - consistency_losses[-1]) / consistency_losses[0] * 100 if consistency_losses[0] > 0 else 0
            print(f"   - loss_consistency: 下降 {consistency_reduction:.1f}%")
            if consistency_losses[-1] < 0.001:
                print(f"     → 当前值非常小 ({consistency_losses[-1]:.4f})，可以考虑增大权重")
    
    if 'loss_spatial_align' in all_loss_types:
        spatial_losses = [loss_data[e]['loss_spatial_align'] for e in epochs if 'loss_spatial_align' in loss_data[e]]
        if len(spatial_losses) >= 2:
            spatial_reduction = (spatial_losses[0] - spatial_losses[-1]) / spatial_losses[0] * 100 if spatial_losses[0] > 0 else 0
            print(f"   - loss_spatial_align: 下降 {spatial_reduction:.1f}%")
            if spatial_losses[-1] < 0.01:
                print(f"     → 当前值较小 ({spatial_losses[-1]:.4f})，可以考虑增大权重")
    
    print("\n3. 推荐权重调整方案:")
    print("   方案1（保守）:")
    print("     contrastive_weight: 0.1 → 0.2")
    print("     consistency_weight: 0.1 → 0.3")
    print("     spatial_align_loss_weight: 0.1 → 0.2")
    print("\n   方案2（激进）:")
    print("     contrastive_weight: 0.1 → 0.5")
    print("     consistency_weight: 0.1 → 0.5")
    print("     spatial_align_loss_weight: 0.1 → 0.3")
    
    return epochs, loss_data, all_loss_types

def plot_loss_curves(epochs, loss_data, all_loss_types, output_dir='./'):
    """绘制loss曲线图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    # 1. 总Loss曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    if 'total_loss' in all_loss_types:
        total_losses = [loss_data[e]['total_loss'] for e in epochs]
        ax.plot(epochs, total_losses, 'b-', linewidth=2, label='Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Training Loss')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'total_loss.png', dpi=150)
        plt.close()
    
    # 2. 检测Loss曲线
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    detection_losses = ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'loss_mask']
    colors = ['r', 'g', 'b', 'm', 'c']
    
    for idx, loss_type in enumerate(detection_losses):
        if loss_type in all_loss_types:
            losses = [loss_data[e].get(loss_type, np.nan) for e in epochs]
            axes[idx].plot(epochs, losses, f'{colors[idx]}-', linewidth=2, label=loss_type)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].set_title(loss_type)
            axes[idx].legend()
            axes[idx].grid(True)
    
    # 总检测Loss
    det_total = []
    for e in epochs:
        total = 0
        for loss_type in detection_losses:
            if loss_type in loss_data[e]:
                total += loss_data[e][loss_type]
        det_total.append(total)
    
    axes[5].plot(epochs, det_total, 'k-', linewidth=2, label='Detection Total')
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('Loss')
    axes[5].set_title('Detection Loss Total')
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_losses.png', dpi=150)
    plt.close()
    
    # 3. 辅助Loss曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    auxiliary_losses = ['loss_contrastive', 'loss_consistency', 'loss_spatial_align']
    colors = ['r', 'g', 'b']
    
    for idx, loss_type in enumerate(auxiliary_losses):
        if loss_type in all_loss_types:
            losses = [loss_data[e].get(loss_type, np.nan) for e in epochs]
            axes[idx].plot(epochs, losses, f'{colors[idx]}-', linewidth=2, label=loss_type)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].set_title(loss_type)
            axes[idx].legend()
            axes[idx].grid(True)
    
    # 总辅助Loss
    aux_total = []
    for e in epochs:
        total = 0
        for loss_type in auxiliary_losses:
            if loss_type in loss_data[e]:
                total += loss_data[e][loss_type]
        aux_total.append(total)
    
    axes[3].plot(epochs, aux_total, 'k-', linewidth=2, label='Auxiliary Total')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Loss')
    axes[3].set_title('Auxiliary Loss Total')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auxiliary_losses.png', dpi=150)
    plt.close()
    
    # 4. 准确率曲线
    if 'acc' in all_loss_types:
        fig, ax = plt.subplots(figsize=(12, 6))
        accs = [loss_data[e].get('acc', np.nan) for e in epochs]
        ax.plot(epochs, accs, 'g-', linewidth=2, label='Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Classification Accuracy')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy.png', dpi=150)
        plt.close()
    
    print(f"\n图表已保存至: {output_dir.absolute()}")

def main():
    log_path = "/home/wangcheng/project/SCVT3/portable_sam_fusion/logs/rsprompter_fusion_depth_aware_v2_20260220_214615.log"
    output_dir = "/home/wangcheng/project/SCVT3/portable_sam_fusion/logs/analysis"
    
    print("正在解析日志文件...")
    loss_data = parse_log(log_path)
    
    if loss_data:
        epochs = list(loss_data.keys())
        print(f"成功解析 {len(epochs)} 个epoch的数据")
        
        epochs, loss_data, all_loss_types = analyze_loss_trends(loss_data)
        
        print("\n正在生成loss曲线图...")
        plot_loss_curves(epochs, loss_data, all_loss_types, output_dir)

if __name__ == "__main__":
    main()
