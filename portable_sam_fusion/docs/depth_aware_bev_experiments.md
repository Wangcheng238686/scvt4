# 深度感知BEV融合实验记录

## 实验概述

本实验探索将单目深度估计集成到无人机BEV构建流程中，以改进卫星视角下的建筑物实例分割效果。

---

## 版本对比

### 基线版本（无深度信息）

- **配置文件**: `configs/rsprompter_anchor_satS_drone_guidance_v1.py`
- **训练脚本**: `scripts/start_rsprompter_fusion_v10_satS_v1.sh`
- **推理脚本**: `scripts/inference_rsprompter_fusion_v1.sh`
- **结果目录**: `inference_results/metrics_20260219_095002/`

**架构特点**:
- 标准BEV构建（2D平面假设）
- ContrastiveSpatialFusion 融合模块
- 无深度估计模块

---

### V1: 深度感知BEV + 高度引导融合

- **配置文件**: `configs/rsprompter_anchor_satS_drone_depth_aware_v1.py`
- **训练脚本**: `scripts/start_rsprompter_fusion_depth_aware_v1.sh`
- **推理脚本**: `scripts/inference_rsprompter_fusion_depth_aware_v1.sh`

**新增模块**:
1. `uav/depth_aware_bev.py`
   - `LightweightDepthEncoder`: 基于Depth Anything V2的轻量级深度编码器
   - `DepthAwareCrossViewAttention`: 深度感知的跨视角注意力
   - `DepthAwareCVTEncoder`: 完整的深度感知BEV编码器

2. `models/height_guided_fusion.py`
   - `HeightGuidedSpatialFusion`: 高度引导的空间融合
   - `MultiLevelHeightGuidedFusion`: 多尺度高度引导融合

**架构特点**:
- 深度估计集成到BEV构建流程
- 高度图引导卫星-BEV特征融合
- 建筑物区域获得更高注意力权重
- DINOv2 backbone冻结

---

### V2: 多视角深度一致性增强版

- **配置文件**: `configs/rsprompter_anchor_satS_drone_depth_aware_v2.py`
- **训练脚本**: `scripts/start_rsprompter_fusion_depth_aware_v2.sh`
- **推理脚本**: `scripts/inference_rsprompter_fusion_depth_aware_v2.sh`
- **结果目录**: `inference_results/metrics_depth_aware_v2_20260221_101417/`

**新增模块**:
- `uav/multiview_depth_consistency.py`
  - `MultiViewDepthConsistency`: 多视角深度一致性计算
  - `DepthWeightedCrossViewAttention`: 深度置信度加权注意力
  - `EnhancedDepthAwareBEV`: 增强版深度感知BEV

**架构特点**:
- 包含V1所有功能
- 多视角深度置信度估计
- 深度一致性正则化损失
- 深度置信度影响注意力权重

---

## 实验结果对比

### 检测指标 (bbox)

| 指标 | 基线版本 | V2版本 | 变化 |
|------|---------|--------|------|
| mAP | 0.6577 | 0.6613 | +0.54% |
| mAP_50 | 0.8853 | 0.8840 | -0.15% |
| mAP_75 | 0.7705 | 0.7509 | -2.54% |
| **mAP_m** | 0.4123 | **0.4774** | **+15.8%** |
| mAP_l | 0.6812 | 0.6800 | -0.18% |
| AR@100 | 0.1995 | 0.2061 | +3.3% |
| AR@300 | 0.7323 | 0.7167 | -2.1% |
| AR@1000 | 0.7407 | 0.7231 | -2.4% |

### 分割指标 (segm)

| 指标 | 基线版本 | V2版本 | 变化 |
|------|---------|--------|------|
| mAP | 0.6674 | 0.6463 | -3.2% |
| mAP_50 | 0.8684 | 0.8445 | -2.75% |
| mAP_75 | 0.7403 | 0.7090 | -4.23% |
| **mAP_m** | 0.3402 | **0.3770** | **+10.8%** |
| mAP_l | 0.6985 | 0.6738 | -3.54% |
| AR@100 | 0.2099 | 0.2114 | +0.7% |
| AR@300 | 0.7470 | 0.7128 | -4.6% |
| AR@1000 | 0.7515 | 0.7138 | -5.0% |

### 损失值对比

| 损失项 | 基线版本 | V2版本 |
|--------|---------|--------|
| avg_total | 0.3724 | 0.3914 |
| loss_rpn_cls | 0.0257 | 0.0383 |
| loss_rpn_bbox | 0.0036 | 0.0039 |
| loss_cls | 0.2058 | 0.2094 |
| acc | 93.66% | 94.71% |
| loss_bbox | 0.0928 | 0.1000 |
| loss_mask | 0.0379 | 0.0391 |
| loss_contrastive | 0.0 | 0.0 |
| loss_consistency | 0.0015 | 0.00007 |
| loss_spatial_align | 0.0051 | 0.0006 |

---

## 结论与分析

### 正面效果

1. **中等目标显著提升**
   - bbox/mAP_m 提升 15.8%
   - segm/mAP_m 提升 10.8%
   - 深度信息对中等尺度建筑物有明显帮助

2. **分类准确率提升**
   - acc 从 93.66% 提升到 94.71%

### 负面效果

1. **大目标分割略有下降**
   - segm/mAP_l 下降 3.5%
   - 可能因为深度模块增加了模型复杂度

2. **总体分割mAP下降**
   - segm/mAP 下降 3.2%
   - 需要进一步调优

### 可能原因

1. **训练不充分**: 深度模块增加了大量参数，可能需要更多epoch
2. **深度估计质量**: 预训练模型对无人机俯视视角的适应性有限
3. **超参数未调优**: 一致性损失权重、高度门控权重等需要调整
4. **V2版本过于复杂**: 多视角一致性模块可能增加了不必要的复杂度

---

## 后续优化建议

### 1. 延长训练时间
```bash
EPOCHS=100 ./scripts/start_rsprompter_fusion_depth_aware_v2.sh
```

### 2. 调整损失权重
```python
depth_encoder=dict(
    consistency_weight=0.05,  # 降低一致性损失权重
)
spatial_fusion_cfg=dict(
    height_loss_weight=0.005,  # 降低高度损失权重
)
```

### 3. 微调深度编码器
```python
depth_encoder=dict(
    freeze=False,  # 解冻深度编码器
)
```

### 4. 对比V1版本
V1版本（无多视角一致性）可能更适合当前数据集规模

---

## 文件结构

```
portable_sam_fusion/
├── uav/
│   ├── depth_aware_bev.py          # 深度感知BEV模块
│   └── multiview_depth_consistency.py  # 多视角深度一致性
├── models/
│   └── height_guided_fusion.py     # 高度引导融合模块
├── configs/
│   ├── rsprompter_anchor_satS_drone_guidance_v1.py    # 基线配置
│   ├── rsprompter_anchor_satS_drone_depth_aware_v1.py # V1配置
│   └── rsprompter_anchor_satS_drone_depth_aware_v2.py # V2配置
├── scripts/
│   ├── start_rsprompter_fusion_v10_satS_v1.sh         # 基线训练
│   ├── start_rsprompter_fusion_depth_aware_v1.sh      # V1训练
│   ├── start_rsprompter_fusion_depth_aware_v2.sh      # V2训练
│   ├── inference_rsprompter_fusion_v1.sh              # 基线推理
│   ├── inference_rsprompter_fusion_depth_aware_v1.sh  # V1推理
│   └── inference_rsprompter_fusion_depth_aware_v2.sh  # V2推理
└── inference_results/
    ├── metrics_20260219_095002/              # 基线结果
    └── metrics_depth_aware_v2_20260221_101417/  # V2结果
```

---

## 实验时间线

| 日期 | 版本 | 描述 |
|------|------|------|
| 2026-02-19 | 基线版本 | 无深度信息的融合模型 |
| 2026-02-21 | V2版本 | 深度感知BEV + 多视角一致性 |
