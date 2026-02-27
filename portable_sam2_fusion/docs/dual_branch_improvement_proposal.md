# 双分支融合模型改进方案

## 目录
- [问题分析](#问题分析)
- [改进方案](#改进方案)
  - [方案1: 空间感知注意力融合](#方案1-空间感知注意力融合)
  - [方案2: 多尺度特征金字塔融合](#方案2-多尺度特征金字塔融合)
  - [方案3: 训练策略改进](#方案3-训练策略改进)
  - [方案4: BEV分辨率提升](#方案4-bev分辨率提升)
  - [方案5: 损失函数增强](#方案5-损失函数增强)
- [实施优先级](#实施优先级)
- [预期效果](#预期效果)

---

## 问题分析

### 当前架构问题

#### 1. 融合方式过于简单

当前实现位于 `models/rsprompter_anchor_drone_guidance.py`:

```python
# DroneSemanticGuidance.forward()
context = self.pool(drone_bev).flatten(1)  # 全局池化，丢失空间信息

for x, gate_mlp in zip(feats, self.level_gates):
    x_context = self.pool(x).flatten(1)
    combined_context = torch.cat([context, x_context], dim=1)
    
    gate_offset = gate_mlp(combined_context).unsqueeze(-1).unsqueeze(-1)
    guided.append(x * (1.0 + self.gate_scale * gate_offset))  # gate_scale=0.1
```

**问题点**:
- `gate_scale=0.1` 太小，无人机信息对最终特征的影响有限
- 使用全局平均池化，**丢失了所有空间位置信息**
- 门控是逐通道的标量调制，没有像素级别的交互
- 卫星特征和 BEV 特征之间没有显式的空间对应关系

#### 2. 空间对齐缺失

| 特征 | 尺寸 | 说明 |
|------|------|------|
| 卫星特征 (FPN输出) | `(B, 256, 256, 256)` | 原图 1024x1024 的 1/4 |
| 无人机 BEV 特征 | `(B, 128, 80, 80)` | 100m x 100m 区域 |

**问题**: 两个特征图没有建立显式的空间对应关系，仅通过全局池化后的向量交互。

#### 3. 训练策略问题

- 无人机编码器学习率 `drone_lr_mult=0.5` 可能不足以充分学习
- 没有采用多阶段训练策略
- `gate_image_embeddings=False`，未利用 SAM 的 image_embeddings

#### 4. 当前性能基线

| 指标 | 单分支 | 双分支 | 提升 |
|------|--------|--------|------|
| bbox/mAP | 59.40% | 62.71% | +5.6% |
| segm/mAP | 51.68% | 58.06% | +12.3% |
| bbox/mAP_m | 37.06% | 46.23% | +24.7% |

---

## 改进方案

### 方案1: 空间感知注意力融合

**核心思想**: 将全局门控改为空间级别的注意力融合，保留空间位置信息。

#### 1.1 空间注意力融合模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialGuidance(nn.Module):
    """空间感知的无人机引导模块
    
    使用跨注意力机制让卫星特征关注无人机BEV特征的空间位置。
    """
    
    def __init__(
        self, 
        bev_dim: int, 
        sat_dim: int, 
        num_heads: int = 4,
        use_deformable: bool = False,
    ):
        super().__init__()
        self.bev_dim = bev_dim
        self.sat_dim = sat_dim
        
        # BEV 特征投影
        self.bev_proj = nn.Sequential(
            nn.Conv2d(bev_dim, sat_dim, 1, bias=False),
            nn.BatchNorm2d(sat_dim),
            nn.ReLU(inplace=True),
        )
        
        # 卫星特征投影
        self.sat_proj = nn.Sequential(
            nn.Conv2d(sat_dim, sat_dim, 1, bias=False),
            nn.BatchNorm2d(sat_dim),
            nn.ReLU(inplace=True),
        )
        
        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(
            sat_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        # 融合后的精炼
        self.refine = nn.Sequential(
            nn.Conv2d(sat_dim, sat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(sat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(sat_dim, sat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(sat_dim),
        )
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, sat_feat: Tensor, bev_feat: Tensor) -> Tensor:
        """
        Args:
            sat_feat: 卫星特征 (B, C_sat, H_sat, W_sat)
            bev_feat: 无人机BEV特征 (B, C_bev, H_bev, W_bev)
        
        Returns:
            融合后的特征 (B, C_sat, H_sat, W_sat)
        """
        B, C, H, W = sat_feat.shape
        
        # 1. 投影到相同维度
        bev_proj = self.bev_proj(bev_feat)  # (B, C_sat, H_bev, W_bev)
        
        # 2. 上采样 BEV 到卫星特征尺寸
        bev_upsampled = F.interpolate(
            bev_proj, size=(H, W), 
            mode='bilinear', align_corners=False
        )
        
        # 3. 展平为序列格式
        sat_flat = sat_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        bev_flat = bev_upsampled.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # 4. 跨注意力: 卫星特征作为 Query，BEV 作为 Key/Value
        attn_out, attn_weights = self.spatial_attn(
            query=sat_flat,
            key=bev_flat,
            value=bev_flat,
        )
        
        # 5. 恢复空间维度
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # 6. 精炼和残差连接
        refined = self.refine(attn_out)
        
        # 7. 可学习的加权融合
        alpha = torch.sigmoid(self.fusion_weight)
        output = alpha * sat_feat + (1 - alpha) * refined
        
        return output
```

#### 1.2 多层级空间融合

```python
class MultiLevelSpatialGuidance(nn.Module):
    """多层级空间感知融合"""
    
    def __init__(
        self,
        bev_dim: int,
        sat_channels: tuple = (256, 256, 256, 256, 256),
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_levels = len(sat_channels)
        
        # 每个层级一个空间融合模块
        self.spatial_guides = nn.ModuleList([
            SpatialGuidance(bev_dim, c, num_heads) 
            for c in sat_channels
        ])
        
        # 自顶向下的特征增强
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
            for c in sat_channels[:-1]
        ])
        
    def forward(
        self, 
        sat_feats: tuple, 
        bev_feat: Tensor
    ) -> tuple:
        """
        Args:
            sat_feats: 多尺度卫星特征元组
            bev_feat: BEV特征 (B, C_bev, H_bev, W_bev)
        
        Returns:
            融合后的多尺度特征元组
        """
        # 1. 各层级独立融合
        guided_feats = []
        for feat, guide in zip(sat_feats, self.spatial_guides):
            guided_feats.append(guide(feat, bev_feat))
        
        # 2. 自顶向下增强 (深层特征指导浅层)
        for i in range(self.num_levels - 1, 0, -1):
            upsampled = F.interpolate(
                guided_feats[i], 
                size=guided_feats[i-1].shape[-2:],
                mode='bilinear', 
                align_corners=False
            )
            guided_feats[i-1] = guided_feats[i-1] + self.td_convs[i-1](upsampled)
        
        return tuple(guided_feats)
```

---

### 方案2: 多尺度特征金字塔融合

**核心思想**: 构建无人机特征的FPN，与卫星FPN进行双向交互。

#### 2.1 BEV特征金字塔

```python
class BEVFeaturePyramid(nn.Module):
    """BEV特征金字塔网络"""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        num_levels: int = 5,
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # 自底向上路径
        self.bottom_up = nn.ModuleList()
        for i in range(num_levels):
            if i == 0:
                self.bottom_up.append(
                    nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.bottom_up.append(
                    nn.Sequential(
                        nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, 1) 
            for _ in range(num_levels)
        ])
        
    def forward(self, bev_feat: Tensor) -> list:
        """
        Args:
            bev_feat: 原始BEV特征 (B, C, H, W)
        
        Returns:
            多尺度BEV特征列表
        """
        feats = [bev_feat]
        
        # 自底向上构建金字塔
        for i, layer in enumerate(self.bottom_up):
            if i == 0:
                feats[0] = layer(bev_feat)
            else:
                downsampled = layer(feats[-1])
                feats.append(downsampled)
        
        # 横向连接
        pyramid = [l(f) for l, f in zip(self.lateral_convs, feats)]
        
        return pyramid


class BidirectionalFusion(nn.Module):
    """双向特征融合"""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        
        # 卫星 -> BEV 融合
        self.sat_to_bev = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(5)
        ])
        
        # BEV -> 卫星 融合
        self.bev_to_sat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(5)
        ])
        
    def forward(
        self, 
        sat_feats: tuple, 
        bev_feats: list
    ) -> tuple:
        """
        双向融合卫星和BEV特征
        """
        fused_sat = []
        fused_bev = []
        
        for i, (sat, bev) in enumerate(zip(sat_feats, bev_feats)):
            # 尺寸对齐
            if sat.shape[-2:] != bev.shape[-2:]:
                bev_aligned = F.interpolate(
                    bev, size=sat.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                bev_aligned = bev
            
            # 卫星 + BEV -> 增强卫星
            concat_sat = torch.cat([sat, bev_aligned], dim=1)
            fused_sat.append(self.bev_to_sat[i](concat_sat))
            
            # BEV + 卫星 -> 增强BEV
            sat_aligned = F.interpolate(
                sat, size=bev.shape[-2:],
                mode='bilinear', align_corners=False
            )
            concat_bev = torch.cat([bev, sat_aligned], dim=1)
            fused_bev.append(self.sat_to_bev[i](concat_bev))
        
        return tuple(fused_sat), fused_bev
```

---

### 方案3: 训练策略改进

#### 3.1 多阶段训练

| 阶段 | 训练内容 | 学习率配置 | Epochs | 说明 |
|------|----------|-----------|--------|------|
| Stage 1 | 冻结无人机编码器，训练融合模块和主网络 | `sat_backbone: 1e-4, sat_other: 2e-4, drone: 0` | 10 | 快速适应融合 |
| Stage 2 | 联合训练，无人机编码器低学习率 | `sat_backbone: 5e-5, sat_other: 1e-4, drone: 1e-5` | 20 | 精细调整 |
| Stage 3 | 全模型微调 | `all: 2e-5` | 10 | 最终收敛 |

#### 3.2 训练脚本修改

```bash
# scripts/start_rsprompter_fusion_v2.sh

# Stage 1: 冻结无人机编码器
EPOCHS_STAGE1=10
DRONE_LR_MULT_STAGE1=0
SAT_BACKBONE_LR_MULT_STAGE1=0.5
SAT_OTHER_LR_MULT_STAGE1=1.0

# Stage 2: 联合训练
EPOCHS_STAGE2=20
DRONE_LR_MULT_STAGE2=0.1
SAT_BACKBONE_LR_MULT_STAGE2=0.25
SAT_OTHER_LR_MULT_STAGE2=0.5

# Stage 3: 微调
EPOCHS_STAGE3=10
DRONE_LR_MULT_STAGE3=0.1
SAT_BACKBONE_LR_MULT_STAGE3=0.1
SAT_OTHER_LR_MULT_STAGE3=0.2
```

#### 3.3 渐进式解冻策略

```python
# train/train_rsprompter_fusion_v2.py

def configure_training_stages(model, epoch, total_epochs):
    """根据训练阶段调整模型参数的可训练性"""
    
    # Stage 1: 前 1/3 轮次
    if epoch < total_epochs // 3:
        # 冻结无人机编码器
        if hasattr(model, 'drone_encoder'):
            for param in model.drone_encoder.parameters():
                param.requires_grad = False
    
    # Stage 2: 中间 1/3 轮次
    elif epoch < 2 * total_epochs // 3:
        # 解冻无人机编码器的部分层
        if hasattr(model, 'drone_encoder'):
            # 只解冻最后几层
            for name, param in model.drone_encoder.named_parameters():
                if 'layers.3' in name or 'layers.2' in name:
                    param.requires_grad = True
    
    # Stage 3: 最后 1/3 轮次
    else:
        # 全部解冻
        for param in model.parameters():
            param.requires_grad = True
```

#### 3.4 配置文件修改

```python
# configs/rsprompter_anchor_satS_drone_guidance_v2.py

model = dict(
    type="RSPrompterAnchorDroneGuidance",
    enable_drone_branch=True,
    
    # ... 其他配置 ...
    
    drone_branch=dict(
        pretrained=True,  # 使用预训练权重
        freeze=False,
        dim=128,
        middle=[2, 2, 2, 2],
        scale=1.0,
        cross_view=dict(
            image_height=512,
            image_width=512,
            qkv_bias=False,
            heads=4,
            dim_head=32,
            skip=True,
        ),
        bev_embedding=dict(
            sigma=1.0,
            bev_height=160,  # 提升分辨率
            bev_width=160,
            h_meters=100.0,
            w_meters=100.0,
            offset=0.0,
            decoder_blocks=[2, 2],
            grid_refine_scale=0.5,
        ),
    ),
    
    guidance=dict(
        bev_dim=128,
        level_channels=[256, 256, 256, 256, 256],
        gate_image_embeddings=True,  # 启用 image_embeddings 门控
        gate_scale=0.3,  # 增大门控强度
        use_spatial_attention=True,  # 启用空间注意力
    ),
)
```

---

### 方案4: BEV分辨率提升

#### 4.1 当前配置 vs 改进配置

| 参数 | 当前值 | 改进值 | 影响 |
|------|--------|--------|------|
| bev_height | 80 | **160** | 空间分辨率提升 4x |
| bev_width | 80 | **160** | 空间分辨率提升 4x |
| h_meters | 100.0 | 100.0 | 初始化值（V_inv可学习） |
| w_meters | 100.0 | 100.0 | 初始化值（V_inv可学习） |
| init_scale | 0.0 | **0.1** | 添加随机扰动，增强探索 |

> **注意**: `h_meters` 和 `w_meters` 仅用于 `V_inv` 的初始化。`V_inv` 是可学习参数，会在训练过程中自适应调整几何映射关系。物理映射关系由模型隐式学习，不应假设固定的"每像素代表X米"。

#### 4.2 内存优化

提升分辨率会增加显存占用，可采用以下优化：

```python
# 使用梯度检查点
class CVTEncoderWithCheckpoint(nn.Module):
    def forward(self, batch):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, batch
            )
        return self._forward_impl(batch)
    
    def _forward_impl(self, batch):
        # 原始 forward 逻辑
        ...
```

---

### 方案5: 损失函数增强

#### 5.1 添加一致性损失

```python
class ConsistencyLoss(nn.Module):
    """卫星-BEV 特征一致性损失"""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        
    def forward(
        self, 
        sat_feat: Tensor, 
        bev_feat: Tensor
    ) -> Tensor:
        """
        计算卫星特征和BEV特征的一致性损失
        """
        # 投影到相同空间
        sat_embed = self.projector(sat_feat)
        bev_embed = self.projector(bev_feat)
        
        # L2 归一化
        sat_embed = F.normalize(sat_embed, dim=1)
        bev_embed = F.normalize(bev_embed, dim=1)
        
        # 余弦相似度损失
        similarity = torch.sum(sat_embed * bev_embed, dim=1)
        loss = 1 - similarity.mean()
        
        return loss
```

#### 5.2 添加对比学习损失

```python
class ContrastiveLoss(nn.Module):
    """跨视角对比学习损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        sat_feat: Tensor, 
        bev_feat: Tensor
    ) -> Tensor:
        """
        对比学习损失：同一场景的卫星-BEV特征应该相似
        """
        B = sat_feat.shape[0]
        
        # 全局池化
        sat_global = F.adaptive_avg_pool2d(sat_feat, 1).flatten(1)
        bev_global = F.adaptive_avg_pool2d(bev_feat, 1).flatten(1)
        
        # L2 归一化
        sat_global = F.normalize(sat_global, dim=1)
        bev_global = F.normalize(bev_global, dim=1)
        
        # 计算相似度矩阵
        logits = torch.matmul(sat_global, bev_global.T) / self.temperature
        
        # 对角线为正样本
        labels = torch.arange(B, device=sat_feat.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        
        return loss / 2
```

---

## 实施优先级

### 立即可做（无需改代码）

| 改进项 | 修改位置 | 预期效果 |
|--------|----------|----------|
| 启用 `gate_image_embeddings=True` | 配置文件 | +1-2% |
| 增大 `gate_scale=0.3` | 配置文件 | +1-2% |
| 提高 BEV 分辨率到 160x160 | 配置文件 | +1-2% |
| 调整学习率倍率 | 训练脚本 | +1-2% |

### 短期改进（1-2天）

| 改进项 | 实现难度 | 预期效果 |
|--------|----------|----------|
| 空间注意力融合模块 | 中等 | +3-5% |
| 多阶段训练策略 | 低 | +2-4% |
| 一致性损失 | 低 | +1-2% |

### 中期优化（1周）

| 改进项 | 实现难度 | 预期效果 |
|--------|----------|----------|
| 多尺度特征金字塔融合 | 中等 | +2-3% |
| 对比学习损失 | 中等 | +1-2% |
| 无人机编码器预训练 | 高 | +2-3% |

---

## 预期效果

### 单项改进预期

| 改进项 | 预期 mAP 提升 | 实现难度 | 开发时间 |
|--------|--------------|----------|----------|
| 配置调优 | +2-4% | 低 | 1小时 |
| 空间注意力融合 | +3-5% | 中等 | 1-2天 |
| 多阶段训练 | +2-4% | 低 | 0.5天 |
| BEV 分辨率提升 | +1-2% | 低 | 1小时 |
| 一致性损失 | +1-2% | 低 | 0.5天 |
| 多尺度金字塔融合 | +2-3% | 中等 | 2-3天 |
| 对比学习损失 | +1-2% | 中等 | 1天 |

### 综合改进预期

| 方案组合 | 预期 segm/mAP | 相比当前提升 |
|----------|--------------|--------------|
| 当前基线 | 58.06% | - |
| 配置调优 | 60-62% | +2-4% |
| + 空间注意力 | 64-66% | +6-8% |
| + 多阶段训练 | 66-68% | +8-10% |
| + 损失函数增强 | 68-70% | +10-12% |

---

## 参考文献

1. **Cross-View Attention**: "Cross-view Transformers for real-time Map-view Semantic Segmentation" (CVPR 2022)
2. **BEV Fusion**: "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's Eye View Representation" (ICRA 2023)
3. **Feature Pyramid Networks**: "Feature Pyramid Networks for Object Detection" (CVPR 2017)
4. **Deformable Attention**: "Deformable DETR: Deformable Transformers for End-to-End Object Detection" (ICLR 2021)

---

## 文件修改清单

### 需要修改的文件

| 文件路径 | 修改内容 |
|----------|----------|
| `models/rsprompter_anchor_drone_guidance.py` | 添加空间注意力融合模块 |
| `configs/rsprompter_anchor_satS_drone_guidance_v2.py` | 新建配置文件 |
| `train/train_rsprompter_fusion.py` | 添加多阶段训练逻辑 |
| `scripts/start_rsprompter_fusion_v2.sh` | 新建训练脚本 |
| `uav/cross_view_attention.py` | 可选：添加可变形注意力 |

### 需要新建的文件

| 文件路径 | 说明 |
|----------|------|
| `models/spatial_guidance.py` | 空间注意力融合模块 |
| `models/bev_pyramid.py` | BEV特征金字塔 |
| `models/losses.py` | 一致性损失和对比学习损失 |
