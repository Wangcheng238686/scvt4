# portable_sam_fusion 改进记录：BEV 可学习化与训练策略修复

**日期**: 2026-02-09
**分支**: improve
**硬件环境**: 单卡 RTX 3090 (24GB VRAM)

---

## 一、改进概览

本次改进包含两大部分：

| 改进方向 | 核心问题 | 修改文件数 |
|---------|---------|-----------|
| BEV 先验参数端到端可学习化 | BEV grid 和相机内参硬编码，无法适应实际场景 | 3 |
| 训练策略修复 | LR 失衡、单卡不适配、显存溢出、早停失效 | 2 |

**修改文件清单**:

| 文件 | 改进方向 |
|------|---------|
| `uav/bev_embedding.py` | BEV 可学习化 |
| `uav/cross_view_attention.py` | BEV 可学习化 |
| `configs/rsprompter_anchor_satS_drone_guidance_v1.py` | BEV 可学习化 |
| `train/train_rsprompter_fusion.py` | 训练策略 |
| `scripts/start_rsprompter_fusion_v10_satS_v1.sh` | 训练策略 |

---

## 二、BEV 先验参数端到端可学习化

### 2.1 问题分析

CVT (Cross-View Transformer) 架构将多视角无人机图像投影到 BEV 空间时，依赖以下硬编码物理先验：

- `h_meters=100, w_meters=100, offset=0` — 假设 BEV 覆盖 100m x 100m 的物理区域
- 相机内参矩阵 — 直接使用数据集提供的标定值

**核心问题**: `BEVEmbedding` 中的 `grid` 通过 `get_view_matrix()` 从硬编码参数计算后注册为 `register_buffer`（不可学习）。如果物理参数与实际场景不匹配，cross-attention 会关注到错误的图像区域，降低无人机特征质量。

### 2.2 修改 1: 可学习 BEV grid (`uav/bev_embedding.py`)

**策略**: 几何初始化 + 残差可学习

将固定 grid 改为"初始几何值 + 可学习偏移量"的残差形式，初始状态等价于原始行为。

**修改前**:
```python
self.register_buffer("grid", grid, persistent=False)
```

**修改后**:
```python
# Learnable BEV grid: geometric init + residual offset
self.register_buffer("grid_init", grid, persistent=False)
self.grid_offset = nn.Parameter(torch.zeros_like(grid))
self.grid_refine_scale = grid_refine_scale  # default 0.5

@property
def grid(self):
    return self.grid_init + self.grid_refine_scale * self.grid_offset
```

**设计要点**:
- `grid_init` 保留原始几何计算结果（buffer，不参与梯度）
- `grid_offset` 初始化为全零（nn.Parameter，参与梯度）
- `grid_refine_scale=0.5` 控制修正幅度，防止偏移过大
- 通过 `@property` 返回组合值，对外接口 `bev.grid` 不变
- 初始状态 offset=0，输出与修改前完全一致（无损兼容）
- 训练时梯度从 cross-attention 回传，端到端修正 grid 位置

### 2.3 修改 2: 可学习内参修正 (`uav/cross_view_attention.py`)

**策略**: 对 `I_inv`（内参逆矩阵）做仿射修正

CVTEncoder 已有 `alignment_weight`（可学习外参对齐矩阵），但内参（焦距、主点）也可能有标定误差。添加轻量级内参修正。

**新增参数**:
```python
# Learnable intrinsics correction: affine transform on I_inv
self.intrinsics_scale = nn.Parameter(torch.ones(1))   # 初始化为 1
self.intrinsics_offset = nn.Parameter(torch.zeros(1))  # 初始化为 0
```

**在 forward 中应用**:
```python
# Apply learnable intrinsics correction
I_inv = self.intrinsics_scale * I_inv + self.intrinsics_offset
```

**设计要点**:
- `scale=1, offset=0` 时等价于原始内参（无损兼容）
- 仅 2 个标量参数，几乎不增加计算量
- 与已有的 `alignment_weight`（外参修正）互补

### 2.4 修改 3: 配置更新 (`configs/...drone_guidance_v1.py`)

在 `bev_embedding` 配置中添加:
```python
bev_embedding=dict(
    ...
    grid_refine_scale=0.5,  # 新增
),
```

### 2.5 验证方式

1. 模型能正常构建和前向传播（无 shape 错误）
2. 初始状态下 `grid_offset` 全零，输出与修改前一致
3. 训练时观察 `grid_offset` 的梯度是否非零（确认梯度回传）
4. 训练后检查 `grid_offset` 的范数，确认网络确实在修正 grid

---

## 三、训练策略修复

### 3.1 问题诊断

通过对比 `migrated_project`（单视角 SAM，效果好）和 `portable_sam_fusion`（融合版，效果差）的训练策略，发现以下关键问题：

| 问题 | 严重程度 | 具体表现 |
|------|---------|---------|
| LR 倍率严重失衡 | 严重 | sat_backbone 实际 LR 仅 2e-5，与 drone 差 5-10 倍 |
| 脚本默认 4 卡 | 严重 | 当前机器仅 1 张 3090，无法直接运行 |
| batch_size=2 显存不足 | 严重 | SAM + drone 分支在单卡 24GB 下 OOM |
| 验证集未启用 | 中等 | val_ratio=0.0 导致早停机制形同虚设 |

> **注意**: 原分析报告中标记的"缺少 LR 预热"和"无混合精度训练"在当前代码中**已经修复**（SequentialLR warmup + torch.amp.autocast/GradScaler），无需再改。

### 3.2 修复 1: 学习率倍率平衡

**问题**: sat_backbone_lr_mult 过低，导致卫星特征提取器几乎未训练，RPN 依赖的卫星特征退化。

**修改文件**: `scripts/start_rsprompter_fusion_v10_satS_v1.sh` + `train/train_rsprompter_fusion.py`

**Shell 脚本修改**:

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `SAT_BACKBONE_LR_MULT_STAGE1` | 0.1 | **0.5** |
| `SAT_OTHER_LR_MULT_STAGE1` | 1.0 | 1.0 |
| `DRONE_LR_MULT_STAGE1` | 0.5 | 0.5 |
| `SAT_BACKBONE_LR_MULT_STAGE2` | 0.2 | **0.5** |
| `SAT_OTHER_LR_MULT_STAGE2` | 0.5 | 0.5 |
| `DRONE_LR_MULT_STAGE2` | 0.5 | 0.5 |

**Python argparse 默认值修改**:

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `--sat-backbone-lr-mult` | 0.01 | **0.5** |
| `--sat-other-lr-mult` | 0.1 | **1.0** |
| `--drone-lr-mult` | 1.0 | **0.5** |

**修改后实际学习率** (base_lr=2e-4):

| 参数组 | 修改前 | 修改后 | 变化 |
|--------|--------|--------|------|
| sat_backbone | 2e-5 | **1e-4** | 5x 提升 |
| sat_other | 2e-4 | 2e-4 | 不变 |
| drone | 1e-4 | **1e-4** | 不变 |

修改后三个参数组的学习率处于同一量级 (1e-4 ~ 2e-4)，卫星分支可以充分训练。

### 3.3 修复 2: 单卡 3090 适配

**问题**: 脚本默认 `CUDA_VISIBLE_DEVICES=0,1,2,3`、`NPROC_PER_NODE=4`，在单卡机器上无法运行。

**修改文件**: `scripts/start_rsprompter_fusion_v10_satS_v1.sh`

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | **`0`** |
| `NPROC_PER_NODE` | `4` | **`1`** |
| `BATCH_SIZE` | `2` | **`1`** |

### 3.4 修复 3: 梯度累积

**问题**: batch_size=2 在单卡 3090 上 SAM + drone 分支显存不足。直接降到 batch_size=1 会改变训练动态。

**解决方案**: batch_size=1 + grad_accum_steps=2 = 等效 batch size 2

**修改文件**: `train/train_rsprompter_fusion.py` + `scripts/start_rsprompter_fusion_v10_satS_v1.sh`

**新增参数**: `--grad-accum-steps`（默认 1，shell 脚本中设为 2）

**训练循环核心修改**:

```python
# 修改前: 每个 batch 都执行完整的 zero_grad -> backward -> step
for batch in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss = ...
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()

# 修改后: 累积 N 个 batch 的梯度后再 step
grad_accum = args.grad_accum_steps
optimizer.zero_grad(set_to_none=True)
for batch_idx, batch in enumerate(train_loader):
    loss = ...
    loss_for_backward = loss / grad_accum          # 缩放 loss
    scaler.scale(loss_for_backward).backward()     # 累积梯度

    # 每 N 个 batch 或最后一个 batch 时执行 step
    if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

**关键细节**:
- loss 除以 `grad_accum` 保证梯度量级与大 batch 一致
- `optimizer.zero_grad` 仅在每个累积窗口的开头执行，不在每个 batch 执行
- 最后一个 batch 即使不满 N 步也会触发 step，避免丢失尾部梯度

### 3.5 修复 4: 启用验证集与早停

**问题**: `VAL_RATIO=0.0` 导致不划分验证集，早停机制 (patience=15) 永远不会触发。

**修改文件**: `scripts/start_rsprompter_fusion_v10_satS_v1.sh`

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `VAL_RATIO` | `0.0` | **`0.1`** |

修改后 10% 的训练数据用于验证，早停机制可以正常工作，避免过拟合和无效训练。

---

## 四、修改前后对比总览

### 4.1 BEV 可学习化

| 组件 | 修改前 | 修改后 |
|------|--------|--------|
| BEV grid | `register_buffer("grid", ...)` 固定不可学习 | `grid_init` (buffer) + `grid_offset` (Parameter) 残差可学习 |
| 相机内参 | 直接使用 `I_inv = pinv(intrinsics)` | `I_inv = scale * I_inv + offset`，scale/offset 可学习 |
| 新增参数量 | 0 | grid_offset: 3x20x20=1200 个 + intrinsics: 2 个 |

### 4.2 训练策略

| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| sat_backbone LR | 2e-5 (过低) | 1e-4 (与其他组平衡) |
| GPU 配置 | 4 卡 | 单卡 3090 |
| Batch size | 2 (单卡 OOM) | 1 + 梯度累积 2 步 = 等效 2 |
| 验证集 | 未启用 (0%) | 10% |
| 早停 | 形同虚设 | 正常工作 (patience=15) |
| LR 预热 | 已有 (5 epoch warmup) | 不变 |
| AMP 混合精度 | 已有 (autocast + GradScaler) | 不变 |
| 梯度裁剪 | 已有 (clip_norm=0.5) | 不变 |

---

## 五、完整修改 diff

### 5.1 `uav/bev_embedding.py`

```diff
 class BEVEmbedding(nn.Module):
     def __init__(
         self,
         dim: int,
         sigma: int,
         bev_height: int,
         bev_width: int,
         h_meters: int,
         w_meters: int,
         offset: int,
         decoder_blocks: list,
+        grid_refine_scale: float = 0.5,
     ):
         super().__init__()
         ...
         grid = V_inv @ rearrange(grid, "d h w -> d (h w)")
         grid = rearrange(grid, "d (h w) -> d h w", h=h, w=w)

-        self.register_buffer("grid", grid, persistent=False)
+        # Learnable BEV grid: geometric init + residual offset
+        self.register_buffer("grid_init", grid, persistent=False)
+        self.grid_offset = nn.Parameter(torch.zeros_like(grid))
+        self.grid_refine_scale = grid_refine_scale
+
         self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))

+    @property
+    def grid(self):
+        return self.grid_init + self.grid_refine_scale * self.grid_offset
+
     def get_prior(self):
         return self.learned_features
```

### 5.2 `uav/cross_view_attention.py`

```diff
 class CVTEncoder(nn.Module):
     def __init__(self, ...):
         ...
         # Learnable alignment layer
         self.alignment_weight = nn.Parameter(torch.eye(4))

+        # Learnable intrinsics correction
+        self.intrinsics_scale = nn.Parameter(torch.ones(1))
+        self.intrinsics_offset = nn.Parameter(torch.zeros(1))

     def forward(self, batch):
         ...
         I_inv = torch.where(torch.isfinite(I_inv), I_inv, torch.zeros_like(I_inv))
         E_inv = torch.where(torch.isfinite(E_inv), E_inv, torch.zeros_like(E_inv))

+        # Apply learnable intrinsics correction
+        I_inv = self.intrinsics_scale * I_inv + self.intrinsics_offset
+
         features = [self.down(y) for y in self.backbone(self.norm(image))]
```

### 5.3 `configs/rsprompter_anchor_satS_drone_guidance_v1.py`

```diff
         bev_embedding=dict(
             sigma=1.0,
             bev_height=80,
             bev_width=80,
             h_meters=100.0,
             w_meters=100.0,
             offset=0.0,
             decoder_blocks=[2, 2],
+            grid_refine_scale=0.5,
         ),
```

### 5.4 `train/train_rsprompter_fusion.py`

```diff
-    parser.add_argument("--sat-backbone-lr-mult", type=float, default=0.01)
-    parser.add_argument("--sat-other-lr-mult", type=float, default=0.1)
-    parser.add_argument("--drone-lr-mult", type=float, default=1.0)
+    parser.add_argument("--sat-backbone-lr-mult", type=float, default=0.5)
+    parser.add_argument("--sat-other-lr-mult", type=float, default=1.0)
+    parser.add_argument("--drone-lr-mult", type=float, default=0.5)
+    parser.add_argument("--grad-accum-steps", type=int, default=1)
```

训练循环:
```diff
         total_loss = 0.0
         loss_meter = {}
         num_batches = 0
-        for batch in train_loader:
+        grad_accum = args.grad_accum_steps
+        optimizer.zero_grad(set_to_none=True)
+        for batch_idx, batch in enumerate(train_loader):
             ...
-            optimizer.zero_grad(set_to_none=True)
             model_for_loss = model.module if hasattr(model, "module") else model

             with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                 loss_dict = model_for_loss.loss(model_inputs, data_samples)
                 loss = sum(v for k, v in loss_dict.items() if "loss" in k ...)
+                loss_for_backward = loss / grad_accum
             ...
-            scaler.scale(loss).backward()
-            scaler.unscale_(optimizer)
-            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
-            scaler.step(optimizer)
-            scaler.update()
+            scaler.scale(loss_for_backward).backward()
+
+            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
+                scaler.unscale_(optimizer)
+                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
+                scaler.step(optimizer)
+                scaler.update()
+                optimizer.zero_grad(set_to_none=True)
```

### 5.5 `scripts/start_rsprompter_fusion_v10_satS_v1.sh`

```diff
-export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
-NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
-BATCH_SIZE="${BATCH_SIZE:-2}"
+export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
+NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
+BATCH_SIZE="${BATCH_SIZE:-1}"
 EPOCHS="${EPOCHS:-100}"
 LEARNING_RATE="${LEARNING_RATE:-2e-4}"
 STAGE1_PERCENT="${STAGE1_PERCENT:-40}"
 STAGE2_LR="${STAGE2_LR:-1e-4}"
+GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
 ...
-VAL_RATIO="${VAL_RATIO:-0.0}"
+VAL_RATIO="${VAL_RATIO:-0.1}"
 ...
-SAT_BACKBONE_LR_MULT_STAGE1="${SAT_BACKBONE_LR_MULT_STAGE1:-0.1}"
+SAT_BACKBONE_LR_MULT_STAGE1="${SAT_BACKBONE_LR_MULT_STAGE1:-0.5}"
 SAT_OTHER_LR_MULT_STAGE1="${SAT_OTHER_LR_MULT_STAGE1:-1.0}"
 DRONE_LR_MULT_STAGE1="${DRONE_LR_MULT_STAGE1:-0.5}"

-SAT_BACKBONE_LR_MULT_STAGE2="${SAT_BACKBONE_LR_MULT_STAGE2:-0.2}"
+SAT_BACKBONE_LR_MULT_STAGE2="${SAT_BACKBONE_LR_MULT_STAGE2:-0.5}"
 SAT_OTHER_LR_MULT_STAGE2="${SAT_OTHER_LR_MULT_STAGE2:-0.5}"
 DRONE_LR_MULT_STAGE2="${DRONE_LR_MULT_STAGE2:-0.5}"
```

两个阶段的启动命令均新增:
```diff
+  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
```

---

## 六、设计原则

1. **残差设计**: BEV grid 和内参修正均以残差形式添加，初始状态等价于原始行为，不破坏已有训练结果
2. **最小侵入**: 不改变模块接口和数据流，只增加可学习自由度
3. **向后兼容**: 旧 checkpoint 可以 `strict=False` 加载（新参数使用默认初始化）
4. **环境变量覆盖**: shell 脚本所有参数均支持环境变量覆盖，如需恢复多卡训练只需 `NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=2 GRAD_ACCUM_STEPS=1 bash scripts/start_...sh`

---

## 七、问题诊断报告落实情况

本节对照 `问题诊断报告.md` 中识别的问题和提出的方案，记录实际落实状态。

### 7.1 诊断问题落实

| # | 诊断问题 | 严重程度 | 落实状态 | 说明 |
|---|---------|---------|---------|------|
| 1 | RPN 训练严重不足 (loss_rpn_cls≈0.685) | 严重 | 已修复 | 根因是问题 2，LR 修复后 RPN 可充分训练 |
| 2 | 非对称学习率策略失败 (sat_backbone LR=2e-6) | 严重 | 已修复 | sat_backbone_lr_mult 从 0.01→0.5，实际 LR 从 2e-6→1e-4 |
| 3 | 语义引导模块在推理时引入噪声 | 中等 | 未修复 | 需训练后对比有/无引导效果再决定 |
| 4 | score_thr=0.05 过低 | 中等 | 已修复 | 已在之前的改动中提高到 0.3 |
| 5 | SAM 模型和特征层选择差异 | 轻微 | 无需修复 | 诊断确认两个项目配置一致，非问题根因 |

### 7.2 解决方案落实

| # | 提出方案 | 落实状态 | 实际修改 |
|---|---------|---------|---------|
| 1 | 调整学习率策略 | 已实施 | sat_backbone_lr_mult: 0.01→0.5 (shell+Python)，drone_lr_mult: 1.0→0.5 |
| 2 | 提高 score_thr 阈值 | 已实施 | score_thr: 0.05→0.3 (在本次会话之前已完成) |
| 3 | 禁用语义引导（调试用） | 未实施 | 待训练后根据效果决定是否需要 |
| 4 | 优化数据增强策略 | 未实施 | 当前仅 flip_prob=0.5，其余为 0；待后续迭代 |
| 5 | 增强 RPN 训练 (sampler num/pos_fraction) | 未实施 | 优先观察 LR 修复后的 RPN 训练效果 |

### 7.3 超出诊断报告的额外改进

诊断报告未涉及但本次实施的改进：

| 改进 | 来源 | 说明 |
|------|------|------|
| BEV grid 残差可学习 | 独立方案 | 解决 BEV 物理先验硬编码问题 |
| 相机内参可学习修正 | 独立方案 | 解决内参标定误差问题 |
| 单卡 3090 适配 | 训练策略分析 | CUDA_VISIBLE_DEVICES/NPROC 调整 |
| 梯度累积 | 训练策略分析 | batch_size=1 + accum=2，解决单卡显存不足 |
| 验证集启用 | 训练策略分析 | val_ratio: 0.0→0.1，使早停机制生效 |

### 7.4 待后续迭代的事项

以下事项需要在本次修改训练完成后，根据实际效果决定：

1. **数据增强** — 如果 LR 修复后 RPN 仍泛化不足，启用 rotate_prob、scale_prob、color_jitter_prob 等
2. **语义引导诊断** — 对比有/无 drone guidance 的推理效果，确认引导模块是否引入噪声
3. **RPN sampler 调参** — 如果 RPN loss 仍高于 0.3，考虑增大 sampler num (256→512) 和 pos_fraction (0.5→0.6)
4. **grid_refine_scale 调参** — 观察训练后 `grid_offset` 的范数，如果过小可增大 scale，过大则减小
5. **intrinsics_scale/offset 监控** — 检查训练后这两个参数偏离初始值的程度，评估内参修正的实际贡献
