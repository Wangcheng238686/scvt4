# 方案A: 可学习 V_inv 实施记录

## 背景

原始 `BEVEmbedding` 使用固定的 `h_meters`/`w_meters` 计算 `V_inv` 矩阵（BEV 像素坐标 → 世界坐标的映射），无法随训练自适应调整 BEV 覆盖范围。同时 `CVTEncoder` 中的 `intrinsics_scale`/`intrinsics_offset` 与可学习 V_inv 功能冗余。

## 设计决策

- **V_inv (learnable) + grid_offset (learnable) 共存**：V_inv 负责全局仿射变换，grid_offset 负责局部逐像素微调
- **保留** `CVTEncoder.alignment_weight`（处理 3D 世界坐标对齐，与 V_inv 职责不同）
- **移除** `intrinsics_scale` / `intrinsics_offset`（与可学习 V_inv 功能冗余）

## 修改清单

### 1. `uav/bev_embedding.py` — V_inv 改为可学习参数

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| V_inv | 固定张量，由 `get_view_matrix` 计算后取逆 | `nn.Parameter`，以几何值初始化，可随训练更新 |
| grid_pixels | 不存在（直接与 V_inv 相乘后存为 `grid_init` buffer） | `register_buffer`，保存原始像素坐标 |
| grid_init | `register_buffer`，存储预计算的世界坐标 grid | 已移除 |
| grid 属性 | `grid_init + scale * grid_offset`（静态基础 + 偏移） | `V_inv @ grid_pixels + scale * grid_offset`（动态计算） |
| self.h / self.w | 不存在 | 新增，用于 `grid` 属性中 rearrange 的 reshape |

构造函数签名不变，`h_meters`/`w_meters`/`offset` 仍作为参数传入用于 V_inv 初始值计算。

### 2. `uav/cross_view_attention.py` — 移除 intrinsics 校正参数

**`CVTEncoder.__init__` 中删除：**
- `self.intrinsics_scale = nn.Parameter(torch.ones(1))`
- `self.intrinsics_offset = nn.Parameter(torch.zeros(1))`

**`CVTEncoder.forward` 中删除：**
- `I_inv = self.intrinsics_scale * I_inv + self.intrinsics_offset`

**保留：**
- `self.alignment_weight = nn.Parameter(torch.eye(4))` 及其在 forward 中的使用

### 3. 配置文件 — 无需修改

`configs/rsprompter_anchor_satS_drone_guidance_v1.py` 中 `bev_embedding` 配置保持不变，`h_meters`/`w_meters`/`offset` 现用于初始化可学习 V_inv。

## 验证要点

1. `bev_embedding.V_inv` 出现在 `model.named_parameters()` 中
2. `intrinsics_scale` / `intrinsics_offset` 不再出现在参数列表中
3. `alignment_weight` 仍在参数列表中
4. 前向传播 shape 正确，无报错

## 注意事项

- 加载旧 checkpoint 时需注意：`grid_init` buffer 已不存在，`V_inv` 从 buffer 变为 parameter，需做 key 映射或 `strict=False`
- V_inv 学习率建议与主网络一致或略低，避免初期剧烈偏离几何初始值
