# Portable SAM Fusion 项目代码审计文档 (2026-02-09)

## 1. 概述
本轮开发主要针对跨视角（卫星图 + 无人机图）融合模型在实际应用中遇到的三个核心挑战进行了系统性优化：
1. **内外参尺度不匹配**：解决由于无人机图像缩放导致的投影偏移。
2. **坐标系几何偏差**：通过可学习层解决 COLMAP 世界坐标系与卫星图 BEV 空间的不对齐。
3. **时相特征冲突**：通过自适应门控机制解决由于拍摄时间不同导致的特征不一致。

---

## 2. 核心改动审计清单

### 2.1 数据预处理优化 (Data Preprocessing)
*   **文件**: [satellite_drone_dataset.py](file:///home/wangcheng2021/project/SCVT3/portable_sam_fusion/data/satellite_drone_dataset.py)
*   **问题**: 无人机图像在加载时被 resize 到 512x512，但原始相机内参矩阵未同步缩放，导致反向投影射线方向错误。
*   **实现**: 
    *   在 `_load_drone_data` 中捕获 `orig_w, orig_h`。
    *   计算缩放比例 `scale_x, scale_y`。
    *   同步更新内参矩阵 `intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]`。
*   **审计要点**: 确保内参缩放逻辑与图像 `Resize` 算子完全匹配。

### 2.2 隐式几何对齐层 (Learnable Alignment)
*   **文件**: [cross_view_attention.py](file:///home/wangcheng2021/project/SCVT3/portable_sam_fusion/uav/cross_view_attention.py)
*   **问题**: COLMAP 建立的坐标系是以场景中心为原点且方向随机的，无法直接与以卫星图中心为原点的 BEV 空间对齐。
*   **实现**:
    *   在 `CVTEncoder` 中引入 `alignment_weight` (4x4 nn.Parameter)，初始化为单位矩阵。
    *   在 `forward` 过程中执行 $E_{aligned\_inv} = T_{learnable} \times E_{inv}$。
*   **原理**: 利用反向传播自动学习 COLMAP 到卫星图的旋转、平移和缩放变换。
*   **审计要点**: 检查 `alignment_transform` 在 Batch 维度的 `repeat` 逻辑以及 `einsum` 矩阵乘法的正确性。

### 2.3 自适应时相冲突过滤 (Adaptive Gating)
*   **文件**: [rsprompter_anchor_drone_guidance.py](file:///home/wangcheng2021/project/SCVT3/portable_sam_fusion/models/rsprompter_anchor_drone_guidance.py)
*   **问题**: 极个别场景下，无人机与卫星图拍摄时间不一，导致地物特征（如车辆、植被、阴影）冲突。
*   **实现**:
    *   升级 `DroneSemanticGuidance` 中的门控逻辑。
    *   从单一输入（仅 Drone）升级为双输入（Satellite + Drone）拼接。
    *   引入 2 层 MLP 代替单层线性映射。
*   **原理**: 模型根据两个视角的全局上下文对比，自动判断无人机信息的“信任度”。若冲突严重，Gate 值趋于 0。
*   **审计要点**: 检查 `torch.cat([context, x_context], dim=1)` 后的维度匹配情况。

---

## 3. 辅助功能增强
*   **训练/验证集划分**: 在 [satellite_drone_dataset.py](file:///home/wangcheng2021/project/SCVT3/portable_sam_fusion/data/satellite_drone_dataset.py) 中增加了基于 `hashlib` 的场景划分逻辑，确保在 `use_drone` 模式下训练集与验证集场景完全隔离。
*   **配置一致性**: 统一了训练与推理脚本中的 `--num-views` 默认参数（设置为 15），确保模型输入规模的一致。

---

## 4. 后续建议
1. **对齐层监控**: 建议在训练日志中记录 `alignment_weight` 的变化，若其 Frobenius 范数过大，可能需要检查初始坐标偏离是否超出了学习范围。
2. **尺度确认**: 目前 BEV 的 `h_meters` / `w_meters` 需与卫星图实际物理尺寸严格对应，否则 `alignment_weight` 需额外承担繁重的缩放学习任务。
