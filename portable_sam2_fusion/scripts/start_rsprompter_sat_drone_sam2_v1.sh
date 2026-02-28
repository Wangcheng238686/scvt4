#!/bin/bash

echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合训练 v1"
echo "========================================"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export SAM2_CKPT="${SAM2_CKPT:-./portable_sam2_fusion/checkpoints/sam2.1_hiera_base_plus.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./portable_sam2_fusion/checkpoints/rsprompter-sat-drone-sam2-v1}"
DATA_ROOT="${DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-s-train}"
DRONE_DATA_ROOT="${DRONE_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-d-train-selected-5}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
DRONE_IMAGE_SIZE="${DRONE_IMAGE_SIZE:-512}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BASE_LR="${BASE_LR:-4e-4}"
SAT_BACKBONE_LR_MULT="${SAT_BACKBONE_LR_MULT:-0.5}"
SAT_OTHER_LR_MULT="${SAT_OTHER_LR_MULT:-1.0}"
DRONE_LR_MULT="${DRONE_LR_MULT:-1.0}"
NUM_VIEWS="${NUM_VIEWS:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_SCENES="${MAX_SCENES:-1000}"
GPU_ID="${GPU_ID:-0}"

echo "训练参数:"
echo "  - 模型: SAM2 Hiera-Base+ (卫星+无人机独立backbone)"
echo "  - 权重: ${SAM2_CKPT}"
echo "  - 数据目录: ${DATA_ROOT}"
echo "  - 无人机数据目录: ${DRONE_DATA_ROOT}"
echo "  - 图像尺寸: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - 无人机图像尺寸: ${DRONE_IMAGE_SIZE}x${DRONE_IMAGE_SIZE}"
echo "  - 总轮数: ${EPOCHS}"
echo "  - 基础学习率: ${BASE_LR}"
echo "  - 批大小: ${BATCH_SIZE} (梯度累积: ${GRAD_ACCUM_STEPS}) [effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS))]"
echo "  - LR倍率: sat_backbone=${SAT_BACKBONE_LR_MULT} sat_other=${SAT_OTHER_LR_MULT} drone=${DRONE_LR_MULT}"
echo "  - 视角数: ${NUM_VIEWS}"
echo "  - 最大场景数: ${MAX_SCENES}"
echo "  - 权重目录: ${CHECKPOINT_DIR}"
echo "  - 特性: 深度感知 + 高度引导融合 + 独立backbone"
echo "========================================"

mkdir -p "${CHECKPOINT_DIR}"

CONFIG_FILE="portable_sam2_fusion/configs/rsprompter_sat_drone_sam2_v1.py"

python portable_sam2_fusion/train/train_rsprompter_fusion.py \
    --config "${CONFIG_FILE}" \
    --data-root "${DATA_ROOT}" \
    --drone-data-root "${DRONE_DATA_ROOT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${BASE_LR} \
    --image-size ${IMAGE_SIZE} ${IMAGE_SIZE} \
    --drone-image-size ${DRONE_IMAGE_SIZE} ${DRONE_IMAGE_SIZE} \
    --num-views ${NUM_VIEWS} \
    --use-drone \
    --normalize-drone \
    --sat-backbone-lr-mult ${SAT_BACKBONE_LR_MULT} \
    --sat-other-lr-mult ${SAT_OTHER_LR_MULT} \
    --drone-lr-mult ${DRONE_LR_MULT} \
    --grad-accum-steps ${GRAD_ACCUM_STEPS} \
    --max-scenes ${MAX_SCENES} \
    --gpu-id ${GPU_ID} \
    --val-ratio 0.1 \
    --val-every-n-epochs 1 \
    --early-stopping-patience 15

echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合训练完成"
echo "========================================"
