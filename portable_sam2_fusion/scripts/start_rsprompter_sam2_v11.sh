#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/wangcheng/anaconda3/envs/cvt2/bin/python}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -z "${SAM2_CKPT}" ]; then
  export SAM2_CKPT="${PROJECT_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt"
fi

SAT_DATA_ROOT="${SAT_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-s-train}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/rsprompter-sam2-v11}"

USE_FSDP="${USE_FSDP:-0}"
FSDP_SHARDING="${FSDP_SHARDING:-FULL_SHARD}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-1000000}"

VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"

SAM2_BACKBONE_LR_MULT="${SAM2_BACKBONE_LR_MULT:-0.5}"
SAM2_OTHER_LR_MULT="${SAM2_OTHER_LR_MULT:-1.0}"

FSDP_ARGS=()
if [ "$USE_FSDP" = "1" ]; then
  FSDP_ARGS+=(--use-fsdp --fsdp-sharding "$FSDP_SHARDING" --fsdp-min-num-params "$FSDP_MIN_NUM_PARAMS")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/rsprompter_sam2_v11_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "${LOG_DIR}/rsprompter_sam2_v11_latest.log"

CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_anchor_satS_v8_sam2.py"
TRAIN_BIN="${PROJECT_ROOT}/train/train_rsprompter_fusion.py"

echo "========================================"
echo "RSPrompter(SAM2) Hiera-Base 卫星图像分割训练 v11"
echo "========================================"
echo "训练参数:"
echo "  - 模型: SAM2 Hiera-Base+"
echo "  - 权重: $SAM2_CKPT"
echo "  - 图像尺寸: 512x512"
echo "  - 总轮数: $EPOCHS"
echo "  - 基础学习率: $LEARNING_RATE"
echo "  - 批大小: $BATCH_SIZE (梯度累积: $GRAD_ACCUM_STEPS) [effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS))]"
echo "  - LR倍率: backbone=${SAM2_BACKBONE_LR_MULT} other=${SAM2_OTHER_LR_MULT}"
echo "  - 验证: ratio=${VAL_RATIO} patience=${EARLY_STOPPING_PATIENCE}"
echo "  - 权重目录: $CHECKPOINT_DIR"
echo "  - 修复: 特征聚合代码修复"
echo "========================================"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  "$TRAIN_BIN" \
  --config "$CONFIG_PATH" \
  --data-root "$SAT_DATA_ROOT" \
  --image-size 512 512 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --val-ratio "$VAL_RATIO" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --sat-backbone-lr-mult "$SAM2_BACKBONE_LR_MULT" \
  --sat-other-lr-mult "$SAM2_OTHER_LR_MULT" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  "${FSDP_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "RSPrompter(SAM2) 训练完成"
echo "========================================"
