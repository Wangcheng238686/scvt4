#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/wangcheng/anaconda3/envs/cvt/bin/python}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -z "${SAT_SAM_HF_PRETRAIN}" ]; then
  if [ -d "${PROJECT_ROOT}/checkpoints/facebook/sam-vit-base" ]; then
    export SAT_SAM_HF_PRETRAIN="${PROJECT_ROOT}/checkpoints/facebook/sam-vit-base"
  elif [ -d "/home/wangcheng2021/project/RSPrompter-release/checkpoints/facebook/sam-vit-base" ]; then
    export SAT_SAM_HF_PRETRAIN="/home/wangcheng2021/project/RSPrompter-release/checkpoints/facebook/sam-vit-base"
  fi
fi

if [ -z "${SAT_SAM_CKPT}" ]; then
  export SAT_SAM_CKPT="${PROJECT_ROOT}/checkpoints/sam_vit_b_01ec64.pth"
fi

SAT_DATA_ROOT="${SAT_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-s-train}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/rsprompter-satS-only-v1}"

USE_FSDP="${USE_FSDP:-0}"
FSDP_SHARDING="${FSDP_SHARDING:-FULL_SHARD}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-1000000}"

VAL_RATIO="${VAL_RATIO:-0.0}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"

SAT_BACKBONE_LR_MULT="${SAT_BACKBONE_LR_MULT:-0.5}"
SAT_OTHER_LR_MULT="${SAT_OTHER_LR_MULT:-1.0}"

FSDP_ARGS=()
if [ "$USE_FSDP" = "1" ]; then
  FSDP_ARGS+=(--use-fsdp --fsdp-sharding "$FSDP_SHARDING" --fsdp-min-num-params "$FSDP_MIN_NUM_PARAMS")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/rsprompter_satS_only_v1_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "${LOG_DIR}/rsprompter_satS_only_latest_v1.log"

CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_anchor_satS_only_v1.py"
TRAIN_BIN="${PROJECT_ROOT}/train/train_rsprompter_fusion.py"

echo "========================================"
echo "RSPrompter(SAM) 卫星单分支训练 (消融实验)"
echo "========================================"
echo "训练参数:"
echo "  - 总轮数: $EPOCHS"
echo "  - 学习率: $LEARNING_RATE"
echo "  - 批大小: $BATCH_SIZE (梯度累积: $GRAD_ACCUM_STEPS)"
echo "  - LR倍率: backbone=${SAT_BACKBONE_LR_MULT} other=${SAT_OTHER_LR_MULT}"
echo "  - 验证: ratio=${VAL_RATIO} patience=${EARLY_STOPPING_PATIENCE}"
echo "  - 权重目录: $CHECKPOINT_DIR"
echo "  - 注意: 不使用无人机分支"
echo "========================================"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  "$TRAIN_BIN" \
  --config "$CONFIG_PATH" \
  --data-root "$SAT_DATA_ROOT" \
  --image-size 1024 1024 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --val-ratio "$VAL_RATIO" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --sat-backbone-lr-mult "$SAT_BACKBONE_LR_MULT" \
  --sat-other-lr-mult "$SAT_OTHER_LR_MULT" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  "${FSDP_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "RSPrompter(SAM) 卫星单分支训练完成"
echo "========================================"
