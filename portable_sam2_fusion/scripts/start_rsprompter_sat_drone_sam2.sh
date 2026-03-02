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
DRONE_DATA_ROOT="${DRONE_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-d-train-selected-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/rsprompter-sat-drone-sam2-v1}"

USE_FSDP="${USE_FSDP:-0}"
FSDP_SHARDING="${FSDP_SHARDING:-FULL_SHARD}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-1000000}"

VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"

SAT_BACKBONE_LR_MULT="${SAT_BACKBONE_LR_MULT:-0.5}"
SAT_OTHER_LR_MULT="${SAT_OTHER_LR_MULT:-1.0}"
DRONE_LR_MULT="${DRONE_LR_MULT:-1.0}"
SCENE_ALIGN_LR_MULT="${SCENE_ALIGN_LR_MULT:-2.0}"
NUM_VIEWS="${NUM_VIEWS:-4}"
MAX_SCENES="${MAX_SCENES:-1000}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
DRONE_IMAGE_SIZE="${DRONE_IMAGE_SIZE:-512}"

FSDP_ARGS=()
if [ "$USE_FSDP" = "1" ]; then
  FSDP_ARGS+=(--use-fsdp --fsdp-sharding "$FSDP_SHARDING" --fsdp-min-num-params "$FSDP_MIN_NUM_PARAMS")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/rsprompter_sat_drone_sam2_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "${LOG_DIR}/rsprompter_sat_drone_sam2_latest.log"

CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_sat_drone_sam2_v1.py"
TRAIN_BIN="${PROJECT_ROOT}/train/train_rsprompter_fusion.py"

echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合训练"
echo "========================================"
echo "训练参数:"
echo "  - 模型: SAM2 Hiera-Base+ (卫星+无人机独立backbone)"
echo "  - 权重: $SAM2_CKPT"
echo "  - 数据目录: $SAT_DATA_ROOT"
echo "  - 无人机数据目录: $DRONE_DATA_ROOT"
echo "  - 图像尺寸: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - 无人机图像尺寸: ${DRONE_IMAGE_SIZE}x${DRONE_IMAGE_SIZE}"
echo "  - 总轮数: $EPOCHS"
echo "  - 基础学习率: $LEARNING_RATE"
echo "  - 批大小: $BATCH_SIZE (梯度累积: $GRAD_ACCUM_STEPS) [effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))]"
echo "  - LR倍率: sat_backbone=${SAT_BACKBONE_LR_MULT} sat_other=${SAT_OTHER_LR_MULT} drone=${DRONE_LR_MULT} scene_align=${SCENE_ALIGN_LR_MULT}"
echo "  - 视角数: $NUM_VIEWS"
echo "  - 最大场景数: $MAX_SCENES"
echo "  - 验证: ratio=${VAL_RATIO} patience=${EARLY_STOPPING_PATIENCE}"
echo "  - 分布式: nproc=$NPROC_PER_NODE fsdp=$USE_FSDP"
echo "  - 权重目录: $CHECKPOINT_DIR"
echo "  - 日志文件: $LOG_FILE"
echo "  - 特性: 深度感知 + 高度引导融合 + 独立backbone"
echo "========================================"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  "$TRAIN_BIN" \
  --config "$CONFIG_PATH" \
  --data-root "$SAT_DATA_ROOT" \
  --drone-data-root "$DRONE_DATA_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --image-size "$IMAGE_SIZE" "$IMAGE_SIZE" \
  --drone-image-size "$DRONE_IMAGE_SIZE" "$DRONE_IMAGE_SIZE" \
  --num-views "$NUM_VIEWS" \
  --use-drone \
  --normalize-drone \
  --sat-backbone-lr-mult "$SAT_BACKBONE_LR_MULT" \
  --sat-other-lr-mult "$SAT_OTHER_LR_MULT" \
  --drone-lr-mult "$DRONE_LR_MULT" \
  --scene-align-lr-mult "$SCENE_ALIGN_LR_MULT" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-scenes "$MAX_SCENES" \
  --val-ratio "$VAL_RATIO" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  "${FSDP_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合训练完成"
echo "========================================"
