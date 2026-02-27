#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/wangcheng/anaconda3/envs/cvt/bin/python3}"

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

SAT_TEST_DATA_ROOT="${SAT_TEST_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-s-test}"
DRONE_TEST_DATA_ROOT="${DRONE_TEST_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-d-test-selected-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID:-0}"

INFERENCE_BIN="${PROJECT_ROOT}/inference/inference_rsprompter_fusion.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SHOW_DIR="${PROJECT_ROOT}/inference_results/vis_auto_${TIMESTAMP}"
WORK_DIR="${PROJECT_ROOT}/inference_results/metrics_auto_${TIMESTAMP}"

mkdir -p "$SHOW_DIR"
mkdir -p "$WORK_DIR"

NUM_VIEWS="${NUM_VIEWS:-4}"
SCORE_THR="${SCORE_THR:-0.3}"
MAX_NUM="${MAX_NUM:-100}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1024}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1024}"
DRONE_IMAGE_SIZE_H="${DRONE_IMAGE_SIZE_H:-512}"
DRONE_IMAGE_SIZE_W="${DRONE_IMAGE_SIZE_W:-512}"

if [ -z "$CONFIG_PATH" ]; then
  echo "错误: 必须设置 CONFIG_PATH 环境变量"
  echo "用法: CONFIG_PATH=/path/to/config.py $0"
  exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
  echo "错误: 必须设置 CHECKPOINT_PATH 环境变量"
  echo "用法: CHECKPOINT_PATH=/path/to/checkpoint.pth $0"
  exit 1
fi

echo "========================================"
echo "RSPrompter(SAM) 自动模式推理"
echo "========================================"
echo "配置: $CONFIG_PATH"
echo "权重: $CHECKPOINT_PATH"
echo "卫星数据: $SAT_TEST_DATA_ROOT"
echo "无人机数据: $DRONE_TEST_DATA_ROOT"
echo "推理模式: 自动检测 (根据模型配置)"
echo "结果保存至: $SHOW_DIR"
echo "========================================"

DRONE_ARGS=""
if [ "${FORCE_NO_DRONE:-0}" = "1" ]; then
  DRONE_ARGS="--no-drone"
  echo "注意: 已强制禁用无人机分支 (--no-drone)"
fi

"$PYTHON_BIN" "$INFERENCE_BIN" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  --data-root "$SAT_TEST_DATA_ROOT" \
  --drone-data-root "$DRONE_TEST_DATA_ROOT" \
  --num-views "$NUM_VIEWS" \
  --image-size "$IMAGE_SIZE_H" "$IMAGE_SIZE_W" \
  --drone-image-size "$DRONE_IMAGE_SIZE_H" "$DRONE_IMAGE_SIZE_W" \
  --show-dir "$SHOW_DIR" \
  --work-dir "$WORK_DIR" \
  --show-score-thr "$SCORE_THR" \
  --max-num "$MAX_NUM" \
  --gpu-id "$GPU_ID" \
  $DRONE_ARGS

echo ""
echo "========================================"
echo "自动模式推理完成！"
echo "可视化结果: $SHOW_DIR"
echo "评估指标: $WORK_DIR"
echo "========================================"
