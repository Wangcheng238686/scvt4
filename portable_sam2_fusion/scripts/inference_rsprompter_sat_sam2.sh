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

SAT_TEST_DATA_ROOT="${SAT_TEST_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-s-test}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID:-0}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/rsprompter-sam2-v11/best_model.pth}"
CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_anchor_satS_v8_sam2.py"
INFERENCE_BIN="${PROJECT_ROOT}/inference/inference_rsprompter_fusion.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="${PROJECT_ROOT}/inference_results/rsprompter-sam2-v11-${TIMESTAMP}"
SHOW_DIR="${SHOW_DIR:-${WORK_DIR}/visualizations}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 权重文件不存在: $CHECKPOINT_PATH"
    echo "请设置 CHECKPOINT_PATH 环境变量"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

mkdir -p "$WORK_DIR"
mkdir -p "$SHOW_DIR"

SCORE_THR="${SCORE_THR:-0.3}"
MAX_NUM="${MAX_NUM:-100}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-512}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-512}"

echo "========================================"
echo "RSPrompter(SAM2) Hiera-Base 推理 v11"
echo "========================================"
echo "配置: $CONFIG_PATH"
echo "权重: $CHECKPOINT_PATH"
echo "SAM2权重: $SAM2_CKPT"
echo "测试数据: $SAT_TEST_DATA_ROOT"
echo "图像尺寸: ${IMAGE_SIZE_H}x${IMAGE_SIZE_W}"
echo "可视化目录: $SHOW_DIR"
echo "指标目录: $WORK_DIR"
echo "========================================"

"$PYTHON_BIN" "$INFERENCE_BIN" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  --data-root "$SAT_TEST_DATA_ROOT" \
  --image-size "$IMAGE_SIZE_H" "$IMAGE_SIZE_W" \
  --show-dir "$SHOW_DIR" \
  --work-dir "$WORK_DIR" \
  --show-score-thr "$SCORE_THR" \
  --max-num "$MAX_NUM" \
  --gpu-id "$GPU_ID"

echo ""
echo "========================================"
echo "RSPrompter(SAM2) 推理完成"
echo "========================================"
echo "可视化结果: $SHOW_DIR"
echo "评估指标: $WORK_DIR"
echo "========================================"
