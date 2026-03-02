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
DRONE_TEST_DATA_ROOT="${DRONE_TEST_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-d-test-selected-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID:-0}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/rsprompter-sat-drone-sam2-v1/best_model.pth}"
CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_sat_drone_sam2_v1.py"
INFERENCE_BIN="${PROJECT_ROOT}/inference/inference_rsprompter_fusion.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="${PROJECT_ROOT}/inference_results/rsprompter-sat-drone-sam2-v1-${TIMESTAMP}"
SHOW_DIR="${SHOW_DIR:-${WORK_DIR}/visualizations}"

SCORE_THR="${SCORE_THR:-0.3}"
MAX_NUM="${MAX_NUM:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
DRONE_IMAGE_SIZE="${DRONE_IMAGE_SIZE:-512}"
NUM_VIEWS="${NUM_VIEWS:-4}"
USE_DRONE="${USE_DRONE:-true}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-false}"

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

echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合推理"
echo "========================================"
echo "配置: $CONFIG_PATH"
echo "权重: $CHECKPOINT_PATH"
echo "SAM2权重: $SAM2_CKPT"
echo "卫星测试数据: $SAT_TEST_DATA_ROOT"
echo "无人机测试数据: $DRONE_TEST_DATA_ROOT"
echo "图像尺寸: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "无人机图像尺寸: ${DRONE_IMAGE_SIZE}x${DRONE_IMAGE_SIZE}"
echo "视角数: $NUM_VIEWS"
echo "分数阈值: $SCORE_THR"
echo "最大可视化数: $MAX_NUM"
echo "GPU ID: $GPU_ID"
echo "使用无人机: $USE_DRONE"
echo "随机采样: $RANDOM_SAMPLE"
echo "可视化目录: $SHOW_DIR"
echo "指标目录: $WORK_DIR"
echo "========================================"

CMD="\"$PYTHON_BIN\" \"$INFERENCE_BIN\""
CMD="${CMD} \"$CONFIG_PATH\""
CMD="${CMD} \"$CHECKPOINT_PATH\""
CMD="${CMD} --work-dir \"$WORK_DIR\""
CMD="${CMD} --show-dir \"$SHOW_DIR\""
CMD="${CMD} --show-score-thr $SCORE_THR"
CMD="${CMD} --max-num $MAX_NUM"
CMD="${CMD} --gpu-id $GPU_ID"
CMD="${CMD} --data-root \"$SAT_TEST_DATA_ROOT\""
CMD="${CMD} --drone-data-root \"$DRONE_TEST_DATA_ROOT\""
CMD="${CMD} --image-size $IMAGE_SIZE $IMAGE_SIZE"
CMD="${CMD} --drone-image-size $DRONE_IMAGE_SIZE $DRONE_IMAGE_SIZE"
CMD="${CMD} --num-views $NUM_VIEWS"

if [ "$USE_DRONE" = "true" ]; then
    CMD="${CMD} --use-drone"
fi

if [ "$RANDOM_SAMPLE" = "true" ]; then
    CMD="${CMD} --random-sample"
fi

eval $CMD

echo ""
echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合推理完成"
echo "========================================"
echo "可视化结果: $SHOW_DIR"
echo "评估指标: $WORK_DIR"
echo "========================================"
