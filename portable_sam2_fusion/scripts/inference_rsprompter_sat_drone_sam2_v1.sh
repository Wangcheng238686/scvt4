#!/bin/bash

echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合推理 v1"
echo "========================================"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 配置参数
CONFIG_FILE="${CONFIG_FILE:-portable_sam2_fusion/configs/rsprompter_sat_drone_sam2_v1.py}"
CHECKPOINT="${CHECKPOINT:-portable_sam2_fusion/checkpoints/rsprompter-sat-drone-sam2-v1/best_model.pth}"
DATA_ROOT="${DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-s-test}"
DRONE_DATA_ROOT="${DRONE_DATA_ROOT:-/home/wangcheng/data/unversity-big-after-without-negative/university-d-test-selected-5}"
WORK_DIR="${WORK_DIR:-./portable_sam2_fusion/inference_results/rsprompter-sat-drone-sam2-v1-${TIMESTAMP}}"
SHOW_DIR="${SHOW_DIR:-${WORK_DIR}/visualizations}"

# 图像尺寸
IMAGE_SIZE="${IMAGE_SIZE:-512}"
DRONE_IMAGE_SIZE="${DRONE_IMAGE_SIZE:-512}"
NUM_VIEWS="${NUM_VIEWS:-4}"

# 推理参数
SCORE_THR="${SCORE_THR:-0.3}"
MAX_NUM="${MAX_NUM:-100}"
GPU_ID="${GPU_ID:-0}"

# 是否使用无人机分支
USE_DRONE="${USE_DRONE:-true}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-false}"

# 检查必需参数
if [ -z "$CHECKPOINT" ]; then
    echo "错误: 必须指定 CHECKPOINT 环境变量或修改脚本"
    echo "示例: CHECKPOINT=./checkpoints/model.pth bash $0"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 权重文件不存在: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "推理参数:"
echo "  - 配置: ${CONFIG_FILE}"
echo "  - 权重: ${CHECKPOINT}"
echo "  - 卫星数据: ${DATA_ROOT}"
echo "  - 无人机数据: ${DRONE_DATA_ROOT}"
echo "  - 图像尺寸: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - 无人机图像尺寸: ${DRONE_IMAGE_SIZE}x${DRONE_IMAGE_SIZE}"
echo "  - 视角数: ${NUM_VIEWS}"
echo "  - 分数阈值: ${SCORE_THR}"
echo "  - 最大可视化数: ${MAX_NUM}"
echo "  - GPU ID: ${GPU_ID}"
echo "  - 使用无人机: ${USE_DRONE}"
echo "  - 随机采样: ${RANDOM_SAMPLE}"
echo "  - 工作目录: ${WORK_DIR}"
echo "  - 可视化目录: ${SHOW_DIR}"
echo "========================================"

mkdir -p "${WORK_DIR}"
mkdir -p "${SHOW_DIR}"

# 构建命令
CMD="python portable_sam2_fusion/inference/inference_rsprompter_fusion.py"
CMD="${CMD} ${CONFIG_FILE}"
CMD="${CMD} ${CHECKPOINT}"
CMD="${CMD} --work-dir ${WORK_DIR}"
CMD="${CMD} --show-dir ${SHOW_DIR}"
CMD="${CMD} --show-score-thr ${SCORE_THR}"
CMD="${CMD} --max-num ${MAX_NUM}"
CMD="${CMD} --gpu-id ${GPU_ID}"
CMD="${CMD} --data-root ${DATA_ROOT}"
CMD="${CMD} --drone-data-root ${DRONE_DATA_ROOT}"
CMD="${CMD} --image-size ${IMAGE_SIZE} ${IMAGE_SIZE}"
CMD="${CMD} --drone-image-size ${DRONE_IMAGE_SIZE} ${DRONE_IMAGE_SIZE}"
CMD="${CMD} --num-views ${NUM_VIEWS}"

if [ "$USE_DRONE" = "true" ]; then
    CMD="${CMD} --use-drone"
fi

if [ "$RANDOM_SAMPLE" = "true" ]; then
    CMD="${CMD} --random-sample"
fi

echo "执行命令:"
echo "${CMD}"
echo "========================================"

# 执行推理
${CMD}

echo "========================================"
echo "RSPrompter(SAM2) 卫星-无人机融合推理完成"
echo "========================================"
echo "结果保存在: ${WORK_DIR}"