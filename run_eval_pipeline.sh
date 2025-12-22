#!/bin/bash
# 运行评估流程的 Docker 启动脚本
# 使用 syncnet 环境运行 eval_pipline

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "==========================================="
echo "NeRFFaceSpeech 评估流程启动脚本"
echo "==========================================="
echo ""

# 配置参数
INPUT_DIR="$PROJECT_ROOT/data/geneface_datasets/data/raw/videos"
OUTPUT_DIR="$PROJECT_ROOT/output/eval_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="$PROJECT_ROOT/NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl"
SEGMENT_SEC=8
MAX_SEGMENTS=8
DOCKER_IMAGE="nerffacespeech:latest"
CONTAINER_NAME="nerffacespeech-eval"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ 错误: 输入目录不存在: $INPUT_DIR${NC}"
    echo ""
    echo "请确保以下目录存在："
    echo "  $INPUT_DIR"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ 错误: 模型文件不存在: $MODEL_PATH${NC}"
    echo ""
    echo "请确保模型文件存在，或修改脚本中的 MODEL_PATH 变量"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓${NC} 输出目录: $OUTPUT_DIR"
echo ""

# 检查 Docker 镜像是否存在
if ! docker images | grep -q "^${DOCKER_IMAGE%:*}"; then
    echo -e "${YELLOW}⚠️  警告: Docker 镜像 $DOCKER_IMAGE 不存在${NC}"
    echo "正在尝试构建镜像..."
    cd "$PROJECT_ROOT/docker"
    docker build -t "$DOCKER_IMAGE" ..
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✓${NC} Docker 镜像构建完成"
    echo ""
fi

# 检查容器是否已存在
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo -e "${YELLOW}⚠️  检测到已存在的容器: $CONTAINER_NAME${NC}"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        echo -e "${GREEN}✓${NC} 已删除旧容器"
    else
        echo "使用现有容器"
    fi
fi

# 启动容器（如果不存在）
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "启动 Docker 容器..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        -v "$PROJECT_ROOT/data:/app/data:ro" \
        -v "$PROJECT_ROOT/output:/app/output:rw" \
        -v "$PROJECT_ROOT/NeRFFaceSpeech_Code:/app/NeRFFaceSpeech_Code:ro" \
        -v "$PROJECT_ROOT/eval_pipline:/app/eval_pipline:ro" \
        -v "$PROJECT_ROOT/weights:/app/weights:rw" \
        -v "$PROJECT_ROOT/Hugging_Face:/app/Hugging_Face:rw" \
        -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
        -e TORCH_HOME=/app/weights \
        -e HF_ENDPOINT=https://hf-mirror.com \
        -e HF_HOME=/app/Hugging_Face \
        -e CUDA_VISIBLE_DEVICES=0 \
        "$DOCKER_IMAGE" \
        tail -f /dev/null
    
    echo -e "${GREEN}✓${NC} 容器已启动: $CONTAINER_NAME"
    echo ""
    
    # 等待容器就绪
    sleep 2
fi

# 在容器内运行评估流程
echo "==========================================="
echo "开始运行评估流程"
echo "==========================================="
echo ""
echo "配置参数:"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型文件: $MODEL_PATH"
echo "  每段时长: ${SEGMENT_SEC}秒"
echo "  每视频段数: ${MAX_SEGMENTS}段（随机选择）"
echo "  对齐方式: FFHQFaceAlignment"
echo ""

# 激活 syncnet 环境并运行评估流程
OUTPUT_DIR_BASENAME=$(basename "$OUTPUT_DIR")
docker exec -it "$CONTAINER_NAME" bash -c "
    set -e
    
    # 激活 syncnet 环境
    source /opt/conda/etc/profile.d/conda.sh
    
    # 检查 syncnet 环境是否存在
    if [ -d '/app/environment/syncnet' ]; then
        echo '[INFO] 激活 syncnet 环境...'
        conda activate /app/environment/syncnet
    else
        echo '[WARNING] syncnet 环境不存在，使用 base 环境'
        conda activate base
    fi
    
    # 设置工作目录
    cd /app
    
    # 运行评估流程
    python -m eval_pipline \\
        --input-dir /app/data/geneface_datasets/data/raw/videos \\
        --output-dir /app/output/$OUTPUT_DIR_BASENAME \\
        --network /app/NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl \\
        --segment-sec $SEGMENT_SEC \\
        --max-segments $MAX_SEGMENTS \\
        --random-segments \\
        --ffhq-alignment \\
        --device cuda
    
    echo ''
    echo '==========================================='
    echo '评估流程完成！'
    echo '==========================================='
    echo ''
    echo '结果保存在: /app/output/$OUTPUT_DIR_BASENAME'
    echo '  - 切分视频: /app/output/$OUTPUT_DIR_BASENAME/videos_split/'
    echo '  - 裁剪视频: /app/output/$OUTPUT_DIR_BASENAME/videos_cropped/'
    echo '  - 推理结果: /app/output/$OUTPUT_DIR_BASENAME/videos_infer/'
    echo '  - 指标结果: /app/output/$OUTPUT_DIR_BASENAME/metrics.json'
"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo -e "${GREEN}✅ 评估流程执行成功！${NC}"
    echo "==========================================="
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo "  - 切分视频: $OUTPUT_DIR/videos_split/"
    echo "  - 裁剪视频: $OUTPUT_DIR/videos_cropped/"
    echo "  - 推理结果: $OUTPUT_DIR/videos_infer/"
    echo "  - 指标结果: $OUTPUT_DIR/metrics.json"
    echo ""
    echo "查看结果:"
    echo "  ls -lh $OUTPUT_DIR/"
    echo ""
else
    echo ""
    echo "==========================================="
    echo -e "${RED}❌ 评估流程执行失败！${NC}"
    echo "==========================================="
    echo ""
    echo "退出码: $EXIT_CODE"
    echo "请检查容器日志:"
    echo "  docker logs $CONTAINER_NAME"
    echo ""
    exit $EXIT_CODE
fi

