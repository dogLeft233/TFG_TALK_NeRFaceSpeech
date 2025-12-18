#!/bin/bash
set -e

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "==========================================="
echo "NeRFFaceSpeech Docker 构建和运行脚本"
echo "==========================================="

# 检查模型文件
MODEL_DIR="NeRFFaceSpeech_Code/pretrained_networks"
MISSING_MODELS=()

if [ ! -d "$MODEL_DIR" ]; then
    MISSING_MODELS+=("模型目录不存在")
else
    if [ ! -f "$MODEL_DIR/ffhq_1024.pkl" ]; then
        MISSING_MODELS+=("ffhq_1024.pkl")
    fi
    if [ ! -f "$MODEL_DIR/seg.pth" ]; then
        MISSING_MODELS+=("seg.pth")
    fi
    if [ ! -f "$MODEL_DIR/LipaintNet.pt" ]; then
        MISSING_MODELS+=("LipaintNet.pt")
    fi
fi

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  警告: 以下模型文件缺失："
    for model in "${MISSING_MODELS[@]}"; do
        echo "   - $model"
    done
    echo ""
    echo "==========================================="
    echo "请下载模型文件到: $MODEL_DIR/"
    echo "==========================================="
    echo ""
    echo "模型下载地址："
    echo "https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i"
    echo ""
    echo "需要的模型文件："
    echo "1. ffhq_1024.pkl (184.3 MB)"
    echo "2. seg.pth (50.8 MB)"
    echo "3. LipaintNet.pt (12.4 MB)"
    echo "4. sad_talker_pretrained/ (目录)"
    echo "5. BFM_for_3DMM-Fitting-Pytorch/ (目录)"
    echo "6. Deep3DFaceRecon_pytorch/ (目录)"
    echo ""
    read -p "是否继续构建镜像？(模型文件可以通过 volume 挂载) (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装"
    exit 1
fi

# 检查 docker-compose 命令（支持新版本 docker compose 和旧版本 docker-compose）
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "❌ 错误: 未找到 docker-compose 命令"
    echo ""
    echo "解决方案："
    echo "1. 安装 docker-compose:"
    echo "   sudo apt-get update && sudo apt-get install docker-compose"
    echo ""
    echo "2. 或使用 Docker Desktop（推荐），确保启用 WSL 集成"
    echo "   在 Docker Desktop 设置中启用 WSL 2 集成"
    echo ""
    echo "3. 或使用新版本的 Docker（包含 compose 插件）"
    echo "   确保 Docker 版本 >= 20.10"
    exit 1
fi

# 检查 Docker 权限
if ! docker ps &> /dev/null; then
    echo ""
    echo "❌ 错误: 无法连接到 Docker daemon"
    echo ""
    echo "可能的原因："
    echo "1. Docker 服务未启动"
    echo "2. 当前用户没有 Docker 权限"
    echo ""
    echo "解决方案："
    echo "方案1：运行权限修复脚本（推荐）"
    echo "  ./docker/fix_docker_permissions.sh"
    echo ""
    echo "方案2：手动修复"
    echo "  sudo usermod -aG docker $USER"
    echo "  然后重新登录或执行: newgrp docker"
    echo ""
    echo "方案3：使用 sudo 运行（不推荐，但可以临时使用）"
    echo "  sudo ./build_and_run.sh"
    echo ""
    echo "方案4：启动 Docker 服务"
    echo "  sudo systemctl start docker"
    echo ""
    exit 1
fi

# 检查 NVIDIA Docker（可选）
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    echo "⚠️  警告: GPU 支持可能不可用（如果不需要 GPU 可以忽略）"
fi

# 检查镜像是否已存在
IMAGE_NAME="nerffacespeech:latest"
echo ""
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}\$"; then
    echo "✅ 检测到已存在镜像: ${IMAGE_NAME}"
    echo "   如需强制重新构建，请手动运行："
    echo "   ${DOCKER_COMPOSE_CMD} -f docker/docker-compose.yml build --no-cache"
else
    echo "开始构建 Docker 镜像..."
    echo "（这可能需要 30-60 分钟，取决于网络速度）"
    $DOCKER_COMPOSE_CMD -f docker/docker-compose.yml build
fi

# 启动容器
echo ""
echo "启动容器..."
$DOCKER_COMPOSE_CMD -f docker/docker-compose.yml up -d

# 等待服务启动
echo ""
echo "等待服务启动..."
sleep 5

# 检查服务状态
echo ""
echo "检查服务状态..."
$DOCKER_COMPOSE_CMD -f docker/docker-compose.yml ps

# 检查日志
echo ""
echo "查看容器日志（最后 20 行）..."
$DOCKER_COMPOSE_CMD -f docker/docker-compose.yml logs --tail=20

echo ""
echo "==========================================="
echo "构建和启动完成！"
echo "==========================================="
echo ""
echo "服务地址："
echo "  - 后端 API: http://localhost:8000/"
echo "  - API 文档: http://localhost:8000/docs"
echo "  - 前端界面: http://localhost:7860/"
echo ""
echo "查看日志："
echo "  ${DOCKER_COMPOSE_CMD} -f docker/docker-compose.yml logs -f"
echo ""
echo "停止服务："
echo "  ${DOCKER_COMPOSE_CMD} -f docker/docker-compose.yml down"
echo "==========================================="

