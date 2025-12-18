# NeRFFaceSpeech Docker 配置
# 支持 GPU (CUDA 11.8)
# 所有下载使用国内镜像源
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    TORCH_HOME=/app/weights \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/Hugging_Face \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# ==================== 配置 APT 镜像源（清华源）====================
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://ports.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list || true

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ==================== 安装 Miniconda（使用清华镜像）====================
RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# ==================== 配置 conda 镜像源（清华源）====================
RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    /opt/conda/bin/conda config --set show_channel_urls yes && \
    /opt/conda/bin/conda config --set channel_priority strict

# ==================== 配置 pip 镜像源（已在环境变量中设置）====================
# PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件（排除模型文件，通过 .dockerignore 处理）
COPY . /app/

# 创建必要的目录
RUN mkdir -p \
    /app/weights \
    /app/Hugging_Face \
    /app/database/videos \
    /app/outputs/video \
    /app/outputs/audio \
    /app/NeRFFaceSpeech_Code/pretrained_networks

# ==================== 安装 Python 环境 ====================
# 安装 llm_talk 环境（如果环境文件存在）
RUN if [ -f "environment/llm_talk.yaml" ]; then \
        echo "安装 llm_talk 环境..." && \
        /opt/conda/bin/conda env create -f environment/llm_talk.yaml --prefix environment/llm_talk || \
        echo "警告: llm_talk 环境创建失败，将使用 pip 安装依赖"; \
    else \
        echo "警告: environment/llm_talk.yaml 不存在，跳过 llm_talk 环境安装"; \
    fi

# 安装 nerffacespeech 环境（如果环境文件存在）
RUN if [ -f "environment/nerffacespeech.yaml" ]; then \
        echo "安装 nerffacespeech 环境..." && \
        /opt/conda/bin/conda env create -f environment/nerffacespeech.yaml --prefix environment/nerffacespeech || \
        echo "警告: nerffacespeech 环境创建失败，将使用 pip 安装依赖"; \
    else \
        echo "警告: environment/nerffacespeech.yaml 不存在，跳过 nerffacespeech 环境安装"; \
    fi

# 安装 syncnet 环境（如果环境文件存在）
RUN if [ -f "environment/syncnet.yaml" ]; then \
        echo "安装 syncnet 环境..." && \
        /opt/conda/bin/conda env create -f environment/syncnet.yaml --prefix environment/syncnet || \
        echo "警告: syncnet 环境创建失败，LSE 相关功能可能不可用"; \
    else \
        echo "警告: environment/syncnet.yaml 不存在，跳过 syncnet 环境安装"; \
    fi

# ==================== 下载 SyncNet 模型（用于 LSE） ====================
RUN if [ -f "metrics/scores_LSE/syncnet_python/download_model.sh" ]; then \
        echo "下载 SyncNet 模型和示例数据..." && \
        cd metrics/scores_LSE/syncnet_python && \
        bash download_model.sh || \
        echo "警告: SyncNet 模型下载失败，请在容器内手动运行 download_model.sh"; \
    else \
        echo "警告: 未找到 metrics/scores_LSE/syncnet_python/download_model.sh，跳过 SyncNet 模型下载"; \
    fi

# ==================== 安装 nvdiffrast ====================
RUN cd NeRFFaceSpeech_Code/Deep3DFaceRecon_pytorch && \
    if [ ! -d "nvdiffrast" ]; then \
        echo "克隆 nvdiffrast 仓库..." && \
        git clone https://github.com/NVlabs/nvdiffrast || \
        (echo "警告: nvdiffrast clone 失败，尝试使用 gitee 镜像..." && \
         git clone https://gitee.com/mirrors/nvdiffrast.git || echo "nvdiffrast clone 失败"); \
    fi && \
    if [ -d "nvdiffrast" ]; then \
        cd nvdiffrast && \
        echo "安装 nvdiffrast..." && \
        /opt/conda/bin/pip install . || echo "警告: nvdiffrast 安装失败"; \
    fi

# ==================== 安装 API 环境依赖 ====================
RUN echo "安装 FastAPI 等 API 依赖..." && \
    /opt/conda/bin/pip install --no-cache-dir \
    fastapi==0.119.1 \
    uvicorn==0.38.0 \
    python-multipart==0.0.20 \
    aiofiles==24.1.0 \
    requests==2.32.5 \
    pydantic==2.11.10 || \
    echo "警告: API 依赖安装失败"

# 如果 requirements.txt 存在，安装其中的依赖
RUN if [ -f "requirements.txt" ]; then \
        echo "安装 requirements.txt 中的依赖..." && \
        /opt/conda/bin/pip install --no-cache-dir -r requirements.txt || \
        echo "警告: requirements.txt 安装失败"; \
    fi

# ==================== 创建模型检查脚本 ====================
RUN echo '#!/bin/bash\n\
# 检查模型文件是否存在\n\
MODEL_DIR="/app/NeRFFaceSpeech_Code/pretrained_networks"\n\
MISSING_MODELS=()\n\
\n\
echo "==========================================="\n\
echo "检查模型文件..."\n\
echo "==========================================="\n\
\n\
# 检查主要模型文件\n\
if [ ! -f "$MODEL_DIR/ffhq_1024.pkl" ]; then\n\
    MISSING_MODELS+=("ffhq_1024.pkl")\n\
fi\n\
\n\
if [ ! -f "$MODEL_DIR/seg.pth" ]; then\n\
    MISSING_MODELS+=("seg.pth")\n\
fi\n\
\n\
if [ ! -f "$MODEL_DIR/LipaintNet.pt" ]; then\n\
    MISSING_MODELS+=("LipaintNet.pt")\n\
fi\n\
\n\
# 检查目录\n\
if [ ! -d "$MODEL_DIR/sad_talker_pretrained" ]; then\n\
    MISSING_MODELS+=("sad_talker_pretrained/")\n\
fi\n\
\n\
if [ ! -d "$MODEL_DIR/BFM_for_3DMM-Fitting-Pytorch" ]; then\n\
    MISSING_MODELS+=("BFM_for_3DMM-Fitting-Pytorch/")\n\
fi\n\
\n\
if [ ! -d "$MODEL_DIR/Deep3DFaceRecon_pytorch" ]; then\n\
    MISSING_MODELS+=("Deep3DFaceRecon_pytorch/")\n\
fi\n\
\n\
if [ ${#MISSING_MODELS[@]} -gt 0 ]; then\n\
    echo ""\n\
    echo "❌ 缺少以下模型文件："\n\
    for model in "${MISSING_MODELS[@]}"; do\n\
        echo "   - $model"\n\
    done\n\
    echo ""\n\
    echo "==========================================="\n\
    echo "请下载模型文件到以下目录："\n\
    echo "$MODEL_DIR"\n\
    echo "==========================================="\n\
    echo ""\n\
    echo "模型下载地址："\n\
    echo "https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i"\n\
    echo ""\n\
    echo "需要的模型文件："\n\
    echo "1. ffhq_1024.pkl (184.3 MB)"\n\
    echo "2. seg.pth (50.8 MB)"\n\
    echo "3. LipaintNet.pt (12.4 MB)"\n\
    echo "4. sad_talker_pretrained/ (目录)"\n\
    echo "5. BFM_for_3DMM-Fitting-Pytorch/ (目录)"\n\
    echo "6. Deep3DFaceRecon_pytorch/ (目录)"\n\
    echo ""\n\
    echo "下载后，请将文件放置到："\n\
    echo "NeRFFaceSpeech_Code/pretrained_networks/"\n\
    echo ""\n\
    echo "或者使用 Docker volume 挂载："\n\
    echo "docker run -v /path/to/pretrained_networks:/app/NeRFFaceSpeech_Code/pretrained_networks:ro ..."\n\
    echo "==========================================="\n\
    exit 1\n\
else\n\
    echo "✅ 所有必需的模型文件都存在"\n\
    echo "==========================================="\n\
fi\n\
' > /app/check_models.sh && chmod +x /app/check_models.sh

# ==================== 创建启动脚本 ====================
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# 设置环境变量\n\
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple\n\
export TORCH_HOME=/app/weights\n\
export HF_ENDPOINT=https://hf-mirror.com\n\
export HF_HOME=/app/Hugging_Face\n\
\n\
# 激活 conda base 环境\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate base\n\
\n\
# 检查模型文件（可选，如果挂载了模型目录则跳过）\n\
# 如果模型目录是通过 volume 挂载的，则检查挂载的目录\n\
if [ -f "/app/check_models.sh" ]; then\n\
    /app/check_models.sh || echo "警告: 模型文件检查失败，但继续启动服务..."\n\
fi\n\
\n\
# 启动后端服务器\n\
cd /app/fastapi_server\n\
echo "启动后端服务器..."\n\
if [ -f "/app/environment/api/bin/python" ]; then\n\
    echo "使用 API 环境启动后端..."\n\
    /app/environment/api/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
else\n\
    echo "使用 base 环境启动后端..."\n\
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
fi\n\
\n\
# 等待后端启动\n\
sleep 3\n\
\n\
# 启动前端服务器\n\
echo "启动前端服务器..."\n\
python simple_web.py &\n\
\n\
# 等待服务启动\n\
sleep 2\n\
\n\
echo ""\n\
echo "==========================================="\n\
echo "NeRFFaceSpeech 服务已启动"\n\
echo "后端地址: http://localhost:8000/"\n\
echo "前端地址: http://localhost:7860/"\n\
echo "API 文档: http://localhost:8000/docs"\n\
echo "==========================================="\n\
echo ""\n\
\n\
# 保持容器运行\n\
wait\n\
' > /app/start.sh && chmod +x /app/start.sh

# 暴露端口
EXPOSE 8000 7860

# 设置默认命令
CMD ["/app/start.sh"]

