# 评估流程 Docker 启动脚本使用说明

## 概述

`run_eval_pipeline.sh` 是一个用于在 Docker 环境中运行评估流程的启动脚本。它会自动启动 Docker 容器，使用 syncnet 环境运行完整的评估流程。

## 功能特性

- ✅ 自动检查 Docker 镜像和容器
- ✅ 使用 syncnet 环境运行评估流程
- ✅ 自动创建带时间戳的输出目录
- ✅ 每8秒切分视频
- ✅ 每个视频随机选择8段
- ✅ 使用 FFHQFaceAlignment 进行人脸对齐
- ✅ 使用 ffhq_1024.pkl 模型进行推理
- ✅ 完整的错误处理和日志输出

## 使用方法

### 基本使用

```bash
./run_eval_pipeline.sh
```

### 前置条件

1. **Docker 环境**
   - 已安装 Docker 和 nvidia-docker
   - Docker 镜像 `nerffacespeech:latest` 已构建（脚本会自动检查并构建）

2. **输入数据**
   - 输入视频目录: `data/geneface_datasets/data/raw/videos/`
   - 目录中应包含 `.mp4` 视频文件

3. **模型文件**
   - 模型路径: `NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl`
   - 模型文件必须存在

## 配置参数

脚本中的默认配置：

```bash
INPUT_DIR="data/geneface_datasets/data/raw/videos"
OUTPUT_DIR="output/eval_$(date +%Y%m%d_%H%M%S)"  # 自动生成时间戳
MODEL_PATH="NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl"
SEGMENT_SEC=8          # 每8秒一切
MAX_SEGMENTS=8         # 每个视频随机取8段
```

### 修改配置

如果需要修改配置，编辑 `run_eval_pipeline.sh` 文件中的相应变量：

```bash
# 修改输入目录
INPUT_DIR="$PROJECT_ROOT/data/your_videos"

# 修改输出目录（固定路径）
OUTPUT_DIR="$PROJECT_ROOT/output/my_eval_result"

# 修改模型路径
MODEL_PATH="$PROJECT_ROOT/path/to/your_model.pkl"

# 修改切分参数
SEGMENT_SEC=10         # 每10秒一切
MAX_SEGMENTS=5         # 每个视频随机取5段
```

## 评估流程步骤

脚本会执行以下步骤：

1. **视频切分** (`videos_split/`)
   - 将输入视频每8秒切分为一段
   - 每个视频随机选择8段（如果视频足够长）

2. **人脸检测和裁剪** (`videos_cropped/`)
   - 使用 FFHQFaceAlignment 进行人脸对齐
   - 从第一帧计算对齐参数，应用到所有帧
   - 确保 GT 和生成视频在同一坐标系

3. **模型推理** (`videos_infer/`)
   - 使用 `ffhq_1024.pkl` 模型进行推理
   - 输入：对齐后的视频（第一帧图像 + 完整音频）
   - 输出：生成的视频

4. **指标计算** (`metrics.json`)
   - 计算 FID、LSE-C、LSE-D 等指标
   - 使用对齐后的 GT 视频和生成视频进行比较

## 输出结构

```
output/eval_YYYYMMDD_HHMMSS/
├── videos_split/          # 切分后的视频
│   ├── video1_seg_000.mp4
│   ├── video1_seg_001.mp4
│   └── ...
├── videos_cropped/       # 对齐和裁剪后的视频
│   ├── video1_seg_000.mp4
│   ├── video1_seg_001.mp4
│   └── ...
├── videos_infer/         # 模型推理结果
│   ├── video1_seg_000.mp4
│   ├── video1_seg_001.mp4
│   └── ...
└── metrics.json          # 评估指标结果
```

## 环境要求

### Docker 环境

- **镜像**: `nerffacespeech:latest`
- **环境**: syncnet conda 环境
- **GPU**: 需要 NVIDIA GPU 支持（通过 `--gpus all` 传递）

### 挂载的目录

脚本会自动挂载以下目录：

- `data/` → `/app/data` (只读)
- `output/` → `/app/output` (读写)
- `NeRFFaceSpeech_Code/` → `/app/NeRFFaceSpeech_Code` (只读)
- `eval_pipline/` → `/app/eval_pipline` (只读)
- `weights/` → `/app/weights` (读写，模型缓存)
- `Hugging_Face/` → `/app/Hugging_Face` (读写，HuggingFace 缓存)

## 故障排查

### 1. Docker 镜像不存在

**问题**: 脚本提示镜像不存在

**解决**:
```bash
cd docker
docker build -t nerffacespeech:latest ..
```

### 2. 容器启动失败

**问题**: 容器无法启动

**解决**:
- 检查 Docker 是否运行: `docker ps`
- 检查 GPU 支持: `nvidia-smi`
- 检查端口占用: `docker ps -a`

### 3. 输入目录不存在

**问题**: 脚本提示输入目录不存在

**解决**:
- 确保 `data/geneface_datasets/data/raw/videos/` 目录存在
- 或修改脚本中的 `INPUT_DIR` 变量

### 4. 模型文件不存在

**问题**: 脚本提示模型文件不存在

**解决**:
- 确保 `NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl` 存在
- 或修改脚本中的 `MODEL_PATH` 变量

### 5. syncnet 环境不存在

**问题**: 容器内 syncnet 环境不存在

**解决**:
- 确保 Docker 镜像构建时包含了 syncnet 环境
- 检查 `environment/syncnet.yaml` 是否存在
- 重新构建 Docker 镜像

### 6. FFHQFaceAlignment 依赖缺失

**问题**: FFHQFaceAlignment 相关错误

**解决**:
```bash
# 进入容器
docker exec -it nerffacespeech-eval bash

# 激活 syncnet 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/environment/syncnet

# 安装依赖
cd /app/eval_pipline/FFHQFaceAlignment
pip install -r requirements.txt
python download.py
```

### 7. 查看容器日志

如果评估流程失败，可以查看容器日志：

```bash
docker logs nerffacespeech-eval
```

### 8. 进入容器调试

```bash
docker exec -it nerffacespeech-eval bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/environment/syncnet
cd /app
```

## 高级用法

### 自定义参数

如果需要使用不同的参数，可以修改脚本中的命令部分，或直接使用 Docker 命令：

```bash
docker exec -it nerffacespeech-eval bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate /app/environment/syncnet
    cd /app
    python -m eval_pipline \\
        --input-dir /app/data/your_videos \\
        --output-dir /app/output/your_output \\
        --network /app/path/to/model.pkl \\
        --segment-sec 10 \\
        --max-segments 5 \\
        --random-segments \\
        --ffhq-alignment \\
        --device cuda
"
```

### 跳过某些步骤

如果需要跳过某些步骤（例如只运行推理），可以修改脚本添加相应的跳过参数：

```bash
--skip-split    # 跳过视频切分
--skip-crop     # 跳过人脸裁剪
--skip-infer    # 跳过模型推理
--skip-eval     # 跳过指标计算
```

### 并行处理

如果需要并行处理多个视频，可以：

1. 修改脚本，为每个视频创建单独的输出目录
2. 使用多个容器并行运行
3. 使用任务队列系统（如 Celery）

## 性能优化

1. **GPU 内存**: 确保 GPU 有足够内存（建议至少 8GB）
2. **批量大小**: 可以调整 `--batch-size` 参数（默认 32）
3. **最大帧数**: 可以使用 `--max-frames` 限制处理的帧数
4. **跳过 LSE**: 如果不需要 LSE 指标，可以使用 `--skip-lse`

## 注意事项

1. **首次运行**: 首次运行可能需要下载一些模型文件（如 FFHQFaceAlignment 模型）
2. **存储空间**: 确保有足够的存储空间（输出目录可能很大）
3. **运行时间**: 完整的评估流程可能需要较长时间，取决于视频数量和长度
4. **容器状态**: 脚本会保持容器运行，可以重复使用

## 清理

### 停止并删除容器

```bash
docker stop nerffacespeech-eval
docker rm nerffacespeech-eval
```

### 清理输出目录

```bash
rm -rf output/eval_*
```

## 相关文件

- `run_eval_pipeline.sh` - 启动脚本
- `eval_pipline/__main__.py` - 评估流程主入口
- `docker/Dockerfile` - Docker 镜像定义
- `docker/docker-compose.yml` - Docker Compose 配置

## 更新日志

- **2025-12-20**: 初始版本
  - ✓ 创建 Docker 启动脚本
  - ✓ 支持 syncnet 环境
  - ✓ 支持 FFHQFaceAlignment
  - ✓ 自动创建输出目录
  - ✓ 完整的错误处理

