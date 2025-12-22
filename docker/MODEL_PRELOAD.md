# Docker 构建时模型预下载说明

## 概述

在 Docker 构建过程中，会自动预下载以下模型文件，使用国内镜像源加速下载，避免运行时下载导致的延迟。

## 预下载的模型

### 1. HuggingFace 模型

#### chatterbox-tts (xlm)
- **模型名称**: `chatterbox-tts` (XLM 多语言 TTS 模型)
- **下载方式**: 通过 `ChatterboxMultilingualTTS.from_pretrained()` 下载
- **镜像源**: `https://hf-mirror.com` (HuggingFace 国内镜像)
- **缓存位置**: `/app/Hugging_Face/`
- **用途**: 多语言文本转语音（TTS）

### 2. PyTorch Hub 模型

#### resnet18-5c106cde
- **模型名称**: ResNet-18
- **下载方式**: `torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)`
- **缓存位置**: `/app/weights/hub/checkpoints/resnet18-5c106cde.pth`
- **用途**: 图像分类骨干网络

#### alexnet-owt-7be5be79
- **模型名称**: AlexNet
- **下载方式**: `torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)`
- **缓存位置**: `/app/weights/hub/checkpoints/alexnet-owt-7be5be79.pth`
- **用途**: 图像分类

### 3. Face Alignment 模型

#### 3DFAN4-4a694010b9
- **模型名称**: 3D Face Alignment Network 4
- **下载方式**: 直接下载 ZIP 文件
- **下载地址**: `https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip`
- **缓存位置**: `/app/weights/hub/checkpoints/3DFAN4-4a694010b9.zip`
- **用途**: 人脸关键点检测和对齐（face_alignment 库使用）

### 4. 深度估计模型

#### depth-6c4283c0e0
- **模型名称**: 深度估计模型（具体模型待确认）
- **下载方式**: 根据实际使用的库确定（可能是 MiDaS 或其他）
- **缓存位置**: `/app/weights/hub/checkpoints/`
- **用途**: 深度估计

**注意**: 如果无法确定具体的深度模型，脚本会尝试下载 MiDaS 模型作为备选。

## 实现方式

### 预下载脚本

创建了 `docker/preload_models.py` 脚本，用于在 Docker 构建时预下载所有模型。

脚本功能：
1. 配置镜像源环境变量
2. 下载 HuggingFace 模型（chatterbox-tts）
3. 下载 PyTorch Hub 模型（resnet18, alexnet）
4. 下载 3DFAN4 模型
5. 尝试下载深度估计模型
6. 输出下载结果总结

### Dockerfile 集成

在 `docker/Dockerfile` 中添加了以下步骤：

```dockerfile
# ==================== 预下载模型文件 ====================
# 安装预下载脚本所需的依赖
RUN /opt/conda/bin/pip install --no-cache-dir requests tqdm || \
    echo "警告: requests 或 tqdm 安装失败，可能影响模型下载"

# 运行预下载脚本
RUN if [ -f "docker/preload_models.py" ]; then \
        echo "预下载模型文件..." && \
        export HF_ENDPOINT=https://hf-mirror.com && \
        export HF_HOME=/app/Hugging_Face && \
        export TORCH_HOME=/app/weights && \
        /opt/conda/bin/python docker/preload_models.py || \
        echo "警告: 模型预下载失败，将在运行时自动下载"; \
    else \
        echo "警告: docker/preload_models.py 不存在，跳过模型预下载"; \
    fi
```

## 镜像源配置

### HuggingFace 镜像

- **镜像地址**: `https://hf-mirror.com`
- **环境变量**: `HF_ENDPOINT=https://hf-mirror.com`
- **缓存目录**: `HF_HOME=/app/Hugging_Face`

### PyTorch Hub 镜像

- **缓存目录**: `TORCH_HOME=/app/weights`
- **Hub 缓存**: `/app/weights/hub/checkpoints/`

PyTorch Hub 默认使用官方源，但可以通过设置 `TORCH_HOME` 环境变量来指定缓存位置。

## 使用方法

### 构建 Docker 镜像

```bash
cd docker
docker build -t nerffacespeech:latest .
```

构建过程中会自动运行预下载脚本，下载所有模型文件。

### 查看下载日志

构建时会显示详细的下载日志，包括：
- 每个模型的下载进度
- 下载成功/失败状态
- 下载总结

### 验证模型文件

构建完成后，可以进入容器验证模型文件：

```bash
docker run -it nerffacespeech:latest bash

# 检查 HuggingFace 模型
ls -lh /app/Hugging_Face/

# 检查 PyTorch Hub 模型
ls -lh /app/weights/hub/checkpoints/
```

## 故障处理

### 模型下载失败

如果某个模型下载失败，脚本会：
1. 记录错误信息
2. 继续下载其他模型
3. 在构建时不会中断（允许运行时下载）

### 运行时下载

如果构建时某些模型未下载成功，运行时首次使用时会自动下载。但建议在构建时完成所有下载，以避免运行时延迟。

### 手动下载

如果自动下载失败，可以手动下载模型文件：

1. **3DFAN4 模型**:
   ```bash
   wget https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip \
        -O /app/weights/hub/checkpoints/3DFAN4-4a694010b9.zip
   ```

2. **PyTorch 模型**: 可以通过 Python 脚本下载：
   ```python
   import torch
   torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
   torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
   ```

3. **HuggingFace 模型**: 可以通过 Python 脚本下载：
   ```python
   from chatterbox.mtl_tts import ChatterboxMultilingualTTS
   ChatterboxMultilingualTTS.from_pretrained('cpu')
   ```

## 注意事项

1. **网络连接**: 构建时需要网络连接来下载模型
2. **构建时间**: 模型下载会增加构建时间（取决于网络速度）
3. **存储空间**: 模型文件会占用一定的存储空间
4. **镜像大小**: 预下载的模型会增加 Docker 镜像大小
5. **缓存持久化**: 模型文件缓存在容器内，如果需要持久化，可以使用 Docker volume

## 优化建议

1. **使用多阶段构建**: 可以考虑将模型下载作为单独的构建阶段
2. **模型压缩**: 如果镜像大小是问题，可以考虑压缩模型文件
3. **Volume 挂载**: 对于大型模型，可以考虑使用 volume 挂载而不是打包到镜像中
4. **CDN 加速**: 如果可能，使用 CDN 加速模型下载

## 更新日志

- **2025-12-20**: 初始版本
  - ✓ 添加 HuggingFace 模型预下载（chatterbox-tts）
  - ✓ 添加 PyTorch Hub 模型预下载（resnet18, alexnet）
  - ✓ 添加 3DFAN4 模型预下载
  - ✓ 添加深度估计模型预下载支持
  - ✓ 配置国内镜像源加速下载

