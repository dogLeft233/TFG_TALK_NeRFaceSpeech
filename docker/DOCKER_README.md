# NeRFFaceSpeech Docker 部署指南

## 📋 目录

- [概述](#概述)
- [模型文件准备](#模型文件准备)
- [构建镜像](#构建镜像)
- [运行容器](#运行容器)
- [验证部署](#验证部署)
- [常见问题](#常见问题)

---

## 概述

本项目使用 Docker 容器化部署，所有下载都通过国内镜像源加速：

- **APT 镜像源**：清华大学镜像源
- **pip 镜像源**：清华大学 PyPI 镜像
- **conda 镜像源**：清华大学 Anaconda 镜像
- **HuggingFace 镜像**：hf-mirror.com

---

## 模型文件准备

### ⚠️ 重要提示

模型文件**不会**被打包到 Docker 镜像中，需要通过以下方式之一提供：

1. **使用 Docker Volume 挂载**（推荐）
2. **在容器启动前下载到主机**

### 模型下载地址

**Google Drive**: https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i

### 需要的模型文件

| 文件/目录 | 大小 | 说明 |
|---------|------|------|
| `ffhq_1024.pkl` | 184.3 MB | 主要的 StyleGAN 模型 |
| `seg.pth` | 50.8 MB | 分割模型 |
| `LipaintNet.pt` | 12.4 MB | Lipaint 模型 |
| `sad_talker_pretrained/` | - | SadTalker 预训练模型目录 |
| `BFM_for_3DMM-Fitting-Pytorch/` | - | BFM 3DMM 拟合模型目录 |
| `Deep3DFaceRecon_pytorch/` | - | Deep3D 人脸重建模型目录 |

### 下载步骤

1. **访问 Google Drive**
   ```
   https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i
   ```

2. **下载所有模型文件到本地**
   ```bash
   # 创建模型目录
   mkdir -p NeRFFaceSpeech_Code/pretrained_networks
   
   # 下载文件到该目录
   # 1. ffhq_1024.pkl
   # 2. seg.pth
   # 3. LipaintNet.pt
   # 4. sad_talker_pretrained/ (整个目录)
   # 5. BFM_for_3DMM-Fitting-Pytorch/ (整个目录)
   # 6. Deep3DFaceRecon_pytorch/ (整个目录)
   ```

3. **验证文件结构**
   ```bash
   ls -lh NeRFFaceSpeech_Code/pretrained_networks/
   # 应该看到：
   # - ffhq_1024.pkl
   # - seg.pth
   # - LipaintNet.pt
   # - sad_talker_pretrained/
   # - BFM_for_3DMM-Fitting-Pytorch/
   # - Deep3DFaceRecon_pytorch/
   ```

---

## 构建镜像

### 前置要求

1. **Docker** (版本 >= 20.10)
   ```bash
   # 检查 Docker 是否安装
   docker --version
   
   # 检查 Docker 权限（如果失败，需要修复权限）
   docker ps
   ```

2. **Docker 权限**（如果遇到权限问题）
   ```bash
   # 运行权限修复脚本
   ./docker/fix_docker_permissions.sh
   
   # 或手动修复
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **NVIDIA Docker** (用于 GPU 支持，可选)
   ```bash
   # 检查 NVIDIA Docker 是否安装
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

### 构建步骤

#### 方法1：使用 docker-compose（推荐）

```bash
# 1. 进入项目根目录
cd /path/to/TFG_TALK_NeRFaceSpeech

# 2. 构建镜像（使用国内镜像源）
docker-compose -f docker/docker-compose.yml build

# 或者不使用缓存重新构建
docker-compose -f docker/docker-compose.yml build --no-cache
```

#### 方法2：使用 docker build

```bash
# 进入项目根目录
cd /path/to/TFG_TALK_NeRFaceSpeech

# 构建镜像（注意：构建上下文是项目根目录，dockerfile 在 docker/ 目录）
docker build -f docker/Dockerfile -t nerffacespeech:latest .

# 或者指定标签
docker build -f docker/Dockerfile -t nerffacespeech:v1.0.0 -t nerffacespeech:latest .
```

### 构建时间

- 首次构建：约 30-60 分钟（取决于网络速度）
- 后续构建：约 10-20 分钟（使用缓存）

---

## 运行容器

### 使用 docker-compose（推荐）

```bash
# 进入项目根目录
cd /path/to/TFG_TALK_NeRFaceSpeech

# 启动容器（后台运行）
docker-compose -f docker/docker-compose.yml up -d

# 查看日志
docker-compose -f docker/docker-compose.yml logs -f

# 停止容器
docker-compose -f docker/docker-compose.yml down

# 重启容器
docker-compose -f docker/docker-compose.yml restart
```

### 使用 docker run

```bash
docker run -d \
  --name nerffacespeech-app \
  --gpus all \
  -p 8000:8000 \
  -p 7860:7860 \
  -v $(pwd)/NeRFFaceSpeech_Code/pretrained_networks:/app/NeRFFaceSpeech_Code/pretrained_networks:ro \
  -v $(pwd)/data:/app/data:rw \
  -v $(pwd)/outputs:/app/outputs:rw \
  -v $(pwd)/database:/app/database:rw \
  -v $(pwd)/weights:/app/weights:rw \
  -v $(pwd)/Hugging_Face:/app/Hugging_Face:rw \
  -v $(pwd)/assets:/app/assets:ro \
  -e CUDA_VISIBLE_DEVICES=0 \
  nerffacespeech:latest
```

### 模型文件检查

容器启动时会自动检查模型文件：

- ✅ **如果模型文件存在**：正常启动服务
- ❌ **如果模型文件缺失**：显示下载提示并退出

**如果看到模型文件缺失提示**：

1. 按照提示下载模型文件
2. 将文件放置到 `NeRFFaceSpeech_Code/pretrained_networks/` 目录
3. 重新启动容器

---

## 验证部署

### 1. 检查容器状态

```bash
docker ps | grep nerffacespeech
```

### 2. 检查容器日志

```bash
docker logs nerffacespeech-app
```

### 3. 检查模型文件

```bash
docker exec nerffacespeech-app ls -lh /app/NeRFFaceSpeech_Code/pretrained_networks/
```

### 4. 测试 API

```bash
# 测试后端 API
curl http://localhost:8000/docs

# 测试模型列表接口
curl http://localhost:8000/models
```

### 5. 访问服务

- **后端 API**: http://localhost:8000/
- **API 文档**: http://localhost:8000/docs
- **前端界面**: http://localhost:7860/

---

## 常见问题

### Q1: Docker 镜像拉取超时或 DNS 解析失败

**问题**：`failed to resolve source metadata`、`i/o timeout` 或 `no such host`

**原因**：访问 Docker Hub 网络较慢或 DNS 解析失败

**解决方案**：

#### 方案1：自动修复镜像加速器（推荐）

```bash
# 自动测试并配置可用的镜像源
./docker/fix_docker_mirror.sh
```

#### 方案2：手动配置镜像加速器

```bash
# 编辑 /etc/docker/daemon.json
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://hub-mirror.c.163.com",
    "https://mirror.ccs.tencentyun.com"
  ]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

#### 方案3：使用代理（如果有）

```bash
# 设置代理环境变量
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 或者在 Docker daemon.json 中配置代理
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "proxies": {
    "http-proxy": "http://your-proxy:port",
    "https-proxy": "http://your-proxy:port"
  }
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

#### 方案4：手动拉取镜像后构建

```bash
# 1. 使用代理或 VPN 手动拉取基础镜像
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 然后构建（会使用本地镜像）
docker-compose -f docker/docker-compose.yml build
```

#### 方案5：修复 DNS（如果是 DNS 问题）

```bash
# 检查 DNS 设置
cat /etc/resolv.conf

# 如果 DNS 有问题，可以临时修改
sudo tee /etc/resolv.conf > /dev/null <<EOF
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 114.114.114.114
EOF
```

### Q2: Docker 权限错误

**问题**：`permission denied while trying to connect to the Docker daemon socket`

**解决方案**：
```bash
# 方案1：运行权限修复脚本（推荐）
./docker/fix_docker_permissions.sh

# 方案2：手动修复
sudo usermod -aG docker $USER
newgrp docker

# 方案3：临时使用 sudo（不推荐）
sudo docker-compose -f docker/docker-compose.yml build
```

### Q3: 构建失败 - 网络问题

**问题**：构建时下载包失败

**解决方案**：
```bash
# 检查网络连接
ping mirrors.tuna.tsinghua.edu.cn

# 如果使用代理，设置代理环境变量
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
docker-compose -f docker/docker-compose.yml build
```

### Q4: GPU 不可用

**问题**：容器无法访问 GPU

**解决方案**：
```bash
# 1. 检查 NVIDIA Docker 运行时
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 2. 如果失败，安装 nvidia-docker2
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 3. 验证安装
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Q5: 模型文件找不到

**问题**：容器启动时提示模型文件缺失

**解决方案**：
```bash
# 1. 检查主机上的模型文件
ls -lh NeRFFaceSpeech_Code/pretrained_networks/

# 2. 检查挂载路径是否正确
docker exec nerffacespeech-app ls -la /app/NeRFFaceSpeech_Code/pretrained_networks/

# 3. 如果文件不存在，按照提示下载：
# https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i
```

### Q6: 端口被占用

**问题**：端口 8000 或 7860 已被占用

**解决方案**：
```bash
# 1. 检查端口占用
netstat -tuln | grep -E '8000|7860'

# 2. 修改 docker-compose.yml 中的端口映射
# 例如：- "18000:8000" - "17860:7860"
```

### Q7: 权限问题

**问题**：容器无法访问挂载的目录

**解决方案**：
```bash
# 确保目录权限正确
chmod -R 755 NeRFFaceSpeech_Code/pretrained_networks
chmod -R 755 outputs database weights Hugging_Face
```

### Q8: 镜像源访问慢

**问题**：即使使用国内镜像源，下载仍然很慢

**解决方案**：
```bash
# 1. 尝试其他镜像源（修改 Dockerfile）
# 阿里云：https://mirrors.aliyun.com
# 中科大：https://mirrors.ustc.edu.cn

# 2. 使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

---

## 镜像源配置说明

### APT 镜像源

已配置为清华大学镜像源：
- `https://mirrors.tuna.tsinghua.edu.cn`

### pip 镜像源

通过环境变量设置：
- `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`

### conda 镜像源

已配置为清华大学镜像源：
- `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`
- `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`
- `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/`

### HuggingFace 镜像

通过环境变量设置：
- `HF_ENDPOINT=https://hf-mirror.com`

---

## 快速开始脚本

使用项目提供的 `docker/build_and_run.sh` 脚本：

```bash
# 进入项目根目录
cd /path/to/TFG_TALK_NeRFaceSpeech

# 运行脚本（会自动检查模型文件、构建镜像、启动容器）
./docker/build_and_run.sh
```

或者：

```bash
# 直接运行（脚本会自动切换到项目根目录）
bash docker/build_and_run.sh
```

脚本功能：
- ✅ 自动检查模型文件是否存在
- ✅ 提示模型下载地址（如果缺失）
- ✅ 检查 Docker 和 GPU 支持
- ✅ 构建 Docker 镜像
- ✅ 启动容器
- ✅ 显示服务状态和访问地址

---

## 联系与支持

- **项目地址**: TFG_TALK_NeRFaceSpeech
- **模型下载**: https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i

---

**祝使用愉快！** 🎉

