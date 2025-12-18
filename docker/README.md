# Docker 部署说明

本目录包含 NeRFFaceSpeech 项目的所有 Docker 相关配置文件。

## 📁 文件说明

- **Dockerfile** - Docker 镜像构建文件（使用国内镜像源）
- **docker-compose.yml** - Docker Compose 配置文件
- **.dockerignore** - Docker 构建时忽略的文件列表
- **DOCKER_README.md** - 详细的部署文档和使用说明
- **build_and_run.sh** - 一键构建和运行脚本

## 🚀 快速开始

### 前置步骤

#### 1. 配置 Docker 镜像加速器（推荐，解决拉取镜像超时问题）

如果遇到镜像拉取超时或 DNS 解析失败，先修复镜像加速器：

```bash
# 在项目根目录下运行（自动测试并配置可用的镜像源）
cd /path/to/TFG_TALK_NeRFaceSpeech
./docker/fix_docker_mirror.sh
```

或者手动配置（网易云镜像，通常更稳定）：

```bash
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

如果仍然失败，可以尝试：
- 使用代理
- 手动拉取镜像：`docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- 检查 DNS 设置

#### 2. 修复 Docker 权限（如需要）

如果遇到 `permission denied` 错误，先运行权限修复脚本：

```bash
# 在项目根目录下运行
cd /path/to/TFG_TALK_NeRFaceSpeech
./docker/fix_docker_permissions.sh
```

或者手动修复：

```bash
# 将用户添加到 docker 组
sudo usermod -aG docker $USER

# 重新登录或执行
newgrp docker
```

### 方法1：使用一键脚本（推荐）

```bash
# 在项目根目录下运行
cd /path/to/TFG_TALK_NeRFaceSpeech
./docker/build_and_run.sh
```

### 方法2：使用 docker-compose

```bash
# 在项目根目录下运行
cd /path/to/TFG_TALK_NeRFaceSpeech

# 构建镜像
docker-compose -f docker/docker-compose.yml build

# 启动容器
docker-compose -f docker/docker-compose.yml up -d
```

### 方法3：使用 docker build

```bash
# 在项目根目录下运行
cd /path/to/TFG_TALK_NeRFaceSpeech

# 构建镜像
docker build -f docker/Dockerfile -t nerffacespeech:latest .

# 运行容器
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
  nerffacespeech:latest
```

## 📖 详细文档

查看 [DOCKER_README.md](./DOCKER_README.md) 获取完整的部署指南，包括：

- 模型文件准备
- 构建镜像详细步骤
- 运行容器说明
- 常见问题解决
- 镜像源配置说明

## ⚠️ 重要提示

1. **模型文件**：模型文件不会打包到镜像中，需要通过 volume 挂载或提前下载
2. **模型下载地址**：https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i
3. **GPU 支持**：需要安装 NVIDIA Docker 运行时
4. **国内镜像源**：所有下载都配置为使用国内镜像源加速

## 🔗 相关链接

- 项目根目录 README: ../README.md
- 详细部署文档: [DOCKER_README.md](./DOCKER_README.md)

