# Docker 镜像重新构建指南

## 概述

当修改了 `Dockerfile` 后，需要重新构建 Docker 镜像以使更改生效。本文档提供了多种重新构建镜像的方法。

## 方法1: 使用 docker build 命令（推荐）

### 基本命令

**重要**: Dockerfile 在 `docker/` 目录下，构建上下文是项目根目录，需要使用 `-f` 参数指定 Dockerfile 路径。

```bash
# 方法1: 在项目根目录执行（推荐）
cd /path/to/TFG_TALK_NeRFaceSpeech
docker build -f docker/Dockerfile -t nerffacespeech:latest .

# 方法2: 在 docker 目录执行
cd docker
docker build -f Dockerfile -t nerffacespeech:latest ..
```

### 参数说明

- `-t nerffacespeech:latest`: 指定镜像名称和标签
- `..`: 构建上下文目录（项目根目录）

### 完整命令（带清理）

```bash
# 进入项目根目录
cd /path/to/TFG_TALK_NeRFaceSpeech

# 停止并删除旧容器（如果存在）
docker stop nerffacespeech-app nerffacespeech-eval 2>/dev/null || true
docker rm nerffacespeech-app nerffacespeech-eval 2>/dev/null || true

# 删除旧镜像（可选）
docker rmi nerffacespeech:latest 2>/dev/null || true

# 重新构建镜像（注意：使用 -f 指定 Dockerfile 路径）
docker build -f docker/Dockerfile -t nerffacespeech:latest .
```

## 方法2: 使用 docker-compose

### 基本命令

```bash
cd docker
docker-compose build
```

### 强制重新构建（不使用缓存）

```bash
cd docker
docker-compose build --no-cache
```

### 重新构建并启动

```bash
cd docker
docker-compose up --build -d
```

参数说明：
- `--build`: 构建镜像
- `-d`: 后台运行

## 方法3: 使用项目构建脚本

项目根目录的 `build_and_run.sh` 脚本已经包含了构建逻辑：

```bash
./build_and_run.sh
```

该脚本会：
1. 检查模型文件
2. 构建 Docker 镜像
3. 启动容器

## 方法4: 完全清理后重建

如果需要完全清理所有相关资源：

```bash
cd docker

# 1. 停止并删除所有相关容器
docker stop nerffacespeech-app nerffacespeech-eval 2>/dev/null || true
docker rm nerffacespeech-app nerffacespeech-eval 2>/dev/null || true

# 2. 删除镜像
docker rmi nerffacespeech:latest 2>/dev/null || true

# 3. 清理未使用的资源（可选）
docker system prune -f

# 4. 重新构建
docker build -t nerffacespeech:latest ..
```

## 常用构建选项

### 不使用缓存构建

```bash
# 在项目根目录执行
docker build -f docker/Dockerfile --no-cache -t nerffacespeech:latest .
```

**使用场景**: 
- 确保所有层都重新构建
- 依赖包有更新时
- 怀疑缓存导致的问题

### 指定构建参数

```bash
docker build \
    --build-arg PYTHON_VERSION=3.10 \
    --build-arg CUDA_VERSION=11.7 \
    -t nerffacespeech:latest ..
```

### 查看构建进度

```bash
docker build --progress=plain -t nerffacespeech:latest ..
```

### 输出构建日志到文件

```bash
docker build -t nerffacespeech:latest .. 2>&1 | tee build.log
```

## 验证构建结果

### 检查镜像是否创建成功

```bash
docker images | grep nerffacespeech
```

### 查看镜像详细信息

```bash
docker inspect nerffacespeech:latest
```

### 查看镜像大小

```bash
docker images nerffacespeech:latest
```

### 测试镜像

```bash
docker run --rm nerffacespeech:latest echo "镜像构建成功"
```

## 构建过程中的常见问题

### 1. 构建失败：找不到文件

**问题**: `COPY failed: file not found`

**解决**: 检查 `.dockerignore` 文件，确保没有排除需要的文件

### 2. 构建失败：权限错误

**问题**: `permission denied`

**解决**: 
```bash
# 使用 sudo（不推荐）
sudo docker build -t nerffacespeech:latest ..

# 或添加用户到 docker 组（推荐）
sudo usermod -aG docker $USER
# 然后重新登录
```

### 3. 构建失败：网络超时

**问题**: 下载依赖时超时

**解决**: 
- 检查网络连接
- 使用国内镜像源（已在 Dockerfile 中配置）
- 使用代理：
```bash
docker build --build-arg HTTP_PROXY=http://proxy:port --build-arg HTTPS_PROXY=http://proxy:port -t nerffacespeech:latest ..
```

### 4. 构建失败：磁盘空间不足

**问题**: `no space left on device`

**解决**:
```bash
# 清理未使用的镜像和容器
docker system prune -a

# 清理构建缓存
docker builder prune
```

### 5. 构建时间过长

**优化建议**:
- 使用多阶段构建
- 合理使用 `.dockerignore`
- 利用 Docker 缓存（将变化少的层放在前面）

## 增量构建 vs 完全重建

### 增量构建（使用缓存）

```bash
docker build -t nerffacespeech:latest ..
```

**优点**: 快速，只重建变化的层

**适用场景**: 修改了少量文件

### 完全重建（不使用缓存）

```bash
docker build --no-cache -t nerffacespeech:latest ..
```

**优点**: 确保所有依赖都是最新的

**适用场景**: 
- 修改了基础镜像
- 依赖包有重大更新
- 怀疑缓存导致的问题

## 最佳实践

### 1. 使用版本标签

不要只使用 `latest` 标签，建议同时使用版本号：

```bash
docker build -t nerffacespeech:latest -t nerffacespeech:v1.0.0 ..
```

### 2. 构建前检查

```bash
# 检查 Dockerfile 语法（使用 hadolint）
hadolint docker/Dockerfile

# 或使用 docker build 的 dry-run（需要 Docker BuildKit）
DOCKER_BUILDKIT=1 docker build --dry-run -t nerffacespeech:latest ..
```

### 3. 保存构建日志

```bash
docker build -t nerffacespeech:latest .. 2>&1 | tee build_$(date +%Y%m%d_%H%M%S).log
```

### 4. 多阶段构建优化

如果 Dockerfile 使用多阶段构建，可以只构建特定阶段：

```bash
docker build --target stage_name -t nerffacespeech:latest ..
```

## 自动化脚本

可以创建一个快速重建脚本 `rebuild_docker.sh`:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/docker"

echo "==========================================="
echo "重新构建 Docker 镜像"
echo "==========================================="

# 停止并删除旧容器
echo "清理旧容器..."
docker stop nerffacespeech-app nerffacespeech-eval 2>/dev/null || true
docker rm nerffacespeech-app nerffacespeech-eval 2>/dev/null || true

# 询问是否删除旧镜像
read -p "是否删除旧镜像? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rmi nerffacespeech:latest 2>/dev/null || true
fi

# 询问是否使用缓存
read -p "是否使用缓存构建? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    BUILD_ARGS="--no-cache"
else
    BUILD_ARGS=""
fi

# 构建镜像
echo "开始构建镜像..."
docker build $BUILD_ARGS -t nerffacespeech:latest ..

echo ""
echo "==========================================="
echo "构建完成！"
echo "==========================================="
docker images | grep nerffacespeech
```

## 快速参考

### 最常用的命令

```bash
# 快速重建（使用缓存）- 在项目根目录执行
docker build -f docker/Dockerfile -t nerffacespeech:latest .

# 完全重建（不使用缓存）- 在项目根目录执行
docker build -f docker/Dockerfile --no-cache -t nerffacespeech:latest .

# 使用 docker-compose（推荐，自动处理路径）
cd docker && docker-compose build

# 使用 docker-compose 完全重建
cd docker && docker-compose build --no-cache
```

### 检查构建结果

```bash
# 查看镜像
docker images nerffacespeech:latest

# 查看镜像历史
docker history nerffacespeech:latest

# 查看镜像层
docker inspect nerffacespeech:latest | jq '.[0].RootFS.Layers'
```

## 相关文件

- `docker/Dockerfile` - Docker 镜像定义文件
- `docker/docker-compose.yml` - Docker Compose 配置
- `build_and_run.sh` - 项目构建和运行脚本
- `.dockerignore` - Docker 构建忽略文件列表

## 更新日志

- **2025-12-20**: 初始版本
  - ✓ 添加基本构建命令
  - ✓ 添加 docker-compose 构建方法
  - ✓ 添加故障排查指南
  - ✓ 添加最佳实践

