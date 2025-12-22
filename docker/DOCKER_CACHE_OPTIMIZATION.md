# Docker 分层构建缓存优化指南

## 概述

优化后的 Dockerfile 采用**分层构建策略**，充分利用 Docker 的层缓存机制。当构建过程中某个步骤失败时，重新构建会从**失败的层开始**，而不是从头开始。

---

## 优化策略

### 1. 分层原则

按照**变化频率从低到高**的顺序组织层：

```
变化频率低 → 变化频率高
系统依赖 → 环境文件 → 项目代码
```

### 2. 关键优化点

#### ✅ 分离依赖安装和代码复制

**优化前（❌ 不好）：**
```dockerfile
COPY . /app/                    # 代码变化频繁，导致后续层全部失效
RUN conda env create ...        # 即使环境文件没变，也要重新执行
```

**优化后（✅ 好）：**
```dockerfile
COPY environment/*.yaml /app/environment/  # 只复制环境文件
RUN conda env create ...                    # 环境文件变化时才重新执行
COPY . /app/                                # 代码放在最后
```

#### ✅ 将耗时操作单独分层

每个 conda 环境创建、pip 安装都应该分开：

```dockerfile
# 第 8 层：llm_talk 环境
RUN conda env create -f environment/llm_talk.yaml ...

# 第 10 层：nerffacespeech 环境
RUN conda env create -f environment/nerffacespeech.yaml ...

# 第 11 层：syncnet 环境
RUN conda env create -f environment/syncnet.yaml ...
```

**优势：** 如果 `llm_talk` 环境创建失败，修复后重新构建时，前面的层（系统依赖、Miniconda 等）都会使用缓存。

---

## 分层结构说明

### 第 1-6 层：基础环境（缓存命中率：⭐⭐⭐⭐⭐）

```dockerfile
# 第 1 层：APT 镜像源配置
# 第 2 层：系统依赖安装
# 第 3 层：Miniconda 安装
# 第 4 层：conda 镜像源配置
# 第 5 层：Conda ToS 接受
# 第 6 层：创建目录结构
```

**特点：**
- 这些层**几乎不会变化**
- 缓存命中率接近 100%
- 即使后续步骤失败，这些层也不会重新构建

---

### 第 7 层：环境文件复制（缓存命中率：⭐⭐⭐⭐）

```dockerfile
COPY environment/*.yaml /app/environment/
COPY environment/env_tar/ /app/environment/env_tar/
```

**特点：**
- 只复制环境定义文件，不复制整个项目
- 环境文件变化频率较低
- 如果环境文件没变，后续环境创建层可以使用缓存

---

### 第 8-11 层：Conda 环境创建（缓存命中率：⭐⭐⭐）

```dockerfile
# 第 8 层：llm_talk 环境
# 第 9 层：PyTorch 和 pkuseg 安装
# 第 10 层：nerffacespeech 环境
# 第 11 层：syncnet 环境
```

**特点：**
- 每个环境单独一层
- 如果某个环境创建失败，修复后重新构建时，其他环境层可以使用缓存
- 耗时最长，但缓存效果最好

---

### 第 12 层：API 依赖（缓存命中率：⭐⭐⭐⭐）

```dockerfile
RUN pip install fastapi uvicorn ...
```

**特点：**
- API 依赖变化频率较低
- 单独一层，便于管理和缓存

---

### 第 13 层：项目代码复制（缓存命中率：⭐）

```dockerfile
COPY . /app/
```

**特点：**
- 代码变化最频繁
- 放在最后，最大化前面的缓存
- 使用 `.dockerignore` 排除不必要的文件

---

### 第 14-18 层：项目特定操作（缓存命中率：⭐⭐）

```dockerfile
# 第 14 层：requirements.txt 安装
# 第 15 层：SyncNet 模型下载
# 第 16 层：nvdiffrast 安装
# 第 17 层：模型预下载
# 第 18 层：创建辅助脚本
```

**特点：**
- 这些操作依赖项目代码
- 如果代码没变，可以使用缓存

---

## 使用优化后的 Dockerfile

### 方式 1：直接替换（推荐）

```bash
cd /mnt/e/Documents/TFG_TALK_NeRFaceSpeech/docker
cp Dockerfile Dockerfile.original  # 备份原文件
cp Dockerfile.optimized Dockerfile  # 使用优化版本
```

### 方式 2：手动合并

将 `Dockerfile.optimized` 中的优化点手动应用到 `Dockerfile`。

---

## 缓存效果对比

### 场景 1：代码修改后重新构建

**优化前：**
```
从头开始构建：~30 分钟
```

**优化后：**
```
使用缓存到第 13 层：~2 分钟（只重新构建代码相关层）
```

### 场景 2：环境文件修改后重新构建

**优化前：**
```
从头开始构建：~30 分钟
```

**优化后：**
```
使用缓存到第 7 层：~15 分钟（只重新构建环境相关层）
```

### 场景 3：构建过程中失败（如网络问题）

**优化前：**
```
修复后重新构建：~30 分钟（从头开始）
```

**优化后：**
```
修复后重新构建：从失败层开始，使用前面所有缓存
例如：第 9 层失败 → 修复后从第 9 层开始，前 8 层使用缓存
```

---

## 验证缓存效果

### 查看构建过程

```bash
docker build -t nerffacespeech:test -f docker/Dockerfile.optimized .
```

观察输出中的 `CACHED` 标记：

```
Step 1/18 : FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
 ---> Using cache
 ---> abc123def456

Step 2/18 : WORKDIR /app
 ---> Using cache
 ---> def456ghi789
```

### 强制不使用缓存（测试）

```bash
docker build --no-cache -t nerffacespeech:test -f docker/Dockerfile.optimized .
```

---

## 最佳实践

### 1. 使用 .dockerignore

确保 `.dockerignore` 排除不必要的文件：

```
# 模型文件（太大，通过 volume 挂载）
**/pretrained_networks/
**/weights/
**/Hugging_Face/

# 输出文件
**/outputs/
**/database/videos/

# 临时文件
**/.git/
**/__pycache__/
**/*.pyc
```

### 2. 环境文件版本控制

- ✅ 提交 `environment/*.yaml` 到 git
- ❌ 不提交 `environment/env_tar/*.tar`（太大）
- ✅ 在 CI/CD 中自动生成 tar 文件

### 3. 构建参数化

对于可能变化的配置，使用构建参数：

```dockerfile
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.0.1

RUN conda create -n myenv python=${PYTHON_VERSION}
RUN pip install torch==${TORCH_VERSION}
```

### 4. 多阶段构建（可选）

对于更复杂的场景，可以使用多阶段构建：

```dockerfile
# 阶段 1：构建环境
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS builder
# ... 安装所有依赖

# 阶段 2：运行环境
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
COPY --from=builder /opt/conda /opt/conda
# ... 只复制运行时需要的文件
```

---

## 故障排查

### Q1: 为什么缓存没有生效？

**可能原因：**
1. Dockerfile 指令顺序改变
2. COPY 的文件内容变化
3. RUN 命令中的变量或时间戳导致变化

**解决：**
- 检查 Dockerfile 指令顺序
- 使用 `docker history` 查看层变化
- 确保 `.dockerignore` 正确配置

### Q2: 如何清理缓存？

```bash
# 清理所有构建缓存
docker builder prune

# 清理未使用的镜像
docker image prune

# 清理所有未使用的资源
docker system prune -a
```

### Q3: 如何查看层大小？

```bash
docker history nerffacespeech:latest
```

---

## 总结

优化后的 Dockerfile 通过**分层构建策略**实现了：

1. ✅ **快速重建**：失败后从失败层开始，而不是从头开始
2. ✅ **高效缓存**：变化频率低的层缓存命中率高
3. ✅ **易于调试**：每个步骤独立，便于定位问题
4. ✅ **节省时间**：代码修改后只需重建相关层

**建议：** 在生产环境中使用优化后的 Dockerfile，可以显著提高构建效率。

