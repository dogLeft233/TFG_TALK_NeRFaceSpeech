# CUDA 11.7 环境配置说明

## 概述

本项目已针对 **CUDA 11.7** 进行了优化配置，解决了以下关键问题：

1. ✅ **PyTorch 版本匹配**：使用 `torch==2.0.1 cu117`（而非 `torch==2.6.0 cu126`）
2. ✅ **pkuseg 构建问题**：使用 `--no-use-pep517` 解决构建隔离问题
3. ✅ **Python 版本**：使用 Python 3.10（更稳定，兼容性更好）
4. ✅ **依赖安装顺序**：确保构建依赖（numpy / cython）在编译型包之前安装

---

## 关键修改

### 1. PyTorch 版本（CUDA 11.7 兼容）

**❌ 错误配置（之前）：**
```dockerfile
pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

**✅ 正确配置（现在）：**
```dockerfile
pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

**原因：**
- CUDA 11.7 基础镜像必须使用 `cu117` 版本的 PyTorch
- `torch==2.6.0 cu126` 是为 CUDA 12.6 设计的，不匹配

---

### 2. pkuseg 安装方式（解决构建隔离问题）

**❌ 错误配置（之前）：**
```dockerfile
pip install --no-build-isolation pkuseg==0.0.25
```

**✅ 正确配置（现在）：**
```dockerfile
# 1. 先确保构建依赖已安装
pip install --upgrade numpy==1.26.4 setuptools wheel cython

# 2. 使用 --no-use-pep517 禁用 PEP517，直接使用环境中的 numpy
pip install --no-use-pep517 pkuseg==0.0.25
pip install --no-use-pep517 spacy-pkuseg==1.0.1
```

**原因：**
- `pkuseg` 使用 `pyproject.toml`，pip 默认启用 PEP517 构建隔离
- 即使 `--no-build-isolation`，PEP517 仍会创建隔离环境
- `--no-use-pep517` 强制使用传统 `setup.py`，直接访问环境中的 numpy

---

### 3. Python 版本（从 3.11 降级到 3.10）

**修改位置：** `environment/llm_talk.yaml`

```yaml
dependencies:
  - python=3.10  # 从 3.11 改为 3.10
```

**原因：**
- Python 3.10 在 CUDA 11.7 + PyTorch 2.0 环境下更稳定
- 减少潜在的兼容性问题

---

### 4. 依赖安装顺序优化

**Dockerfile 中的安装顺序：**

```dockerfile
# [1/4] 构建依赖（numpy / setuptools / wheel / cython）
pip install --upgrade numpy==1.26.4 setuptools wheel cython

# [2/4] PyTorch（CUDA 11.7 版本）
pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# [3/4] pkuseg（禁用 PEP517）
pip install --no-use-pep517 pkuseg==0.0.25

# [4/4] spacy-pkuseg（禁用 PEP517）
pip install --no-use-pep517 spacy-pkuseg==1.0.1
```

---

## 版本对照表

| 组件 | CUDA 11.7 版本 | 说明 |
|------|---------------|------|
| 基础镜像 | `nvidia/cuda:11.7.1-devel-ubuntu20.04` | CUDA 11.7 开发环境 |
| Python | `3.10` | 稳定版本 |
| PyTorch | `2.0.1` | cu117 版本 |
| torchvision | `0.15.2` | 匹配 PyTorch 2.0.1 |
| torchaudio | `2.0.2` | 匹配 PyTorch 2.0.1 |
| numpy | `1.26.4` | 构建依赖 |
| pkuseg | `0.0.25` | 使用 `--no-use-pep517` |

---

## 构建和运行

### 1. 构建 Docker 镜像

```bash
cd /mnt/e/Documents/TFG_TALK_NeRFaceSpeech/docker
docker-compose build
```

### 2. 验证安装

进入容器后，验证 PyTorch 和 pkuseg：

```bash
docker exec -it nerffacespeech-eval bash

# 激活环境
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/environment/llm_talk

# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证 pkuseg
python -c "import pkuseg; print('pkuseg 安装成功')"
```

---

## 常见问题

### Q1: 为什么不用 `torch==2.6.0`？

**A:** `torch==2.6.0` 需要 CUDA 12.6，而基础镜像是 CUDA 11.7。版本不匹配会导致运行时错误。

### Q2: `--no-use-pep517` 和 `--no-build-isolation` 的区别？

**A:**
- `--no-build-isolation`：禁用构建隔离，但 PEP517 仍可能生效
- `--no-use-pep517`：完全禁用 PEP517，强制使用传统 `setup.py`，直接访问环境依赖

### Q3: 如果我想升级到 CUDA 12.6 怎么办？

**A:** 需要：
1. 修改基础镜像：`FROM nvidia/cuda:12.6.2-devel-ubuntu22.04`
2. 修改 PyTorch 版本：`torch==2.6.0 cu126`
3. 可能需要调整 Python 版本和其他依赖

---

## 参考

- [PyTorch 官方安装指南](https://pytorch.org/get-started/previous-versions/)
- [pip PEP517 说明](https://peps.python.org/pep-0517/)
- [pkuseg GitHub](https://github.com/lancopku/pkuseg-python)

