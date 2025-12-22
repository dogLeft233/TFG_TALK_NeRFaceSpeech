# Conda 环境激活方式检查

## 激活方式验证

对于使用 `--prefix` 创建的环境，激活方式有两种：

### 方式1: 使用 conda activate（推荐）

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/environment/syncnet
```

**这是正确的方式** ✅

### 方式2: 直接使用 activate 脚本

```bash
source /app/environment/syncnet/bin/activate
```

**这也是正确的方式** ✅

## 当前脚本中的激活方式

在 `run_eval_pipeline.sh` 中：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/environment/syncnet
```

**激活方式是正确的** ✅

## 问题诊断

如果看到 `EnvironmentLocationNotFound: Not a conda environment: /app/environment/syncnet`，说明：

1. **环境目录不存在** - 环境没有被创建
2. **环境目录存在但不是有效的 conda 环境** - 环境创建失败或不完整

## 检查环境是否存在

```bash
# 检查环境目录
docker exec nerffacespeech-eval ls -la /app/environment/syncnet

# 检查是否是有效的 conda 环境
docker exec nerffacespeech-eval test -f /app/environment/syncnet/bin/python && echo "有效环境" || echo "无效环境"
```

## 如果环境不存在，手动创建

```bash
docker exec -it nerffacespeech-eval bash

# 在容器内
source /opt/conda/etc/profile.d/conda.sh
cd /app

# 检查 yaml 文件是否存在
ls -la environment/syncnet.yaml

# 创建环境
conda env create -f environment/syncnet.yaml --prefix environment/syncnet

# 验证
conda activate /app/environment/syncnet
python --version
which python
```

## 验证激活是否成功

```bash
docker exec nerffacespeech-eval bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate /app/environment/syncnet
    echo 'Python 路径: ' \$(which python)
    echo 'Python 版本: ' \$(python --version)
    echo '环境路径: ' \$(python -c 'import sys; print(sys.prefix)')
"
```

应该输出：
```
Python 路径: /app/environment/syncnet/bin/python
Python 版本: Python 3.11.14
环境路径: /app/environment/syncnet
```

## 总结

**激活方式是正确的**，问题在于：
- syncnet 环境在 Docker 构建时可能没有被创建
- 或者环境创建失败但没有报错

**解决方案**：
1. 检查 Docker 构建日志，确认 syncnet 环境是否创建成功
2. 如果环境不存在，重新构建 Docker 镜像
3. 或者在容器内手动创建环境

