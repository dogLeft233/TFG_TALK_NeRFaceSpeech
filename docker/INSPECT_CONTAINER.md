# 进入 Docker 容器检查环境

## 快速进入容器

### 方法1: 使用 docker exec（推荐）

```bash
docker exec -it nerffacespeech-eval bash
```

如果容器名称不同，先查看容器名称：

```bash
docker ps
```

然后使用实际的容器名称：

```bash
docker exec -it <容器名称> bash
```

### 方法2: 使用 docker-compose

```bash
cd docker
docker-compose exec nerffacespeech bash
```

## 检查环境是否存在

### 1. 检查环境目录

```bash
# 进入容器后
ls -la /app/environment/

# 应该看到类似：
# drwxr-xr-x  syncnet
# drwxr-xr-x  nerffacespeech
# drwxr-xr-x  llm_talk
```

### 2. 检查 syncnet 环境

```bash
# 检查目录是否存在
ls -la /app/environment/syncnet

# 检查是否是有效的 conda 环境（应该有 bin/python）
test -f /app/environment/syncnet/bin/python && echo "✅ 有效环境" || echo "❌ 无效环境"

# 检查环境大小
du -sh /app/environment/syncnet
```

### 3. 列出所有 conda 环境

```bash
source /opt/conda/etc/profile.d/conda.sh
conda env list
```

应该看到类似输出：

```
# conda environments:
#
base                     /opt/conda
                         /app/environment/syncnet
                         /app/environment/nerffacespeech
                         /app/environment/llm_talk
```

### 4. 尝试激活环境

```bash
source /opt/conda/etc/profile.d/conda.sh

# 尝试激活 syncnet 环境
conda activate /app/environment/syncnet

# 如果成功，检查 Python 路径
which python
python --version

# 检查环境路径
python -c "import sys; print(sys.prefix)"
```

## 完整检查脚本

在容器内运行以下命令进行完整检查：

```bash
#!/bin/bash
echo "==========================================="
echo "检查 Conda 环境"
echo "==========================================="

# 初始化 conda
source /opt/conda/etc/profile.d/conda.sh

echo ""
echo "1. 检查环境目录："
ls -la /app/environment/ 2>/dev/null || echo "❌ /app/environment/ 不存在"

echo ""
echo "2. 检查 syncnet 环境："
if [ -d "/app/environment/syncnet" ]; then
    echo "✅ syncnet 目录存在"
    if [ -f "/app/environment/syncnet/bin/python" ]; then
        echo "✅ syncnet 是有效的 conda 环境"
        echo "   Python 版本: $(/app/environment/syncnet/bin/python --version 2>&1)"
        echo "   环境大小: $(du -sh /app/environment/syncnet 2>/dev/null | cut -f1)"
    else
        echo "❌ syncnet 目录存在但不是有效的 conda 环境"
    fi
else
    echo "❌ syncnet 目录不存在"
fi

echo ""
echo "3. 检查 nerffacespeech 环境："
if [ -d "/app/environment/nerffacespeech" ]; then
    echo "✅ nerffacespeech 目录存在"
    if [ -f "/app/environment/nerffacespeech/bin/python" ]; then
        echo "✅ nerffacespeech 是有效的 conda 环境"
        echo "   Python 版本: $(/app/environment/nerffacespeech/bin/python --version 2>&1)"
    else
        echo "❌ nerffacespeech 目录存在但不是有效的 conda 环境"
    fi
else
    echo "❌ nerffacespeech 目录不存在"
fi

echo ""
echo "4. Conda 环境列表："
conda env list

echo ""
echo "5. 检查环境 YAML 文件："
ls -la /app/environment/*.yaml 2>/dev/null || echo "❌ 没有找到 YAML 文件"

echo ""
echo "6. 测试激活 syncnet 环境："
if conda activate /app/environment/syncnet 2>/dev/null; then
    echo "✅ syncnet 环境可以激活"
    echo "   Python 路径: $(which python)"
    echo "   Python 版本: $(python --version)"
    conda deactivate
else
    echo "❌ syncnet 环境无法激活"
fi

echo ""
echo "==========================================="
```

## 一键检查命令

### 在主机上直接运行检查

```bash
docker exec nerffacespeech-eval bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    echo '=== 环境目录检查 ==='
    ls -la /app/environment/ 2>/dev/null || echo '环境目录不存在'
    echo ''
    echo '=== syncnet 环境检查 ==='
    if [ -d '/app/environment/syncnet' ]; then
        echo '✅ syncnet 目录存在'
        if [ -f '/app/environment/syncnet/bin/python' ]; then
            echo '✅ syncnet 是有效环境'
            /app/environment/syncnet/bin/python --version
        else
            echo '❌ syncnet 不是有效环境'
        fi
    else
        echo '❌ syncnet 目录不存在'
    fi
    echo ''
    echo '=== Conda 环境列表 ==='
    conda env list
"
```

## 如果环境不存在，手动创建

### 在容器内创建 syncnet 环境

```bash
# 1. 进入容器
docker exec -it nerffacespeech-eval bash

# 2. 初始化 conda
source /opt/conda/etc/profile.d/conda.sh

# 3. 检查 YAML 文件是否存在
cd /app
ls -la environment/syncnet.yaml

# 4. 创建环境（这可能需要较长时间）
conda env create -f environment/syncnet.yaml --prefix environment/syncnet

# 5. 验证环境
conda activate /app/environment/syncnet
python --version
which python

# 6. 测试导入关键包
python -c "import cv2; print('opencv-python:', cv2.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
```

## 检查环境中的包

### 查看已安装的包

```bash
# 激活环境后
conda list

# 或查看特定包
conda list | grep opencv
conda list | grep torch
pip list | grep opencv
```

### 检查关键依赖

```bash
# 激活 syncnet 环境后
python -c "
import sys
print('Python:', sys.version)
try:
    import cv2
    print('✅ opencv-python:', cv2.__version__)
except ImportError:
    print('❌ opencv-python 未安装')

try:
    import torch
    print('✅ torch:', torch.__version__)
except ImportError:
    print('❌ torch 未安装')

try:
    import numpy
    print('✅ numpy:', numpy.__version__)
except ImportError:
    print('❌ numpy 未安装')
"
```

## 常见问题排查

### 问题1: 容器不存在

```bash
# 查看所有容器（包括停止的）
docker ps -a | grep nerffacespeech

# 如果容器不存在，需要先启动
cd docker
docker-compose up -d
```

### 问题2: 无法进入容器

```bash
# 检查容器状态
docker ps | grep nerffacespeech

# 如果容器未运行，启动它
docker start nerffacespeech-eval

# 然后再进入
docker exec -it nerffacespeech-eval bash
```

### 问题3: 权限问题

```bash
# 如果遇到权限问题，使用 root 用户
docker exec -it --user root nerffacespeech-eval bash
```

### 问题4: 环境目录存在但无法激活

```bash
# 检查环境是否完整
ls -la /app/environment/syncnet/bin/
ls -la /app/environment/syncnet/lib/

# 检查 Python 可执行文件
file /app/environment/syncnet/bin/python

# 尝试直接运行 Python
/app/environment/syncnet/bin/python --version
```

## 快速参考命令

```bash
# 进入容器
docker exec -it nerffacespeech-eval bash

# 检查环境目录
ls -la /app/environment/

# 列出 conda 环境
source /opt/conda/etc/profile.d/conda.sh && conda env list

# 激活 syncnet 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/environment/syncnet

# 检查 Python
which python
python --version

# 退出容器
exit
```

## 诊断脚本

创建一个诊断脚本 `check_env.sh`：

```bash
#!/bin/bash
CONTAINER_NAME="nerffacespeech-eval"

echo "检查容器: $CONTAINER_NAME"
docker exec $CONTAINER_NAME bash -c "
    echo '=== 环境目录 ==='
    ls -la /app/environment/ 2>/dev/null || echo '目录不存在'
    
    echo ''
    echo '=== syncnet 环境 ==='
    if [ -d '/app/environment/syncnet' ]; then
        echo '目录存在'
        if [ -f '/app/environment/syncnet/bin/python' ]; then
            echo '有效环境'
            /app/environment/syncnet/bin/python --version
        else
            echo '无效环境'
        fi
    else
        echo '目录不存在'
    fi
    
    echo ''
    echo '=== Conda 环境列表 ==='
    source /opt/conda/etc/profile.d/conda.sh
    conda env list
"
```

使用方法：

```bash
chmod +x check_env.sh
./check_env.sh
```

