# 训练功能测试说明

## 测试脚本

`test_training.py` - 训练功能的综合测试脚本

## 运行测试

### 基本用法（自动启动后端）

```bash
cd gradio_app/test
python test_training.py
```

测试脚本会**自动启动后端服务**（如果后端未运行），测试结束后自动停止。

### 禁用自动启动后端

如果后端已经在运行，可以禁用自动启动：

```bash
AUTO_START_BACKEND=false python test_training.py
```

### 指定后端地址

```bash
API_BASE_URL=http://localhost:8000 python test_training.py
```

### 组合使用

```bash
# 禁用自动启动 + 指定后端地址
AUTO_START_BACKEND=false API_BASE_URL=http://remote-server:8000 python test_training.py
```

## 测试内容

测试脚本会执行以下测试：

1. **测试1: 检查训练脚本是否存在**
   - 验证 `run_train.py` 文件是否存在

2. **测试2: 测试训练脚本帮助信息**
   - 运行 `python run_train.py --help`
   - 验证帮助信息正常显示

3. **测试3: 测试训练脚本参数验证**
   - 测试缺少必需参数时的错误处理

4. **测试4: 测试后端连接**
   - 检查后端 API 是否可访问

5. **测试5: 测试列出数据集 API**
   - 调用 `GET /train/datasets`
   - 验证返回的数据集列表

6. **测试6: 测试启动训练 API**
   - 调用 `POST /train/start`
   - 需要有效的训练数据目录和基础模型
   - 使用很小的 `kimg=1` 进行快速测试

7. **测试7: 测试训练状态查询 API**
   - 调用 `GET /train/status/{task_id}`
   - 验证状态信息返回

8. **测试8: 测试列出训练任务 API**
   - 调用 `GET /train/tasks`
   - 验证任务列表返回

9. **测试9: 测试训练脚本干运行**
   - 使用不存在的路径测试参数验证
   - 验证脚本在参数错误时正确退出

## 前置条件

### 必需

1. **训练脚本存在**
   - `NeRFFaceSpeech_Code/StyleNeRF/run_train.py` 必须存在

2. **Python 环境**
   - 需要激活 NeRF conda 环境（用于运行训练脚本）
   - 需要 API conda 环境（用于自动启动后端）

### 可选（用于完整测试）

1. **后端服务运行**
   - **自动启动**: 测试脚本会自动启动后端服务（默认启用）
   - **手动启动**: 如果禁用自动启动，需要手动启动后端服务
   - 后端地址: `http://localhost:8000`
   - 如果没有运行且禁用自动启动，API 相关测试会被跳过

2. **训练数据**
   - **自动生成**: 测试脚本会自动创建随机测试图像数据（如果不存在）
   - 手动指定: 测试数据目录 `test_data/training_images/`
   - 或修改脚本中的 `TEST_DATA_DIR` 变量

3. **基础模型**
   - 预训练模型: `pretrained_networks/ffhq_1024.pkl`
   - 或者修改脚本中的 `TEST_MODEL` 变量

## 自动生成测试数据

测试脚本现在支持**自动生成随机测试数据**，无需手动准备训练数据集：

- **自动检测**: 如果测试数据目录不存在，脚本会自动创建随机图像
- **随机图像**: 生成包含简单几何形状的随机 RGB 图像
- **临时存储**: 测试数据存储在临时目录中，测试结束后自动清理
- **可配置**: 可以指定图像数量和分辨率

### 随机数据特性

- 图像格式: PNG
- 默认数量: 5-10 张（根据测试类型）
- 默认分辨率: 128x128 或 256x256（根据测试类型）
- 图像内容: 随机 RGB 像素 + 简单的圆形几何形状

## 测试输出示例

```
============================================================
  StyleNeRF 训练功能测试
============================================================

============================================================
  测试1: 检查训练脚本是否存在
============================================================
✅ 训练脚本存在: /path/to/run_train.py

============================================================
  测试2: 测试训练脚本帮助信息
============================================================
✅ 训练脚本帮助信息正常

帮助信息预览:
  Usage: run_train.py [OPTIONS]
  
  Train StyleNeRF model
  
  Options:
    --outdir TEXT    Where to save the training results [required]
    --data TEXT      Path to training dataset [required]
    ...

============================================================
  测试总结
============================================================
总测试数: 9
通过: 7
失败: 0
跳过: 2

详细结果:
  script_exists: ✅ 通过
  script_help: ✅ 通过
  parameter_validation: ✅ 通过
  backend_connection: ✅ 通过
  list_datasets: ✅ 通过
  start_training: ⚠️  跳过
  training_status: ⚠️  跳过
  list_tasks: ✅ 通过
  dry_run: ✅ 通过

🎉 所有可执行的测试都通过了！
```

## 自动启动后端功能

测试脚本支持**自动启动后端服务**，无需手动启动：

### 功能特性

- ✅ **自动检测**: 检测后端是否已运行
- ✅ **自动启动**: 如果后端未运行，自动启动
- ✅ **自动停止**: 测试结束后自动停止后端
- ✅ **日志输出**: 实时显示后端日志
- ✅ **信号处理**: 支持 Ctrl+C 优雅退出

### 配置

通过环境变量控制：

- `AUTO_START_BACKEND=true` (默认): 启用自动启动
- `AUTO_START_BACKEND=false`: 禁用自动启动

### 工作原理

1. 测试开始时检查后端是否运行
2. 如果未运行且启用自动启动，启动后端服务
3. 等待后端就绪（最多30秒）
4. 执行测试
5. 测试结束后自动停止后端

## 故障排除

### 后端连接失败

如果看到 "无法连接到后端" 错误：

1. **检查自动启动是否启用**:
   - 默认情况下自动启动是启用的
   - 如果禁用，设置 `AUTO_START_BACKEND=true`

2. **手动启动后端服务**:
   ```bash
   cd gradio_app
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

3. **使用一键启动脚本**:
   ```bash
   cd gradio_app
   python start_all.py
   ```

4. **检查端口占用**:
   ```bash
   # 检查8000端口是否被占用
   lsof -i :8000
   # 或
   netstat -tuln | grep 8000
   ```

### 训练脚本不存在

如果看到 "训练脚本不存在" 错误：

1. 确认 `NeRFFaceSpeech_Code/StyleNeRF/run_train.py` 文件存在
2. 检查路径配置是否正确

### 测试数据不存在

如果看到 "测试数据目录不存在" 警告：

1. 创建测试数据目录：
   ```bash
   mkdir -p test_data/training_images
   ```

2. 添加一些测试图像到该目录

3. 或者修改脚本中的 `TEST_DATA_DIR` 变量指向现有数据集

### Python 环境问题

如果看到导入错误：

1. 确保在正确的 conda 环境中运行
2. 检查 `NERF_CONDA_PYTHON` 配置是否正确

## 手动测试

如果自动测试脚本有问题，可以手动测试各个功能：

### 1. 测试训练脚本帮助

```bash
cd NeRFFaceSpeech_Code
python StyleNeRF/run_train.py --help
```

### 2. 测试后端 API

```bash
# 列出数据集
curl http://localhost:8000/train/datasets

# 启动训练（需要有效数据）
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "/path/to/data",
    "base_model": "ffhq_1024.pkl",
    "kimg": 1
  }'

# 查询训练状态
curl http://localhost:8000/train/status/{task_id}

# 列出所有任务
curl http://localhost:8000/train/tasks
```

## 注意事项

1. **测试训练启动**: 如果实际启动训练，会使用 `kimg=1` 进行快速测试，但仍可能需要一些时间
2. **资源占用**: 训练测试会占用 GPU 资源，确保有足够的资源
3. **数据要求**: 训练需要有效的图像数据集，格式参考 `TRAINING_README.md`
4. **环境要求**: 确保在正确的 conda 环境中运行测试

