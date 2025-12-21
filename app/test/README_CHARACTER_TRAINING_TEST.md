# 角色训练功能测试说明

## 测试文件

`test_character_training.py` - 角色训练功能的完整测试程序

## 功能测试

该测试程序会测试以下功能：

1. **后端连接测试** - 验证后端服务是否正常运行
2. **列出角色 API** - 测试获取所有已训练角色的列表
3. **角色训练 API** - 测试提交角色训练任务
4. **训练状态查询** - 测试查询训练任务状态（轮询直到完成）
5. **角色状态查询** - 测试查询指定角色的训练数据状态

## 使用方法

### 基本使用

```bash
cd gradio_app
python test/test_character_training.py
```

### 指定输入视频文件

```bash
# 使用指定的视频文件
python test/test_character_training.py --video /path/to/video.mp4

# 指定视频文件和角色名称
python test/test_character_training.py --video /path/to/video.mp4 --character my_character

# 指定视频文件、角色名称和 API 地址
python test/test_character_training.py --video /path/to/video.mp4 --character my_character --api-url http://localhost:8000

# 使用短参数
python test/test_character_training.py -v /path/to/video.mp4 -c my_character
```

### 命令行参数

- `--video`, `-v`: 输入视频文件路径（支持 .mp4, .avi, .mov, .mkv 格式）
- `--character`, `-c`: 角色名称（默认: `test_student`）
- `--api-url`: 后端 API 地址（默认: `http://localhost:8000`）
- `--no-auto-start`: 不自动启动后端服务（需要手动启动后端）
- `--max-wait-time`: 最大等待时间（秒，默认: 600）

### 环境变量配置

- `API_BASE_URL`: 后端 API 地址（默认: `http://localhost:8000`）
- `AUTO_START_BACKEND`: 是否自动启动后端（默认: `true`）

示例：
```bash
export API_BASE_URL=http://localhost:8000
export AUTO_START_BACKEND=true
python test/test_character_training.py --video /path/to/video.mp4
```

## 测试流程

1. **自动启动后端**（如果 `AUTO_START_BACKEND=true`）
   - 使用 `API_CONDA_PYTHON` 启动后端服务
   - 等待后端就绪（最多30秒）

2. **创建测试视频**（如果 OpenCV 可用）
   - 创建一个简单的测试视频（3秒，25fps）
   - 如果 OpenCV 不可用，会尝试查找现有测试视频

3. **提交训练任务**
   - 上传测试视频
   - 指定角色名称：`test_student`
   - 使用默认参数（face_ratio=0.6, output_size=1024x1024）

4. **轮询任务状态**
   - 每3秒查询一次任务状态
   - 最多等待600秒
   - 显示处理进度

5. **验证结果**
   - 检查图像目录是否存在
   - 检查音频文件是否存在
   - 验证提取的帧数

6. **清理资源**
   - 清理临时测试视频（如果创建了）
   - 停止后端服务（如果自动启动的）

## 测试输出示例

```
============================================================
  角色训练功能测试
============================================================

自动启动后端: 已启用 (AUTO_START_BACKEND=True)

============================================================
  测试1: 测试后端连接
============================================================
✅ 后端连接成功: http://localhost:8000

============================================================
  测试2: 测试列出角色 API
============================================================
✅ 成功获取角色列表，共 1 个角色
  1. ayanami
     图像数量: 0
     音频存在: False

============================================================
  测试3: 测试角色训练 API
============================================================
✅ 找到测试视频: /path/to/test_video.mp4
✅ 角色训练任务提交成功
   任务ID: xxx-xxx-xxx
   角色名称: test_student

============================================================
  测试4: 测试训练任务状态查询 API
============================================================
任务状态: processing
   等待中... (已等待 0秒)
...
任务状态: completed
✅ 训练任务完成！
   角色目录: /path/to/assets/charactor/test_student
   图像目录: /path/to/assets/charactor/test_student/images
   音频文件: /path/to/assets/charactor/test_student/audio.wav
   提取帧数: 7032
   ✅ 图像目录存在，包含 7032 张图像
   ✅ 音频文件存在: 8790.08 KB

============================================================
  测试总结
============================================================
总测试数: 5
通过: 5
失败: 0

🎉 所有测试都通过了！
```

## 注意事项

1. **OpenCV 依赖**
   - 如果 OpenCV 未安装，测试会尝试使用现有测试视频
   - 建议安装 OpenCV: `pip install opencv-python`

2. **测试视频**
   - **优先使用**：如果通过 `--video` 参数指定了视频文件，会使用指定的视频
   - 如果未指定视频且 OpenCV 可用，会自动创建测试视频
   - 如果未指定视频且 OpenCV 不可用，会在以下目录查找现有视频：
     - `data/geneface_datasets/data/raw/videos/`
     - `test_data/`

3. **处理时间**
   - 视频处理可能需要较长时间（取决于视频长度）
   - 测试会等待最多600秒（10分钟）

4. **端口占用**
   - 确保端口 8000 未被占用
   - 如果后端已在运行，测试会直接使用现有服务

5. **Conda 环境**
   - 确保 `NERF_CONDA_PYTHON` 配置正确
   - 该环境需要包含 OpenCV 和其他视频处理依赖

## 测试结果

- ✅ **成功**: 所有测试通过，返回码 0
- ❌ **失败**: 部分测试失败，返回码 1，查看输出了解失败原因

## 故障排除

### 后端无法启动
- 检查 `API_CONDA_PYTHON` 路径是否正确
- 检查端口 8000 是否被占用
- 查看后端日志了解错误信息

### 视频处理失败
- 检查 `NERF_CONDA_PYTHON` 环境是否包含 OpenCV
- 检查 `video_face_crop.py` 脚本是否存在
- 查看错误日志了解具体问题

### 测试超时
- 增加 `max_wait_time` 参数
- 检查视频文件大小（大文件需要更长时间）
- 检查系统资源（CPU/内存/磁盘）

