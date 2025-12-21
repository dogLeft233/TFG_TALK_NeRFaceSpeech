# 角色训练功能说明

## 功能概述

角色训练功能允许用户上传一个人物讲话的视频，系统会自动：
1. 使用 `video_face_crop.py` 处理视频（人脸检测、对齐、裁剪）
2. 从处理后的视频中提取所有帧（保存为图像）
3. 从原始视频中提取音频
4. 调用 `main_NeRFFaceSpeech_audio_driven_w_given_poses.py` 生成 PTI 模型文件（G_PTI.pt, w_PTI.pt, bg_PTI.pt）
5. 将处理后的数据和模型文件保存到 `assets/charactor/{角色名}/` 目录

## API 接口

### 1. 提交角色训练任务

**接口**: `POST /character/train`

**请求参数**:
- `video` (文件): 人物讲话视频文件（支持 .mp4, .avi, .mov, .mkv）
- `character_name` (字符串): 角色名称（用于创建输出目录）
- `face_ratio` (浮点数, 可选): 人脸占画面的比例，默认 0.6
- `output_size_w` (整数, 可选): 输出视频宽度，默认 1024
- `output_size_h` (整数, 可选): 输出视频高度，默认 1024
- `ffhq_alignment` (布尔值, 可选): 是否使用 FFHQ 对齐，默认 True
- `overwrite` (布尔值, 可选): 是否覆盖已存在的训练数据，默认 False

**响应示例**:
```json
{
    "success": true,
    "task_id": "uuid-string",
    "status": "pending",
    "character_name": "角色名",
    "message": "角色训练任务已提交，正在后台处理中..."
}
```

### 2. 查询训练任务状态

**接口**: `GET /character/train/status/{task_id}`

**响应示例**:
```json
{
    "success": true,
    "task_id": "uuid-string",
    "status": "completed",
    "character_name": "角色名",
    "video_filename": "video.mp4",
    "result": {
        "success": true,
        "message": "角色 角色名 的训练数据已准备完成",
        "character_dir": "/path/to/assets/charactor/角色名",
        "images_dir": "/path/to/assets/charactor/角色名/images",
        "audio_file": "/path/to/assets/charactor/角色名/audio.wav",
        "num_frames": 300,
        "pti_models": {
            "G_PTI": "/path/to/assets/charactor/角色名/G_PTI.pt",
            "w_PTI": "/path/to/assets/charactor/角色名/w_PTI.pt",
            "bg_PTI": "/path/to/assets/charactor/角色名/bg_PTI.pt"
        }
    }
}
```

**任务状态**:
- `pending`: 任务已提交，等待处理
- `processing`: 正在处理中
- `completed`: 处理完成
- `failed`: 处理失败

### 3. 列出所有角色

**接口**: `GET /character/list`

**响应示例**:
```json
{
    "success": true,
    "characters": [
        {
            "exists": true,
            "character_name": "角色名",
            "character_dir": "/path/to/assets/charactor/角色名",
            "images_dir": "/path/to/assets/charactor/角色名/images",
            "num_images": 300,
            "audio_file": "/path/to/assets/charactor/角色名/audio.wav",
            "audio_exists": true,
            "pti_models": {
                "G_PTI": true,
                "w_PTI": true,
                "bg_PTI": true
            },
            "pti_models_exist": true
        }
    ]
}
```

### 4. 查询角色状态

**接口**: `GET /character/{character_name}/status`

**响应示例**:
```json
{
    "success": true,
    "data": {
        "exists": true,
        "character_name": "角色名",
        "character_dir": "/path/to/assets/charactor/角色名",
        "images_dir": "/path/to/assets/charactor/角色名/images",
        "num_images": 300,
        "audio_file": "/path/to/assets/charactor/角色名/audio.wav",
        "audio_exists": true,
        "pti_models": {
            "G_PTI": true,
            "w_PTI": true,
            "bg_PTI": true
        },
        "pti_models_exist": true
    }
}
```

## 输出目录结构

处理完成后，会在 `assets/charactor/{角色名}/` 目录下创建以下结构：

```
assets/charactor/{角色名}/
├── images/              # 提取的视频帧（图像）
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── audio.wav            # 提取的音频文件（16kHz 单声道）
├── G_PTI.pt             # PTI 训练生成的生成器模型
├── w_PTI.pt             # PTI 训练生成的潜在代码
├── bg_PTI.pt            # PTI 训练生成的背景潜在代码
└── output_NeRFFaceSpeech.mp4  # 训练过程中生成的测试视频（可选）
```

### 模型文件说明

- **G_PTI.pt**: 经过 PTI（Pivotal Tuning Inversion）微调后的生成器模型，用于生成特定角色的图像
- **w_PTI.pt**: 输入图像的潜在代码，用于身份保持
- **bg_PTI.pt**: 背景的潜在代码，用于背景保持

这些模型文件可以用于后续的音频驱动视频生成。

## 使用示例

### Python 示例

```python
import requests

# 提交训练任务
with open("character_video.mp4", "rb") as f:
    files = {"video": f}
    data = {
        "character_name": "my_character",
        "face_ratio": 0.6,
        "output_size_w": 1024,
        "output_size_h": 1024,
        "ffhq_alignment": True,
        "overwrite": False
    }
    response = requests.post("http://localhost:8000/character/train", files=files, data=data)
    result = response.json()
    task_id = result["task_id"]

# 查询任务状态
import time
while True:
    response = requests.get(f"http://localhost:8000/character/train/status/{task_id}")
    status = response.json()
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(2)
```

### cURL 示例

```bash
# 提交训练任务
curl -X POST "http://localhost:8000/character/train" \
  -F "video=@character_video.mp4" \
  -F "character_name=my_character" \
  -F "face_ratio=0.6" \
  -F "output_size_w=1024" \
  -F "output_size_h=1024" \
  -F "ffhq_alignment=true" \
  -F "overwrite=false"

# 查询任务状态
curl "http://localhost:8000/character/train/status/{task_id}"

# 列出所有角色
curl "http://localhost:8000/character/list"
```

## 注意事项

1. **视频格式**: 支持 .mp4, .avi, .mov, .mkv 格式
2. **处理时间**: 
   - 视频处理和帧提取可能需要几分钟
   - **PTI 模型训练可能需要较长时间（10-30分钟或更长）**，取决于视频长度和硬件配置
   - 建议在 GPU 环境下运行以获得更好的性能
3. **存储空间**: 确保有足够的磁盘空间存储提取的图像、音频和模型文件（模型文件可能较大）
4. **FFHQ 对齐**: 推荐使用 FFHQ 对齐以获得更好的效果，但需要确保相关依赖已安装
5. **角色名称**: 角色名称会用作目录名，建议使用英文和数字，避免特殊字符
6. **模型文件**: 
   - 如果模型文件已存在，会跳过 PTI 训练步骤
   - PTI 训练需要基础模型文件（ffhq_1024.pkl），请确保该文件存在于 `pretrained_networks` 目录
7. **GPU 要求**: PTI 训练需要 GPU 支持，建议使用 CUDA 环境

## 依赖要求

### 基础依赖
- `video_face_crop.py` 脚本及其依赖
- FFHQFaceAlignment（如果使用 FFHQ 对齐）
- ffmpeg（用于音频提取）
- OpenCV（用于视频处理）

### PTI 训练依赖
- **Ninja 构建工具**（必需）
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ninja-build
  
  # 或使用 conda
  conda install ninja
  ```
- CUDA 和 GPU 支持（推荐）
- 完整的 nerffacespeech 环境依赖
- 预训练模型文件：
  - `pretrained_networks/ffhq_1024.pkl`
  - `pretrained_networks/seg.pth`
  - `pretrained_networks/LipaintNet.pt`
  - `pretrained_networks/Deep3DFaceRecon_pytorch/face_recon/epoch_20.pth`

