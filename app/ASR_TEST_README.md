# ASR API 测试说明

## 概述

本目录包含两个测试程序，用于测试 gradio_app 后端的语音识别（ASR）功能：

1. **test_asr_api.py** - 独立的测试程序，测试已运行的后端服务
2. **start_and_test_asr.py** - 启动服务并运行测试的完整脚本

## 前置要求

1. **安装依赖**
   ```bash
   pip install openai-whisper requests
   ```

2. **准备测试音频文件**
   - 默认使用: `assets/charactors/Ayanami/绫波丽.wav`
   - 或使用 `--audio-file` 参数指定其他音频文件

## 使用方法

### 方法1: 使用独立测试程序（推荐）

适用于后端服务已在运行的情况。

#### 1. 启动后端服务（如果未启动）

```bash
cd gradio_app/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 2. 启动ASR服务（可选，如果使用ASR服务）

```bash
cd gradio_app/services
python asr_service.py --port 8002 --model base
```

#### 3. 运行测试

```bash
cd gradio_app
python test_asr_api.py --audio-file <音频文件路径>
```

**常用参数：**

```bash
# 使用默认音频文件，运行所有测试
python test_asr_api.py

# 指定音频文件和模型
python test_asr_api.py --audio-file /path/to/audio.wav --model small

# 只测试Base64识别
python test_asr_api.py --test base64

# 只测试文件路径识别
python test_asr_api.py --test file

# 只测试聊天接口
python test_asr_api.py --test chat

# 指定语言
python test_asr_api.py --language zh

# 指定后端地址
python test_asr_api.py --backend-url http://localhost:8000
```

### 方法2: 使用启动脚本（一键测试）

自动启动服务并运行测试。

```bash
cd gradio_app
python start_and_test_asr.py --audio-file <音频文件路径>
```

**常用参数：**

```bash
# 使用默认配置运行所有测试
python start_and_test_asr.py

# 指定音频文件和模型
python start_and_test_asr.py --audio-file /path/to/audio.wav --model base

# 跳过启动后端（假设已在运行）
python start_and_test_asr.py --skip-backend

# 跳过启动ASR服务（假设已在运行）
python start_and_test_asr.py --skip-asr

# 测试完成后保持服务运行
python start_and_test_asr.py --keep-running

# 只运行特定测试
python start_and_test_asr.py --test base64
```

## 测试内容

### 1. 健康检查测试 (`health`)
- 检查后端服务是否可用
- 检查ASR服务健康状态

### 2. Base64音频识别测试 (`base64`)
- 将音频文件编码为Base64
- 调用 `/asr/transcribe` API
- 验证识别结果

### 3. 文件路径识别测试 (`file`)
- 直接使用文件路径调用API
- 验证识别结果

### 4. 聊天接口音频输入测试 (`chat`)
- 使用音频输入调用 `/chat` API
- 验证语音识别和LLM回复

## 测试输出示例

```
============================================================
ASR API 测试程序
============================================================
后端地址: http://localhost:8000
音频文件: /path/to/audio.wav
模型: base
语言: 自动检测
测试类型: all
============================================================

🔍 检查后端服务...
✅ 后端服务可用

🔍 检查ASR服务健康状态...
✅ ASR服务状态检查成功
   服务可用: True
   服务健康: True
   服务地址: http://localhost:8002

📂 读取音频文件: /path/to/audio.wav
✅ 文件读取成功
   文件大小: 123,456 bytes (120.56 KB)
   Base64长度: 164,608 字符

============================================================
测试1: Base64音频识别
============================================================
📤 发送请求到: http://localhost:8000/asr/transcribe
   模型: base
   语言: 自动检测
   任务: transcribe
   Base64长度: 164,608 字符
⏱️  请求耗时: 2.34 秒
📥 响应状态码: 200
✅ 识别成功！
📝 识别文本: 你好，这是一个测试音频。
🌐 检测语言: zh

============================================================
测试结果汇总
============================================================
✅ 成功: 3/3
❌ 失败: 0/3

🎉 所有测试通过！
```

## 故障排除

### 1. 后端服务不可用

**错误：** `❌ 后端服务不可用: http://localhost:8000`

**解决方法：**
```bash
cd gradio_app/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. ASR服务不可用

**错误：** `⚠️ ASR服务不可用，将使用直接调用方式`

**解决方法：**
- 这是正常的，系统会自动回退到直接调用ASR模块
- 如果想使用ASR服务，启动它：
  ```bash
  cd gradio_app/services
  python asr_service.py --port 8002 --model base
  ```

### 3. 音频文件不存在

**错误：** `❌ 音频文件不存在: /path/to/audio.wav`

**解决方法：**
- 使用 `--audio-file` 参数指定正确的音频文件路径
- 或使用项目中的默认音频文件

### 4. Whisper模型未安装

**错误：** `ModuleNotFoundError: No module named 'whisper'`

**解决方法：**
```bash
pip install openai-whisper
```

### 5. 端口被占用

**错误：** `⚠️ 端口 8000 已被占用`

**解决方法：**
- 使用 `--skip-backend` 跳过启动后端
- 或使用 `--backend-port` 指定其他端口
- 或停止占用端口的进程

## 参数说明

### test_asr_api.py 参数

- `--backend-url`: 后端服务地址（默认: http://localhost:8000）
- `--audio-file`: 测试音频文件路径
- `--model`: Whisper模型名称（tiny, base, small, medium, large）
- `--language`: 语言代码（如'zh', 'en'），None表示自动检测
- `--test`: 要运行的测试（all, health, base64, file, chat）
- `--character`: 聊天测试使用的角色（默认: ayanami）

### start_and_test_asr.py 参数

- `--backend-port`: 后端服务端口（默认: 8000）
- `--asr-port`: ASR服务端口（默认: 8002）
- `--audio-file`: 测试音频文件路径
- `--model`: Whisper模型名称（默认: base）
- `--language`: 语言代码
- `--test`: 要运行的测试（默认: all）
- `--character`: 聊天测试使用的角色（默认: ayanami）
- `--skip-backend`: 跳过启动后端服务
- `--skip-asr`: 跳过启动ASR服务
- `--keep-running`: 测试完成后保持服务运行

## 注意事项

1. **模型选择**：
   - `tiny`: 最快，准确度最低
   - `base`: 平衡速度和准确度（推荐）
   - `small`: 更好的准确度
   - `medium`: 高准确度
   - `large`: 最高准确度，但速度最慢

2. **首次运行**：
   - Whisper模型会在首次使用时自动下载
   - 下载可能需要一些时间，请耐心等待

3. **性能**：
   - Base64识别需要先编码，可能稍慢
   - 文件路径识别更高效
   - 使用ASR服务可以避免重复加载模型

4. **超时设置**：
   - 测试程序设置了5分钟的超时
   - 如果音频文件很大或模型很慢，可能需要更长时间

## 示例命令

```bash
# 快速测试（使用默认配置）
cd gradio_app
python test_asr_api.py

# 完整测试（启动服务并测试）
python start_and_test_asr.py

# 测试特定功能
python test_asr_api.py --test base64 --model base

# 使用中文音频测试
python test_asr_api.py --audio-file chinese_audio.wav --language zh

# 测试聊天功能
python test_asr_api.py --test chat --character ayanami
```

