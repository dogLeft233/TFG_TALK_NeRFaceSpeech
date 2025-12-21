# TTS 服务

独立的 TTS 服务，模型常驻内存，避免每次请求都重新加载模型。

## 优势

- ✅ **性能优化**：模型在服务启动时加载一次，后续请求直接使用，无需重新加载
- ✅ **资源节约**：避免重复占用 GPU 内存
- ✅ **响应快速**：后续请求响应速度显著提升
- ✅ **向后兼容**：如果 TTS 服务不可用，自动回退到 subprocess 方式

## 启动方式

### 方式 1：使用 start_all.py（推荐）

```bash
cd gradio_app
python start_all.py
```

这会自动启动：
1. TTS 服务（端口 8001）
2. 后端服务（端口 8000）
3. 前端服务（端口 7860）

### 方式 2：单独启动 TTS 服务

```bash
cd gradio_app/services
python start_tts_service.py
```

或者直接使用 Python：

```bash
cd gradio_app
python -m services.tts_service --host 0.0.0.0 --port 8001
```

### 方式 3：使用 LLM conda 环境

```bash
source activate /path/to/environment/llm_talk
cd gradio_app
python services/tts_service.py --host 0.0.0.0 --port 8001
```

## 配置

TTS 服务地址可通过环境变量配置：

```bash
export TTS_SERVICE_URL=http://localhost:8001
```

默认地址：`http://localhost:8001`

## API 接口

### 健康检查

```bash
curl http://localhost:8001/health
```

### 生成 TTS 音频（返回音频数据）

```bash
curl -X POST http://localhost:8001/api/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，世界",
    "language_id": "zh",
    "audio_prompt_path": "/path/to/prompt.wav"
  }'
```

### 生成 TTS 音频文件

```bash
curl -X POST http://localhost:8001/api/tts/generate_file \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，世界",
    "output_path": "/path/to/output.wav",
    "language_id": "zh",
    "audio_prompt_path": "/path/to/prompt.wav"
  }'
```

### 完整对话（LLM + TTS）

```bash
curl -X POST http://localhost:8001/api/talk \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "你好",
    "language_id": "zh",
    "audio_prompt_path": "/path/to/prompt.wav",
    "combine_audio": true,
    "split_sentences": true
  }'
```

### 模型管理

```bash
# 检查模型状态
curl http://localhost:8001/api/model/status

# 重新加载模型
curl -X POST http://localhost:8001/api/model/reload
```

## 工作原理

1. **服务启动时**：加载 Chatterbox TTS 模型到内存
2. **请求处理**：直接使用已加载的模型生成音频
3. **模型常驻**：模型保持在内存中，直到服务关闭

## 性能对比

### 使用 TTS 服务（推荐）
- 首次请求：~5-10秒（模型已加载）
- 后续请求：~1-3秒（直接使用模型）

### 使用 subprocess 方式（旧方式）
- 每次请求：~10-20秒（每次重新加载模型）

## 故障排除

### TTS 服务无法启动

1. 检查 LLM conda 环境是否正确
2. 检查端口 8001 是否被占用
3. 查看服务日志

### 后端无法连接到 TTS 服务

1. 确认 TTS 服务正在运行：`curl http://localhost:8001/health`
2. 检查 `TTS_SERVICE_URL` 环境变量
3. 后端会自动回退到 subprocess 方式（性能较差）

### 模型加载失败

1. 检查模型文件是否存在
2. 检查 GPU 内存是否充足
3. 查看服务启动日志

## 注意事项

- TTS 服务需要 LLM conda 环境（包含 chatterbox 和相关依赖）
- 模型加载需要一定时间（首次启动约 30-60 秒）
- 建议使用 1 个工作进程（`--workers 1`），因为模型在内存中共享
- 如果使用多个工作进程，每个进程都会加载一份模型副本（占用更多内存）

