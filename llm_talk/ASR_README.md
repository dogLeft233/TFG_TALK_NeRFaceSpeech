# Whisper语音识别模块使用说明

## 安装依赖

首先需要安装Whisper库：

```bash
pip install openai-whisper
```

## 基本使用

### 1. 从音频文件识别

```python
from llm_talk.asr import transcribe_audio_file

# 识别音频文件
result = transcribe_audio_file(
    audio_path="audio.wav",
    model_name="base",  # 可选: tiny, base, small, medium, large
    language=None,  # None表示自动检测，也可以指定如'zh', 'en'
    task="transcribe"  # 'transcribe'或'translate'
)

if result['success']:
    print(f"识别文本: {result['data']['text']}")
    print(f"检测语言: {result['data']['language']}")
else:
    print(f"识别失败: {result['error']['message']}")
```

### 2. 从音频数据识别

```python
from llm_talk.asr import transcribe_audio_data
import soundfile as sf

# 读取音频文件
audio_data, sample_rate = sf.read("audio.wav")

# 识别
result = transcribe_audio_data(
    audio_data,
    sample_rate=sample_rate,
    model_name="base"
)

if result['success']:
    print(f"识别文本: {result['data']['text']}")
```

### 3. 从Base64数据识别

```python
from llm_talk.asr import transcribe_base64_audio
import base64

# 读取音频文件并编码为Base64
with open("audio.wav", "rb") as f:
    wav_bytes = f.read()
base64_data = base64.b64encode(wav_bytes).decode('utf-8')

# 识别
result = transcribe_base64_audio(
    base64_data,
    model_name="base"
)

if result['success']:
    print(f"识别文本: {result['data']['text']}")
```

### 4. 使用统一API接口

```python
from llm_talk.asr import get_asr_response_api

# 支持多种输入格式：文件路径、Base64字符串、bytes、numpy数组
result = get_asr_response_api(
    audio_input="audio.wav",  # 可以是文件路径、Base64字符串、bytes或numpy数组
    model_name="base",
    language=None,  # 自动检测语言
    task="transcribe"
)

if result['success']:
    print(f"识别文本: {result['data']['text']}")
```

### 5. 模型管理

```python
from llm_talk.asr import manage_asr_model

# 检查模型状态
status = manage_asr_model('status')
print(status)

# 加载模型
manage_asr_model('load', model_name='base')

# 释放模型
manage_asr_model('unload')

# 重新加载模型
manage_asr_model('reload', model_name='small')
```

## 模型选择

Whisper提供多个模型，按准确度和速度排序：

- `tiny`: 最快，准确度最低，适合快速测试
- `base`: 平衡速度和准确度（推荐）
- `small`: 更好的准确度
- `medium`: 高准确度
- `large`: 最高准确度，但速度最慢

## 语言支持

Whisper支持多种语言，可以：

1. **自动检测语言**: 设置 `language=None`
2. **指定语言**: 设置 `language='zh'` (中文) 或 `language='en'` (英文)

常见语言代码：
- `zh`: 中文
- `en`: 英文
- `ja`: 日文
- `ko`: 韩文
- `es`: 西班牙文
- `fr`: 法文
- `de`: 德文

## 任务类型

- `transcribe`: 转录（保持原语言）
- `translate`: 翻译（翻译成英文）

## 错误处理

所有函数都会返回标准化的响应格式：

```python
{
    'success': True/False,
    'data': {
        'text': '识别的文本',
        'language': '检测到的语言',
        'segments': [...],  # 分段信息
        'info': {...}  # 其他信息
    },
    'error': {
        'code': '错误代码',
        'message': '错误消息',
        'type': '错误类型'
    } or None
}
```

## 测试

运行测试程序：

```bash
python test_asr.py --audio_file <音频文件路径> --model base
```

测试选项：
- `--audio_file`: 音频文件路径（必需）
- `--model`: 模型名称（默认: base）
- `--test`: 测试类型（all, model, file, data, base64, api, language）

示例：

```bash
# 运行所有测试
python test_asr.py --audio_file test.wav --model base

# 只测试文件识别
python test_asr.py --audio_file test.wav --test file
```

