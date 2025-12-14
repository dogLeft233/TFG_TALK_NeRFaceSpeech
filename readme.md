# 开发日志
- 先到`develop`分支中开发，测试成功再合并到`main`分支中
- 尽量不要擅自修改别人写的代码
---
- 文档文件夹`/doc`,目前使用AI分析了一些项目代码，放在`/doc/AI_Analysis`中
- `/doc`中加入了个人运行环境构建过程，只能保证跑通`llm_talk`和`metrics`
- 加了`FID/FVD`评价方法，在`metrics/FID_FVD.py`

# 接口

## llm_talk 模块接口

### 核心功能
`llm_talk` 模块实现了完整的对话功能：用户输入 → LLM回答 → TTS音频生成

### 主要接口

#### 1. Talk 对话接口
```python
from llm_talk import get_talk_response_api

# 完整对话功能
result = get_talk_response_api(
    user_input="用户问题",
    language_id='zh',           # TTS语言ID
    combine_audio=True,         # 是否合并音频
    release_tts_model=False,    # 是否在完成后释放TTS模型
    split_sentences=True        # 是否分句处理
)
```

**返回格式：**
```json
{
    "success": true,
    "data": {
        "user_input": "用户问题",
        "llm_answer": "LLM回答",
        "sentences": ["句子1", "句子2"],
        "audio_results": [...],
        "combined_audio": {
            "combined_audio_data": "WAV二进制数据",
            "combined_base64_data": "Base64编码音频",
            "total_duration": 5.2,
            "sample_rate": 22050
        },
        "processing_info": {
            "total_sentences": 2,
            "successful_audio": 2,
            "failed_audio": 0,
            "split_sentences_enabled": true,
            "tts_model_released": false
        }
    },
    "error": null
}
```

#### 2. LLM 大语言模型接口
```python
from llm_talk import get_llm_response_api

# LLM问答
result = get_llm_response_api("用户问题")
```

**返回格式：**
```json
{
    "success": true,
    "data": {
        "answer": "LLM回答内容",
        "question": "用户问题"
    },
    "error": null
}
```

#### 3. TTS 语音合成接口
```python
from llm_talk import get_tts_response_api, manage_tts_model

# 文本转语音
result = get_tts_response_api(
    text="要转换的文本",
    language_id='zh',
    exaggeration=0.5,
    cfg_weight=0.7,
    temperature=0.3
)

# TTS模型管理
model_status = manage_tts_model('status')      # 查看状态
model_unload = manage_tts_model('unload')      # 释放模型
model_reload = manage_tts_model('reload')      # 重新加载
```

**TTS返回格式：**
```json
{
    "success": true,
    "data": {
        "wav_data": "WAV二进制数据",
        "base64_data": "Base64编码音频",
        "sample_rate": 22050,
        "duration": 3.5,
        "text": "原始文本",
        "audio_info": {
            "sample_rate": 22050,
            "duration": 3.5,
            "channels": 1,
            "format": "WAV"
        }
    },
    "error": null
}
```

### 工具函数

#### 1. 音频保存
```python
from llm_talk import save_wav_to_file

# 保存WAV文件
success = save_wav_to_file(wav_data, "output.wav")
```

### 使用示例

#### 1. 基础使用
```python
from llm_talk import get_talk_response_api

# 简单对话
result = get_talk_response_api("你好，请介绍一下人工智能")
if result['success']:
    print(f"回答: {result['data']['llm_answer']}")
    # 播放音频
    audio_data = result['data']['combined_audio']['combined_base64_data']
```

#### 2.高级配置
```python
# 不分句处理，完成后释放模型
result = get_talk_response_api(
    "请用一句话总结机器学习",
    split_sentences=False,
    release_tts_model=True
)
```

#### 3.批量处理
```python
questions = ["问题1", "问题2", "问题3"]
for question in questions:
    result = get_talk_response_api(
        question,
        release_tts_model=True  # 每个问题后释放模型节省内存
    )
```

### 4.错误处理
所有接口都返回统一的错误格式：
```json
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "错误描述",
        "type": "ErrorType"
    },
    "data": null
}
```

### 运行测试
```bash
# 在项目根目录运行
python -m llm_talk.talk
```