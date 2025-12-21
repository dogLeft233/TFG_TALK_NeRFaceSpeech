# 后端服务内容清单

本文档列出了 `fastapi_server` 中所有需要迁移到 `gradio_app` 的后端服务内容。

## 📁 目录结构

```
fastapi_server/
├── main.py              # FastAPI主应用（1959行）
├── config.py            # 配置管理（139行）
├── start.py             # 启动脚本（710行）
├── simple_web.py        # 前端HTTP服务器（92行）
├── utils/               # 工具模块
│   ├── run_llm_talk.py          # LLM+TTS音频生成
│   ├── run_nerffacespeech.py    # NeRF视频生成
│   ├── run_chat.py              # 聊天功能
│   ├── run_training.py          # 训练功能
│   ├── llm_talk_api_bridge.py   # LLM API桥接
│   └── llm_talk_with_text_bridge.py
├── database/            # 数据库模块
│   ├── settings_db.py           # 设置数据库
│   ├── video_records_db.py      # 视频记录数据库
│   └── chat_db.py               # 聊天数据库
└── webui/               # 前端静态文件（不迁移）
```

## 🔌 API端点清单（33个）

### 1. 静态文件服务

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/videos/{filename}` | GET/HEAD | 提供视频文件 | 支持Range请求，用于视频seek |
| `/audios/{filename}` | GET/HEAD | 提供音频文件 | 支持Range请求 |
| `/texts/{filename}` | GET/HEAD | 提供文本文件 | UTF-8编码 |
| `/` | GET | 后端根路径 | 显示后端输出页面 |
| `/favicon.ico` | GET | 网站图标 | 返回204 |

### 2. 设置管理 API（4个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/api/settings` | GET | 获取所有设置 | 返回所有设置项 |
| `/api/settings/{key}` | GET | 获取指定设置 | 根据key获取单个设置 |
| `/api/settings/{key}` | POST | 更新设置 | 更新单个设置项 |
| `/api/settings` | POST | 批量更新设置 | 批量更新多个设置 |

**设置项包括**:
- `nerf_theme`: 主题（tech/warm/minimal）
- `nerf_font`: 字体
- `nerf_font_size`: 字号

### 3. 数据库管理 API（3个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/api/databases` | GET | 列出所有数据库文件 | 返回database目录下的.db文件 |
| `/api/databases/{db_name}/content` | GET | 获取数据库内容 | 可指定table参数获取特定表 |
| `/api/databases/{db_name}/update` | POST | 更新数据库内容 | 支持update/insert/delete操作 |

### 4. 模型管理 API（1个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/models` | GET | 获取模型列表 | 返回.pkl模型文件列表 |

### 5. 视频生成 API（5个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/generate_video` | POST | 提交视频生成任务 | 异步模式，立即返回task_id |
| `/generate_video_sync` | POST | 同步生成视频 | 同步模式，等待完成（不推荐） |
| `/generate_video/status/{unique_id}` | GET | 查询任务状态 | 返回任务状态和结果 |
| `/generate_video/running` | GET | 获取正在运行的任务 | 用于页面初始化检查 |
| `/generate_video/cancel/{unique_id}` | POST | 终止任务 | 取消正在运行的任务 |

**请求参数**:
- `text`: 输入文本
- `character`: 角色（ayanami/Aerith）
- `model_name`: 模型文件名（.pkl）

**响应字段**:
- `unique_id`: 任务ID
- `status`: 状态（pending/running/completed/failed/cancelled）
- `video_path`: 视频文件路径
- `audio_path`: 音频文件路径
- `text_path`: 文本文件路径
- `generation_time`: 生成耗时（秒）

### 6. 聊天对话 API（4个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/chat` | POST | 发送聊天消息 | 支持文本输入，返回LLM回答和音频 |
| `/llm_only` | POST | 仅LLM问答 | 不生成音频，快速响应 |
| `/chat/sessions` | GET | 获取聊天会话列表 | 支持limit和offset参数 |
| `/chat/sessions/{session_id}/messages` | GET | 获取会话消息 | 获取指定会话的所有消息 |
| `/chat/sessions/{session_id}` | DELETE | 删除聊天会话 | 删除会话及其所有消息 |

**请求参数**:
- `text`: 用户输入的文本
- `character`: 角色（ayanami/Aerith）
- `enable_audio`: 是否生成音频回复
- `session_id`: 会话ID（可选，不存在则创建新会话）

**响应字段**:
- `llm_answer`: LLM回答文本
- `audio_base64`: 音频base64编码（如果启用）
- `audio_url`: 音频文件URL（如果已保存）
- `session_id`: 会话ID
- `user_message_id`: 用户消息ID
- `assistant_message_id`: AI回复消息ID

### 7. 训练相关 API（5个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/train/start` | POST | 启动模型训练 | 提交训练任务 |
| `/train/status/{task_id}` | GET | 获取训练状态 | 返回训练状态和日志 |
| `/train/tasks` | GET | 列出所有训练任务 | 返回任务列表 |
| `/train/stop/{task_id}` | POST | 停止训练任务 | 终止正在运行的训练 |
| `/train/datasets` | GET | 列出可用数据集 | 返回数据集路径列表 |

**训练请求参数**:
- `data_path`: 数据集路径
- `base_model`: 基础模型（默认ffhq_1024.pkl）
- `kimg`: 训练迭代数（默认50）
- `snap`: 快照间隔（默认5）
- `imgsnap`: 图像快照间隔（默认1）
- `aug`: 数据增强（默认noaug）
- `mirror`: 镜像翻转（默认False）
- `config_name`: 配置名称（默认style_ffhq_ae_basic）

### 8. 日志查看 API（2个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/logs` | GET | 获取日志输出 | 支持limit参数（默认500） |
| `/logs/full` | GET | 获取完整日志 | 无长度限制 |

**参数**:
- `limit`: 返回的日志条数限制
- `debug`: 是否只返回debug日志

### 9. 生成记录 API（3个）

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/generation_records` | GET | 获取生成记录列表 | 支持record_type/limit/offset |
| `/generation_records/{unique_id}` | GET | 获取单个记录详情 | 返回记录的完整信息 |
| `/generation_records/{unique_id}` | DELETE | 删除生成记录 | 删除指定记录 |

**参数**:
- `record_type`: 记录类型（'video'/'chat'，None表示所有）
- `limit`: 返回记录数量限制（默认100）
- `offset`: 偏移量（默认0）

## 🗄️ 数据库模块

### 1. settings_db.py - 设置数据库
**功能**:
- `init_database()`: 初始化数据库
- `get_setting(key)`: 获取设置
- `set_setting(key, value)`: 设置值
- `get_all_settings()`: 获取所有设置

**表结构**:
```sql
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### 2. video_records_db.py - 视频记录数据库
**功能**:
- `init_database()`: 初始化数据库
- `add_video_record()`: 添加视频记录
- `list_generation_records()`: 列出生成记录
- `get_generation_record()`: 获取单个记录
- `delete_generation_record()`: 删除记录
- `create_or_update_task()`: 创建或更新任务
- `get_task()`: 获取任务
- `get_running_task()`: 获取正在运行的任务
- `list_tasks()`: 列出所有任务
- `delete_task()`: 删除任务

**表结构**:
```sql
-- generation_records表
CREATE TABLE generation_records (
    unique_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    text TEXT,
    character TEXT,
    model_name TEXT,
    video_path TEXT,
    audio_path TEXT,
    text_path TEXT,
    llm_response TEXT,
    generation_time REAL,
    config TEXT,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- tasks表
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    text TEXT,
    character TEXT,
    model_name TEXT,
    video_path TEXT,
    audio_path TEXT,
    text_path TEXT,
    config TEXT,
    error_message TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    generation_time REAL
)
```

### 3. chat_db.py - 聊天数据库
**功能**:
- `init_database()`: 初始化数据库
- `create_chat_session()`: 创建聊天会话
- `get_chat_session()`: 获取会话
- `list_chat_sessions()`: 列出所有会话
- `add_chat_message()`: 添加消息
- `get_chat_messages()`: 获取会话消息
- `delete_chat_session()`: 删除会话

**表结构**:
```sql
-- chat_sessions表
CREATE TABLE chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    character TEXT,
    config TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- chat_messages表
CREATE TABLE chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_type TEXT NOT NULL,
    content_type TEXT,
    text_content TEXT,
    text_path TEXT,
    audio_path TEXT,
    audio_base64 TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
)
```

## 🛠️ 工具模块（utils/）

### 1. run_llm_talk.py
**功能**: LLM + TTS 音频生成
**主要函数**:
- `generate_audio(text, output_path, character)`: 生成音频文件

### 2. run_nerffacespeech.py
**功能**: NeRF 视频生成
**主要函数**:
- `generate_video(audio_path, character, output_path, model_name)`: 生成视频

### 3. run_chat.py
**功能**: 聊天对话
**主要函数**:
- `chat_with_llm(user_input, character, enable_audio)`: 聊天对话
- `get_llm_only(text)`: 仅获取LLM回答

### 4. run_training.py
**功能**: 模型训练
**主要函数**:
- `start_training(...)`: 启动训练
- `get_training_status(task_id)`: 获取训练状态
- `list_training_tasks()`: 列出训练任务
- `stop_training(task_id)`: 停止训练

## 📦 核心功能模块

### 1. 日志系统
- **BufferLogHandler**: 自定义日志处理器
- **LOG_BUFFER**: 日志缓冲区（最大1000条）
- **DEBUG_LOG_BUFFER**: Debug日志缓冲区
- **setup_logging()**: 配置日志系统

### 2. 任务管理系统
- **TASKS**: 内存中的任务字典
- **TASKS_LOCK**: 线程锁
- **run_video_generation_task()**: 后台任务执行函数

### 3. 文件服务
- **视频文件**: 存储在 `VIDEOS_STORAGE_DIR`
- **音频文件**: 存储在 `AUDIOS_STORAGE_DIR`
- **文本文件**: 存储在 `TEXTS_STORAGE_DIR`
- 支持Range请求（视频seek功能）

## 🔧 配置管理（config.py）

**主要配置项**:
- `PROJECT_ROOT`: 项目根目录
- `NERF_CODE_DIR`: NeRFFaceSpeech代码目录
- `OUTPUT_VIDEO_DIR`: 视频输出目录
- `OUTPUT_AUDIO_DIR`: 音频输出目录
- `MODEL_DIR`: 模型目录
- `WEBUI_DIR`: WebUI目录
- `DATABASE_DIR`: 数据库目录
- `VIDEOS_STORAGE_DIR`: 视频存储目录
- `AUDIOS_STORAGE_DIR`: 音频存储目录
- `TEXTS_STORAGE_DIR`: 文本存储目录
- `DATA_DIR`: 数据目录
- `TRAINING_DATASET_DIR`: 训练数据集目录
- `API_CONDA_ENV`: API环境路径
- `LLM_CONDA_ENV`: LLM环境路径
- `NERF_CONDA_ENV`: NeRF环境路径

## 📋 迁移清单

### 必须迁移的核心模块

1. **配置模块** (`config.py`)
   - ✅ 所有路径配置
   - ✅ 环境配置
   - ✅ 辅助函数

2. **数据库模块** (`database/`)
   - ✅ settings_db.py
   - ✅ video_records_db.py
   - ✅ chat_db.py

3. **工具模块** (`utils/`)
   - ✅ run_llm_talk.py
   - ✅ run_nerffacespeech.py
   - ✅ run_chat.py
   - ✅ run_training.py
   - ✅ llm_talk_api_bridge.py
   - ✅ llm_talk_with_text_bridge.py

4. **核心功能**
   - ✅ 日志系统（BufferLogHandler, setup_logging）
   - ✅ 任务管理系统（TASKS, TASKS_LOCK）
   - ✅ 文件服务逻辑

### 需要转换的API端点

所有33个API端点需要转换为Gradio函数或保留为HTTP端点（如果Gradio需要）。

### 可选迁移

1. **启动脚本** (`start.py`)
   - 可以保留或修改为Gradio启动方式

2. **静态文件服务** (`simple_web.py`)
   - Gradio内置静态文件服务，可能不需要

## 🎯 迁移策略

### 方案A: 完全迁移到Gradio
- 将所有API逻辑转换为Gradio函数
- 使用Gradio的组件和事件系统
- 优点：统一框架，简化架构
- 缺点：需要重写大量代码

### 方案B: 混合架构（推荐）
- 保留FastAPI作为后端API服务
- Gradio作为前端界面，通过HTTP调用API
- 优点：保持API兼容性，迁移成本低
- 缺点：需要同时运行两个服务

### 方案C: 渐进式迁移
- 先迁移核心功能到Gradio
- 保留FastAPI处理复杂任务
- 逐步迁移其他功能

## 📝 下一步行动

1. **选择迁移方案**（建议方案B）
2. **创建新的目录结构**
3. **迁移配置模块**
4. **迁移数据库模块**
5. **迁移工具模块**
6. **实现Gradio界面组件**
7. **集成API调用**

