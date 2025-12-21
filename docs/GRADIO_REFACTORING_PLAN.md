# Gradio 重构计划

## 目标

将现有的 FastAPI + HTML 架构重构为 **Gradio + 直接函数调用** 架构：
- ✅ 前端使用 Gradio（替代 HTML）
- ✅ 尽量还原原有 UI 界面风格
- ✅ 后端直接调用函数（不需要 HTTP 服务器）

## 当前架构分析

### 主要功能模块

1. **视频生成** (`generate.html`)
   - 文本输入 → LLM生成对话 → 音频生成 → NeRF视频生成
   - 模型选择、角色选择
   - 实时日志显示
   - 视频预览和历史记录

2. **人机对话** (`chat.html`)
   - 文本对话
   - 音频对话（可选）
   - 对话历史管理
   - 角色选择

3. **模型训练** (`train.html`)
   - 数据集选择
   - 训练参数配置
   - 训练任务管理
   - 训练日志查看

4. **设置管理** (`settings.js`)
   - 主题切换（tech/warm/minimal）
   - 字体设置
   - 字体大小

5. **数据库管理** (`database.html`)
   - 查看数据库内容
   - 数据库操作

6. **日志查看** (`logs.html`)
   - 实时日志显示
   - 调试日志

### 后端 API 端点

- `/generate_video` - 异步视频生成
- `/generate_video/status/{id}` - 查询任务状态
- `/chat` - 对话接口
- `/train/start` - 开始训练
- `/train/status/{id}` - 查询训练状态
- `/api/settings` - 设置管理
- `/logs` - 日志查看
- `/generation_records` - 生成记录

### 核心函数（utils）

- `utils.run_nerffacespeech.generate_video()` - 视频生成
- `utils.run_chat.chat_with_llm()` - 对话
- `utils.run_training.start_training()` - 训练
- `utils.run_llm_talk.generate_audio()` - 音频生成

## 重构计划

### 阶段 1：项目结构搭建 ✅

1. 创建新的项目结构
   ```
   gradio_app/
   ├── __init__.py
   ├── main.py              # Gradio 主应用
   ├── pages/
   │   ├── __init__.py
   │   ├── generate.py     # 视频生成页面
   │   ├── chat.py          # 对话页面
   │   ├── train.py         # 训练页面
   │   └── settings.py      # 设置页面
   ├── services/
   │   ├── __init__.py
   │   ├── video_service.py  # 视频生成服务（直接调用函数）
   │   ├── chat_service.py   # 对话服务
   │   ├── training_service.py # 训练服务
   │   └── settings_service.py # 设置服务
   ├── utils/
   │   ├── __init__.py
   │   ├── theme.py         # 主题管理
   │   └── database.py      # 数据库工具
   └── config.py            # 配置（复用 fastapi_server/config.py）
   ```

2. 创建启动脚本
   ```
   start_gradio.py          # 启动 Gradio 应用
   ```

### 阶段 2：核心服务层（直接函数调用）✅

1. **视频生成服务** (`services/video_service.py`)
   - 直接调用 `utils.run_nerffacespeech.generate_video()`
   - 直接调用 `utils.run_llm_talk.generate_audio()`
   - 任务状态管理（使用数据库）
   - 日志收集

2. **对话服务** (`services/chat_service.py`)
   - 直接调用 `utils.run_chat.chat_with_llm()`
   - 对话历史管理（使用数据库）

3. **训练服务** (`services/training_service.py`)
   - 直接调用 `utils.run_training.start_training()`
   - 训练状态管理

4. **设置服务** (`services/settings_service.py`)
   - 直接调用数据库函数
   - 主题、字体等设置管理

### 阶段 3：Gradio UI 实现

#### 3.1 视频生成页面 (`pages/generate.py`)

**UI 组件**：
- 文本输入框（多行）
- 角色选择下拉框
- 模型选择下拉框
- 生成按钮
- 日志显示区域（实时更新）
- 视频预览区域
- 历史记录列表

**功能**：
- 提交生成任务
- 实时显示日志（使用 `gr.LoggingComponent` 或自定义）
- 轮询任务状态
- 显示生成的视频
- 历史记录管理

**样式还原**：
- 使用 Gradio 的 `theme` 参数还原深色主题
- 自定义 CSS（通过 `gr.HTML` 或 `gr.CSS`）
- 保持原有的卡片式布局

#### 3.2 对话页面 (`pages/chat.py`)

**UI 组件**：
- 对话历史显示（`gr.Chatbot`）
- 文本输入框
- 发送按钮
- 角色选择
- 音频播放（如果生成音频）

**功能**：
- 发送消息
- 显示对话历史
- 音频播放
- 会话管理

#### 3.3 训练页面 (`pages/train.py`)

**UI 组件**：
- 数据集选择
- 训练参数配置（学习率、epoch等）
- 开始训练按钮
- 训练日志显示
- 训练任务列表

**功能**：
- 配置训练参数
- 启动训练
- 查看训练日志
- 管理训练任务

#### 3.4 设置页面 (`pages/settings.py`)

**UI 组件**：
- 主题选择（Radio）
- 字体选择
- 字体大小选择
- 保存按钮

**功能**：
- 保存设置到数据库
- 应用主题（通过 Gradio 的 `theme` 参数）

### 阶段 4：主应用集成 (`main.py`)

使用 Gradio 的 `Blocks` 或 `TabbedInterface` 创建多页面应用：

```python
import gradio as gr

with gr.Blocks(theme=load_theme()) as app:
    with gr.Tabs():
        with gr.Tab("视频生成"):
            from pages.generate import create_generate_ui
            create_generate_ui()
        
        with gr.Tab("人机对话"):
            from pages.chat import create_chat_ui
            create_chat_ui()
        
        with gr.Tab("模型训练"):
            from pages.train import create_train_ui
            create_train_ui()
        
        with gr.Tab("设置"):
            from pages.settings import create_settings_ui
            create_settings_ui()
```

### 阶段 5：主题和样式还原

1. **主题配置** (`utils/theme.py`)
   - 定义三个主题（tech/warm/minimal）
   - 转换为 Gradio 的 `Theme` 对象
   - 自定义 CSS（通过 `gr.CSS`）

2. **样式还原**
   - 深色背景
   - 卡片式布局
   - 渐变背景
   - 圆角边框
   - 阴影效果

### 阶段 6：测试和优化

1. **功能测试**
   - 视频生成流程
   - 对话功能
   - 训练功能
   - 设置保存

2. **UI 测试**
   - 界面还原度
   - 响应式布局
   - 主题切换

3. **性能优化**
   - 异步任务处理
   - 日志实时更新优化
   - 视频加载优化

## 技术要点

### 1. 直接函数调用

不需要 HTTP 请求，直接调用函数：
```python
# 旧方式（HTTP）
response = requests.post("http://localhost:8000/generate_video", json={...})

# 新方式（直接调用）
from services.video_service import generate_video
result = generate_video(text="...", character="...", model_name="...")
```

### 2. 异步任务处理

使用 Python 的 `threading` 或 `asyncio` 处理长时间任务：
```python
import threading

def generate_video_async(text, character, model_name, progress_callback):
    def task():
        result = generate_video(text, character, model_name)
        progress_callback(result)
    
    thread = threading.Thread(target=task)
    thread.start()
    return thread
```

### 3. 实时日志更新

使用 Gradio 的 `gr.LoggingComponent` 或自定义组件：
```python
import logging
from gradio import LoggingComponent

# 自定义日志处理器
class GradioLogHandler(logging.Handler):
    def __init__(self, logging_component):
        super().__init__()
        self.logging_component = logging_component
    
    def emit(self, record):
        self.logging_component.update(record.getMessage())
```

### 4. 任务状态轮询

使用 Gradio 的 `gr.Timer` 或 `gr.update` 实现轮询：
```python
def check_task_status(task_id):
    status = get_task_status(task_id)
    if status == "completed":
        return gr.update(value=status, visible=False), get_video_path(task_id)
    return gr.update(value=status), None

# 使用 gr.Timer 定期检查
timer = gr.Timer(value=1.0)  # 每秒检查一次
timer.change(check_task_status, inputs=[task_id], outputs=[status, video])
```

## 实施步骤

### Step 1: 创建项目结构 ✅
- [x] 创建目录结构
- [x] 创建基础文件

### Step 2: 实现服务层 ✅
- [x] 视频生成服务
- [x] 对话服务
- [x] 训练服务
- [x] 设置服务

### Step 3: 实现 Gradio UI
- [ ] 视频生成页面
- [ ] 对话页面
- [ ] 训练页面
- [ ] 设置页面

### Step 4: 主题和样式
- [ ] 主题配置
- [ ] CSS 样式还原

### Step 5: 集成和测试
- [ ] 主应用集成
- [ ] 功能测试
- [ ] UI 测试

## 预期成果

1. **单一入口**：`python start_gradio.py` 启动整个应用
2. **无 HTTP 服务器**：直接函数调用，更简单
3. **UI 还原**：尽量还原原有界面风格
4. **功能完整**：所有原有功能都保留
5. **易于维护**：代码结构清晰，易于扩展

