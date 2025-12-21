# Gradio重构快速开始指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install gradio>=4.0.0
```

### 2. 创建基础结构

```bash
mkdir -p gradio_components gradio_styles gradio_utils
touch gradio_app.py gradio_components/__init__.py gradio_utils/__init__.py
```

## 📝 代码示例

### 示例1: 主应用入口 (gradio_app.py)

```python
"""
Gradio主应用入口
"""
import gradio as gr
from gradio_components.home import create_home_interface
from gradio_components.generate import create_generate_interface
from gradio_components.chat import create_chat_interface
from gradio_components.train import create_train_interface
from gradio_utils.theme_manager import load_theme_css

# 加载默认主题CSS
DEFAULT_THEME_CSS = load_theme_css("tech")

def create_main_app():
    """创建主应用"""
    with gr.Blocks(
        title="NeRFFaceSpeech",
        theme=gr.themes.Soft(),
        css=DEFAULT_THEME_CSS
    ) as app:
        # 全局状态
        app_state = gr.State({
            "current_theme": "tech",
            "current_font": "Inter",
            "font_size": 14
        })
        
        # 使用Tabs实现多页面
        with gr.Tabs() as tabs:
            with gr.Tab("首页", id="home"):
                home_interface = create_home_interface()
            
            with gr.Tab("视频生成", id="generate"):
                generate_interface = create_generate_interface()
            
            with gr.Tab("人机对话", id="chat"):
                chat_interface = create_chat_interface()
            
            with gr.Tab("训练模型", id="train"):
                train_interface = create_train_interface()
        
        # 设置按钮（全局）
        settings_btn = gr.Button("⚙️ 设置", elem_id="settings-btn")
        
        # 设置侧边栏（使用gr.Column实现）
        with gr.Column(visible=False, elem_id="settings-sidebar") as settings_sidebar:
            # 设置内容
            pass
    
    return app

if __name__ == "__main__":
    app = create_main_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

### 示例2: API客户端 (gradio_utils/api_client.py)

```python
"""
API客户端封装
"""
import requests
from typing import Optional, Dict, Any

class APIClient:
    """统一的API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def generate_video(
        self, 
        text: str, 
        character: str, 
        model_name: str
    ) -> Dict[str, Any]:
        """提交视频生成任务"""
        response = requests.post(
            f"{self.base_url}/generate_video",
            json={
                "text": text,
                "character": character,
                "model_name": model_name
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_video_status(self, task_id: str) -> Dict[str, Any]:
        """获取视频生成状态"""
        response = requests.get(
            f"{self.base_url}/generate_video/status/{task_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> list:
        """获取模型列表"""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def chat(
        self,
        text: str,
        character: str = "ayanami",
        enable_audio: bool = True,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """发送聊天消息"""
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "text": text,
                "character": character,
                "enable_audio": enable_audio,
                "session_id": session_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_settings(self) -> Dict[str, str]:
        """获取所有设置"""
        response = requests.get(f"{self.base_url}/api/settings")
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("data", {})
        return {}
    
    def update_setting(self, key: str, value: str) -> bool:
        """更新设置"""
        response = requests.post(
            f"{self.base_url}/api/settings/{key}",
            json={"value": value}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("success", False)
```

### 示例3: 首页组件 (gradio_components/home.py)

```python
"""
首页组件
"""
import gradio as gr

def create_home_interface():
    """创建首页界面"""
    with gr.Column() as home:
        # 标题
        gr.Markdown(
            "# NeRFFaceSpeech\n选择核心模块",
            elem_id="home-title"
        )
        
        # 功能卡片
        with gr.Row():
            with gr.Column():
                generate_card = gr.Button(
                    "视频生成",
                    variant="primary",
                    elem_id="home-card-generate"
                )
                gr.Markdown(
                    "**文本 → LLM+TTS → NeRF 视频**\n\n"
                    "输入文本、选择角色与模型，一键生成视频（自动转码 H.264）。",
                    elem_id="home-card-desc"
                )
            
            with gr.Column():
                chat_card = gr.Button(
                    "人机对话",
                    variant="primary",
                    elem_id="home-card-chat"
                )
                gr.Markdown(
                    "**实时对话 · LLM + TTS**\n\n"
                    "与AI角色进行实时文本对话，支持LLM生成回答和TTS语音合成。",
                    elem_id="home-card-desc"
                )
            
            with gr.Column():
                train_card = gr.Button(
                    "训练模型",
                    variant="primary",
                    elem_id="home-card-train"
                )
                gr.Markdown(
                    "**模型训练/微调**\n\n"
                    "训练和微调NeRF模型，支持自定义数据集和参数配置。",
                    elem_id="home-card-desc"
                )
    
    return home
```

### 示例4: 视频生成组件 (gradio_components/generate.py)

```python
"""
视频生成组件
"""
import gradio as gr
from gradio_utils.api_client import APIClient
import time

api_client = APIClient()

def create_generate_interface():
    """创建视频生成界面"""
    with gr.Column() as generate:
        # 标题
        gr.Markdown("# 视频生成", elem_id="generate-title")
        
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="输入要说的话",
                    placeholder="例如：你好，请简要介绍一下人工智能",
                    lines=5
                )
                
                with gr.Row():
                    character_select = gr.Dropdown(
                        choices=["ayanami", "Aerith"],
                        value="ayanami",
                        label="选择角色"
                    )
                    model_select = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="选择模型（pkl）"
                    )
                
                with gr.Row():
                    refresh_btn = gr.Button("刷新模型列表", variant="secondary")
                    generate_btn = gr.Button("🚀 开始生成视频", variant="primary")
                
                # 进度显示
                progress_bar = gr.Progress()
                progress_text = gr.Markdown("", visible=False)
                
                # 状态日志
                status_log = gr.Textbox(
                    label="状态 / 日志",
                    value="就绪",
                    lines=10,
                    interactive=False
                )
            
            # 右侧：视频预览
            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="视频预览",
                    elem_id="generate-video"
                )
                
                with gr.Row():
                    download_btn = gr.Button("⬇️ 下载视频", variant="secondary")
                    fullscreen_btn = gr.Button("⛶ 全屏播放", variant="secondary")
        
        # 历史记录
        history_df = gr.Dataframe(
            label="历史记录",
            headers=["时间", "文本", "角色", "模型"],
            visible=False
        )
        
        # 事件绑定
        def load_models():
            """加载模型列表"""
            try:
                models = api_client.list_models()
                return gr.Dropdown(choices=models, value=models[0] if models else None)
            except Exception as e:
                return gr.Dropdown(choices=[], value=None)
        
        def start_generation(text, character, model_name, progress=gr.Progress()):
            """开始生成视频"""
            if not text or not model_name:
                return None, "请填写完整信息"
            
            try:
                # 提交任务
                result = api_client.generate_video(text, character, model_name)
                task_id = result.get("unique_id")
                
                if not task_id:
                    return None, "任务提交失败"
                
                # 轮询任务状态
                progress(0, desc="任务已提交，等待开始...")
                while True:
                    time.sleep(2)
                    status = api_client.get_video_status(task_id)
                    task_data = status.get("data", {})
                    task_status = task_data.get("status")
                    
                    if task_status == "completed":
                        progress(1.0, desc="生成完成")
                        video_path = task_data.get("video_path")
                        if video_path:
                            # 构造视频URL
                            video_url = f"http://localhost:8000/videos/{video_path.split('/')[-1]}"
                            return video_url, "生成成功"
                        return None, "视频路径无效"
                    
                    elif task_status == "failed":
                        error = task_data.get("error", "生成失败")
                        return None, f"生成失败: {error}"
                    
                    elif task_status in ["pending", "running"]:
                        progress(0.5, desc="正在生成中...")
                        continue
                    
                    else:
                        return None, f"未知状态: {task_status}"
            
            except Exception as e:
                return None, f"错误: {str(e)}"
        
        # 绑定事件
        refresh_btn.click(
            fn=load_models,
            outputs=model_select
        )
        
        generate_btn.click(
            fn=start_generation,
            inputs=[text_input, character_select, model_select],
            outputs=[video_output, status_log],
            show_progress=True
        )
        
        # 页面加载时自动加载模型列表
        generate.load(
            fn=load_models,
            outputs=model_select
        )
    
    return generate
```

### 示例5: 聊天组件 (gradio_components/chat.py)

```python
"""
聊天组件
"""
import gradio as gr
from gradio_utils.api_client import APIClient

api_client = APIClient()

def create_chat_interface():
    """创建聊天界面"""
    with gr.Column() as chat:
        # 标题
        gr.Markdown("# 人机对话", elem_id="chat-title")
        
        # 聊天机器人
        chatbot = gr.Chatbot(
            label="对话",
            height=500,
            elem_id="chat-messages"
        )
        
        # 输入区域
        with gr.Row():
            text_input = gr.Textbox(
                label="输入消息",
                placeholder="输入消息... (Shift+Enter换行，Enter发送)",
                lines=3,
                scale=4
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
        
        # 设置
        with gr.Row():
            character_select = gr.Dropdown(
                choices=["ayanami", "Aerith"],
                value="ayanami",
                label="角色"
            )
            enable_audio = gr.Checkbox(
                label="启用音频",
                value=True
            )
            volume_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=50,
                label="音量"
            )
        
        # 事件处理
        def send_message(message, history, character, enable_audio_flag):
            """发送消息"""
            if not message.strip():
                return history, ""
            
            # 添加用户消息
            history.append([message, None])
            
            try:
                # 调用API
                result = api_client.chat(
                    text=message,
                    character=character,
                    enable_audio=enable_audio_flag
                )
                
                if result.get("success"):
                    answer = result.get("data", {}).get("llm_answer", "")
                    audio_url = result.get("data", {}).get("audio_url")
                    
                    # 添加AI回复
                    history[-1][1] = answer
                    
                    # 如果有音频，添加到消息中
                    if audio_url and enable_audio_flag:
                        # Gradio Chatbot支持音频，但需要特殊格式
                        # 这里可以返回文本，音频通过其他方式播放
                        pass
                    
                    return history, ""
                else:
                    error_msg = result.get("error", "未知错误")
                    history[-1][1] = f"❌ 错误: {error_msg}"
                    return history, ""
            
            except Exception as e:
                history[-1][1] = f"❌ 网络错误: {str(e)}"
                return history, ""
        
        # 绑定事件
        send_btn.click(
            fn=send_message,
            inputs=[text_input, chatbot, character_select, enable_audio],
            outputs=[chatbot, text_input]
        )
        
        text_input.submit(
            fn=send_message,
            inputs=[text_input, chatbot, character_select, enable_audio],
            outputs=[chatbot, text_input]
        )
    
    return chat
```

### 示例6: 主题管理 (gradio_utils/theme_manager.py)

```python
"""
主题管理
"""
from pathlib import Path

THEME_DIR = Path(__file__).parent.parent / "gradio_styles"

def load_theme_css(theme_name: str) -> str:
    """加载主题CSS"""
    theme_file = THEME_DIR / f"theme_{theme_name}.css"
    if theme_file.exists():
        return theme_file.read_text(encoding="utf-8")
    else:
        # 返回默认主题
        default_file = THEME_DIR / "theme_tech.css"
        if default_file.exists():
            return default_file.read_text(encoding="utf-8")
        return ""

def apply_theme_css(theme_name: str) -> str:
    """应用主题CSS（返回JavaScript代码）"""
    css_content = load_theme_css(theme_name)
    # 将CSS注入到页面中
    js_code = f"""
    <script>
    (function() {{
        const style = document.createElement('style');
        style.textContent = {repr(css_content)};
        document.head.appendChild(style);
    }})();
    </script>
    """
    return js_code
```

### 示例7: 主题CSS (gradio_styles/theme_tech.css)

```css
/* 科技风主题 - Gradio适配版 */

:root {
  --primary: #1e40af;
  --primary-2: #3b82f6;
  --bg: #0b1020;
  --card: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.14);
  --text: #e5e7eb;
  --muted: #9ca3af;
  --g1: rgba(59,130,246,0.16);
  --g2: rgba(14,165,233,0.14);
  --g3: rgba(56,189,248,0.12);
  --card-bg: rgba(59,130,246,0.25);
}

/* 全局样式 */
body {
  background: radial-gradient(circle at 20% 20%, var(--g1), transparent 25%),
              radial-gradient(circle at 80% 30%, var(--g2), transparent 25%),
              radial-gradient(circle at 50% 80%, var(--g3), transparent 22%),
              var(--bg) !important;
  color: var(--text) !important;
}

/* Gradio组件样式覆盖 */
.gradio-container {
  background: transparent !important;
}

/* 卡片样式 */
.gr-box {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
}

/* 按钮样式 */
button.primary {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  border-color: var(--primary-2) !important;
  color: #fff !important;
}

/* 输入框样式 */
input, textarea, select {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}

/* 设置按钮 */
#settings-btn {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  padding: 10px 16px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--card-bg);
  color: var(--text);
}
```

## 🔄 迁移步骤

### 步骤1: 创建基础框架
1. 创建文件结构
2. 实现API客户端
3. 实现主题管理

### 步骤2: 实现首页
1. 创建home.py
2. 实现导航功能
3. 应用基础样式

### 步骤3: 实现各功能页面
1. 按优先级实现（视频生成 > 聊天 > 训练）
2. 逐个测试功能
3. 应用样式

### 步骤4: 完善样式
1. 提取原有CSS
2. 转换为Gradio兼容格式
3. 测试主题切换

### 步骤5: 集成测试
1. 功能测试
2. 样式测试
3. 性能测试

## 📌 注意事项

1. **Gradio版本**: 建议使用Gradio 4.0+，支持更多自定义功能
2. **CSS优先级**: 使用`!important`确保自定义样式生效
3. **组件ID**: 使用`elem_id`为组件添加唯一ID，方便CSS选择
4. **异步处理**: 长时间任务使用`gr.Progress`和异步函数
5. **状态管理**: 使用`gr.State`管理复杂状态

## 🎨 样式还原技巧

1. **CSS变量**: 保持原有的CSS变量系统
2. **组件类名**: 使用`elem_classes`添加自定义类
3. **内联样式**: 对于特殊样式，可以使用`style`参数
4. **JavaScript**: 对于复杂交互，可以使用`gr.HTML`嵌入JavaScript

## 🚀 启动方式

### 开发模式
```bash
python gradio_app.py
```

### 生产模式
```bash
# 使用gunicorn等WSGI服务器
# 或直接使用Gradio的launch方法
```

## 📚 参考资源

- Gradio文档: https://www.gradio.app/docs/
- Gradio示例: https://www.gradio.app/demos/
- CSS定制: https://www.gradio.app/guides/custom-CSS-and-JavaScript

