#!/usr/bin/env python3
"""
Gradio 前端应用
从 fastapi_server/webui/ 迁移而来，使用 Gradio 实现前端界面
尽量还原原有样式和功能
"""
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import requests
import json
import uuid
from typing import Optional, Tuple
from datetime import datetime

# 导入共享模块
from shared.config import (
    PROJECT_ROOT, MODEL_DIR, VIDEOS_STORAGE_DIR, 
    AUDIOS_STORAGE_DIR, TEXTS_STORAGE_DIR
)
try:
    from shared.config import get_character_list
except ImportError:
    # 如果函数不存在，使用默认实现
    def get_character_list():
        from shared.config import CHARACTER_AUDIO_PROMPTS
        return list(CHARACTER_AUDIO_PROMPTS.keys())
from shared.database import settings_db
from shared.database import video_records_db
from shared.database import chat_db

# API 基础 URL 配置
# 优先级：环境变量 > 默认值
# 如果通过端口转发访问，需要设置为实际的后端地址
# 例如：如果通过 ssh -L 8000:localhost:8000 转发，使用 http://localhost:8000
# 如果后端在远程服务器，使用 http://服务器IP:8000
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# 如果设置了 GRADIO_SERVER_NAME，尝试从当前请求推断后端地址
# 这对于端口转发场景很有用
if os.environ.get("GRADIO_SERVER_NAME"):
    # 如果 Gradio 运行在特定地址，可以自动推断后端地址
    pass  # 保持使用环境变量或默认值

# 自定义 CSS 样式（还原原有样式）
CUSTOM_CSS = """
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
  --grad1: #0b1220;
  --grad2: #0f172a;
  --card-bg: rgba(59,130,246,0.25);
}

body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  background: radial-gradient(circle at 20% 20%, var(--g1), transparent 25%),
              radial-gradient(circle at 80% 30%, var(--g2), transparent 25%),
              radial-gradient(circle at 50% 80%, var(--g3), transparent 22%),
              var(--bg) !important;
  color: var(--text) !important;
}

.gradio-container {
  background: transparent !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
}

.gradio-card {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 18px !important;
  box-shadow: 0 12px 32px rgba(0,0,0,0.25) !important;
}

.gradio-button.primary {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  border: 1px solid var(--primary-2) !important;
  color: #fff !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
}

.gradio-button.secondary {
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-weight: 600 !important;
  border-radius: 12px !important;
}

.gradio-textbox textarea,
.gradio-dropdown select,
.gradio-textbox input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  padding: 12px 14px !important;
}

.gradio-textbox textarea:focus,
.gradio-dropdown select:focus,
.gradio-textbox input:focus {
  border-color: rgba(96,165,250,0.8) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
}

.gradio-video video {
  border-radius: 16px !important;
  border: 1px solid #1f2937 !important;
  background: transparent !important;
}

h1, h2, h3 {
  color: var(--text) !important;
}

label {
  color: var(--text) !important;
  font-weight: 700 !important;
}
"""


def get_settings():
    """获取设置"""
    try:
        settings = settings_db.get_all_settings()
        print(f"[前端] 获取设置: 找到 {len(settings)} 个设置项")
        return settings
    except Exception as e:
        print(f"[前端] ❌ 获取设置失败: {e}")
        return {}


def get_models():
    """获取可用模型列表"""
    try:
        models = []
        if MODEL_DIR.exists():
            for pkl_file in MODEL_DIR.glob("*.pkl"):
                models.append(pkl_file.name)
        result = models if models else ["未找到模型"]
        print(f"[前端] 获取模型列表: 找到 {len(result)} 个模型")
        return result
    except Exception as e:
        print(f"[前端] ❌ 获取模型列表失败: {e}")
        return ["错误"]


def check_backend_connection():
    """检查后端连接"""
    try:
        url = f"{API_BASE_URL}/docs"
        print(f"[前端] 检查后端连接: {url}")
        response = requests.get(url, timeout=3)
        result = response.status_code == 200
        if result:
            print(f"[前端] ✅ 后端连接成功: {url}")
        else:
            print(f"[前端] ❌ 后端连接失败: {url} (状态码: {response.status_code})")
        return result
    except requests.exceptions.ConnectionError as e:
        print(f"[前端] ❌ 后端连接失败: {API_BASE_URL}/docs")
        print(f"[前端]    错误详情: {str(e)}")
        print(f"[前端]    解决方案:")
        print(f"[前端]      1. 确保后端服务已启动: uvicorn backend.main:app --host 0.0.0.0 --port 8000")
        print(f"[前端]      2. 或使用一键启动脚本: python start_all.py")
        print(f"[前端]      3. 检查 API_BASE_URL 配置是否正确")
        return False
    except Exception as e:
        print(f"[前端] ❌ 后端连接失败: {API_BASE_URL}/docs (错误: {e})")
        return False


def generate_video_async(text: str, character: str, model_name: str, progress=gr.Progress()):
    """异步生成视频"""
    try:
        if not text.strip():
            return None, "请输入文本"
        
        if not model_name or model_name == "未找到模型":
            return None, "请选择模型"
        
        # 检查后端连接
        if not check_backend_connection():
            error_msg = (
                f"❌ 无法连接到后端服务\n\n"
                f"**目标地址**: {API_BASE_URL}\n\n"
                f"**解决方案**:\n"
                f"1. 启动后端服务:\n"
                f"   ```bash\n"
                f"   cd gradio_app\n"
                f"   uvicorn backend.main:app --host 0.0.0.0 --port 8000\n"
                f"   ```\n\n"
                f"2. 或使用一键启动脚本:\n"
                f"   ```bash\n"
                f"   cd gradio_app\n"
                f"   python start_all.py\n"
                f"   ```\n\n"
                f"3. 检查 API_BASE_URL 环境变量配置"
            )
            return None, error_msg
        
        unique_id = str(uuid.uuid4())
        
        # 调用后端 API
        api_url = f"{API_BASE_URL}/generate_video"
        print(f"[前端] 提交视频生成任务")
        print(f"[前端]   目标地址: {api_url}")
        print(f"[前端]   参数: text={text[:50]}..., character={character}, model={model_name}")
        
        progress(0.1, desc="提交生成任务...")
        response = requests.post(
            api_url,
            json={
                "text": text,
                "character": character,
                "model_name": model_name
            },
            timeout=30
        )
        
        print(f"[前端]   响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                task_id = result.get("unique_id", unique_id)
                
                # 轮询任务状态
                max_wait = 600  # 最多等待10分钟
                wait_time = 0
                progress(0.2, desc="生成中，请稍候...")
                
                while wait_time < max_wait:
                    try:
                        status_url = f"{API_BASE_URL}/generate_video/status/{task_id}"
                        if wait_time == 0:  # 只在第一次打印
                            print(f"[前端] 开始轮询任务状态")
                            print(f"[前端]   状态查询地址: {status_url}")
                        
                        status_response = requests.get(
                            status_url,
                            timeout=10
                        )
                        
                        if status_response.status_code == 200:
                            status_response_data = status_response.json()
                            # 后端返回格式: {"success": True, "data": {...}}
                            if status_response_data.get("success") and status_response_data.get("data"):
                                status_data = status_response_data.get("data")
                            else:
                                status_data = status_response_data  # 兼容直接返回数据的情况
                            
                            status = status_data.get("status", "unknown")
                            
                            # 打印状态信息（每10秒打印一次，避免日志过多）
                            if wait_time % 10 == 0:
                                print(f"[前端] 任务状态: {status}, 已等待: {wait_time}秒")
                                if status_data.get("video_path"):
                                    print(f"[前端]   视频路径: {status_data.get('video_path')}")
                            
                            if status == "completed":
                                # 优先使用 HTTP URL（后端静态文件服务）
                                # 后端会返回 video_path，我们可以根据 task_id 构建 URL
                                video_http_url = f"{API_BASE_URL}/videos/{task_id}.mp4"
                                print(f"[前端] ✅ 视频生成完成，使用 HTTP URL: {video_http_url}")
                                progress(1.0, desc="生成完成！")
                                # Gradio Video 组件支持 HTTP URL
                                return video_http_url, "✅ 视频生成成功！"
                                
                                # 备用方案：尝试从 video_path 获取本地文件路径
                                video_path = status_data.get("video_path")
                                if video_path:
                                    from pathlib import Path as PathLib
                                    video_path_obj = PathLib(video_path)
                                    
                                    # 如果是绝对路径，检查文件是否存在
                                    if video_path_obj.is_absolute() and video_path_obj.exists():
                                        print(f"[前端] ✅ 找到视频文件（绝对路径）: {video_path_obj}")
                                        progress(1.0, desc="生成完成！")
                                        return str(video_path_obj), "✅ 视频生成成功！"
                                    
                                    # 如果是相对路径，相对于 PROJECT_ROOT
                                    if not video_path_obj.is_absolute():
                                        full_path = PROJECT_ROOT / video_path
                                        if full_path.exists():
                                            print(f"[前端] ✅ 找到视频文件（相对路径）: {full_path}")
                                            progress(1.0, desc="生成完成！")
                                            return str(full_path), "✅ 视频生成成功！"
                                
                                # 如果都没有找到，尝试使用 task_id 构建本地路径
                                print(f"[前端] ⚠️ 尝试使用 task_id 构建本地路径: {task_id}")
                                possible_paths = [
                                    VIDEOS_STORAGE_DIR / f"{task_id}.mp4",
                                    PROJECT_ROOT / "database" / "videos" / f"{task_id}.mp4"
                                ]
                                for possible_path in possible_paths:
                                    if possible_path.exists():
                                        print(f"[前端] ✅ 在备用路径找到视频: {possible_path}")
                                        progress(1.0, desc="生成完成！")
                                        return str(possible_path), "✅ 视频生成成功！"
                                
                                # 最后尝试使用 HTTP URL
                                print(f"[前端] ⚠️ 使用 HTTP URL 作为最后尝试: {video_http_url}")
                                progress(1.0, desc="生成完成！")
                                return video_http_url, "✅ 视频生成成功！"
                            
                            elif status == "failed":
                                error = status_data.get("error", "未知错误")
                                return None, f"❌ 生成失败: {error}"
                            
                            elif status == "running":
                                progress(0.3 + (wait_time / max_wait) * 0.6, desc=f"生成中... ({status})")
                    
                    except requests.RequestException:
                        pass  # 继续等待
                    
                    import time
                    time.sleep(2)
                    wait_time += 2
                
                return None, "⏱️ 生成超时，请检查后端日志"
            else:
                return None, f"❌ 生成失败: {result.get('error', '未知错误')}"
        else:
            return None, f"❌ API 请求失败: {response.status_code}"
    
    except requests.RequestException as e:
        return None, f"❌ 网络错误: {str(e)}"
    except Exception as e:
        return None, f"❌ 错误: {str(e)}"


def chat_with_llm(message: str, character: str, history: list):
    """与 LLM 聊天"""
    try:
        if not message.strip():
            return history, "", None
        
        # 检查后端连接
        if not check_backend_connection():
            error_msg = (
                f"❌ 无法连接到后端服务\n\n"
                f"**目标地址**: {API_BASE_URL}\n\n"
                f"请启动后端服务或检查配置"
            )
            history.append((message, error_msg))
            return history, "", None
        
        api_url = f"{API_BASE_URL}/chat"
        print(f"[前端] 发送聊天消息")
        print(f"[前端]   目标地址: {api_url}")
        print(f"[前端]   参数: message={message[:50]}..., character={character}")
        
        response = requests.post(
            api_url,
            json={
                "user_input": message,
                "character": character
            },
            timeout=60
        )
        
        print(f"[前端]   响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result.get("data", {})
                llm_answer = data.get("llm_answer", "")
                audio_base64 = data.get("audio_base64")
                
                # 更新历史记录
                history.append((message, llm_answer))
                
                # 处理音频（如果有）
                audio_path = None
                if audio_base64:
                    # 保存音频到临时文件
                    import base64
                    import tempfile
                    try:
                        audio_data = base64.b64decode(audio_base64)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                            f.write(audio_data)
                            audio_path = f.name
                    except Exception as e:
                        print(f"音频处理失败: {e}")
                
                return history, "", audio_path
            else:
                error = result.get("error", "未知错误")
                history.append((message, f"❌ 错误: {error}"))
                return history, "", None
        else:
            history.append((message, f"❌ API 请求失败: {response.status_code}"))
            return history, "", None
    
    except requests.RequestException as e:
        history.append((message, f"❌ 网络错误: {str(e)}"))
        return history, "", None
    except Exception as e:
        history.append((message, f"❌ 错误: {str(e)}"))
        return history, "", None


def create_video_generation_tab():
    """创建视频生成标签页"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎬 视频生成设置")
            
            text_input = gr.Textbox(
                label="输入文本",
                placeholder="输入要生成的文本...",
                lines=5,
                value="",
                info="输入你想要生成的文本内容"
            )
            
            character_dropdown = gr.Dropdown(
                label="角色",
                choices=get_character_list(),
                value="ayanami" if "ayanami" in get_character_list() else (get_character_list()[0] if get_character_list() else None),
                info="选择角色形象"
            )
            
            model_dropdown = gr.Dropdown(
                label="模型",
                choices=get_models(),
                value=get_models()[0] if get_models() and get_models()[0] != "未找到模型" else None,
                info="选择生成模型"
            )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 生成视频", variant="primary", scale=2)
                refresh_models_btn = gr.Button("🔄 刷新模型", variant="secondary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📹 生成结果")
            
            video_output = gr.Video(
                label="生成的视频",
                height=400
            )
            
            status_output = gr.Textbox(
                label="状态",
                value="等待生成...",
                interactive=False,
                lines=3
            )
    
    def refresh_models():
        """刷新模型列表"""
        models = get_models()
        return gr.Dropdown.update(choices=models, value=models[0] if models and models[0] != "未找到模型" else None)
    
    generate_btn.click(
        fn=generate_video_async,
        inputs=[text_input, character_dropdown, model_dropdown],
        outputs=[video_output, status_output]
    )
    
    refresh_models_btn.click(
        fn=refresh_models,
        outputs=[model_dropdown]
    )
    
    return text_input, character_dropdown, model_dropdown, generate_btn, video_output, status_output


def create_chat_tab():
    """创建聊天标签页"""
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 💬 人机对话")
            
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500,
                show_label=False
            )
            
            msg_input = gr.Textbox(
                label="输入消息",
                placeholder="输入你的消息，按 Enter 发送...",
                lines=2,
                show_label=False
            )
            
            with gr.Row():
                character_dropdown = gr.Dropdown(
                    label="角色",
                    choices=get_character_list(),
                    value="ayanami" if "ayanami" in get_character_list() else (get_character_list()[0] if get_character_list() else None),
                    scale=2
                )
                send_btn = gr.Button("📤 发送", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ 清空", variant="secondary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 语音回复")
            audio_output = gr.Audio(
                label="语音回复",
                type="filepath",
                autoplay=True
            )
            gr.Markdown("""
            **使用说明：**
            - 输入消息后按 Enter 或点击发送按钮
            - AI 会生成文本回复和语音
            - 语音会自动播放
            """)
    
    def send_message(message, character, history):
        if not message.strip():
            return history, "", None
        
        history, new_msg, audio = chat_with_llm(message, character, history)
        return history, new_msg, audio
    
    def clear_chat():
        return [], "", None
    
    send_btn.click(
        fn=send_message,
        inputs=[msg_input, character_dropdown, chatbot],
        outputs=[chatbot, msg_input, audio_output]
    )
    
    msg_input.submit(
        fn=send_message,
        inputs=[msg_input, character_dropdown, chatbot],
        outputs=[chatbot, msg_input, audio_output]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_input, audio_output]
    )
    
    return chatbot, character_dropdown, msg_input, send_btn, clear_btn, audio_output


def create_main_app():
    """创建主应用"""
    with gr.Blocks(title="NeRFFaceSpeech") as app:
        # 显示后端连接状态
        backend_status = gr.Markdown(
            f"""
            <div style="padding: 10px; margin-bottom: 10px; border-radius: 8px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3);">
            <strong>后端服务地址：</strong> {API_BASE_URL}<br>
            <strong>连接状态：</strong> <span id="backend-status">检查中...</span>
            </div>
            <script>
            // 检查后端连接
            fetch('{API_BASE_URL}/docs', {{ method: 'GET', mode: 'no-cors' }})
                .then(() => {{
                    document.getElementById('backend-status').textContent = '✅ 已连接';
                    document.getElementById('backend-status').style.color = '#10b981';
                }})
                .catch(() => {{
                    document.getElementById('backend-status').textContent = '❌ 未连接';
                    document.getElementById('backend-status').style.color = '#ef4444';
                }});
            </script>
            """,
            visible=True
        )
        
        gr.Markdown(
            """
            # 🎭 NeRFFaceSpeech
            ### AI 驱动的语音视频生成系统
            
            ---
            """
        )
        
        with gr.Tabs():
            with gr.Tab("🎬 视频生成"):
                create_video_generation_tab()
            
            with gr.Tab("💬 人机对话"):
                create_chat_tab()
            
            with gr.Tab("⚙️ 设置"):
                gr.Markdown("### 系统设置")
                settings_display = gr.JSON(label="当前设置", value=get_settings())
                
                gr.Markdown("### 后端配置")
                api_url_display = gr.Textbox(
                    label="后端 API 地址",
                    value=API_BASE_URL,
                    interactive=False,
                    info="可通过环境变量 API_BASE_URL 修改"
                )
                
                gr.Markdown("""
                **设置说明：**
                - 设置功能正在开发中
                - 当前显示的是系统默认设置
                
                **后端连接配置：**
                - 默认地址：`http://localhost:8000`
                - 通过环境变量修改：`export API_BASE_URL=http://your-server:8000`
                - 如果使用端口转发，确保转发配置正确
                """)
    
    return app


if __name__ == "__main__":
    # 打印配置信息
    print("=" * 60)
    print("Gradio 前端应用启动")
    print("=" * 60)
    print(f"后端 API 地址: {API_BASE_URL}")
    print(f"前端服务地址: http://0.0.0.0:7860")
    print("=" * 60)
    print("\n提示：")
    print("1. 如果后端连接失败，请检查后端服务是否已启动")
    print("2. 可以通过环境变量设置 API 地址：export API_BASE_URL=http://your-server:8000")
    print("3. 如果使用端口转发，确保 SSH 转发配置正确")
    print("4. 所有后端 API 请求都会在控制台打印目标地址")
    print("=" * 60)
    print()
    
    # 测试后端连接
    print("[前端] 正在检查后端连接...")
    if check_backend_connection():
        print("[前端] ✅ 后端连接正常，可以开始使用")
    else:
        print("[前端] ⚠️  后端连接失败，请检查后端服务")
    print()
    
    app = create_main_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate"
        )
    )

