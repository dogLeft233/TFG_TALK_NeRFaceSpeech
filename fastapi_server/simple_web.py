import gradio as gr
import requests
import subprocess
import os
import tempfile
from pathlib import Path

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_VIDEO_DIR

FASTAPI_URL = "http://localhost:8000/generate_video"
MODELS_URL = "http://localhost:8000/models"

# -----------------------------
# 加载模型列表
# -----------------------------
def load_models():
    try:
        return requests.get(MODELS_URL).json()
    except:
        return []

# -----------------------------
# 视频转码
# -----------------------------
def ensure_h264(src_path):
    if not os.path.exists(src_path):
        return None

    tmp = tempfile.NamedTemporaryFile(
        suffix="_h264.mp4", delete=False
    )
    output_path = tmp.name
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, check=True)

    return output_path

# -----------------------------
# 主流程
# -----------------------------
def run_pipeline(text, character, model_name):
    if not text.strip():
        return "请输入文本", None

    payload = {
        "text": text,
        "character": character,
        "model_name": model_name
    }

    resp = requests.post(FASTAPI_URL, json=payload).json()

    if not resp.get("success"):
        return "❌ 后端生成失败：" + str(resp), None

    video_local_path = resp["video_url"]
    converted = ensure_h264(video_local_path)

    return "✅ 视频生成成功！", converted

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎬 NeRFFaceSpeech 一键生成视频")

    with gr.Row():
        text_input = gr.Textbox(
            label="输入要说的话",
            placeholder="请输入文本内容"
        )
        character_input = gr.Dropdown(
            label="选择角色",
            choices=["ayanami", "Aerith"],
            value="ayanami"
        )
        model_input = gr.Dropdown(
            label="选择模型（pkl）",
            choices=load_models()
        )

    run_btn = gr.Button("开始生成视频")

    status = gr.Textbox(label="状态显示")
    video_output = gr.Video(label="生成视频预览")

    run_btn.click(
        run_pipeline,
        inputs=[text_input, character_input, model_input],
        outputs=[status, video_output]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    allowed_paths=[
        str(OUTPUT_VIDEO_DIR)
    ]
)
