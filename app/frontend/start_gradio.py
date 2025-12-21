#!/usr/bin/env python3
"""
启动 Gradio 前端应用
"""
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 不强制设置环境变量，使用系统默认值

# 设置 API 基础 URL（如果未设置）
os.environ.setdefault('API_BASE_URL', 'http://localhost:8000')

from frontend.app import create_main_app

import gradio as gr
from frontend.app import CUSTOM_CSS

if __name__ == "__main__":
    app = create_main_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate"
        )
    )

