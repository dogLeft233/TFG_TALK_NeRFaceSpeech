from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Body
import uuid
import os
from pathlib import Path

from utils.run_llm_talk import generate_audio
from utils.run_nerffacespeech import generate_video
from config import OUTPUT_VIDEO_DIR, OUTPUT_AUDIO_DIR, MODEL_DIR

app = FastAPI()

app.mount(
    "/videos",
    StaticFiles(directory=str(OUTPUT_VIDEO_DIR)),
    name="videos"
)

# ---------------------------
# 返回所有 pkl 模型
# ---------------------------
@app.get("/models")
def list_models():
    return [
        f for f in os.listdir(str(MODEL_DIR))
        if f.endswith(".pkl")
    ]


# ---------------------------
# 生成视频接口
# ---------------------------
@app.post("/generate_video")
def generate_video_api(
    text: str = Body(..., embed=True),
    character: str = Body(..., embed=True),
    model_name: str = Body(..., embed=True),
):
    unique_id = str(uuid.uuid4())

    audio_output = OUTPUT_AUDIO_DIR / f"{unique_id}.wav"
    video_dir = OUTPUT_VIDEO_DIR / unique_id
    video_output = video_dir / "output_NeRFFaceSpeech.mp4"

    video_dir.mkdir(parents=True, exist_ok=True)

    ok1 = generate_audio(
        text=text,
        output_path=audio_output,
        character=character
    )
    if not ok1:
        return {"success": False, "error": "LLM语音生成失败"}

    ok2 = generate_video(
        audio_path=audio_output,
        character=character,
        output_path=video_dir,
        model_name=model_name
    )
    if not ok2:
        return {"success": False, "error": "NeRF 视频生成失败"}

    return {
        "success": True,
        "video_url": str(video_output)
    }
