from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import os
from pathlib import Path
import collections
from datetime import datetime
import logging
import sys
import warnings

from utils.run_llm_talk import generate_audio
from utils.run_nerffacespeech import generate_video
from utils.run_chat import chat_with_llm, get_llm_only
from utils.run_training import start_training, get_training_status, list_training_tasks, stop_training
from config import OUTPUT_VIDEO_DIR, OUTPUT_AUDIO_DIR, MODEL_DIR, WEBUI_DIR, VIDEOS_STORAGE_DIR, DATA_DIR, TRAINING_DATASET_DIR
from database.settings_db import get_setting, set_setting, get_all_settings, DB_DIR
from database.video_records_db import add_video_record
import sqlite3
import shutil
import time

# 日志缓冲系统
LOG_BUFFER_SIZE = 10000  # 最大保存 10000 条日志
LOG_BUFFER = collections.deque(maxlen=LOG_BUFFER_SIZE)
DEBUG_LOG_BUFFER = collections.deque(maxlen=LOG_BUFFER_SIZE)


# 自定义日志处理器，将日志添加到缓冲区
class BufferLogHandler(logging.Handler):
    """将日志输出添加到缓冲区的处理器"""
    
    def emit(self, record):
        try:
            # 确定日志级别
            level_map = {
                logging.DEBUG: "debug",
                logging.INFO: "info",
                logging.WARNING: "warning",
                logging.ERROR: "error",
                logging.CRITICAL: "error",
            }
            level = level_map.get(record.levelno, "info")
            
            # 格式化日志消息，保持完整格式
            message = record.getMessage()
            
            # 对于所有日志，保持完整的原始格式，不简化
            # 这样可以确保所有输出都能完整显示
            # 如果是Uvicorn的日志，保持原始格式
            if record.name.startswith("uvicorn"):
                message = record.getMessage()
            # 对于其他日志，也保持原始格式，但可以添加模块名前缀
            elif record.name and record.name != "root":
                # 只在模块名不是root时添加，避免重复
                message = record.getMessage()
            
            # 添加到日志缓冲区
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message
            }
            
            if level == "debug":
                DEBUG_LOG_BUFFER.append(log_entry)
            else:
                LOG_BUFFER.append(log_entry)
        except Exception:
            # 防止日志处理出错导致程序崩溃
            pass


# 配置日志系统
def setup_logging():
    """配置日志系统，捕获所有日志输出"""
    # 创建自定义处理器（只添加到缓冲区，不输出到控制台）
    buffer_handler = BufferLogHandler()
    buffer_handler.setLevel(logging.DEBUG)
    
    # 设置简洁的日志格式（匹配Uvicorn的格式）
    # 注意：实际上我们不在这里使用formatter，而是在emit中直接处理消息
    formatter = logging.Formatter('%(message)s')
    buffer_handler.setFormatter(formatter)
    
    # 配置Uvicorn的日志记录器（添加缓冲区处理器，保留原有的处理器）
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    # 只添加缓冲区处理器，不替换原有的
    if not any(isinstance(h, BufferLogHandler) for h in uvicorn_logger.handlers):
        uvicorn_logger.addHandler(buffer_handler)
    
    # 配置Uvicorn的access日志（HTTP请求日志）
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)
    # 只添加缓冲区处理器，不替换原有的
    if not any(isinstance(h, BufferLogHandler) for h in access_logger.handlers):
        access_logger.addHandler(buffer_handler)
    
    # 配置FastAPI的日志记录器
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.INFO)
    if not any(isinstance(h, BufferLogHandler) for h in fastapi_logger.handlers):
        fastapi_logger.addHandler(buffer_handler)
    
    # 配置根日志记录器（捕获其他模块的日志，包括 WARNING 级别）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # 设置为 WARNING 以捕获所有警告
    if not any(isinstance(h, BufferLogHandler) for h in root_logger.handlers):
        root_logger.addHandler(buffer_handler)
    
    # 配置所有子模块的日志记录器，确保捕获所有输出
    # 特别是捕获 warnings 模块的输出
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: root_logger.warning(
        f"{filename}:{lineno}: {category.__name__}: {message}"
    )


# 初始化日志系统
setup_logging()

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """应用启动事件，确保日志系统正常工作"""
    # 在启动时确保日志处理器已配置
    setup_logging()
    # 添加启动日志
    add_log("FastAPI应用启动完成", "success")


# 允许前端跨域访问 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/videos",
    StaticFiles(directory=str(VIDEOS_STORAGE_DIR)),
    name="videos"
)

# 挂载 webui 静态文件
app.mount(
    "/webui",
    StaticFiles(directory=str(WEBUI_DIR)),
    name="webui"
)

# ---------------------------
# 根路径 - 后端输出页面（后端显示屏）
# ---------------------------
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    """后端根路径 - 显示后端输出页面（后端显示屏）"""
    logs_html_path = WEBUI_DIR / "logs.html"
    if logs_html_path.exists():
        with open(logs_html_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 替换settings.js路径为/webui/settings.js，确保从根路径访问时能找到文件
            # 替换settings.js路径为/webui/settings.js
            content = content.replace('src="settings.js"', 'src="/webui/settings.js"')
            return HTMLResponse(content=content)
    
    # 如果 logs.html 不存在，返回简单的 API 信息页面
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>TFG_TALK_NeRFaceSpeech API - 后端显示屏</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 40px; background: #0b1020; color: #e5e7eb; }
                .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.06); padding: 30px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.14); }
                h1 { color: #e5e7eb; }
                .endpoint { margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.03); border-left: 4px solid #3b82f6; }
                a { color: #60a5fa; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .link-box { margin-top: 20px; padding: 15px; background: rgba(59,130,246,0.2); border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>TFG_TALK_NeRFaceSpeech API - 后端显示屏</h1>
                <p>这是后端服务器的主页。请访问 <a href="/webui/logs.html">/webui/logs.html</a> 查看后端输出。</p>
                <p>API 文档: <a href="/docs">/docs</a></p>
                <div class="link-box">
                    <strong>前端应用入口：</strong> 请访问 <a href="http://localhost:7860/" target="_blank">http://localhost:7860/</a> 使用前端功能（需要先运行 python simple_web.py）。
                </div>
                <h2>可用端点:</h2>
                <div class="endpoint"><strong>GET</strong> /models - 获取模型列表</div>
                <div class="endpoint"><strong>POST</strong> /generate_video - 生成视频</div>
                <div class="endpoint"><strong>POST</strong> /chat - 聊天对话</div>
                <div class="endpoint"><strong>POST</strong> /llm_only - 仅LLM问答</div>
                <div class="endpoint"><strong>POST</strong> /train/start - 启动训练</div>
                <div class="endpoint"><strong>GET</strong> /train/status/{task_id} - 查询训练状态</div>
                <div class="endpoint"><strong>GET</strong> /train/tasks - 列出训练任务</div>
                <div class="endpoint"><strong>POST</strong> /train/stop/{task_id} - 停止训练</div>
                <div class="endpoint"><strong>GET</strong> /logs - 获取日志输出</div>
            </div>
        </body>
        </html>
        """)

# ---------------------------
# 设置管理 API
# ---------------------------

@app.get("/api/settings")
async def get_settings_api():
    """获取所有设置"""
    try:
        settings = get_all_settings()
        return {"success": True, "data": settings}
    except Exception as e:
        add_log(f"获取设置失败: {str(e)}", "error")
        import traceback
        traceback.print_exc()
        # 返回默认设置，确保前端不会因为错误而无法加载
        return {
            "success": True,
            "data": {
                "nerf_theme": "tech",
                "nerf_font": "Inter",
                "nerf_font_size": "medium",
                "nerf_custom_font_size": "14"
            }
        }


@app.get("/api/settings/{key}")
async def get_setting_by_key_api(key: str):
    """获取指定设置"""
    value = get_setting(key)
    return {"success": True, "data": {"key": key, "value": value}}


@app.post("/api/settings/{key}")
async def update_setting_api(key: str, request: dict = Body(...)):
    """更新设置"""
    # 支持两种格式：{"value": "xxx"} 或直接传递字符串
    value = request.get("value") if isinstance(request, dict) else str(request)
    if value is None:
        # 如果没有value字段，尝试直接使用请求体作为值
        value = str(request)
    set_setting(key, str(value))
    return {"success": True, "message": f"Setting {key} updated"}


@app.post("/api/settings")
async def update_settings_api(settings: dict = Body(...)):
    """批量更新设置"""
    for key, value in settings.items():
        set_setting(key, str(value))
    return {"success": True, "message": "Settings updated"}


# ---------------------------
# 数据库管理 API
# ---------------------------

@app.get("/api/databases")
async def list_databases_api():
    """列出database文件夹下的所有数据库文件"""
    try:
        db_files = []
        if DB_DIR.exists():
            for file in DB_DIR.iterdir():
                if file.is_file() and file.suffix.lower() == '.db':
                    db_files.append({
                        "name": file.name,
                        "path": str(file),
                        "size": file.stat().st_size
                    })
        return {"success": True, "data": sorted(db_files, key=lambda x: x["name"])}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/databases/{db_name}/content")
async def get_database_content_api(db_name: str, table: str = None):
    """获取指定数据库的内容
    如果指定table，返回该表的数据；否则返回所有表的结构和数据
    """
    try:
        db_path = DB_DIR / db_name
        if not db_path.exists() or not db_path.suffix.lower() == '.db':
            return {"success": False, "error": f"数据库文件 {db_name} 不存在"}
        
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # 返回字典格式
        cursor = conn.cursor()
        
        result = {}
        
        if table:
            # 获取指定表的数据
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            result[table] = {
                "columns": columns,
                "rows": [dict(row) for row in rows]
            }
        else:
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                result[table_name] = {
                    "columns": columns,
                    "rows": [dict(row) for row in rows]
                }
        
        conn.close()
        
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/databases/{db_name}/update")
async def update_database_content_api(db_name: str, update_data: dict = Body(...)):
    """更新数据库内容
    update_data格式: {
        "table": "table_name",
        "operation": "update|insert|delete",
        "where": {"key": "value"},  # 用于update和delete
        "data": {"key": "value"}  # 用于update和insert
    }
    """
    try:
        db_path = DB_DIR / db_name
        if not db_path.exists() or not db_path.suffix.lower() == '.db':
            return {"success": False, "error": f"数据库文件 {db_name} 不存在"}
        
        table = update_data.get("table")
        operation = update_data.get("operation")
        where_clause = update_data.get("where", {})
        data = update_data.get("data", {})
        
        if not table or not operation:
            return {"success": False, "error": "缺少必需参数: table 和 operation"}
        
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        cursor = conn.cursor()
        
        if operation == "update":
            if not where_clause or not data:
                return {"success": False, "error": "update操作需要where和data参数"}
            set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
            where_conditions = " AND ".join([f"{k} = ?" for k in where_clause.keys()])
            values = list(data.values()) + list(where_clause.values())
            cursor.execute(f"UPDATE {table} SET {set_clause} WHERE {where_conditions}", values)
            
        elif operation == "insert":
            if not data:
                return {"success": False, "error": "insert操作需要data参数"}
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            cursor.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", list(data.values()))
            
        elif operation == "delete":
            if not where_clause:
                return {"success": False, "error": "delete操作需要where参数"}
            where_conditions = " AND ".join([f"{k} = ?" for k in where_clause.keys()])
            cursor.execute(f"DELETE FROM {table} WHERE {where_conditions}", list(where_clause.values()))
        else:
            return {"success": False, "error": f"不支持的操作: {operation}"}
        
        conn.commit()
        conn.close()
        
        return {"success": True, "message": f"{operation}操作成功"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------
# 返回所有 pkl 模型
# ---------------------------
@app.get("/models")
def list_models():
    try:
        # 检查模型目录是否存在
        if not MODEL_DIR.exists():
            error_msg = f"模型目录不存在: {MODEL_DIR}\n请确保目录 /root/autodl-tmp/TFG_TALK_NeRFaceSpeech/NeRFFaceSpeech_Code/pretrained_networks/ 存在"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        
        # 检查是否有 .pkl 文件
        try:
            files = os.listdir(str(MODEL_DIR))
        except PermissionError:
            error_msg = f"无权限访问模型目录: {MODEL_DIR}"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"读取模型目录失败: {str(e)}\n目录路径: {MODEL_DIR}"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        
        # 只查找 .pkl 文件，排除子目录
        models = []
        for f in files:
            file_path = os.path.join(str(MODEL_DIR), f)
            if f.endswith(".pkl") and os.path.isfile(file_path):
                models.append(f)
        
        if not models:
            # 列出目录中的所有文件（最多20个）用于调试
            file_list = ', '.join(files[:20]) if files else '(空)'
            error_msg = f"模型目录中没有找到 .pkl 文件\n目录路径: {MODEL_DIR}\n目录内容: {file_list}\n请确保 ffhq_1024.pkl 等模型文件在该目录中"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        
        add_log(f"找到 {len(models)} 个模型文件: {', '.join(models)}", "info")
        return models
    except Exception as e:
        error_msg = f"获取模型列表失败: {str(e)}\n目录路径: {MODEL_DIR}"
        add_log(error_msg, "error")
        return {"success": False, "error": error_msg}


# ---------------------------
# 生成视频接口（与可运行版本保持一致）
# ---------------------------
@app.post("/generate_video")
def generate_video_api(
    text: str = Body(..., embed=True),
    character: str = Body(..., embed=True),
    model_name: str = Body(..., embed=True),
):
    unique_id = str(uuid.uuid4())
    start_time = time.time()

    audio_output = f"{OUTPUT_AUDIO_DIR}/{unique_id}.wav"
    video_dir = f"{OUTPUT_VIDEO_DIR}/{unique_id}"
    temp_video_output = f"{video_dir}/output_NeRFFaceSpeech.mp4"

    os.makedirs(video_dir, exist_ok=True)

    # 步骤1: 生成音频
    ok1 = generate_audio(
        text=text,
        output_path=audio_output,
        character=character
    )
    if not ok1:
        return {"success": False, "error": "LLM语音生成失败"}

    # 步骤2: 生成视频
    ok2 = generate_video(
        audio_path=audio_output,
        character=character,
        output_path=video_dir,
        model_name=model_name
    )
    if not ok2:
        return {"success": False, "error": "NeRF 视频生成失败"}

    # 步骤3: 移动视频文件到数据库目录下的videos文件夹
    final_video_dir = VIDEOS_STORAGE_DIR / unique_id
    final_video_dir.mkdir(parents=True, exist_ok=True)
    final_video_path = final_video_dir / "output_NeRFFaceSpeech.mp4"
    
    try:
        if os.path.exists(temp_video_output):
            shutil.move(temp_video_output, str(final_video_path))
            # 清理临时目录（如果为空）
            try:
                if os.path.exists(video_dir):
                    os.rmdir(video_dir)
            except OSError:
                pass  # 目录不为空，保留
    except Exception as e:
        add_log(f"移动视频文件失败: {e}", "warning")
        # 如果移动失败，使用原始路径
        final_video_path = Path(temp_video_output)

    # 计算生成时间
    generation_time = time.time() - start_time

    # 保存记录到数据库
    config = {
        "text": text,
        "character": character,
        "model_name": model_name
    }
    
    add_video_record(
        unique_id=unique_id,
        text=text,
        character=character,
        model_name=model_name,
        video_path=str(final_video_path),
        generation_time=generation_time,
        config=config,
        status='completed'
    )

    # 返回视频URL（相对于videos静态文件路径）
    video_url = f"/videos/{unique_id}/output_NeRFFaceSpeech.mp4"

    return {
        "success": True,
        "video_url": video_url,
        "unique_id": unique_id,
        "generation_time": generation_time
    }


# ---------------------------
# 聊天对话接口
# ---------------------------
class ChatRequest(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    character: str = "ayanami"
    enable_audio: bool = True

@app.post("/chat")
def chat_api(request: ChatRequest):
    """
    聊天对话接口
    支持文本输入，返回LLM回答和音频
    
    Args:
        text: 用户输入的文本
        audio_base64: 音频文件的base64编码（暂不支持，需要语音识别API）
        character: 角色名称（ayanami 或 Aerith）
        enable_audio: 是否生成音频回复
    
    Returns:
        dict: 包含LLM回答和音频的响应
    """
    # 目前只支持文本输入
    if not request.text:
        return {
            "success": False,
            "error": "请提供文本输入（音频输入功能待实现）"
        }
    
    # 调用聊天函数
    result = chat_with_llm(
        user_input=request.text,
        character=request.character,
        enable_audio=request.enable_audio
    )
    
    return result


# ---------------------------
# 纯LLM问答接口（不生成音频，快速响应）
# ---------------------------
class LLMOnlyRequest(BaseModel):
    text: str

@app.post("/llm_only")
def llm_only_api(request: LLMOnlyRequest):
    """
    仅获取LLM回答，不生成音频（用于快速响应）
    
    Args:
        request: 包含用户输入文本的请求对象
    
    Returns:
        dict: 包含LLM回答的响应
    """
    result = get_llm_only(request.text)
    return result


# ---------------------------
# 训练相关接口
# ---------------------------
class TrainingRequest(BaseModel):
    data_path: str
    base_model: str = "ffhq_1024.pkl"
    kimg: int = 50
    snap: int = 5
    imgsnap: int = 1
    aug: str = "noaug"
    mirror: bool = False
    config_name: str = "style_ffhq_ae_basic"

@app.post("/train/start")
def start_training_api(request: TrainingRequest):
    """
    启动模型训练
    
    Args:
        request: 训练请求参数对象
    
    Returns:
        dict: 任务ID和状态
    """
    result = start_training(
        data_path=request.data_path,
        base_model=request.base_model,
        kimg=request.kimg,
        snap=request.snap,
        imgsnap=request.imgsnap,
        aug=request.aug,
        mirror=request.mirror,
        model_config=request.config_name
    )
    return result


@app.get("/train/status/{task_id}")
def get_training_status_api(task_id: str):
    """
    获取训练任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 任务状态和日志
    """
    return get_training_status(task_id)


@app.get("/train/tasks")
def list_training_tasks_api():
    """
    列出所有训练任务
    
    Returns:
        dict: 任务列表
    """
    return list_training_tasks()


@app.post("/train/stop/{task_id}")
def stop_training_api(task_id: str):
    """
    停止训练任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 操作结果
    """
    return stop_training(task_id)


@app.get("/train/datasets")
def list_datasets_api():
    """
    列出可用的训练数据集路径
    
    Returns:
        dict: 包含数据集路径列表的响应
    """
    try:
        datasets = []
        
        # 检查默认数据集目录
        if DATA_DIR.exists():
            # 列出data目录下的子目录作为可选数据集
            for item in DATA_DIR.iterdir():
                if item.is_dir():
                    datasets.append({
                        "name": item.name,
                        "path": str(item),
                        "exists": True
                    })
        
        # 添加默认训练数据集路径（如果存在）
        default_dataset = str(TRAINING_DATASET_DIR)
        if TRAINING_DATASET_DIR.exists():
            datasets.insert(0, {
                "name": "默认训练数据集",
                "path": default_dataset,
                "exists": True,
                "default": True
            })
        else:
            datasets.insert(0, {
                "name": "默认训练数据集（不存在）",
                "path": default_dataset,
                "exists": False,
                "default": True
            })
        
        return {
            "success": True,
            "data": datasets,
            "default_path": default_dataset
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"获取数据集列表失败: {str(e)}"
        }


# ---------------------------
# 日志查看接口
# ---------------------------
@app.get("/logs")
def get_logs(limit: int = 500, debug: bool = False):
    """
    获取日志输出
    
    Args:
        limit: 返回的日志条数限制
        debug: 是否只返回 debug 日志
    
    Returns:
        dict: 包含日志列表的响应
    """
    if debug:
        logs = list(DEBUG_LOG_BUFFER)[-limit:]
    else:
        logs = list(LOG_BUFFER)[-limit:]
    
    return {
        "success": True,
        "logs": logs,
        "total": len(DEBUG_LOG_BUFFER) if debug else len(LOG_BUFFER)
    }


@app.get("/logs/full")
def get_full_logs(debug: bool = False):
    """
    获取完整日志输出（无长度限制）
    
    Args:
        debug: 是否只返回 debug 日志
    
    Returns:
        dict: 包含完整日志列表的响应
    """
    if debug:
        logs = list(DEBUG_LOG_BUFFER)
    else:
        logs = list(LOG_BUFFER)
    
    return {
        "success": True,
        "logs": logs,
        "total": len(DEBUG_LOG_BUFFER) if debug else len(LOG_BUFFER)
    }


def add_log(message: str, level: str = "info", is_debug: bool = False):
    """
    添加日志到缓冲区
    
    Args:
        message: 日志消息
        level: 日志级别 (info, warning, error, debug, success)
        is_debug: 是否为 debug 日志
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    
    if is_debug or level == "debug":
        DEBUG_LOG_BUFFER.append(log_entry)
    else:
        LOG_BUFFER.append(log_entry)


# 初始化时添加一条日志
add_log("后端服务已启动", "success")
