"""
视频生成记录数据库管理模块
使用SQLite存储视频生成记录
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# 数据库文件路径 - 使用和settings_db相同的目录
FASTAPI_SERVER_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = FASTAPI_SERVER_DIR.parent.resolve()
DB_DIR = PROJECT_ROOT / "database"
DB_FILE = DB_DIR / "video_records.db"

# 视频文件存储目录（与config.py中的VIDEOS_STORAGE_DIR保持一致）
VIDEOS_DIR = DB_DIR / "videos"

# 确保数据库目录和视频目录存在
DB_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def init_database():
    """初始化数据库，创建表结构"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # 创建视频生成记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            text TEXT NOT NULL,
            character TEXT NOT NULL,
            model_name TEXT NOT NULL,
            video_path TEXT NOT NULL,
            generation_time REAL,
            config_json TEXT,
            status TEXT DEFAULT 'completed'
        )
    """)
    
    # 创建索引以提高查询性能
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_unique_id ON video_records(unique_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at ON video_records(created_at)
    """)
    
    conn.commit()
    conn.close()


def add_video_record(
    unique_id: str,
    text: str,
    character: str,
    model_name: str,
    video_path: str,
    generation_time: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    status: str = 'completed'
) -> bool:
    """添加视频生成记录"""
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        config_json = json.dumps(config) if config else None
        
        cursor.execute("""
            INSERT INTO video_records 
            (unique_id, created_at, text, character, model_name, video_path, generation_time, config_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (unique_id, created_at, text, character, model_name, video_path, generation_time, config_json, status))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"添加视频记录失败: {e}")
        return False


def get_video_record(unique_id: str) -> Optional[Dict[str, Any]]:
    """获取指定ID的视频记录"""
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM video_records WHERE unique_id = ?", (unique_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            record = dict(row)
            if record.get('config_json'):
                record['config'] = json.loads(record['config_json'])
            return record
        return None
    except Exception as e:
        print(f"获取视频记录失败: {e}")
        return None


def list_video_records(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """列出视频记录（按创建时间倒序）"""
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM video_records 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            record = dict(row)
            if record.get('config_json'):
                record['config'] = json.loads(record['config_json'])
            records.append(record)
        
        return records
    except Exception as e:
        print(f"列出视频记录失败: {e}")
        return []


def delete_video_record(unique_id: str) -> bool:
    """删除视频记录"""
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM video_records WHERE unique_id = ?", (unique_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"删除视频记录失败: {e}")
        return False


# 初始化数据库
init_database()

