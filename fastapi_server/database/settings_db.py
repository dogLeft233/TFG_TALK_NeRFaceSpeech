"""
设置数据库管理模块
使用SQLite存储所有页面的共享设置
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any

# 数据库文件路径 - 存储在 fastapi_server 平级的 database 目录下
# 使用当前文件的父目录（database）的父目录（fastapi_server）的父目录（TFG_TALK_NeRFaceSpeech）作为基础路径
FASTAPI_SERVER_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = FASTAPI_SERVER_DIR.parent.resolve()
DB_DIR = PROJECT_ROOT / "database"
DB_FILE = DB_DIR / "settings.db"

# 确保数据库目录存在
try:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[数据库] 数据库目录: {DB_DIR}")
    print(f"[数据库] 数据库文件: {DB_FILE}")
except Exception as e:
    print(f"[数据库] 警告: 无法创建数据库目录 {DB_DIR}: {e}")

# 默认设置
DEFAULT_SETTINGS = {
    "nerf_theme": "tech",
    "nerf_font": "Inter",
    "nerf_font_size": "medium",
    "nerf_custom_font_size": "14"
}


def init_database():
    """初始化数据库，创建表结构"""
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # 创建设置表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # 检查是否有默认设置，如果没有则插入
        cursor.execute("SELECT COUNT(*) FROM settings")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # 插入默认设置
            for key, value in DEFAULT_SETTINGS.items():
                cursor.execute("""
                    INSERT INTO settings (key, value)
                    VALUES (?, ?)
                """, (key, value))
        
        conn.commit()
        conn.close()
        print(f"[数据库] 数据库初始化成功: {db_path}")
    except Exception as e:
        print(f"[数据库] 错误: 数据库初始化失败: {e}")
        print(f"[数据库] 数据库路径: {DB_FILE}")
        raise


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """获取设置值"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return result[0]
    return default if default is not None else DEFAULT_SETTINGS.get(key)


def set_setting(key: str, value: str):
    """设置值"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = ?
    """, (key, value, value))
    
    conn.commit()
    conn.close()


def get_all_settings() -> Dict[str, str]:
    """获取所有设置"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT key, value FROM settings")
    results = cursor.fetchall()
    
    conn.close()
    
    settings = {key: value for key, value in results}
    
    # 确保所有默认设置都存在
    for key, default_value in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings[key] = default_value
    
    return settings


def reset_to_defaults():
    """重置为默认设置"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    for key, value in DEFAULT_SETTINGS.items():
        cursor.execute("""
            INSERT INTO settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?
        """, (key, value, value))
    
    conn.commit()
    conn.close()


# 初始化数据库（如果失败，打印错误但不阻止导入）
try:
    init_database()
except Exception as e:
    print(f"[数据库] 警告: 数据库初始化失败，但将继续运行: {e}")
    import traceback
    traceback.print_exc()

