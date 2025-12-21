# 后端逻辑迁移计划

## 📋 迁移概览

将 `fastapi_server/` 中的后端逻辑迁移到 `gradio_app/` 中，采用**混合架构**方案：
- 保留FastAPI作为后端API服务（用于文件服务、复杂任务等）
- Gradio作为前端界面，通过HTTP调用API
- 共享核心业务逻辑模块

## 🎯 迁移策略

### 方案：混合架构（推荐）

**架构设计**:
```
gradio_app/
├── backend/              # 后端服务（FastAPI）
│   ├── api/             # API端点
│   ├── services/        # 业务逻辑服务
│   └── server.py        # FastAPI服务器
├── frontend/            # Gradio前端
│   ├── components/      # Gradio组件
│   └── app.py           # Gradio主应用
├── shared/              # 共享模块
│   ├── config.py        # 配置（从fastapi_server迁移）
│   ├── database/        # 数据库模块（从fastapi_server迁移）
│   └── utils/           # 工具模块（从fastapi_server迁移）
└── main.py              # 统一启动入口
```

## 📦 迁移文件清单

### 阶段1: 共享模块迁移（核心业务逻辑）

#### 1.1 配置模块
**源文件**: `fastapi_server/config.py`  
**目标位置**: `gradio_app/shared/config.py`  
**操作**: 直接复制，无需修改

**包含内容**:
- 路径配置（PROJECT_ROOT, NERF_CODE_DIR等）
- 目录配置（OUTPUT_VIDEO_DIR, MODEL_DIR等）
- Conda环境配置
- 脚本路径配置
- 资源路径配置
- 辅助函数

#### 1.2 数据库模块
**源目录**: `fastapi_server/database/`  
**目标位置**: `gradio_app/shared/database/`  
**操作**: 复制整个目录

**文件列表**:
- `__init__.py`
- `settings_db.py` - 设置数据库
- `video_records_db.py` - 视频记录数据库
- `chat_db.py` - 聊天数据库

**修改点**:
- 更新import路径（从`config`改为`shared.config`）

#### 1.3 工具模块
**源目录**: `fastapi_server/utils/`  
**目标位置**: `gradio_app/shared/utils/`  
**操作**: 复制整个目录

**文件列表**:
- `run_llm_talk.py` - LLM+TTS音频生成
- `run_nerffacespeech.py` - NeRF视频生成
- `run_chat.py` - 聊天功能
- `run_training.py` - 训练功能
- `llm_talk_api_bridge.py` - LLM API桥接
- `llm_talk_with_text_bridge.py` - LLM文本桥接

**修改点**:
- 更新import路径
- 确保所有依赖正确

### 阶段2: 后端服务迁移（FastAPI）

#### 2.1 API服务模块
**源文件**: `fastapi_server/main.py`  
**目标位置**: `gradio_app/backend/server.py`  
**操作**: 重构为模块化结构

**需要提取的功能**:
1. **日志系统**
   - BufferLogHandler类
   - setup_logging()函数
   - LOG_BUFFER, DEBUG_LOG_BUFFER

2. **任务管理系统**
   - TASKS字典
   - TASKS_LOCK锁
   - run_video_generation_task()函数

3. **API端点**（33个）
   - 静态文件服务（videos, audios, texts）
   - 设置管理API
   - 数据库管理API
   - 模型管理API
   - 视频生成API
   - 聊天对话API
   - 训练相关API
   - 日志查看API
   - 生成记录API

**重构建议**:
- 将API端点按功能分组到不同模块
- 提取业务逻辑到services层
- 保持API接口不变（确保兼容性）

#### 2.2 后端服务结构
```
gradio_app/backend/
├── __init__.py
├── server.py              # FastAPI主应用
├── api/
│   ├── __init__.py
│   ├── settings.py        # 设置API
│   ├── models.py          # 模型API
│   ├── video.py           # 视频生成API
│   ├── chat.py            # 聊天API
│   ├── training.py         # 训练API
│   ├── records.py         # 记录API
│   └── files.py           # 文件服务API
├── services/
│   ├── __init__.py
│   ├── logging_service.py # 日志服务
│   ├── task_service.py    # 任务管理服务
│   └── file_service.py    # 文件服务
└── middleware/
    └── cors.py            # CORS中间件
```

### 阶段3: Gradio前端（新建）

#### 3.1 Gradio组件
**目标位置**: `gradio_app/frontend/components/`

**组件列表**:
- `home.py` - 首页组件
- `generate.py` - 视频生成组件
- `chat.py` - 聊天组件
- `train.py` - 训练组件
- `settings.py` - 设置组件

#### 3.2 API客户端
**目标位置**: `gradio_app/frontend/api_client.py`

**功能**:
- 封装所有后端API调用
- 统一的错误处理
- 请求重试机制

## 🔄 详细迁移步骤

### 步骤1: 创建目录结构

```bash
cd gradio_app
mkdir -p shared/{database,utils}
mkdir -p backend/{api,services,middleware}
mkdir -p frontend/components
mkdir -p frontend/styles
```

### 步骤2: 迁移配置模块

```bash
# 复制配置文件
cp ../fastapi_server/config.py shared/config.py

# 修改import路径（如果需要）
# 检查所有路径引用是否正确
```

### 步骤3: 迁移数据库模块

```bash
# 复制数据库模块
cp -r ../fastapi_server/database/* shared/database/

# 修改import路径
# 将 from config import ... 改为 from shared.config import ...
```

### 步骤4: 迁移工具模块

```bash
# 复制工具模块
cp -r ../fastapi_server/utils/* shared/utils/

# 修改import路径
# 更新所有config和database的引用
```

### 步骤5: 重构FastAPI服务

**5.1 提取日志系统**
- 创建 `backend/services/logging_service.py`
- 迁移BufferLogHandler和setup_logging

**5.2 提取任务管理**
- 创建 `backend/services/task_service.py`
- 迁移TASKS管理和run_video_generation_task

**5.3 重构API端点**
- 按功能分组到不同模块
- 保持接口不变

**5.4 创建主服务器**
- 创建 `backend/server.py`
- 整合所有API模块

### 步骤6: 创建Gradio前端

**6.1 API客户端**
- 创建 `frontend/api_client.py`
- 封装所有API调用

**6.2 Gradio组件**
- 逐个实现各个功能组件
- 使用API客户端调用后端

**6.3 主应用**
- 创建 `frontend/app.py`
- 整合所有组件

### 步骤7: 统一启动入口

**创建 `main.py`**:
```python
"""
统一启动入口
同时启动FastAPI后端和Gradio前端
"""
import subprocess
import sys
from pathlib import Path

def start_backend():
    """启动FastAPI后端"""
    # 启动backend/server.py
    pass

def start_frontend():
    """启动Gradio前端"""
    # 启动frontend/app.py
    pass

if __name__ == "__main__":
    # 启动两个服务
    pass
```

## 📝 文件映射表

| 源文件/目录 | 目标位置 | 操作 | 说明 |
|------------|---------|------|------|
| `config.py` | `shared/config.py` | 复制 | 直接复制，无需修改 |
| `database/` | `shared/database/` | 复制+修改import | 更新import路径 |
| `utils/` | `shared/utils/` | 复制+修改import | 更新import路径 |
| `main.py` | `backend/server.py` | 重构 | 模块化重构 |
| `main.py` (日志系统) | `backend/services/logging_service.py` | 提取 | 提取日志相关代码 |
| `main.py` (任务管理) | `backend/services/task_service.py` | 提取 | 提取任务管理代码 |
| - | `frontend/api_client.py` | 新建 | API客户端封装 |
| - | `frontend/components/` | 新建 | Gradio组件 |
| - | `frontend/app.py` | 新建 | Gradio主应用 |
| - | `main.py` | 新建 | 统一启动入口 |

## ⚠️ 注意事项

### 1. Import路径修改
所有迁移的文件需要更新import路径：
- `from config import ...` → `from shared.config import ...`
- `from database.xxx import ...` → `from shared.database.xxx import ...`
- `from utils.xxx import ...` → `from shared.utils.xxx import ...`

### 2. 路径引用
确保所有路径引用正确：
- 相对路径需要根据新位置调整
- 绝对路径保持不变

### 3. 依赖关系
检查所有依赖：
- Python包依赖
- 文件系统依赖
- 环境变量依赖

### 4. 数据库文件
数据库文件位置不变（在项目根目录的`database/`下），无需迁移。

### 5. API兼容性
保持所有API端点不变，确保：
- URL路径不变
- 请求/响应格式不变
- 功能行为不变

## 🧪 测试清单

迁移完成后需要测试：

### 功能测试
- [ ] 配置模块加载正常
- [ ] 数据库模块功能正常
- [ ] 工具模块功能正常
- [ ] 所有API端点正常响应
- [ ] 文件服务正常
- [ ] 日志系统正常
- [ ] 任务管理正常

### 集成测试
- [ ] FastAPI后端启动正常
- [ ] Gradio前端启动正常
- [ ] 前后端通信正常
- [ ] 视频生成功能正常
- [ ] 聊天功能正常
- [ ] 训练功能正常

### 兼容性测试
- [ ] 与现有数据库兼容
- [ ] 与现有文件兼容
- [ ] API接口兼容

## 📅 时间估算

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| 阶段1 | 共享模块迁移 | 1-2天 |
| 阶段2 | 后端服务迁移 | 3-4天 |
| 阶段3 | Gradio前端 | 5-7天 |
| 测试 | 功能测试和修复 | 2-3天 |
| **总计** | | **11-16天** |

## 🚀 开始迁移

建议按以下顺序进行：

1. **第一步**: 创建目录结构
2. **第二步**: 迁移配置模块（最简单，验证路径）
3. **第三步**: 迁移数据库模块（验证import）
4. **第四步**: 迁移工具模块（验证功能）
5. **第五步**: 重构FastAPI服务（最复杂）
6. **第六步**: 创建Gradio前端
7. **第七步**: 测试和修复

每一步完成后进行测试，确保功能正常后再进行下一步。

