# Gradio 前端应用

从 `fastapi_server/webui/` 迁移而来，使用 Gradio 实现前端界面，尽量还原原有样式和功能。

## 目录结构

```
gradio_app/
├── frontend/          # Gradio 前端应用
│   ├── __init__.py
│   ├── app.py         # 主应用文件
│   └── start_gradio.py  # 启动脚本
├── backend/           # FastAPI 后端服务
│   ├── __init__.py
│   └── main.py        # FastAPI 主服务
├── shared/            # 共享模块
│   ├── config.py      # 配置模块
│   ├── database/      # 数据库模块
│   └── utils/         # 工具模块
└── test/              # 测试程序
    ├── test_config.py
    ├── test_database.py
    ├── test_utils.py
    ├── test_backend.py
    └── test_frontend.py
```

## 功能特性

### 1. 视频生成
- 文本输入
- 角色选择
- 模型选择
- 异步生成视频
- 实时状态显示

### 2. 人机对话
- 聊天界面
- 角色选择
- 语音回复
- 对话历史

### 3. 系统设置
- 设置查看
- 配置管理（开发中）

## 样式还原

Gradio 前端应用使用自定义 CSS 尽量还原原有样式：

- **深色主题**：使用原有的 tech 主题配色
- **渐变背景**：还原原有的径向渐变效果
- **卡片样式**：半透明卡片，圆角边框
- **按钮样式**：渐变按钮，圆角设计
- **输入框样式**：半透明背景，聚焦效果

## 安装依赖

```bash
# 安装 Gradio
pip install gradio

# 安装其他依赖
pip install requests
```

## 启动方式

### 方式 1：一键启动（推荐）
同时启动后端和前端服务：
```bash
cd gradio_app
python start_all.py
```

启动时会：
- ✅ 检查环境变量（使用系统默认值，不强制设置）
- ✅ 初始化数据库
- ✅ 启动后端服务（FastAPI）
- ✅ 启动前端服务（Gradio）

### 方式 2：分别启动

**启动后端服务**：
```bash
cd gradio_app
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**启动前端服务**（新终端）：
```bash
cd gradio_app
python frontend/start_gradio.py
```

### 方式 3：直接运行前端
```bash
cd gradio_app
python frontend/app.py
```

> **注意**：系统使用默认的环境变量配置（如 `TORCH_HOME`、`HF_ENDPOINT`、`HF_HOME` 等）。如果需要自定义，请在启动前设置相应的环境变量。

### 方式 3：在 conda 环境中运行
```bash
# 激活 API conda 环境
conda activate api  # 或使用对应的 conda 环境

# 运行应用
cd gradio_app
python frontend/app.py
```

## 配置

### API 基础 URL

默认后端 API 地址为 `http://localhost:8000`，可以通过以下方式修改：

#### 方式 1：环境变量（推荐）

```bash
export API_BASE_URL=http://localhost:8000
python frontend/app.py
```

#### 方式 2：修改代码

在 `frontend/app.py` 中修改：

```python
API_BASE_URL = "http://your-server:8000"
```

#### 方式 3：端口转发场景

如果通过 SSH 端口转发访问：

```bash
# 在本地终端执行端口转发
ssh -L 8000:localhost:8000 user@server

# 然后在服务器上运行前端（使用 localhost）
export API_BASE_URL=http://localhost:8000
python frontend/app.py
```

**注意**：如果后端和前端在同一服务器上，使用 `http://localhost:8000` 即可。

### 端口配置

默认端口为 `7860`，可在启动时修改：

```python
app.launch(
    server_name="0.0.0.0",
    server_port=7860,  # 修改端口
    share=False
)
```

## 使用说明

### 基本使用

1. **启动后端服务**（如果使用 FastAPI 后端）：
   ```bash
   cd gradio_app
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. **启动 Gradio 前端**：
   ```bash
   cd gradio_app
   python frontend/app.py
   ```

3. **访问应用**：
   - 打开浏览器访问 `http://localhost:7860`
   - 或使用服务器 IP 地址访问

### 端口转发场景

如果通过 SSH 端口转发远程访问：

1. **在本地终端设置端口转发**：
   ```bash
   # 转发前端端口（7860）和后端端口（8000）
   ssh -L 7860:localhost:7860 -L 8000:localhost:8000 user@server
   ```

2. **在服务器上启动服务**：
   ```bash
   # 终端 1：启动后端
   cd gradio_app
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   
   # 终端 2：启动前端
   cd gradio_app
   export API_BASE_URL=http://localhost:8000  # 确保使用 localhost
   python frontend/app.py
   ```

3. **在本地浏览器访问**：
   - 前端：`http://localhost:7860`
   - 后端 API：`http://localhost:8000`

### 故障排除

**问题：无法连接到后端服务**

1. 检查后端服务是否运行：
   ```bash
   curl http://localhost:8000/docs
   ```

2. 检查 API_BASE_URL 配置：
   ```bash
   echo $API_BASE_URL
   ```

3. 如果使用端口转发，确保：
   - SSH 转发配置正确
   - 后端服务监听 `0.0.0.0:8000`（不是 `127.0.0.1:8000`）
   - 前端使用 `http://localhost:8000`（不是服务器 IP）

4. 查看前端启动日志，确认 API 地址配置

## 测试

运行测试程序：

```bash
cd gradio_app
python test/test_frontend.py
```

## 注意事项

1. **Gradio 依赖**：需要安装 `gradio` 包
2. **后端服务**：如果使用 FastAPI 后端，需要先启动后端服务
3. **数据库**：确保数据库已初始化
4. **模型文件**：确保模型文件存在于配置的模型目录中

## 样式自定义

可以在 `frontend/app.py` 中的 `CUSTOM_CSS` 变量中修改样式，支持：

- CSS 变量（颜色、间距等）
- 组件样式覆盖
- 响应式设计

## 开发计划

- [x] 视频生成功能
- [x] 人机对话功能
- [x] 基础样式还原
- [ ] 训练功能页面
- [ ] 设置管理页面
- [ ] 历史记录查看
- [ ] 主题切换功能
