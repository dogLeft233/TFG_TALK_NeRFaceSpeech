# Gradio App HTML 前端

这是 `gradio_app` 的 HTML 前端实现，仿照 `fastapi_server/webui` 的风格。

## 文件结构

```
webui/
├── start.html      # 开始界面，包含后端连接检查
├── style.css       # 样式文件
├── settings.js     # 配置文件（简化版）
└── README.md       # 本文件
```

## 启动方式

### 方式1：使用 simple_web.py

```bash
cd gradio_app
python3 simple_web.py
```

访问地址: http://localhost:7860/

### 方式2：使用 Python HTTP 服务器

```bash
cd gradio_app/webui
python3 -m http.server 7860
```

访问地址: http://localhost:7860/start.html

## 功能说明

### start.html

开始界面，包含以下功能：

1. **后端连接检查**
   - 自动检测后端服务状态（默认地址: http://localhost:8000）
   - 每5秒自动检查一次
   - 显示连接状态：✅ 连接成功 / ❌ 连接失败 / ⏱️ 连接超时

2. **快速入口**
   - 后端 API 文档：打开后端 Swagger 文档
   - 前端应用：打开前端应用（待实现）

### settings.js

配置文件，提供以下功能：

- `API_BASE_URL`: API 基础地址（默认: http://localhost:8000）
- `getApiBaseUrl()`: 获取 API 地址
- `setApiBaseUrl(url)`: 设置 API 地址（保存到 localStorage）

### style.css

样式文件，使用深色科技风格主题。

## 配置后端地址

### 方式1：修改 settings.js

```javascript
const API_BASE_URL = 'http://your-server:8000';
```

### 方式2：在浏览器控制台设置

```javascript
localStorage.setItem('API_BASE_URL', 'http://your-server:8000');
location.reload();
```

## 后端要求

确保后端服务已启动：

```bash
cd gradio_app
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

后端应提供以下端点：
- `/docs` - Swagger API 文档（用于连接检查）

## 开发计划

- [x] 创建开始界面
- [x] 实现后端连接检查
- [ ] 创建视频生成页面
- [ ] 创建对话页面
- [ ] 创建训练页面
- [ ] 完善设置功能

