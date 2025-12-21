# NeRFFaceSpeech

> AI 驱动的语音视频生成系统 - 基于 NeRF 技术的实时人脸视频合成

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [系统要求](#系统要求)
- [安装指南](#安装指南)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [开发指南](#开发指南)
- [更新日志](#更新日志)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 🎯 项目简介

NeRFFaceSpeech 是一个基于 NeRF（Neural Radiance Fields）技术的实时人脸视频生成系统。系统能够根据文本输入，通过 LLM 生成回复，使用 TTS 合成语音，最终生成逼真的人脸视频。

### 技术栈

- **后端框架**：FastAPI (Python)
- **前端框架**：HTML5 + CSS3 + JavaScript (原生) / Gradio
- **深度学习**：PyTorch, StyleNeRF
- **语音合成**：TTS (Text-to-Speech)
- **语音识别**：ASR (Automatic Speech Recognition)
- **视频处理**：FFmpeg
- **数据库**：SQLite

---

## ✨ 核心特性

### 🎬 视频生成
- **文本输入** → LLM 生成回复 → TTS 语音合成 → NeRF 视频生成
- 支持多种角色和模型
- 实时进度显示
- 自动视频转码（H.264/AAC，浏览器兼容）
- 历史记录管理

### 💬 人机对话
- 实时文本对话
- 语音回复（自动播放）
- 对话历史保存
- 流式输出（逐字显示）

### 🎭 角色训练
- 视频上传 → 自动提取训练数据 → 生成角色模型
- 支持 FFHQ 对齐
- 自动提取帧和音频
- PTI 模型生成

### 🎓 模型训练
- StyleNeRF 模型微调训练
- 实时训练监控
- 训练任务管理
- 损失曲线可视化

### 🎨 个性化设置
- 主题切换（科技风、温馨风、简约风）
- 字体自定义（多种字体可选）
- 字号调整（小、中、大、自定义）
- 设置持久化（数据库存储）

---

## 🚀 快速开始

### 方式一：一键启动（推荐）

**启动应用**：

```bash
cd app
python start_all.py
```

启动脚本会自动：
1. ✅ 检查环境变量（使用系统默认值）
2. ✅ 初始化数据库
3. ✅ 启动 TTS 服务（端口 8001）
4. ✅ 启动 ASR 服务（端口 8002）
5. ✅ 启动后端服务器（FastAPI，端口 8000）
6. ✅ 启动前端服务器（Gradio，端口 7860）

**访问地址**：
- 前端主页：`http://localhost:7860/`
- 后端 API：`http://localhost:8000/`
- API 文档：`http://localhost:8000/docs`
- TTS 服务：`http://localhost:8001/`（如果启动成功）
- ASR 服务：`http://localhost:8002/`（如果启动成功）

### 方式二：Web UI 版本

**一键启动**：

```bash
cd fastapi_server
python start.py
```

**访问地址**：
- 前端主页：`http://localhost:7860/`
- 后端 API：`http://localhost:8000/`
- API 文档：`http://localhost:8000/docs`

### 方式三：Docker 部署

```bash
# 构建镜像
docker build -t nerffacespeech -f docker/Dockerfile .

# 运行容器
docker run -d -p 8000:8000 -p 7860:7860 nerffacespeech
```

---

## 📦 系统要求

### 硬件要求

- **GPU**：NVIDIA GPU（推荐 8GB+ 显存）
- **内存**：16GB+ RAM
- **存储**：50GB+ 可用空间（用于模型和视频）

### 软件要求

- **操作系统**：Linux / macOS / Windows
- **Python**：3.10 或更高版本
- **CUDA**：11.8+（如果使用 GPU）
- **FFmpeg**：用于视频转码

### 依赖环境

- PyTorch
- FastAPI
- Gradio（可选）
- 其他依赖见 `requirements.txt`

---

## 📥 安装指南

### 1. 克隆项目

```bash
git clone <repository-url>
cd TFG_TALK_NeRFaceSpeech
```

### 2. 创建 Conda 环境（推荐）

```bash
# 创建 API 环境
conda create -n api python=3.10
conda activate api

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装 FFmpeg

**Ubuntu/Debian**：
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS**：
```bash
brew install ffmpeg
```

**Windows**：
下载 [FFmpeg](https://ffmpeg.org/download.html) 并添加到 PATH

### 4. 配置环境变量

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export TORCH_HOME=/path/to/weights
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/Hugging_Face
```

### 5. 下载模型文件

将预训练模型文件放置在 `pretrained_networks/` 目录下。

---

## 📖 使用说明

### 视频生成

1. **打开视频生成页面**
   - 访问 `http://localhost:7860/index.html`
   - 点击"视频生成"卡片

2. **输入文本**
   - 在文本框中输入要生成的内容
   - 例如："你好，请简要介绍一下人工智能"

3. **选择角色和模型**
   - 从下拉菜单选择角色（如 `ayanami`、`Aerith`）
   - 选择对应的模型文件

4. **开始生成**
   - 点击"🚀 开始生成视频"按钮
   - 等待生成完成（通常需要 5-10 分钟）

5. **查看结果**
   - 视频生成完成后会自动播放
   - 可以下载视频或查看历史记录

### 人机对话

1. **打开对话页面**
   - 从主页点击"人机对话"

2. **开始对话**
   - 在输入框中输入问题
   - 按 `Enter` 发送
   - AI 会自动回复并播放语音

3. **管理对话**
   - 新建对话：点击"新建对话"按钮
   - 切换对话：从左侧列表选择
   - 导出记录：点击"导出对话"按钮

### 角色训练

1. **打开角色训练页面**
   - 从主页点击"角色训练"

2. **上传视频**
   - 选择视频文件（支持 MP4、AVI 等格式）
   - 输入角色名称

3. **开始训练**
   - 点击"开始训练"按钮
   - 等待训练完成（可能需要较长时间）

4. **使用新角色**
   - 训练完成后，新角色会自动出现在角色列表中
   - 可以在视频生成页面使用新角色

### 个性化设置

1. **打开设置面板**
   - 点击页面右上角的"⚙️ 设置"按钮

2. **切换主题**
   - 选择"主题"标签
   - 点击主题卡片（科技风、温馨风、简约风）

3. **调整字体**
   - 选择"字体"标签
   - 选择字体样式和字号大小
   - 支持自定义字号

4. **保存设置**
   - 设置会自动保存到数据库
   - 刷新页面后设置仍然有效

---

## 📁 项目结构

```
TFG_TALK_NeRFaceSpeech/
├── fastapi_server/          # Web UI 版本（FastAPI + HTML）
│   ├── start.py             # 一键启动脚本
│   ├── main.py              # FastAPI 后端服务器
│   ├── simple_web.py        # 前端 HTTP 服务器
│   ├── config.py            # 配置文件
│   ├── database/            # 数据库模块
│   ├── utils/               # 工具模块
│   └── webui/               # 前端页面
│       ├── start.html       # 选择页面
│       ├── index.html       # 前端主页
│       ├── generate.html    # 视频生成页面
│       ├── talk.html        # 语音对话页面
│       ├── character_train.html  # 角色训练页面
│       ├── settings.js      # 设置管理脚本
│       └── style.css        # 样式文件
│
├── app/                     # Gradio 版本
│   ├── start_all.py         # 一键启动脚本
│   ├── frontend/            # Gradio 前端
│   ├── backend/             # FastAPI 后端
│   └── shared/              # 共享模块
│
├── assets/                  # 资源文件
│   └── charactor/           # 角色资源
│
├── database/                # 数据库文件
│
├── docker/                  # Docker 配置
│   └── Dockerfile
│
├── docs/                    # 文档目录
│
├── environment/             # Conda 环境配置
│
└── README.md                # 本文档
```

---

## ❓ 常见问题

### Q1: 启动失败，提示数据库初始化错误？

**解决方案**：
1. 检查数据库目录权限：`chmod 755 database/`
2. 检查磁盘空间：`df -h`
3. 查看详细错误信息，确认具体问题

### Q2: 视频生成失败？

**可能原因**：
- 后端服务未启动
- 模型文件不存在
- 角色音频提示文件缺失
- 服务器资源不足

**解决方案**：
1. 确认后端服务正常运行：`curl http://localhost:8000/docs`
2. 检查模型文件路径是否正确
3. 查看后端日志获取详细错误信息
4. 确认 GPU/内存资源充足

### Q3: 视频只有音频没有画面？

**可能原因**：
- NeRF 视频生成失败
- 视频转码失败

**解决方案**：
1. 检查 FFmpeg 是否正确安装：`ffmpeg -version`
2. 查看后端日志中的转码信息
3. 确认原始视频文件是否有视频轨道

### Q4: 无法连接到后端服务？

**解决方案**：
1. 确认后端服务已启动：`ps aux | grep uvicorn`
2. 检查端口是否被占用：`netstat -tuln | grep 8000`
3. 如果使用端口转发，确保 SSH 配置正确
4. 检查防火墙设置

### Q5: 角色训练失败？

**可能原因**：
- 视频格式不支持
- 视频中没有检测到人脸
- 训练数据不足

**解决方案**：
1. 使用支持的视频格式（MP4、AVI 等）
2. 确保视频中有清晰的人脸
3. 检查视频时长和帧率

### Q6: 如何修改 API 地址？

**Web UI 版本**：
- 前端会自动检测 API 地址
- 如需手动修改，编辑 `webui/settings.js`

**Gradio 版本**：
```bash
export API_BASE_URL=http://your-server:8000
python frontend/app.py
```

---

## 🛠️ 开发指南

### 开发环境设置

```bash
# 1. 克隆项目
git clone <repository-url>
cd TFG_TALK_NeRFaceSpeech

# 2. 创建开发环境
conda create -n nerffacespeech-dev python=3.10
conda activate nerffacespeech-dev

# 3. 安装开发依赖
pip install -r requirements-dev.txt

# 4. 安装项目依赖
pip install -r requirements.txt
```

### 代码规范

- 使用 `black` 格式化代码
- 使用 `flake8` 检查代码风格
- 遵循 PEP 8 规范

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_backend.py

# 查看覆盖率
pytest --cov=.
```

### 提交代码

1. 创建功能分支：`git checkout -b feature/your-feature`
2. 提交更改：`git commit -m "Add your feature"`
3. 推送到远程：`git push origin feature/your-feature`
4. 创建 Pull Request

---

## 📝 更新日志

### v2.1.0 (当前版本)

**新增功能**：
- ✅ 角色训练功能（视频上传 → 自动提取 → 模型生成）
- ✅ 动态角色加载（自动识别训练的角色）
- ✅ 主题和字体设置功能（所有页面统一）
- ✅ 视频下载功能
- ✅ 优化的日志缓冲区（最多 1000 行）

**改进**：
- ✅ 优化视频转码流程
- ✅ 改进错误处理和提示
- ✅ 优化前端加载性能
- ✅ 改进数据库初始化流程

**修复**：
- ✅ 修复视频转码失败问题
- ✅ 修复角色列表加载问题
- ✅ 修复设置保存问题

### v2.0.0

- ✅ 一键启动脚本
- ✅ 选择页面和后端显示屏
- ✅ 训练页面全面优化
- ✅ 视频生成页面优化
- ✅ 所有页面统一设置功能

### v1.0.0

- ✅ 基础的三个功能模块
- ✅ 主题和字体自定义
- ✅ 对话管理和历史记录

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **Fork 项目**
2. **创建功能分支**：`git checkout -b feature/AmazingFeature`
3. **提交更改**：`git commit -m 'Add some AmazingFeature'`
4. **推送到分支**：`git push origin feature/AmazingFeature`
5. **创建 Pull Request**

### 贡献类型

- 🐛 Bug 修复
- ✨ 新功能
- 📝 文档改进
- 🎨 UI/UX 改进
- ⚡ 性能优化
- 🧪 测试用例

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 👥 团队

**北京理工大学 · NeRFFaceSpeech 团队**

---

## 🔗 相关链接

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Gradio 文档](https://gradio.app/docs/)
- [PyTorch 文档](https://pytorch.org/docs/)
- [FFmpeg 文档](https://ffmpeg.org/documentation.html)

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue：[GitHub Issues](https://github.com/your-repo/issues)
- 发送邮件：your-email@example.com

---

**祝使用愉快！** 🎉

---

## 🚀 快速开始（5 分钟上手）

### 步骤 1：启动服务

```bash
cd app
python start_all.py
```

### 步骤 2：打开浏览器

访问 `http://localhost:7860/`，你会看到 Gradio 界面。

### 步骤 3：生成第一个视频

1. 在"视频生成"标签页
2. 输入文本："你好，世界！"
3. 选择角色和模型
4. 点击"开始生成"按钮
5. 等待完成（5-10 分钟）

### 步骤 4：开始对话

1. 切换到"人机对话"标签页
2. 在输入框中输入问题
3. 按 `Enter` 发送
4. 查看 AI 回复和语音播放

**完成！** 🎉 现在你可以开始使用 NeRFFaceSpeech 了。

---

## 📚 更多文档

- [Web UI 使用指南](fastapi_server/webui/README.md)
- [Gradio 版本文档](app/README.md)
- [API 文档](http://localhost:8000/docs)（启动后访问）
- [服务器使用方法](服务器使用方法%20&%20模型训练方法.md)

---

**最后更新**：2024年

