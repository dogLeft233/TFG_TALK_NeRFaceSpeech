# 视频生成和显示流程

本文档说明前端如何获取生成的视频并显示到界面上。

## 整体流程

```
用户点击生成按钮
    ↓
前端提交任务到后端 (/generate_video)
    ↓
后端返回 task_id，开始后台生成
    ↓
前端开始轮询任务状态 (/generate_video/status/{task_id})
    ↓
后端生成完成后，更新状态为 "completed"
    ↓
前端检测到 completed 状态，获取视频路径
    ↓
前端验证文件存在，返回路径给 Gradio Video 组件
    ↓
Gradio 自动显示视频
```

## 详细步骤

### 1. 提交生成任务

**前端代码位置**: `frontend/app.py` 的 `generate_video_async()` 函数

```python
# 调用后端 API
api_url = f"{API_BASE_URL}/generate_video"
response = requests.post(
    api_url,
    json={
        "text": text,
        "character": character,
        "model_name": model_name
    },
    timeout=30
)

# 获取 task_id
task_id = result.get("task_id", unique_id)
```

**后端处理**: `backend/main.py` 的 `/generate_video` 端点
- 创建后台任务
- 返回 `task_id`
- 任务在后台线程中执行

### 2. 轮询任务状态

**前端轮询逻辑**:
```python
# 每2秒查询一次状态
while wait_time < max_wait:  # 最多等待10分钟
    status_response = requests.get(
        f"{API_BASE_URL}/generate_video/status/{task_id}",
        timeout=10
    )
    
    status_data = status_response.json()
    status = status_data.get("status", "unknown")
    
    if status == "completed":
        # 获取视频路径并显示
        break
    
    time.sleep(2)  # 等待2秒后再次查询
    wait_time += 2
```

**后端状态查询**: `backend/main.py` 的 `/generate_video/status/{unique_id}` 端点
- 优先从内存中获取（正在运行的任务）
- 如果任务已完成，从数据库获取完整信息
- 返回状态和视频路径

### 3. 获取视频路径

**后端返回的数据结构**:
```json
{
    "success": true,
    "data": {
        "status": "completed",
        "video_path": "/root/autodl-tmp/TFG_TALK_NeRFaceSpeech/database/videos/xxx.mp4",
        "video_url": "/videos/xxx.mp4",
        "generation_time": 123.45,
        ...
    }
}
```

**前端路径处理逻辑**:
1. **优先使用 `video_path`**（文件系统路径）
   - 如果是绝对路径，直接使用
   - 如果是相对路径，相对于 `PROJECT_ROOT` 构建完整路径
   - 验证文件是否存在

2. **备用方案：使用 `video_url`**（HTTP URL）
   - 如果 `video_path` 不存在，尝试使用 `video_url`
   - 构建 HTTP URL: `{API_BASE_URL}{video_url}`
   - Gradio Video 组件支持 HTTP URL

3. **最后备用：使用 task_id 构建路径**
   - 尝试常见路径：`VIDEOS_STORAGE_DIR / f"{task_id}.mp4"`

### 4. 显示视频

**Gradio Video 组件**:
```python
video_output = gr.Video(
    label="生成的视频",
    height=400
)

# 返回文件路径或 HTTP URL
return str(full_path), "✅ 视频生成成功！"
```

Gradio 会自动：
- 检测文件格式
- 生成视频播放器
- 显示视频预览

## 可能的问题和解决方案

### 问题1: 视频文件存在但前端找不到

**原因**:
- 路径处理错误（绝对路径 vs 相对路径）
- 文件权限问题
- 路径编码问题

**解决方案**:
- 检查前端日志中的路径信息
- 验证文件是否真的存在
- 检查文件权限

### 问题2: 视频生成完成但前端没有及时显示

**原因**:
- 轮询间隔太长（当前2秒）
- 状态更新延迟
- 前端轮询逻辑错误

**解决方案**:
- 检查后端日志，确认状态是否及时更新
- 检查前端轮询是否正常工作
- 可以缩短轮询间隔（但会增加服务器负载）

### 问题3: 视频路径返回但 Gradio 无法显示

**原因**:
- 文件格式不支持
- 文件损坏
- Gradio 版本问题

**解决方案**:
- 检查视频文件是否完整
- 尝试手动播放视频文件
- 检查 Gradio 版本兼容性

## 调试技巧

### 1. 查看前端日志

前端会在控制台输出详细的调试信息：
```
[前端] 提交视频生成任务
[前端]   目标地址: http://localhost:8000/generate_video
[前端]   响应状态码: 200
[前端] 开始轮询任务状态
[前端]   状态查询地址: http://localhost:8000/generate_video/status/xxx
[前端] 任务状态: running, 已等待: 10秒
[前端] 任务状态: completed, 已等待: 120秒
[前端] ✅ 找到视频文件: /path/to/video.mp4
```

### 2. 查看后端日志

后端会记录任务状态变化：
```
[任务] xxx: 视频生成任务开始
[任务] xxx: 阶段1: LLM + TTS 音频生成
[任务] xxx: 阶段2: NeRF Video Generation
[任务] xxx: 阶段3: Saving Video File
[任务] xxx: 视频生成任务完成，耗时 123.45 秒
```

### 3. 手动测试 API

```bash
# 查询任务状态
curl http://localhost:8000/generate_video/status/{task_id}

# 检查视频文件是否存在
ls -lh /root/autodl-tmp/TFG_TALK_NeRFaceSpeech/database/videos/{task_id}.mp4
```

## 优化建议

1. **缩短轮询间隔**: 如果视频生成很快，可以缩短到1秒
2. **添加 WebSocket 支持**: 实时推送状态更新，避免轮询
3. **添加进度条**: 显示生成进度（需要后端支持）
4. **缓存视频**: 避免重复生成相同内容的视频

## 相关文件

- **前端代码**: `gradio_app/frontend/app.py` - `generate_video_async()` 函数
- **后端代码**: `gradio_app/backend/main.py` - `/generate_video` 和 `/generate_video/status/{unique_id}` 端点
- **视频存储**: `database/videos/` 目录

