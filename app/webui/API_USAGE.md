# 前端 API 请求使用指南

本文档说明前端如何向后端发送 HTTP 请求。

## 请求方式

前端使用 **Fetch API** 向后端发送 HTTP 请求。所有 API 请求都封装在 `api.js` 文件中。

## 基本用法

### 1. 引入 API 客户端

在 HTML 文件中引入 `api.js`：

```html
<script src="api.js"></script>
```

### 2. 使用封装的 API 方法

`api.js` 提供了以下通用方法：

#### GET 请求

```javascript
// 获取模型列表
const models = await getModels();
console.log('模型列表:', models);

// 查询训练任务状态
const status = await getTrainingStatus('task-id-123');
console.log('任务状态:', status);
```

#### POST 请求

```javascript
// 生成视频
const result = await generateVideo({
  text: '你好，世界！',
  character: 'ayanami',
  model_name: 'ffhq_1024.pkl'
});
console.log('生成结果:', result);

// 启动训练
const task = await startTraining({
  data_path: '/path/to/data',
  base_model: 'ffhq_1024.pkl',
  kimg: 50,
  batch_size: 4
});
console.log('训练任务:', task);
```

#### 错误处理

```javascript
try {
  const models = await getModels();
  console.log('成功:', models);
} catch (error) {
  console.error('请求失败:', error.message);
  // 显示错误提示给用户
  alert('获取模型列表失败: ' + error.message);
}
```

## API 方法列表

### 通用方法

- `apiGet(endpoint, params)` - GET 请求
- `apiPost(endpoint, data)` - POST 请求
- `apiPut(endpoint, data)` - PUT 请求
- `apiDelete(endpoint)` - DELETE 请求
- `checkBackendConnection()` - 检查后端连接状态

### 视频生成相关

- `getModels()` - 获取模型列表
- `generateVideo(params)` - 生成视频
- `getVideoGenerationStatus(taskId)` - 查询视频生成状态

### 对话相关

- `chat(params)` - 发送对话请求

### 训练相关

- `getTrainingDatasets()` - 获取训练数据集列表
- `startTraining(params)` - 启动训练任务
- `getTrainingStatus(taskId)` - 查询训练任务状态
- `getTrainingTasks()` - 获取训练任务列表
- `stopTraining(taskId)` - 停止训练任务

## 请求示例

### 示例1：检查后端连接

```javascript
async function checkConnection() {
  const isConnected = await checkBackendConnection();
  if (isConnected) {
    console.log('后端连接正常');
  } else {
    console.log('后端连接失败');
  }
}
```

### 示例2：生成视频（带轮询）

```javascript
async function generateVideoWithPolling(text, character, model) {
  try {
    // 1. 提交生成任务
    const result = await generateVideo({
      text: text,
      character: character,
      model_name: model
    });
    
    if (!result.success) {
      throw new Error(result.error || '生成失败');
    }
    
    const taskId = result.task_id;
    console.log('任务已提交，ID:', taskId);
    
    // 2. 轮询查询状态
    const maxAttempts = 60; // 最多查询60次
    const interval = 2000; // 每2秒查询一次
    
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise(resolve => setTimeout(resolve, interval));
      
      const status = await getVideoGenerationStatus(taskId);
      
      if (status.status === 'completed') {
        console.log('生成完成！', status.video_path);
        return status;
      } else if (status.status === 'failed') {
        throw new Error(status.error || '生成失败');
      }
      
      console.log(`进度: ${status.status} (${i + 1}/${maxAttempts})`);
    }
    
    throw new Error('生成超时');
  } catch (error) {
    console.error('生成视频失败:', error);
    throw error;
  }
}
```

### 示例3：启动训练并监控

```javascript
async function startTrainingAndMonitor(params) {
  try {
    // 1. 启动训练
    const result = await startTraining(params);
    const taskId = result.task_id;
    console.log('训练任务已启动，ID:', taskId);
    
    // 2. 定期查询状态
    const statusInterval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(taskId);
        console.log('训练状态:', status.status);
        console.log('进度:', status.progress + '%');
        
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(statusInterval);
          console.log('训练结束');
        }
      } catch (error) {
        console.error('查询状态失败:', error);
      }
    }, 5000); // 每5秒查询一次
    
  } catch (error) {
    console.error('启动训练失败:', error);
    throw error;
  }
}
```

## 配置 API 地址

### 方式1：修改 settings.js

```javascript
const API_BASE_URL = 'http://your-server:8000';
```

### 方式2：在浏览器控制台设置

```javascript
localStorage.setItem('API_BASE_URL', 'http://your-server:8000');
location.reload();
```

## 请求流程

1. **前端发起请求** → 调用 `api.js` 中的方法
2. **构建请求** → 使用 `fetch` API 构建 HTTP 请求
3. **发送请求** → 发送到后端 API（默认：http://localhost:8000）
4. **处理响应** → 解析 JSON 响应
5. **错误处理** → 捕获并处理错误

## 注意事项

1. **CORS 配置**：确保后端已配置 CORS，允许前端域名访问
2. **错误处理**：始终使用 try-catch 处理异步请求
3. **超时设置**：长时间任务（如视频生成）需要轮询查询状态
4. **API 地址**：确保 `API_BASE_URL` 配置正确

## 后端 API 端点

参考后端 API 文档：http://localhost:8000/docs

主要端点：
- `GET /models` - 获取模型列表
- `POST /generate_video` - 生成视频
- `GET /generate_video/status/{task_id}` - 查询生成状态
- `POST /chat` - 对话接口
- `GET /train/datasets` - 获取训练数据集
- `POST /train/start` - 启动训练
- `GET /train/status/{task_id}` - 查询训练状态

