# MCP聊天API服务

该项目将MCP工具集成到一个HTTP API服务中，允许通过API请求与大模型进行交互并使用各种工具。现在支持两种API格式：简化API和OpenAI兼容API。

## 功能特点

- 将命令行聊天界面转换为HTTP API服务
- 支持使用MCP工具集
- 维护多个会话的上下文
- 自动重试机制和错误处理
- 支持跨域请求(CORS)
- **新增**: 支持OpenAI兼容的API格式
- **新增**: 支持流式响应（Streaming）

## 安装

1. 克隆仓库
2. 安装依赖:

```bash
pip install -r requirements.txt
```

3. 创建一个`.env`文件，设置以下环境变量:

```
OPENAI_API_KEY=你的OpenAI API密钥
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_MODEL=gpt-3.5-turbo
PORT=8000
HOST=0.0.0.0
```

4. 确保`servers_config.json`文件正确配置了所需的MCP服务器

## 使用方法

### 启动服务器

```bash
python main.py
```

服务器默认运行在`http://localhost:8000`

### API端点

#### 简化API

##### GET /

返回简单的欢迎消息。

##### POST /chat

发送聊天消息并获取回复。

请求体格式:
```json
{
  "message": "你的问题或消息",
  "session_id": "可选的会话ID"
}
```

如果不提供`session_id`，服务器会创建一个新的会话。

响应格式:
```json
{
  "response": "大模型的回复",
  "session_id": "会话ID，用于后续请求"
}
```

#### OpenAI兼容API

##### GET /v1/models

获取可用模型列表。

响应格式:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1677610602,
      "owned_by": "organization-owner"
    },
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1677610602,
      "owned_by": "organization-owner"
    }
  ]
}
```

##### POST /v1/chat/completions

发送聊天消息并获取回复，与OpenAI API格式完全兼容。支持普通响应和流式响应。

请求体格式:
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好，介绍一下自己。"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": false  // 设置为true启用流式响应
}
```

**普通响应格式:**
```json
{
  "id": "chatcmpl-123abc456def",
  "object": "chat.completion",
  "created": 1677610602,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是一个AI助手..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 30,
    "completion_tokens": 100,
    "total_tokens": 130
  }
}
```

**流式响应格式:**

当使用`stream=true`参数时，服务器会返回一系列的SSE (Server-Sent Events)事件：

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"！"},"finish_reason":null}]}

... [更多内容块]

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 客户端示例

提供了两个客户端示例：

1. `client.py` - 使用简化API的命令行客户端
2. `test_openai_client.py` - 使用OpenAI兼容API的测试客户端，支持普通和流式响应

运行客户端示例:

```bash
# 简化API客户端
python client.py

# OpenAI兼容API客户端
python test_openai_client.py
```

## 使用OpenAI SDK

由于本服务兼容OpenAI的API格式，您可以直接使用官方的OpenAI SDK或其他第三方库来调用本服务。只需要将base_url设置为本服务的地址即可:

### 普通响应示例

```python
from openai import OpenAI

# 创建客户端时指定base_url
client = OpenAI(
    api_key="任意字符串，不会实际使用",
    base_url="http://localhost:8000/v1"
)

# 使用方法与调用OpenAI API完全一致
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，请问今天天气如何？"}
    ]
)

print(response.choices[0].message.content)
```

### 流式响应示例

```python
from openai import OpenAI

# 创建客户端时指定base_url
client = OpenAI(
    api_key="任意字符串，不会实际使用",
    base_url="http://localhost:8000/v1"
)

# 流式响应调用
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "讲一个关于人工智能的故事"}
    ],
    stream=True  # 启用流式响应
)

# 逐块处理响应
print("AI回复: ", end="")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## 自定义

- 修改`servers_config.json`添加或移除MCP服务器
- 在`.env`文件中更改模型或其他配置
- 调整`main.py`中的超时设置和重试策略

## 注意事项

- 生产环境中应该限制CORS的`allow_origins`
- 考虑添加API认证机制
- 可以根据需要实现会话的持久化存储
- 目前token计数是估算的，不保证与OpenAI的计算完全一致
- 流式响应模式下暂不支持工具调用，如检测到工具调用会转为普通响应

## 流式响应的限制

当使用流式响应时，有以下限制：

1. 不支持MCP工具调用 - 如果检测到模型返回的内容是工具调用（JSON格式），系统会自动切换到非流式模式处理
2. 工具执行结果不会实时流式返回，会等待工具执行完成后一次性返回
3. 流式响应中间不能被中断，必须等待完整响应完成

## 环境要求

- Python 3.7+
- 依赖包：
  - httpx
  - python-dotenv
  - mcp-sdk
  - fastapi
  - uvicorn
  - pydantic
  - requests
  - sseclient-py

## 配置

### 1. 环境变量配置

创建 `.env` 文件，配置以下环境变量：

```env
# LLM API配置
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，默认为OpenAI官方地址
DEFAULT_MODEL=gpt-3.5-turbo  # 可选，默认为gpt-3.5-turbo
PORT=8000
HOST=0.0.0.0

# 简书配置（如果需要）
JIANSHU_USER_ID=你的用户ID
JIANSHU_COOKIES=你的Cookie字符串
```

### 2. 服务器配置

编辑 `servers_config.json` 文件，配置需要连接的服务器：

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "sqlite-server",
      "args": ["database.db"],
      "env": {
        "DB_PATH": "path/to/database.db"
      }
    },
    "jianshu": {
      "type": "sse",
      "url": "http://your-sse-server/sse"
    }
  }
}
```

支持两种类型的服务器：
- 标准输入/输出服务器：需要指定 `command` 和 `args`
- SSE服务器：需要指定 `type: "sse"` 和 `url`

## 使用方法

1. 确保配置文件正确设置
2. 运行聊天机器人：
```bash
python main.py
```

3. 开始对话：
- 输入问题或指令
- 机器人会自动选择合适的工具来处理请求
- 输入 "quit" 或 "exit" 退出程序

## 可用工具

### SQLite工具
- `read_query`: 执行SELECT查询
- `write_query`: 执行INSERT/UPDATE/DELETE查询
- `create_table`: 创建新表
- `list_tables`: 列出所有表
- `describe_table`: 获取表结构
- `append_insight`: 添加业务洞察


## 日志级别

默认使用 INFO 级别的日志，如需调试可以修改 `main.py` 中的日志级别：
```python
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG以获取更详细的日志
    format="%(asctime)s - %(levelname)s - %(message)s"
)
```

## 错误处理

- 工具执行失败会自动重试（默认2次）
- 空响应会提示重新提问
- 服务器连接失败会记录错误并退出
- 资源会在程序退出时自动清理

## 开发说明

### 添加新工具

1. 在服务器端实现工具功能
2. 在 `servers_config.json` 中添加服务器配置
3. 工具会自动被发现并集成到聊天机器人中

### 自定义响应处理

可以通过修改 `process_llm_response` 方法来自定义响应处理逻辑。

### 会话管理

`ChatSession` 类负责管理整个对话过程，包括：
- 初始化服务器连接
- 处理用户输入
- 调用LLM获取响应
- 执行工具调用
- 清理资源

## 注意事项

1. 请确保API密钥安全，不要提交到版本控制系统
2. SSE服务器需要支持长连接
3. 大量调试日志可能会影响性能
4. 请定期检查和更新依赖包版本

## 常见问题

1. 如果遇到连接错误，请检查：
   - 网络连接
   - API密钥是否正确
   - 服务器地址是否可访问

2. 如果工具执行失败，请检查：
   - 工具参数是否正确
   - 服务器状态
   - 日志中的错误信息

3. 如果收到空响应，可以：
   - 重新表述问题
   - 检查API配额
   - 查看详细日志

## 贡献指南

欢迎提交问题和改进建议！请确保：
1. 提供清晰的问题描述
2. 包含必要的日志信息
3. 说明复现步骤

## 许可证

MIT License 