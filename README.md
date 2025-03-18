# OpenAI兼容API服务（MCP后端）

这个项目提供了一个与OpenAI API兼容的HTTP服务，使用MCP（多通道处理器）作为后端。它允许客户端通过标准的OpenAI API格式访问各种工具和服务。

## 功能特点

- 兼容OpenAI API格式，便于集成到现有应用
- 支持ChatGPT风格的聊天完成API
- 提供工具执行功能
- 支持文本嵌入（目前是占位实现）
- 自动初始化和管理MCP服务器连接
- 完善的错误处理和日志记录

## 安装

1. 克隆此仓库：

```bash
git clone <仓库URL>
cd <仓库目录>
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备环境变量（创建`.env`文件）：

```
OPENAI_API_KEY=你的OpenAI_API密钥
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_MODEL=gpt-3.5-turbo
```

## 使用方法

### 启动服务器

```bash
python openai_compatible_api.py
```

服务器默认在`http://localhost:8000`上运行。

### API端点

#### 1. 聊天完成

```
POST /v1/chat/completions
```

请求示例：

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "介绍一下Python语言的主要特点。"}
  ],
  "temperature": 0.7
}
```

#### 2. 获取可用工具列表

```
GET /v1/tools
```

#### 3. 执行工具

```
POST /v1/tools
```

请求示例：

```json
{
  "tool": "工具名称",
  "arguments": {
    "参数1": "值1",
    "参数2": "值2"
  }
}
```

#### 4. 嵌入向量（模拟实现）

```
POST /v1/embeddings
```

请求示例：

```json
{
  "model": "text-embedding-ada-002",
  "input": ["这是一个测试文本"]
}
```

## 测试客户端

我们提供了一个测试客户端`test_client.py`来展示如何调用API。运行以下命令来测试服务：

```bash
python test_client.py
```

## 配置

服务器配置位于`servers_config.json`文件中，它定义了MCP服务器的连接信息。

## 文件结构

- `openai_compatible_api.py` - 主API服务器
- `main.py` - 核心MCP功能
- `test_client.py` - 测试客户端
- `requirements.txt` - 依赖列表
- `servers_config.json` - 服务器配置文件
- `.env` - 环境变量（需要自行创建）

## 注意事项

- 目前嵌入API返回的是随机值，仅用于演示目的
- 在生产环境中，应该更改CORS设置为特定的源，而不是`"*"`
- 确保您的服务器配置正确，并且所有必要的依赖项都已安装

## 环境要求

- Python 3.7+
- 依赖包：
  - httpx
  - python-dotenv
  - mcp-sdk

## 配置

### 1. 环境变量配置

创建 `.env` 文件，配置以下环境变量：

```env
# LLM API配置
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，默认为OpenAI官方地址
DEFAULT_MODEL=gpt-3.5-turbo  # 可选，默认为gpt-3.5-turbo

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