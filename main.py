import asyncio
import json
import logging
import os
import shutil
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import re
import uuid

import httpx
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # 将日志级别从INFO改为DEBUG以显示调试信息
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 创建FastAPI应用
app = FastAPI(title="MCP聊天API", description="通过HTTP API与MCP工具集成的聊天服务，兼容OpenAI API格式")

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该更具体
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义OpenAI兼容的请求和响应模型
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 4096
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

# 定义原有的简化请求和响应模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# 全局变量，保存会话状态
chat_sessions = {}
servers_initialized = False
all_servers = []
llm_client_instance = None

# 添加新类TokenCounter用于估计token数量
class TokenCounter:
    """简单的token计数器，用于估计token数量"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估计文本中的token数量（简化版，仅供参考）
        
        中文和英文的token计数方式不同，这里采用简化方法：
        - 每个中文字符约为1.5个token
        - 每个英文单词约为1.3个token
        - 标点符号和空格约为0.5个token
        """
        if not text:
            return 0
            
        # 计算中文字符数量
        chinese_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        
        # 计算英文单词数量（简化版）
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        
        # 计算标点符号和空格数量
        punctuation_count = sum(1 for char in text if char in ',.!?;:()[]{}"\'`~@#$%^&*_+-=<>/\\| ')
        
        # 计算总token数
        total_tokens = (chinese_char_count * 1.5) + (english_words * 1.3) + (punctuation_count * 0.5)
        
        return max(1, int(total_tokens))

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._streams_context = None
        self._session_context = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        # 对于SSE类型的服务器，使用SSE客户端
        if self.config.get("type") == "sse":
            await self.initialize_sse()
        else:
            await self.initialize_stdio()

    async def initialize_sse(self) -> None:
        """Initialize an SSE server connection."""
        server_url = self.config.get("url")
        if not server_url:
            raise ValueError(f"No URL provided for SSE server {self.name}")

        logging.debug(f"开始初始化SSE服务器 {self.name}，URL: {server_url}")
        try:
            # 创建 SSE 客户端连接上下文管理器，并设置超时参数
            logging.debug(f"创建SSE客户端连接上下文管理器")
            
            # 设置SSE连接的超时时间 - 不使用httpx.Timeout对象，直接传递数值
            # mcp库内部会自己处理timeout参数
            sse_timeout = 30.0  # 设置为30秒
            logging.debug(f"SSE客户端连接超时设置: {sse_timeout}秒")
            
            self._streams_context = await self.exit_stack.enter_async_context(
                sse_client(url=server_url, timeout=sse_timeout)
            )
            logging.debug(f"SSE客户端连接创建成功: {self._streams_context}")
            
            # 使用数据流创建 MCP 客户端会话
            logging.debug(f"使用数据流创建MCP客户端会话")
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(*self._streams_context)
            )
            logging.debug(f"MCP客户端会话创建成功: {self.session}")
            
            # 执行 MCP 协议初始化握手
            logging.debug(f"执行MCP协议初始化握手")
            await self.session.initialize()
            logging.info(f"Successfully initialized SSE server: {self.name}")
        except Exception as e:
            logging.error(f"Error initializing SSE server {self.name}: {e}")
            logging.error("SSE初始化异常详情:", exc_info=True)
            await self.cleanup()
            raise

    async def initialize_stdio(self) -> None:
        """Initialize a stdio server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            logging.info(f"Successfully initialized stdio server: {self.name}")
        except Exception as e:
            logging.error(f"Error initializing stdio server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        logging.debug(f"在服务器 {self.name} 上获取可用工具列表")
        tools_response = await self.session.list_tools()
        logging.debug(f"服务器 {self.name} 返回的原始工具信息: {tools_response}")
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    logging.debug(f"找到工具: {tool.name}")
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        logging.debug(f"服务器 {self.name} 共有 {len(tools)} 个可用工具")
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        logging.debug(f"开始执行工具 {tool_name}，参数：{arguments}")
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                logging.debug(f"调用工具前的session状态: {self.session}")
                
                try:
                    # 尝试捕获更具体的call_tool调用错误
                    result = await self.session.call_tool(tool_name, arguments)
                    logging.debug(f"工具 {tool_name} 执行结果(原始): {result}")
                    
                    # 如果result为None或空，添加警告
                    if result is None:
                        logging.warning(f"工具 {tool_name} 返回了None结果")
                    elif isinstance(result, (str, list, dict)) and not result:
                        logging.warning(f"工具 {tool_name} 返回了空结果")
                        
                    return result
                except asyncio.TimeoutError as e:
                    logging.error(f"工具执行超时: {tool_name}")
                    raise RuntimeError(f"Tool execution timed out: {tool_name}") from e
                except ConnectionError as e:
                    logging.error(f"连接错误: {str(e)}")
                    raise RuntimeError(f"Connection error while calling tool: {tool_name}") from e
                except Exception as e:
                    logging.error(f"调用工具时发生未知错误: {type(e).__name__}: {str(e)}")
                    raise

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                # 添加更详细的异常信息
                logging.error(f"异常详情: {type(e).__name__}: {str(e)}")
                logging.error("异常堆栈:", exc_info=True)
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            if not self.session:  # 如果session已经是None，则不需要清理
                return
                
            logging.debug(f"正在清理服务器 {self.name} 的资源")
            try:
                # 先将session设为None，防止重复清理
                session = self.session
                self.session = None
                self.stdio_context = None
                
                # 安全关闭exit_stack
                try:
                    await self.exit_stack.aclose()
                except Exception as e:
                    logging.error(f"关闭 exit_stack 时出错: {e}")
                    
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")
                logging.debug(f"清理异常详情:", exc_info=True)


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str, base_url: str = None, default_model: str = None) -> None:
        self.api_key: str = api_key
        self.base_url: str = base_url or "https://api.openai.com/v1"
        self.default_model: str = default_model or "gpt-3.5-turbo"

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": self.default_model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }
        
        logging.debug(f"准备发送请求到LLM API，URL: {url}")
        logging.debug(f"请求头信息: {headers}")
        logging.debug(f"请求负载: {payload}")
        
        # 设置重试次数和延迟
        max_retries = 3
        retry_delay = 2  # 秒
        
        # 使用Timeout类设置更详细的超时控制
        # connect=10.0: 连接建立的超时时间为10秒
        # read=60.0: 读取响应的超时时间为60秒
        # write=10.0: 发送请求的超时时间为10秒
        # pool=10.0: 从连接池获取连接的超时时间为10秒
        timeout = httpx.Timeout(10.0, read=60.0, write=10.0, pool=10.0)
        logging.debug(f"设置请求超时: {timeout}")
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    logging.debug(f"开始发送HTTP请求... (尝试 {attempt+1}/{max_retries})")
                    response = client.post(url, headers=headers, json=payload)
                    logging.info(f"HTTP Request: {response.request.method} {response.request.url} \"{response.status_code} {response.reason_phrase}\"")
                    response.raise_for_status()
                    data = response.json()
                    logging.debug(f"收到API响应: {data}")
                    
                    content = data["choices"][0]["message"]["content"]
                    
                    # 如果内容为空或只包含空白字符，记录原始响应
                    if not content or content.strip() == "":
                        logging.warning("收到空响应")
                        logging.debug(f"完整响应数据: {data}")
                        return "[空响应]"
                    
                    return content
                    
            except (httpx.RequestError, KeyError, IndexError) as e:
                logging.error(f"API请求错误 (尝试 {attempt+1}/{max_retries}): {type(e).__name__}: {str(e)}")
                if attempt < max_retries - 1:  # 如果不是最后一次尝试
                    logging.info(f"将在 {retry_delay} 秒后重试...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避策略
                else:
                    logging.error("已达到最大重试次数，放弃请求")
                    logging.error("详细错误信息:", exc_info=True)
                    return f"Error: {str(e)}"


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.messages: List[Dict[str, str]] = []
        self.session_id: str = ""
        self.tools_description: str = ""
        self.system_message: str = ""
        self.initialized: bool = False

    async def initialize(self) -> None:
        """初始化聊天会话，包括准备服务器和加载工具"""
        if self.initialized:
            return
            
        # 初始化服务器，添加重试机制
        initialized_servers = []
        for server in self.servers:
            max_retries = 2
            retry_delay = 1.5
            
            for attempt in range(max_retries):
                try:
                    logging.info(f"正在初始化服务器: {server.name}... (尝试 {attempt+1}/{max_retries})")
                    await server.initialize()
                    initialized_servers.append(server)
                    break  # 初始化成功，跳出重试循环
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"服务器 {server.name} 初始化失败 (尝试 {attempt+1}/{max_retries}): {e}")
                        logging.info(f"将在 {retry_delay} 秒后重试...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        logging.error(f"服务器 {server.name} 初始化失败，已达到最大重试次数: {e}")
        
        # 如果所有服务器都初始化失败，则抛出异常
        if not initialized_servers:
            logging.error("所有服务器初始化失败，会话无法启动")
            raise RuntimeError("所有服务器初始化失败，会话无法启动")
            
        # 更新服务器列表为成功初始化的服务器
        self.servers = initialized_servers
        
        # 获取可用工具
        all_tools = []
        for server in self.servers:
            try:
                tools = await server.list_tools()
                all_tools.extend(tools)
                logging.info(f"服务器 {server.name} 提供的工具数量: {len(tools)}")
            except Exception as e:
                logging.error(f"无法从服务器 {server.name} 获取工具列表: {e}")
        
        if not all_tools:
            logging.warning("没有可用的工具，会话将无法提供工具功能")
            
        self.tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

        self.system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{self.tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )

        self.messages = [{"role": "system", "content": self.system_message}]
        self.initialized = True

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """
        Process LLM response to extract and handle tool calls.

        Args:
            llm_response: Raw response from LLM.

        Returns:
            Processed response or tool execution result.
        """
        import json
        import re

        # 添加调试信息
        logging.debug(f"处理LLM响应: {llm_response}")

        # 尝试直接解析整个响应作为JSON
        try:
            tool_call = json.loads(llm_response.strip())
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"执行工具: {tool_call['tool']}")
                logging.debug(f"工具参数: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    logging.debug(f"服务器 {server.name} 可用工具: {[tool.name for tool in tools]}")
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            logging.debug(f"开始在服务器 {server.name} 上执行工具 {tool_call['tool']}")
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )
                            logging.debug(f"工具执行完成，结果: {result}")

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(f"进度: {progress}/{total} ({percentage:.1f}%)")

                            return f"工具执行结果: {result}"
                        except Exception as e:
                            error_msg = f"工具执行出错: {str(e)}"
                            logging.error(error_msg)
                            logging.error("详细异常信息:", exc_info=True)
                            return error_msg

                    logging.warning(f"在服务器 {server.name} 上未找到工具: {tool_call['tool']}")
                
                logging.error(f"所有服务器中都未找到工具: {tool_call['tool']}")
                return f"未找到工具: {tool_call['tool']}"
        except json.JSONDecodeError:
            # 如果整个响应不是JSON，尝试从响应中提取JSON
            json_pattern = r'({[\s\S]*?})'
            json_matches = re.findall(json_pattern, llm_response)
            
            # 如果找到至少一个JSON模式匹配
            if json_matches:
                # 尝试解析每一个匹配
                for potential_json in json_matches:
                    try:
                        tool_call = json.loads(potential_json.strip())
                        if "tool" in tool_call and "arguments" in tool_call:
                            logging.info(f"执行工具: {tool_call['tool']}")

                            for server in self.servers:
                                tools = await server.list_tools()
                                if any(tool.name == tool_call["tool"] for tool in tools):
                                    try:
                                        logging.debug(f"开始在服务器 {server.name} 上执行工具 {tool_call['tool']}")
                                        result = await server.execute_tool(
                                            tool_call["tool"], tool_call["arguments"]
                                        )
                                        logging.debug(f"工具执行完成，结果: {result}")

                                        if isinstance(result, dict) and "progress" in result:
                                            progress = result["progress"]
                                            total = result["total"]
                                            percentage = (progress / total) * 100
                                            logging.info(f"进度: {progress}/{total} ({percentage:.1f}%)")

                                        return f"工具执行结果: {result}"
                                    except Exception as e:
                                        error_msg = f"工具执行出错: {str(e)}"
                                        logging.error(error_msg)
                                        logging.error("详细异常信息:", exc_info=True)
                                        return error_msg

                            return f"未找到工具: {tool_call['tool']}"
                    except json.JSONDecodeError:
                        continue
        
        # 如果没有识别出工具调用，则返回原始响应
        return llm_response

    async def process_openai_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        处理符合OpenAI格式的请求
        
        Args:
            request: OpenAI格式的请求
            
        Returns:
            OpenAI格式的响应
        """
        # 确保会话已初始化
        if not self.initialized:
            await self.initialize()
        
        # 将OpenAI格式的messages转换为我们系统使用的格式
        self.messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # 如果没有system消息，添加默认system消息
        if not any(msg["role"] == "system" for msg in self.messages):
            self.messages.insert(0, {"role": "system", "content": self.system_message})
        
        try:
            # 调用LLM获取回复
            logging.info(f"处理OpenAI格式请求，消息数量: {len(self.messages)}")
            llm_response = self.llm_client.get_response(self.messages)
            logging.info(f"LLM原始回复: {llm_response}")
            
            # 处理空响应或错误的情况
            if llm_response == "[空响应]" or llm_response.startswith("Error:"):
                error_msg = "我暂时无法回答这个问题，请稍后再试。"
                self.messages.append({"role": "assistant", "content": error_msg})
                return self._format_openai_response(error_msg, request.model)
                
            # 处理工具调用
            result = await self.process_llm_response(llm_response)
            
            final_response = llm_response
            
            # 如果LLM回复是工具调用，则继续处理工具执行结果
            if result != llm_response:
                self.messages.append({"role": "assistant", "content": llm_response})
                self.messages.append({"role": "system", "content": result})

                logging.info("处理工具执行结果...")
                final_response = self.llm_client.get_response(self.messages)
                logging.info(f"最终回复: {final_response}")
                
            # 添加最终回复到会话历史
            self.messages.append({"role": "assistant", "content": final_response})
            
            # 返回OpenAI格式的响应
            return self._format_openai_response(final_response, request.model)
                
        except Exception as e:
            logging.error(f"处理消息时出错: {e}", exc_info=True)
            error_msg = "抱歉，我遇到了内部错误，请重新提问。"
            self.messages.append({"role": "assistant", "content": error_msg})
            return self._format_openai_response(error_msg, request.model)
            
    async def process_openai_stream_request(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """
        处理流式响应请求
        
        Args:
            request: OpenAI格式的请求，带有stream=True
            
        Yields:
            SSE格式的流式响应片段
        """
        # 确保会话已初始化
        if not self.initialized:
            await self.initialize()
        
        # 将OpenAI格式的messages转换为我们系统使用的格式
        self.messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # 如果没有system消息，添加默认system消息
        if not any(msg["role"] == "system" for msg in self.messages):
            self.messages.insert(0, {"role": "system", "content": self.system_message})
        
        response_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
        created_time = int(time.time())
        
        try:
            # 调用LLM获取回复
            logging.info(f"处理OpenAI流式请求，消息数量: {len(self.messages)}")
            llm_response = self.llm_client.get_response(self.messages)
            logging.info(f"LLM原始回复: {llm_response}")
            
            # 处理空响应或错误的情况
            if llm_response == "[空响应]" or llm_response.startswith("Error:"):
                error_msg = "我暂时无法回答这个问题，请稍后再试。"
                self.messages.append({"role": "assistant", "content": error_msg})
                
                # 返回错误消息的流式响应
                yield self._format_stream_chunk(response_id, created_time, request.model, error_msg, 0, "")
                yield self._format_stream_chunk(response_id, created_time, request.model, "", -1, "stop")
                return
                
            # 处理工具调用 - 在流式响应情况下暂不支持工具调用
            # 如果检测到工具调用，转为非流式处理并一次性返回结果
            try:
                json.loads(llm_response.strip())
                # 如果解析成功，说明这是JSON格式，可能是工具调用
                logging.info("检测到可能的工具调用，暂不支持流式处理，转为一次性返回")
                
                result = await self.process_llm_response(llm_response)
                final_response = llm_response
                
                if result != llm_response:
                    self.messages.append({"role": "assistant", "content": llm_response})
                    self.messages.append({"role": "system", "content": result})
                    final_response = self.llm_client.get_response(self.messages)
                
                # 返回最终结果的流式响应
                yield self._format_stream_chunk(response_id, created_time, request.model, final_response, 0, "")
                yield self._format_stream_chunk(response_id, created_time, request.model, "", -1, "stop")
                
                # 更新会话历史
                self.messages.append({"role": "assistant", "content": final_response})
                return
            except json.JSONDecodeError:
                # 不是JSON，继续正常处理
                pass
            
            # 将响应拆分成较小的块（以句子为单位）进行流式返回
            chunks = self._split_into_chunks(llm_response)
            full_content = ""
            
            for i, chunk in enumerate(chunks):
                full_content += chunk
                yield self._format_stream_chunk(response_id, created_time, request.model, chunk, i, "")
                await asyncio.sleep(0.01)  # 添加一点延迟，使流式效果更明显
            
            # 发送完成标记
            yield self._format_stream_chunk(response_id, created_time, request.model, "", len(chunks), "stop")
            
            # 更新会话历史
            self.messages.append({"role": "assistant", "content": full_content})
            
        except Exception as e:
            logging.error(f"流式处理消息时出错: {e}", exc_info=True)
            error_msg = "抱歉，我遇到了内部错误，请重新提问。"
            
            # 发送错误信息和完成标记
            yield self._format_stream_chunk(response_id, created_time, request.model, error_msg, 0, "")
            yield self._format_stream_chunk(response_id, created_time, request.model, "", 1, "stop")
            
            # 更新会话历史
            self.messages.append({"role": "assistant", "content": error_msg})
    
    def _format_stream_chunk(self, id: str, created: int, model: str, content: str, 
                             chunk_index: int, finish_reason: str = None) -> str:
        """
        格式化流式响应的单个数据块为SSE格式
        
        Args:
            id: 响应ID
            created: 创建时间戳
            model: 模型名称
            content: 当前块的内容
            chunk_index: 块索引
            finish_reason: 完成原因，如果是最后一块则为"stop"
            
        Returns:
            SSE格式的数据块
        """
        delta = {"role": "assistant"}
        if content:
            delta["content"] = content
        
        chunk = {
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason
                }
            ]
        }
        
        if chunk_index == -1:  # 表示最后一个块，添加使用情况统计
            prompt_tokens = sum(TokenCounter.estimate_tokens(msg["content"]) for msg in self.messages if msg["role"] != "assistant")
            completion_tokens = TokenCounter.estimate_tokens("".join(msg["content"] for msg in self.messages if msg["role"] == "assistant"))
            
            chunk["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        将文本拆分成较小的块，以便流式返回
        
        Args:
            text: 要拆分的文本
            
        Returns:
            拆分后的文本块列表
        """
        # 使用标点符号作为分割点，确保每个块都是完整的句子
        sentence_endings = ['. ', '! ', '? ', '。', '！', '？', '\n']
        chunks = []
        
        # 如果文本太短，直接作为一个块返回
        if len(text) < 20:
            return [text]
        
        buffer = ""
        for char in text:
            buffer += char
            
            # 如果缓冲区达到一定长度且以句子结尾，则添加为一个块
            if len(buffer) >= 10 and any(buffer.endswith(ending) for ending in sentence_endings):
                chunks.append(buffer)
                buffer = ""
        
        # 添加剩余部分
        if buffer:
            chunks.append(buffer)
        
        # 如果没有拆分成块，则按照固定长度拆分
        if not chunks:
            chunks = [text[i:i+20] for i in range(0, len(text), 20)]
        
        return chunks
            
    def _format_openai_response(self, content: str, model: str) -> ChatCompletionResponse:
        """
        将内容格式化为OpenAI格式的响应
        
        Args:
            content: 助手的回复内容
            model: 使用的模型名称
            
        Returns:
            OpenAI格式的响应对象
        """
        # 估算token数量
        prompt_tokens = sum(TokenCounter.estimate_tokens(msg["content"]) for msg in self.messages if msg["role"] != "assistant")
        completion_tokens = TokenCounter.estimate_tokens(content)
        total_tokens = prompt_tokens + completion_tokens
        
        # 创建响应对象
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )

# API路由定义
@app.get("/")
async def root():
    """API根路径，返回简单的欢迎信息"""
    return {"message": "欢迎使用MCP聊天API，支持简化API('/chat')和OpenAI兼容API('/v1/chat/completions')"}

# 保留原有的简化API端点
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理简化格式的聊天请求"""
    global servers_initialized, all_servers, llm_client_instance, chat_sessions
    
    logging.info(f"接收到聊天请求: {request.message}，会话ID: {request.session_id}")
    
    # 确保服务器已初始化
    if not servers_initialized:
        await initialize_servers()
    
    # 获取或创建会话
    session_id = request.session_id or f"session_{len(chat_sessions) + 1}"
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession(all_servers.copy(), llm_client_instance)
        logging.info(f"创建新会话: {session_id}")
    
    session = chat_sessions[session_id]
    
    # 处理消息并获取回复
    response = await session.process_message(request.message)
    
    return ChatResponse(response=response, session_id=session_id)

# 修改OpenAI兼容的API端点，支持流式响应
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """处理OpenAI兼容格式的聊天请求，支持流式响应"""
    global servers_initialized, all_servers, llm_client_instance, chat_sessions
    
    logging.info(f"接收到OpenAI格式的聊天请求: 模型={request.model}, 消息数量={len(request.messages)}, 流式={request.stream}")
    
    # 确保服务器已初始化
    if not servers_initialized:
        await initialize_servers()
        
    # 使用OpenAI请求中的user字段作为会话ID，如果没有则创建新ID
    session_id = request.user or f"openai_session_{len(chat_sessions) + 1}"
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession(all_servers.copy(), llm_client_instance)
        logging.info(f"创建新的OpenAI兼容会话: {session_id}")
    
    session = chat_sessions[session_id]
    
    # 处理流式响应请求
    if request.stream:
        logging.info("处理流式响应请求")
        
        # 设置用于异步生成响应流的生成器函数
        async def stream_generator():
            async for chunk in session.process_openai_stream_request(request):
                yield chunk
            yield "data: [DONE]\n\n"  # 标志流结束
            
        # 返回流式响应
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
            }
        )
    else:
        # 处理普通（非流式）请求
        response = await session.process_openai_request(request)
        return response

# 添加OpenAI兼容的模型列表端点
@app.get("/v1/models")
async def list_models():
    """返回支持的模型列表(OpenAI兼容格式)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(time.time()) - 10000,
                "owned_by": "organization-owner"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": int(time.time()) - 5000,
                "owned_by": "organization-owner"
            }
        ]
    }

async def initialize_servers():
    """初始化所有服务器，只需要执行一次"""
    global servers_initialized, all_servers, llm_client_instance
    
    logging.info("正在初始化服务器...")
    
    # 加载配置
    config = Configuration()
    
    # 检查jianshu服务器可用性
    jianshu_server_url = "http://38.55.129.183:8015/sse"
    logging.info(f"正在检查jianshu服务器可用性: {jianshu_server_url}")
    
    try:
        timeout = 5.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(jianshu_server_url)
            logging.info(f"检查结果: HTTP {response.status_code}")
            if response.status_code >= 400:
                logging.warning(f"警告: jianshu服务器返回了错误状态码: {response.status_code}")
    except Exception as e:
        logging.warning(f"无法连接到jianshu服务器 ({jianshu_server_url}): {e}")
        logging.warning("简书相关功能可能无法使用。")
    
    # 加载服务器配置和创建实例
    server_config = config.load_config("servers_config.json")
    all_servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    
    # 创建LLM客户端
    llm_client_instance = LLMClient(config.llm_api_key, config.base_url, config.default_model)
    
    servers_initialized = True
    logging.info("服务器初始化完成")

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI关闭事件处理，用于清理资源"""
    logging.info("服务关闭中，清理资源...")
    
    # 清理所有会话的服务器资源
    for session_id, session in chat_sessions.items():
        try:
            await session.cleanup_servers()
            logging.info(f"已清理会话 {session_id} 的资源")
        except Exception as e:
            logging.warning(f"清理会话 {session_id} 资源时出错: {e}")
    
    logging.info("所有资源已清理完毕")

async def main() -> None:
    """初始化并运行API服务器"""
    # 直接启动uvicorn，而不是初始化聊天会话
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logging.info(f"启动API服务器在 {host}:{port}")
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
