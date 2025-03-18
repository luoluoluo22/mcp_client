import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any
import re

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # 将日志级别从DEBUG改回INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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

        try:
            # 创建 SSE 客户端连接上下文管理器
            self._streams_context = await self.exit_stack.enter_async_context(
                sse_client(url=server_url)
            )
            
            # 使用数据流创建 MCP 客户端会话
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(*self._streams_context)
            )
            
            # 执行 MCP 协议初始化握手
            await self.session.initialize()
            logging.info(f"Successfully initialized SSE server: {self.name}")
        except Exception as e:
            logging.error(f"Error initializing SSE server {self.name}: {e}")
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
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

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
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


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
        
        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                
                # 如果内容为空或只包含空白字符，记录原始响应
                if not content or content.strip() == "":
                    logging.warning("收到空响应")
                    return "[空响应]"  # 返回一个标记，表示收到了空响应
                
                return content

        except httpx.RequestError as e:
            error_message = f"获取LLM响应时出错: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                logging.error(f"状态码: {e.response.status_code}")
                logging.error(f"响应详情: {e.response.text}")

            return f"遇到错误: {error_message}。请重试或换个方式提问。"


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

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
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json
        import re

        # 尝试直接解析整个响应作为JSON
        try:
            tool_call = json.loads(llm_response.strip())
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"执行工具: {tool_call['tool']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(f"进度: {progress}/{total} ({percentage:.1f}%)")

                            return f"工具执行结果: {result}"
                        except Exception as e:
                            error_msg = f"工具执行出错: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

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
                                        result = await server.execute_tool(
                                            tool_call["tool"], tool_call["arguments"]
                                        )

                                        if isinstance(result, dict) and "progress" in result:
                                            progress = result["progress"]
                                            total = result["total"]
                                            percentage = (progress / total) * 100
                                            logging.info(f"进度: {progress}/{total} ({percentage:.1f}%)")

                                        return f"工具执行结果: {result}"
                                    except Exception as e:
                                        error_msg = f"工具执行出错: {str(e)}"
                                        logging.error(error_msg)
                                        return error_msg

                            return f"未找到工具: {tool_call['tool']}"
                    except json.JSONDecodeError:
                        continue
        
        # 如果没有识别出工具调用，则返回原始响应
        return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    logging.info(f"正在初始化服务器: {server.name}...")
                    await server.initialize()
                except Exception as e:
                    logging.error(f"服务器初始化失败: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)
                logging.info(f"服务器 {server.name} 提供的工具数量: {len(tools)}")

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
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

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\n退出中...")
                        break

                    messages.append({"role": "user", "content": user_input})
                    
                    logging.info("正在等待回复...")
                    llm_response = self.llm_client.get_response(messages)
                    logging.info("\nAssistant: %s", llm_response)
                    
                    # 处理空响应的情况
                    if llm_response == "[空响应]":
                        logging.warning("收到空响应，请重新提问")
                        messages.append({"role": "assistant", "content": "我暂时无法回答这个问题，请尝试重新表述。"})
                        continue

                    result = await self.process_llm_response(llm_response)

                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})

                        logging.info("处理工具执行结果...")
                        final_response = self.llm_client.get_response(messages)
                        logging.info("\n最终回复: %s", final_response)
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\n退出中...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = LLMClient(config.llm_api_key, config.base_url, config.default_model)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
