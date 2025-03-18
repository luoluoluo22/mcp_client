import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入main.py中的核心组件
from main import Configuration, Server, LLMClient, ChatSession, Tool

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 创建FastAPI应用
app = FastAPI(
    title="OpenAI兼容API",
    description="提供与OpenAI API兼容的聊天和嵌入服务，使用MCP后端",
    version="1.0.0"
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该更具体地指定来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置和服务实例
config = Configuration()
servers = {}  # 用于存储初始化的服务器实例
tools_cache = {}  # 缓存工具信息


# Pydantic模型，用于API请求和响应
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    embedding: List[float]
    index: int
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Dict[str, int]


class ToolRequestBody(BaseModel):
    tool: str
    arguments: Dict[str, Any]


async def initialize_servers():
    """初始化所有配置的服务器。"""
    logging.info("正在初始化服务器...")
    try:
        server_config = config.load_config("servers_config.json")
        for name, srv_config in server_config["mcpServers"].items():
            if name not in servers:
                server = Server(name, srv_config)
                try:
                    await server.initialize()
                    servers[name] = server
                    logging.info(f"服务器 {name} 成功初始化")
                    
                    # 缓存工具信息
                    tools = await server.list_tools()
                    tools_cache[name] = tools
                    logging.info(f"从服务器 {name} 获取到 {len(tools)} 个工具")
                except Exception as e:
                    logging.error(f"初始化服务器 {name} 失败: {e}")
    except Exception as e:
        logging.error(f"初始化服务器时出错: {e}")
        raise


async def get_available_tools() -> List[Tool]:
    """获取所有可用的工具。"""
    all_tools = []
    for server_name, server_tools in tools_cache.items():
        all_tools.extend(server_tools)
    return all_tools


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """执行指定的工具。"""
    for server_name, server in servers.items():
        server_tools = tools_cache.get(server_name, [])
        if any(tool.name == tool_name for tool in server_tools):
            try:
                result = await server.execute_tool(tool_name, arguments)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"执行工具 {tool_name} 失败: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"未找到工具: {tool_name}")


@app.on_event("startup")
async def startup_event():
    """启动时初始化服务器。"""
    await initialize_servers()


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源。"""
    cleanup_tasks = []
    for server_name, server in servers.items():
        cleanup_tasks.append(asyncio.create_task(server.cleanup()))
    
    if cleanup_tasks:
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        except Exception as e:
            logging.warning(f"清理资源时出现警告: {e}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """处理兼容OpenAI的聊天完成请求。"""
    logging.info(f"收到聊天完成请求，使用模型: {request.model}")
    
    try:
        # 初始化LLM客户端
        llm_client = LLMClient(
            api_key=config.llm_api_key, 
            base_url=config.base_url,
            default_model=request.model
        )
        
        # 使用我们内部的消息格式
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.messages
        ]
        
        if not messages:
            raise HTTPException(status_code=400, detail="消息列表不能为空")
        
        # 检查工具调用意图
        if "tool" in messages[-1]["content"].lower() or "function" in messages[-1]["content"].lower():
            # 可能是工具调用请求，尝试解析工具调用
            try:
                # 尝试提取工具调用的JSON
                content = messages[-1]["content"]
                tool_call = None
                
                # 尝试解析完整的JSON
                try:
                    tool_call = json.loads(content)
                except json.JSONDecodeError:
                    # 尝试在文本中查找JSON对象
                    import re
                    json_pattern = r'({[\s\S]*?})'
                    json_matches = re.findall(json_pattern, content)
                    for potential_json in json_matches:
                        try:
                            potential_tool = json.loads(potential_json)
                            if "tool" in potential_tool and "arguments" in potential_tool:
                                tool_call = potential_tool
                                break
                        except json.JSONDecodeError:
                            continue
                
                if tool_call and "tool" in tool_call and "arguments" in tool_call:
                    logging.info(f"识别到工具调用请求: {tool_call['tool']}")
                    # 执行工具
                    result = await execute_tool(tool_call["tool"], tool_call["arguments"])
                    
                    # 添加工具执行结果作为系统消息
                    system_msg = {"role": "system", "content": f"工具执行结果: {result}"}
                    messages.append(system_msg)
            except Exception as e:
                logging.error(f"处理工具调用时出错: {e}")
                # 继续常规流程，不中断
        
        # 获取LLM响应
        logging.debug(f"向LLM发送消息: {messages}")
        llm_response = llm_client.get_response(messages)
        
        if llm_response.startswith("Error:"):
            raise HTTPException(status_code=500, detail=llm_response)
            
        # 创建OpenAI格式的响应
        import time
        import uuid
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=llm_response),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": sum(len(msg["content"]) // 4 for msg in messages),  # 粗略估计
                "completion_tokens": len(llm_response) // 4,  # 粗略估计
                "total_tokens": sum(len(msg["content"]) // 4 for msg in messages) + len(llm_response) // 4
            }
        )
        
        return response
        
    except Exception as e:
        logging.error(f"处理聊天完成请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """获取文本嵌入向量。"""
    logging.info(f"收到嵌入请求，使用模型: {request.model}")
    
    try:
        # 初始化LLM客户端 (此处应该使用专门的嵌入服务，当前用占位符)
        input_texts = [request.input] if isinstance(request.input, str) else request.input
        
        # 目前我们不实际生成嵌入，返回随机值作为示例
        # 在实际实现中，应该调用适当的嵌入服务
        import random
        
        embeddings = []
        for i, text in enumerate(input_texts):
            # 生成随机向量作为示例（实际应用中应替换为真实的嵌入）
            vector = [random.uniform(-1, 1) for _ in range(1536)]  # 1536维是OpenAI嵌入的典型维度
            embeddings.append(
                EmbeddingObject(
                    embedding=vector,
                    index=i,
                    object="embedding"
                )
            )
        
        response = EmbeddingResponse(
            data=embeddings,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(text) // 4 for text in input_texts),
                "total_tokens": sum(len(text) // 4 for text in input_texts)
            }
        )
        
        return response
        
    except Exception as e:
        logging.error(f"处理嵌入请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools")
async def execute_tool_endpoint(request: ToolRequestBody):
    """执行指定的工具。"""
    logging.info(f"收到工具执行请求: {request.tool}")
    
    try:
        result = await execute_tool(request.tool, request.arguments)
        return {"result": result}
    except Exception as e:
        logging.error(f"执行工具时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/tools")
async def list_tools():
    """列出所有可用的工具。"""
    tools = await get_available_tools()
    
    # 转换为OpenAI函数格式
    openai_functions = []
    for tool in tools:
        openai_functions.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        })
    
    return {"functions": openai_functions}


if __name__ == "__main__":
    # 运行FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8000) 