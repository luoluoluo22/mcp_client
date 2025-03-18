import asyncio
import json
import logging
import os
from openai import OpenAI, AsyncOpenAI
import requests
from dotenv import load_dotenv
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载环境变量
load_dotenv()

# 获取OpenAI配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")  # 默认连接到本地服务器
model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

async def test_chat_completion():
    """测试聊天完成API，使用OpenAI官方库"""
    # 配置OpenAI客户端
    client = AsyncOpenAI(
        api_key="sk-ywhprjugjlrzlmlfijphufltvjtokechtglwiigktcyklicz",  # 可以设置为任意值，实际使用的是我们自己的服务器
        base_url="http://localhost:8000/v1",  # 指向我们的本地API服务器
    )
    
    logging.info("使用OpenAI客户端发送聊天完成请求...")
    
    try:
        # 使用OpenAI官方库发送请求
        completion = await client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # 使用与服务端配置匹配的模型
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "介绍一下Python语言的主要特点。"}
            ],
            temperature=0.7,
            timeout=60.0  # 设置超时时间为60秒
        )
        
        # 提取助手的回复
        assistant_reply = completion.choices[0].message.content
        
        # 输出完整的响应对象
        logging.info(f"API响应: {completion.model_dump_json(indent=2)}")
        
        # 输出助手回复
        print(f"\n助手回复:\n{assistant_reply}\n")
        
    except Exception as e:
        logging.error(f"请求异常: {type(e).__name__}: {str(e)}")


async def test_streaming_chat_completion():
    """测试流式聊天完成API，使用OpenAI官方库"""
    # 配置OpenAI客户端
    client = AsyncOpenAI(
        api_key="sk-ywhprjugjlrzlmlfijphufltvjtokechtglwiigktcyklicz",  # 可以设置为任意值，实际使用的是我们自己的服务器
        base_url="http://localhost:8000/v1",  # 指向我们的本地API服务器
    )
    
    logging.info("使用OpenAI客户端发送流式聊天完成请求...")
    
    try:
        # 使用OpenAI官方库发送流式请求
        stream = await client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # 使用与服务端配置匹配的模型
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "用一段简短的文字介绍一下中国。"}
            ],
            temperature=0.7,
            stream=True,  # 启用流式响应
            timeout=60.0  # 设置超时时间为60秒
        )
        
        print("\n流式响应开始接收...")
        full_response = ""
        
        # 处理流式响应
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_delta = chunk.choices[0].delta.content
                print(content_delta, end="", flush=True)
                full_response += content_delta
        
        print("\n\n流式响应完成，总共收到字符数:", len(full_response))
        
    except Exception as e:
        logging.error(f"流式请求异常: {type(e).__name__}: {str(e)}")


async def test_tools_list():
    """测试工具列表API，使用OpenAI官方库"""
    # 配置OpenAI客户端
    client = AsyncOpenAI(
        api_key="sk-ywhprjugjlrzlmlfijphufltvjtokechtglwiigktcyklicz",
        base_url="http://localhost:8000/v1",
    )
    
    logging.info("获取可用工具列表...")
    
    try:
        # 由于OpenAI官方库没有直接获取工具列表的方法，我们使用自定义请求
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get("http://localhost:8000/v1/tools")
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"发现 {len(result['functions'])} 个可用工具")
                
                for i, function in enumerate(result['functions']):
                    print(f"\n工具 {i+1}: {function['function']['name']}")
                    print(f"描述: {function['function']['description']}")
                    print("-" * 50)
            else:
                logging.error(f"请求失败: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"请求异常: {type(e).__name__}: {str(e)}")


async def test_tool_execution():
    """测试工具执行API，结合OpenAI官方库和自定义请求"""
    # 配置OpenAI客户端
    client = AsyncOpenAI(
        api_key="sk-ywhprjugjlrzlmlfijphufltvjtokechtglwiigktcyklicz",
        base_url="http://localhost:8000/v1",
    )
    
    # 首先获取工具列表
    import httpx
    timeout = httpx.Timeout(30.0)
    example_tool = None
    example_arguments = {}
    
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        try:
            response = await http_client.get("http://localhost:8000/v1/tools")
            
            if response.status_code == 200:
                result = response.json()
                if result['functions']:
                    # 选择第一个工具作为示例
                    example_tool = result['functions'][0]['function']['name']
                    
                    # 准备简单的参数
                    parameters = result['functions'][0]['function']['parameters']
                    if 'properties' in parameters:
                        for param_name, param_info in parameters['properties'].items():
                            # 根据参数类型提供示例值
                            if param_info.get('type') == 'string':
                                example_arguments[param_name] = "示例文本"
                            elif param_info.get('type') == 'number' or param_info.get('type') == 'integer':
                                example_arguments[param_name] = 1
                            elif param_info.get('type') == 'boolean':
                                example_arguments[param_name] = True
        except Exception as e:
            logging.error(f"获取工具列表失败: {type(e).__name__}: {str(e)}")
            return
        
    if not example_tool:
        logging.error("没有找到可用的工具")
        return
    
    # 执行工具（通过自定义端点）
    logging.info(f"执行工具 {example_tool}...")
    logging.info(f"参数: {example_arguments}")
    
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        try:
            response = await http_client.post(
                "http://localhost:8000/v1/tools",
                json={
                    "tool": example_tool,
                    "arguments": example_arguments
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"工具执行结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            else:
                logging.error(f"请求失败: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"工具执行失败: {type(e).__name__}: {str(e)}")


async def test_tool_via_openai_chat():
    """使用OpenAI聊天接口和函数调用特性执行工具"""
    # 配置OpenAI客户端
    client = AsyncOpenAI(
        api_key="sk-ywhprjugjlrzlmlfijphufltvjtokechtglwiigktcyklicz",
        base_url="http://localhost:8000/v1",
    )
    
    logging.info("使用OpenAI函数调用特性执行工具...")
    
    # 获取可用工具
    import httpx
    tools = []
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            response = await http_client.get("http://localhost:8000/v1/tools")
            if response.status_code == 200:
                result = response.json()
                tools = result['functions']
                logging.info(f"获取到 {len(tools)} 个工具")
        except Exception as e:
            logging.error(f"获取工具列表失败: {e}")
            return
    
    if not tools:
        logging.error("没有找到可用的工具")
        return
    
    # 选择第一个工具用于测试
    test_tool = tools[0]
    logging.info(f"将使用工具测试: {test_tool['function']['name']}")
    
    try:
        # 创建一个引导模型使用工具的聊天请求
        completion = await client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct", 
            messages=[
                {"role": "system", "content": "你是一个有用的助手。请使用适当的工具来回答问题。"},
                {"role": "user", "content": f"请使用 {test_tool['function']['name']} 工具，随便填写一些合适的参数来展示它的功能。"}
            ],
            tools=tools,  # 传递工具定义
            temperature=0.7,
            timeout=60.0
        )
        
        response_message = completion.choices[0].message
        
        # 检查是否有工具调用
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            logging.info(f"模型请求调用工具: {response_message.tool_calls}")
            print(f"\n模型建议的工具调用:\n{response_message.content}\n")
            
            # 在实际应用中，这里会解析tool_calls并执行工具
            # 由于我们已经有了单独的工具执行测试，这里仅作演示
        else:
            print(f"\n模型响应 (没有工具调用):\n{response_message.content}\n")
            
    except Exception as e:
        logging.error(f"工具调用测试失败: {type(e).__name__}: {str(e)}")


async def main():
    """运行所有测试用例"""
    print("=" * 80)
    print("测试OpenAI兼容API（使用OpenAI官方库）")
    print("=" * 80)
    
    try:
        print("\n1. 测试聊天完成API")
        await test_chat_completion()
        
        print("\n2. 测试流式聊天完成API")
        await test_streaming_chat_completion()
        
        print("\n3. 测试工具列表API")
        await test_tools_list()
        
        print("\n4. 测试工具执行API")
        await test_tool_execution()
        
        print("\n5. 测试通过OpenAI函数调用执行工具")
        await test_tool_via_openai_chat()
        
    except Exception as e:
        logging.error(f"测试过程中出错: {e}", exc_info=True)
    
    print("\n测试完成!")


if __name__ == "__main__":
    # 默认使用流式响应，除非指定--no-stream参数
    use_stream = "--no-stream" not in sys.argv
    asyncio.run(main()) 