import asyncio
import json
import httpx
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

async def test_chat_completion():
    """测试聊天完成API"""
    url = "http://localhost:8000/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "介绍一下Python语言的主要特点。"}
        ],
        "temperature": 0.7
    }
    
    logging.info("发送聊天完成请求...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"API响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 提取助手的回复
            assistant_reply = result['choices'][0]['message']['content']
            print(f"\n助手回复:\n{assistant_reply}\n")
        else:
            logging.error(f"请求失败: {response.status_code} - {response.text}")


async def test_tools_list():
    """测试工具列表API"""
    url = "http://localhost:8000/v1/tools"
    
    logging.info("获取可用工具列表...")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"发现 {len(result['functions'])} 个可用工具")
            
            for i, function in enumerate(result['functions']):
                print(f"\n工具 {i+1}: {function['function']['name']}")
                print(f"描述: {function['function']['description']}")
                print("-" * 50)
        else:
            logging.error(f"请求失败: {response.status_code} - {response.text}")


async def test_tool_execution():
    """测试工具执行API"""
    
    # 首先获取工具列表
    url = "http://localhost:8000/v1/tools"
    example_tool = None
    example_arguments = {}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
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
        
    if not example_tool:
        logging.error("没有找到可用的工具")
        return
    
    # 执行工具
    tool_url = "http://localhost:8000/v1/tools"
    
    payload = {
        "tool": example_tool,
        "arguments": example_arguments
    }
    
    logging.info(f"执行工具 {example_tool}...")
    logging.info(f"参数: {example_arguments}")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(tool_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"工具执行结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            logging.error(f"请求失败: {response.status_code} - {response.text}")


async def main():
    """运行所有测试用例"""
    print("=" * 80)
    print("测试OpenAI兼容API")
    print("=" * 80)
    
    try:
        print("\n1. 测试聊天完成API")
        await test_chat_completion()
        
        print("\n2. 测试工具列表API")
        await test_tools_list()
        
        print("\n3. 测试工具执行API")
        await test_tool_execution()
        
    except Exception as e:
        logging.error(f"测试过程中出错: {e}", exc_info=True)
    
    print("\n测试完成!")


if __name__ == "__main__":
    asyncio.run(main()) 