import requests
import json
import time
import sseclient  # 添加对SSE客户端的支持

# API服务器地址
API_URL = "http://localhost:8000"

def test_list_models():
    """
    测试获取模型列表
    """
    url = f"{API_URL}/v1/models"
    
    print(f"获取模型列表: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        models = response.json()
        print(f"可用模型: {json.dumps(models, indent=2, ensure_ascii=False)}")
        return models
    
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return None

def test_chat_completion(messages, model="gpt-3.5-turbo"):
    """
    测试聊天补全功能
    
    参数:
        messages: 消息列表，格式为[{"role": "user", "content": "消息内容"}, ...]
        model: 使用的模型名称
    """
    url = f"{API_URL}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096
    }
    
    print(f"测试聊天补全: {url}")
    print(f"请求内容: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        print(f"响应时间: {elapsed:.2f}秒")
        print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 提取响应中的文本内容
        if result.get("choices") and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"\n回复内容: {content}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return None

def test_chat_completion_stream(messages, model="gpt-3.5-turbo"):
    """
    测试流式聊天补全功能
    
    参数:
        messages: 消息列表，格式为[{"role": "user", "content": "消息内容"}, ...]
        model: 使用的模型名称
    """
    url = f"{API_URL}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": True  # 启用流式响应
    }
    
    print(f"测试流式聊天补全: {url}")
    print(f"请求内容: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        start_time = time.time()
        # 使用stream=True参数使requests支持流式响应
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        # 使用SSE客户端解析事件流
        client = sseclient.SSEClient(response)
        
        print("\n开始接收流式响应:")
        full_content = ""
        
        # 逐个处理事件
        for event in client.events():
            if event.data == "[DONE]":
                break
                
            try:
                chunk = json.loads(event.data)
                
                # 提取增量内容
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    
                    # 处理结束标志
                    if chunk["choices"][0].get("finish_reason") is not None:
                        continue
                    
                    # 处理内容增量
                    content_delta = delta.get("content", "")
                    if content_delta:
                        full_content += content_delta
                        print(content_delta, end="", flush=True)
            except json.JSONDecodeError:
                print(f"无法解析事件数据: {event.data}")
                
        elapsed = time.time() - start_time
        print(f"\n\n响应完成! 总耗时: {elapsed:.2f}秒")
        
        return full_content
    
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return None

def interactive_chat(stream=False):
    """
    交互式聊天测试
    
    参数:
        stream: 是否使用流式响应
    """
    print("OpenAI API兼容测试客户端")
    print(f"当前模式: {'流式响应' if stream else '普通响应'}")
    print("输入'切换'可以在流式和普通响应之间切换")
    print("输入'退出'或'exit'结束对话")
    
    messages = []
    current_stream = stream
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ["退出", "exit"]:
            print("结束对话")
            break
            
        if user_input.lower() in ["切换", "toggle", "switch"]:
            current_stream = not current_stream
            print(f"已切换到{'流式响应' if current_stream else '普通响应'}模式")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("正在等待回复...")
        
        if current_stream:
            assistant_content = test_chat_completion_stream(messages)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
        else:
            result = test_chat_completion(messages)
            if result and result.get("choices") and len(result["choices"]) > 0:
                assistant_message = result["choices"][0]["message"]
                messages.append(assistant_message)

def main():
    """
    主函数，运行所有测试
    """
    print("==== 测试获取模型列表 ====")
    test_list_models()
    
    print("\n==== 测试单个问题（普通响应）====")
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，介绍一下自己。"}
    ]
    test_chat_completion(messages)
    
    print("\n==== 测试单个问题（流式响应）====")
    test_chat_completion_stream(messages)
    
    print("\n==== 交互式聊天测试 ====")
    interactive_chat(stream=True)  # 默认使用流式响应

if __name__ == "__main__":
    main() 