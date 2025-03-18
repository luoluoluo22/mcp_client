import requests
import json
import time

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

def interactive_chat():
    """
    交互式聊天测试
    """
    print("OpenAI API兼容测试客户端")
    print("输入'退出'或'exit'结束对话")
    
    messages = []
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ["退出", "exit"]:
            print("结束对话")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        print("正在等待回复...")
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
    
    print("\n==== 测试单个问题 ====")
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，介绍一下自己。"}
    ]
    test_chat_completion(messages)
    
    print("\n==== 交互式聊天测试 ====")
    interactive_chat()

if __name__ == "__main__":
    main() 