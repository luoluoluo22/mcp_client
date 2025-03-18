import requests
import json

# API服务器地址
API_URL = "http://localhost:8000"

def chat_with_assistant(message, session_id=None):
    """
    与助手聊天
    
    参数:
        message: 发送给助手的消息
        session_id: 会话ID，用于保持对话上下文。如果为None，服务器会创建新会话
        
    返回:
        tuple: (回复内容, 新的会话ID)
    """
    url = f"{API_URL}/chat"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "message": message
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    print(f"发送请求: {payload}")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 如果响应状态码不是200，抛出异常
        
        result = response.json()
        return result["response"], result["session_id"]
    
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return f"错误: {str(e)}", session_id
    except json.JSONDecodeError:
        print(f"解析响应JSON出错: {response.text}")
        return "服务器返回了无效的JSON响应", session_id

def main():
    print("MCP聊天API客户端示例")
    print("输入'退出'或'exit'结束对话")
    
    session_id = None  # 初始没有会话ID
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ["退出", "exit"]:
            print("结束对话")
            break
        
        print("正在等待回复...")
        response, session_id = chat_with_assistant(user_input, session_id)
        
        print(f"\n助手: {response}")
        print(f"(会话ID: {session_id})")

if __name__ == "__main__":
    main() 