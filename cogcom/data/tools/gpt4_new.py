import os, sys
sys.path.append(os.path.dirname(__file__))
import argparse
from openai import OpenAI  # 更新导入方式
from utils.template_util import *

# 初始化 client
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_API_BASE')  # 如果使用自定义 API 地址
)

class GPT4PI:
    def __init__(self) -> None:
        self.client = client
    
    def get_response(self, prompt):
        if not prompt:
            return -1, "Prompt cannot be empty", 0
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 更新为实际使用的模型
                messages=[{"role": "user", "content": prompt}]
            )
            
            if response.choices:
                # 获取使用的 tokens
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
                return 200, response.choices[0].message.content, tokens_used
            return -1, "No response generated", 0
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return -1, str(e), 0

if __name__ == '__main__':
    default_prompt = get_prompt(question="How old is the man standing behind the black table?", shot=7)
    print(default_prompt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default=default_prompt)
    args = parser.parse_args()

    chat_api = GPT4PI()
    status, response, tokens_used = chat_api.get_response(prompt=args.prompt)  # 修复参数传递
    print(f"Status: {status}")
    print(f"Response: {response}")
    print(f"Tokens used: {tokens_used}")