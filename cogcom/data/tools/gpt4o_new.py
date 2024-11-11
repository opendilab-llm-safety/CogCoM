import os
import sys
import base64
from typing import Optional, Tuple
from pathlib import Path
sys.path.append(os.path.dirname(__file__))

from openai import OpenAI
from utils.template_util import *

def encode_image_to_base64(image_path: str) -> str:
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class GPT4OInterface:
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None) -> None:
        """
        初始化GPT-4o接口
        :param api_key: OpenAI API key，如果为None则从环境变量OPENAI_API_KEY读取
        :param api_base: API基础URL，如果为None则从环境变量OPENAI_API_BASE读取
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            base_url=api_base or os.getenv('OPENAI_API_BASE')
        )

    def get_response(self, prompt: str, image_path: str) -> Tuple[int, Optional[str], int]:
        """
        获取GPT-4o的响应
        :param prompt: 提示文本
        :param image_path: 图片路径
        :return: (状态码, 响应文本, 使用的token数)
        """
        if not prompt or not image_path:
            return -1, "Prompt and image path cannot be empty", 0

        if not Path(image_path).exists():
            return -1, f"Image file not found: {image_path}", 0

        try:
            # 将图片转换为base64
            base64_image = encode_image_to_base64(image_path)

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # 调用API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 使用GPT-4o模型
                messages=messages,
                max_tokens=2000,  # 可根据需要调整
            )

            if response.choices:
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
                return 200, response.choices[0].message.content, tokens_used
            
            return -1, "No response generated", 0

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return -1, str(e), 0


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, 
                       default="What do you see in this image? Please describe in detail.")
    parser.add_argument('--image_path', type=str, required=True,
                       help="Path to the image file")
    args = parser.parse_args()

    # 初始化接口并调用
    gpt4o = GPT4OInterface()
    status, response, tokens_used = gpt4o.get_response(
        prompt=args.prompt,
        image_path=args.image_path
    )

    # 输出结果
    print(f"Status: {status}")
    print(f"Response: {response}")
    print(f"Tokens used: {tokens_used}")