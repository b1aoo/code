import os
import requests
import json
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer 
import re
from datetime import date, datetime
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
import random
from cozepy import Coze, TokenAuth, Stream, WorkflowEvent, WorkflowEventType
from cozepy import COZE_CN_BASE_URL
from data_func import input_deal
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加载环境变量（必须加这行）
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

def json_generate(text):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    try:
        # 构建提示词：把 { 换成 {{，把 } 换成 }}，仅此修复！
        prompt =f"""请严格按以下JSON格式输出批改结果，不添加任何额外文本：
json格式：

{{
  "一、内容和审题": {{
    "优点": ["1. 具体点评内容", "2. 具体点评内容", "3. 具体点评内容"],
    "不足": ["1. 具体点评内容", "2. 具体点评内容"],
    "建议": ["1. 具体建议内容", "2. 具体建议内容", "3. 具体建议内容"]
  }},
  "二、文章结构": {{
    "优点": ["1. ..."],
    "不足": ["1. ..."],
    "建议": ["1. ..."]
  }},
  "三、语言表达": {{
    "优点": ["1. ..."],
    "不足": ["1. ..."],
    "建议": ["1. ..."]
  }}
}}

批改文本：
{text}

格式规则：
1. 三大类名称必须使用中文全称："一、内容和审题"、"二、文章结构"、"三、语言表达"
2. 每个大类下可包含"优点"、"不足"、"建议"三个字段，若某字段无内容则省略该字段（不输出"无"或空数组）
3. 每个字段的值必须是字符串数组，数组元素格式为"序号. 具体内容"（如"1. 开头用'清晨的阳光'营造画面感，很吸引人"）
4. 仅返回纯JSON，无任何前缀、后缀或解释性文字
5. 若存在特殊符号如(*),请删去该符号

**重要提示** 如果要点内需要引用内容，请使用直角引号「」；对于要点内需要使用中文符号；不要对输出内容进行任何转义

需要批改的作文内容：
{text}
            """
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
        }
        
        data = {
            "model": "deepseek-v3.1",
            "input": {"prompt": prompt},
            "parameters": {"temperature": 0.7, "max_tokens": 8000}
        }
        response = requests.post(url, headers=headers, json=data, timeout=300)
        response_data = response.json()
        # 打印调试信息（可保留用于后续问题排查）
        print(f"API响应状态码: {response.status_code}")
        if "output" in response_data and "choices" in response_data["output"]:
            final_result = response_data["output"]["choices"][0]["message"]["content"]
            return final_result
        else:
            return f"错误：API返回格式异常 {response_data}"
    
    except requests.exceptions.Timeout:
        return "错误：API请求超时，请检查网络"
    except requests.exceptions.ConnectionError:
        return "错误：网络连接失败，请检查网络设置"
    except Exception as e:
        return f"处理过程中出错：{str(e)}"
    
if __name__ == "__main__":
    cat = "RAG_dual_stage"
    root_dir = r'D:\zuowen\data\eval\temp'
    behind = '-at.txt'
    for i in range(1,51):

        # 文件名完全保留你原来的格式：20250600 + i + -at.txt
        file_name = "20250600" + str(i) + behind
        # 用os.path.join 自动拼路径，不会出现双斜杠报错
        file_path = os.path.join(root_dir, cat, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                json_result = json_generate(content)
                print(json_result)

                # ===================== 保存 JSON 文件 =====================
                # 自动把 .txt 换成 .json
                json_file_name = file_name.replace(".txt", ".json")
                json_file_path = os.path.join(root_dir, cat, json_file_name)

                # 写入保存
                with open(json_file_path, 'w', encoding='utf-8') as f_json:
                    f_json.write(json_result)

                print(f"✅ 已保存：{json_file_path}")
                # =========================================================

        except FileNotFoundError:
            print(f"文件不存在：{file_path}")