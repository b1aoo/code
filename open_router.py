import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
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
from data_func import write_to_query_file, input_deal
from search_func import processESIndex_Knn,build_problem_knn_query,processESIndex_Knn1,get_sorted_indices
# 删除所有 dashscope 相关导入
from llm_func import grade_essay_test, similar_grade_content,similar_grade_structure,similar_grade_language,join_result,grade_essay_origin,extract_problem_types,grade_essay_origin_content,grade_essay_origin_structure,grade_essay_origin_language,join_result_new,select_top_3
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random

# ===================== 作文批改函数（已修复 403 + SSL 错误） =====================
from openai import OpenAI

def grade_essay_no_change(title, essay_content, sample, model_name):
    try:
        prompt = f"""
请批改以下作文，并从以下几个方面给出详细反馈：
格式要求：先分三个大类，一到三：
一、内容和审题
二、文章结构
三、语言表达
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个维度的每种评价中的每个小点前要加上“1.” 依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”

作文内容：
{essay_content}
"""

        # 修复：稳定可用的 OpenRouter 配置 + 超时设置
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=120,
            default_headers={
                "HTTP-Referer": "https://localhost",
                "X-OpenRouter-Title": "EssayGrader",
            }
        )

        # 稳定调用
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=6000
        )

        if not completion or not completion.choices:
            return "模型返回为空"
        
        message = completion.choices[0].message
        if not message or not message.content:
            return "模型未返回有效内容"

        return message.content.strip()

    except Exception as e:
        return f"异常：{str(e)}"

# ===================== 可配置变量（修复：使用无区域限制的模型） =====================
PARALLEL_NUM = 10
START_NUM = 1
END_NUM = 50

LLM_MODEL = "meta-llama/llama-3.1-70b-instruct"
# ======================================================

coze_api_token = os.environ["COZE_API_TOKEN"]
coze_api_base = COZE_CN_BASE_URL
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)
workflow_id = '7595041545120792611'
load_dotenv()
es_client = Elasticsearch("http://localhost:9200")
try:
    es_client.info()
    print("Elasticsearch连接成功")
except Exception as e:
    print(f"连接失败：{e}")
    exit()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 锁
query_file_lock = threading.Lock()
knn_file_lock = threading.Lock()
output_file_lock = threading.Lock()
error_log_lock = threading.Lock()

def generate_text_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()

def process_essay(input_num, root_dir, output_dir):
    try:
        print(f"开始处理作文 {input_num}...")
        title, content, standard = input_deal(input_num)
        user_input = title+"\n"+content

        with query_file_lock:
            write_to_query_file(user_input, model, input_num)
            query_file_path = rf"D:\zuowen\input\search_query\KNN\input\knn {input_num}.txt"
            with open(query_file_path, 'r', encoding='utf-8') as file:
                search_query_str = file.read().rstrip()
            search_query = json.loads(search_query_str)

        index_name_knn = "research_index_knn"
        docs = processESIndex_Knn(search_query, index_name_knn, es_client)


        knn_output_path_1 = rf"D:\zuowen\output\KNNS\knn_1_{input_num}.txt"
        with knn_file_lock:
            with open(knn_output_path_1, 'w', encoding='utf-8') as output_knn:
                tcnt = 0
                for i in docs:
                    if tcnt >= 6: break
                    if i['_source']['id'] == input_num: continue
                    output_knn.write('标题：'+str(i['_source']['title'])+'\n内容：'+str(i['_source']['text'])+'\n标准评论：'+str(i['_source']['comment'])+'\n')
                    output_knn.write("-"*60+"\n")
                    tcnt += 1

        with knn_file_lock:
            with open(knn_output_path_1, 'r', encoding='utf-8') as file:
                first_samples = file.read()
        
        print(f"作文 {input_num} - 正在使用 {LLM_MODEL} 模型批改...")
        rule_path = r"D:\zuowen\data\eval\rule.txt"
        with open(rule_path, 'r', encoding='utf-8') as f:
            rule = f.read()

        # 修复：调用简化版函数
        result_none = grade_essay_no_change(title, content, first_samples, LLM_MODEL)
        if result_none in ["模型返回为空", "模型未返回有效内容", ""]:
            time.sleep(1)
            result_none = grade_essay_no_change(title, content, first_samples, LLM_MODEL)

        with output_file_lock:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_dir + rf"\20250600{input_num}-at.txt", 'w', encoding='utf-8') as f:
                f.write(result_none)
            with open(output_dir + rf"\20250600{input_num}-q.txt", 'w', encoding='utf-8') as f:
                f.write(title)
            with open(output_dir + rf"\20250600{input_num}-s.txt", 'w', encoding='utf-8') as f:
                f.write(content)
            with open(output_dir + rf"\20250600{input_num}-ta.txt", 'w', encoding='utf-8') as f:
                f.write(standard)

        print(f"作文 {input_num} - 处理完成！")
        return input_num, True, "成功"

    except Exception as e:
        error_msg = f"作文 {input_num} 错误：{str(e)}"
        print(error_msg)
        with error_log_lock:
            with open(os.path.join(output_dir, "error_log.txt"), 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now()} {error_msg}\n")
        return input_num, False, error_msg

if __name__ == "__main__":
    root_dir = r'D:\zuowen\data\EDPO1'
    output_dir = r"D:\zuowen\data\eval\temp\\"+LLM_MODEL
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=PARALLEL_NUM) as executor:
        futures = [executor.submit(process_essay, num, root_dir, output_dir) for num in range(START_NUM, END_NUM+1)]
        results = [f.result() for f in futures]

    total = len(results)
    success = sum(1 for r in results if r[1])
    print(f"\n完成！总耗时：{time.time()-start_time:.2f}s | 成功：{success} | 失败：{total-success}")