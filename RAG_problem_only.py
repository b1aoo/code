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
from llm_func import grade_essay_test, similar_grade_content,similar_grade_structure,similar_grade_language,join_result,grade_essay_origin,extract_problem_types,grade_essay_origin_content,grade_essay_origin_structure,grade_essay_origin_language,join_result_new,select_top_3
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random

# ===================== 可配置变量 =====================
PARALLEL_NUM = 10  # 并行处理数量，可根据需求调整（如5、10、15）
START_NUM = 1      # 起始作文编号
END_NUM = 50    # 结束作文编号
# ======================================================

def grade_essay_no_change(title,essay_content,sample, DASHSCOPE_API_KEY,rule,knowledge,model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        
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
注意：以上的作文内容是唯一的评价对象，所有评论只能以它为基础，不能将下面的参考样例当作评价对象。

参考样例：
{sample}
注意：参考样例仅供学习如何评价相似作文与对应格式，其中的评论不能直接照搬，比如五篇样例均提到紧扣主题，但待批改的作文可能没有完全符合主题，所以评论中不能提到紧扣主题。
"""

        
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
        }
        
        data = {
            "model": model_name,

            "input": {"prompt": prompt},
            "parameters": {"temperature": 0.7, "max_tokens": 8000}
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=300)
        response_data = response.json()
        # 打印调试信息（可保留用于后续问题排查）
        print(f"API响应状态码: {response.status_code}")
        #print("API返回内容: ",response_data["output"]["choices"][0]["message"]["content"])
        
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                
                # 直接检查是否存在output.text字段
                if "output" in response_data and "choices" in response_data["output"] and "message" in response_data["output"]["choices"][0] and "content" in response_data["output"]["choices"][0]["message"]:
                    return response_data["output"]["choices"][0]["message"]["content"]
                else:
                    return f"API返回格式异常：缺少output.choices.message.content字段，响应内容：{response.text}"
            except json.JSONDecodeError:
                return f"API返回内容不是有效的JSON：{response.text}"
        else:
            return f"API请求失败，状态码：{response.status_code}，响应内容：{response.text}"
    
    except requests.exceptions.Timeout:
        return "错误：API请求超时，请检查网络"
    except requests.exceptions.ConnectionError:
        return "错误：网络连接失败，请检查网络设置"
    except Exception as e:
        return f"处理过程中出错：{str(e)}"
# 通过个人访问令牌或oauth获取access_token。
coze_api_token = os.environ["COZE_API_TOKEN"]
coze_api_base = COZE_CN_BASE_URL
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)
workflow_id = '7595041545120792611'
load_dotenv()
es_client = Elasticsearch(
    "http://localhost:9200",  # 强制 https 协议
)
try:
    es_client.info()
    print("Elasticsearch连接成功")
except Exception as e:
    print(f"连接失败：{e}")
    exit()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")




# 创建独立的文件锁（为不同类型的文件操作创建专用锁）
query_file_lock = threading.Lock()      # 查询文件操作锁
knn_file_lock = threading.Lock()        # KNN结果文件锁
output_file_lock = threading.Lock()     # 输出文件锁
error_log_lock = threading.Lock()       # 错误日志锁

def generate_text_embedding(text):
    """生成文本的向量表示"""
    embedding = model.encode(text)
    return embedding.tolist()

def process_essay(input_num, root_dir, output_dir):
    """处理单篇作文的函数（线程安全）"""
    try:
        print(f"开始处理作文 {input_num}...")
        
        # 1. 处理输入并获取正确的作文内容
        title, content, standard = input_deal(input_num)
        
        # 使用实际作文内容作为搜索输入
        user_input = title+"\n"+content
        
        # 添加调试信息
        print(f"作文 {input_num} - 标题: {title[:50]}...")
        print(f"作文 {input_num} - 内容长度: {len(content)} 字符")
        print(f"作文 {input_num} - 内容预览: {content[:100]}...")
        
        # 2. 写入查询文件（使用专用锁）
        with query_file_lock:
            write_to_query_file(user_input, model, input_num)
            
            # 读取查询内容
            query_file_path = rf"D:\zuowen\input\search_query\KNN\input\knn {input_num}.txt"
            if not os.path.exists(query_file_path):
                raise FileNotFoundError(f"查询文件不存在: {query_file_path}")
                
            with open(query_file_path, 'r', encoding='utf-8') as file:
                search_query_str = file.read().rstrip()
            
            if not search_query_str:
                raise ValueError(f"查询文件内容为空: {query_file_path}")
            
            search_query = json.loads(search_query_str)
        
        # 3. KNN搜索（无文件操作，不需要锁）
        index_name_knn = "research_index_knn"
                 # =======================
        # 3. 新增：问题类型抽取
        # =======================
        problem_info = extract_problem_types(user_input, DASHSCOPE_API_KEY)
        problem_types = problem_info["problem_types"]
        problem_summary = problem_info["summary"]

        print("【问题类型】", problem_types)
        print("【问题摘要】", problem_summary)

        # =======================
        # 4. 问题型检索 Query
        # =======================
        problem_search_query = build_problem_knn_query(
            problem_types=problem_types,
            summary=problem_summary,
            k=30
        )

        problem_docs = processESIndex_Knn1(
            problem_search_query,
            index_name_knn,
            es_client
        )
        # 4. 写入KNN结果（使用专用锁）
        knn_output_path_1 = rf"D:\zuowen\output\KNNS\knn_1_{input_num}.txt"


      



        
        with knn_file_lock:
            with open(knn_output_path_1, 'w', encoding='utf-8') as output_knn:
                print(f"作文 {input_num} - ---------------------------------------")
                count = 0
                tcnt=0
                for i in problem_docs:
                    count += 1
                    if count == 1 or i['_source']['id'] == input_num:
                        continue
                    tcnt+=1
                    if tcnt>10:
                        break
                    print(f"作文 {input_num} - {i['_source']['id']}    {i['_score']}")
                    output_knn.write('标题：'+str(i['_source']['title']) + '\n' + '内容：'+str(i['_source']['text']) + '\n' + '标准评论：'+str(i['_source']['comment']))
                    output_knn.write("----------------------------" * 20 + "\n")
              
        
        print(f"作文 {input_num} - knn.txt saved successfully")
        
        # 5. 读取KNN结果（使用专用锁）
        with knn_file_lock:
            with open(knn_output_path_1, 'r', encoding='utf-8') as file:
                first_samples = file.read().rstrip()

        
        # 6. 模型批改（无文件操作，不需要锁）
        print(f"作文 {input_num} - 正在使用deepseek-v3.1模型批改作文，请稍候...(RAG)\n")
        rule_path = r"D:\zuowen\data\eval\rule.txt"

        with open(rule_path, 'r', encoding='utf-8') as file:
            rule = file.read().rstrip()


        result_none = grade_essay_no_change(title, content, first_samples, DASHSCOPE_API_KEY, rule, None,"deepseek-v3.2")



        
        # 7. 保存结果（使用专用锁）
        with output_file_lock:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存各个文件
            with open(output_dir + rf"\20250600{input_num}-at.txt", 'w', encoding='utf-8') as f:
                f.write(result_none)
            with open(output_dir + rf"\20250600{input_num}-q.txt", 'w', encoding='utf-8') as f:
                f.write(title)
            with open(output_dir + rf"\20250600{input_num}-s.txt", 'w', encoding='utf-8') as f:
                f.write(content)
            with open(output_dir + rf"\20250600{input_num}-ta.txt", 'w', encoding='utf-8') as f:
                f.write(standard)
        
        print(f"作文 {input_num} - 处理完成！")
        return input_num, True, "处理成功"
    
    except Exception as e:
        error_msg = f"作文 {input_num} - error: {str(e)}"
        print(error_msg)
        import traceback
        error_detail = traceback.format_exc()
        print(f"详细错误信息: {error_detail}")
        
        # 记录错误日志（使用专用锁）
        with error_log_lock:
            error_log = os.path.join(output_dir, "error_log.txt")
            with open(error_log, 'a', encoding='utf-8') as f:
                f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 作文 {input_num} 错误:\n")
                f.write(f"错误信息: {str(e)}\n")
                f.write(f"详细堆栈: {error_detail}\n")
                f.write("="*80 + "\n")
        
        return input_num, False, error_msg

if __name__ == "__main__":
    # 修改路径配置（移除 file_path）
    root_dir = r'D:\zuowen\data\RAG_problem_only'
    output_dir = r"D:\zuowen\data\eval\temp\RAG_problem_only"  # 输出目录
    
    # 创建输出目录（确保存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印配置信息
    print("=" * 60)
    print(f"并行处理配置：")
    print(f"并行数量：{PARALLEL_NUM}")
    print(f"处理范围：作文 {START_NUM} - {END_NUM}")
    print(f"输出目录：{output_dir}")
    print("=" * 60)
    
    # 记录开始时间
    start_time = time.time()
    
    # 验证处理范围
    if START_NUM > END_NUM:
        print("错误：起始编号大于结束编号！")
        exit()
    
    # 存储结果
    results = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=PARALLEL_NUM) as executor:
        # 提交所有任务（移除 file_path 参数）
        future_to_num = {
            executor.submit(process_essay, num, root_dir, output_dir): num
            for num in range(START_NUM, END_NUM + 1)
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_num):
            results.append(future.result())
    
    # 统计结果
    total = len(results)
    success = sum(1 for r in results if r[1])
    failed = total - success
    total_time = time.time() - start_time
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("并行处理完成！")
    print(f"总耗时：{total_time:.2f} 秒")
    print(f"总任务数：{total}")
    print(f"成功：{success} 篇")
    print(f"失败：{failed} 篇")
    print("=" * 60)
    
    # 输出失败详情
    if failed > 0:
        print("\n失败详情：")
        for r in results:
            if not r[1]:
                print(r[2])
        
        # 提示查看错误日志
        error_log_path = os.path.join(output_dir, "error_log.txt")
        print(f"\n详细错误日志已保存至：{error_log_path}")