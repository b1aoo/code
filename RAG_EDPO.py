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


from openai import OpenAI
import os
import requests

from openai import OpenAI

def grade_essay_no_change(title, essay_content, sample, DASHSCOPE_API_KEY, rule, knowledge, model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        prompt = f"""请批改作文，并按要求给出评价，仅进行评价不修改作文、不创造作文，仅返回一遍评论无需总结，评论内容需丰富、详细、具体，全程基于作文文本本身独立诊断，核心提升批改的全面性、精准性与维度均衡性，助力提升总体匹配质量，要求如下：

评价要求(需逐条落实):
1. 核心问题定位：挖本质，拒表层
先拆解文本主题的核心内涵，明确文本偏离本质的具体表现（例：若文本写 “雕版印刷” 却侧重 “文化传承”，需指出 “偏离‘低谷中的坚持’主线，主题摇摆”）；
识别逻辑断裂的根源（如 “科技与人文” 类文本若仅提 “15 秒便民” 未关联 “铁路网建设”，需点明 “案例与主题的因果链缺失，未体现科技对人文的赋能逻辑”）。
2. 主题聚焦：锚核心，补关联
明确文本需紧扣的核心主题，排查 “内容堆砌” 问题（如记叙文若同时写 “数学、豫剧、体测”，需建议 “聚焦 1 件事，如‘体测突破’，删除无关事件”）；
挖掘题目隐含的潜在关联（如 “传统文化类文本” 需关联 “传统与现代 / 科技”，“时代主题类文本” 需关联 “当代青年视角”，例：“剪纸文化” 需补充 “科技助力剪纸创新 / 避免科技导致的文化浮躁”）。
3. 细节指导：具象化，给方案
针对 “细节缺失”，提供可落地的场景 / 动作 / 心理描写方向（例：写 “外婆做短视频” 需建议 “补充‘学剪辑时反复调整时间轴、向晚辈讨论文案措辞’的细节”；写 “引体向上成长” 需建议 “添加‘肌肉酸痛时握杆的指节发白、想放弃时盯着操场标语的心理活动’”）；
针对 “结构问题”，给出具体逻辑框架 / 过渡示例（如议论文分论点需建议 “按‘精神层面→实践层面’递进，过渡句可写‘若说初心是坚守的内核，那么当代青年的行动便是内核的具象化’”；记叙文过渡需建议 “从‘失败场景’到‘觉醒瞬间’可加‘书包里的错题本滑出来，扉页的 “再试一次” 突然扎进眼里’”）。
4. 思辨与升华：拓维度，强关联
引导社会意义 / 时代背景关联（如 “科研经历类文本” 需建议 “关联‘青年科技人才助力科技强国’，补充‘实验失败时想到 “卡脖子技术突破需坚持” 的时代号召’”；“文化类文本” 需建议 “关联‘文化出海 / 文化自信’，如‘《黑神话》海外玩家讨论孙悟空形象，体现传统文化现代表达的国际影响力’”）；
加入正反思辨（如 “科技与文化” 需建议 “补充‘科技便捷性可能稀释文化深度（如短视频碎片化传播剪纸技艺），需平衡 “传播效率” 与 “文化内涵”’”）；
避免 “个人层面局限”：将 “个人学会书法” 升华为 “书法练习中领悟的‘静心坚持’，正是当代青年应对浮躁社会的重要品质”。

格式要求：先分三个大类，一到三：
一、内容和审题
二、文章结构
三、语言表达

再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个小点前要加上数字标号如"1."依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”

作文内容：
{essay_content}
注意：以上的作文内容是唯一的评价对象，所有评论只能以它为基础，不能将下面的参考样例当作评价对象。

参考样例：
{sample}
注意：参考样例仅供学习如何评价相似作文与对应格式，其中的评论不能直接照搬，比如五篇样例均提到紧扣主题，但待批改的作文可能没有完全符合主题，所以评论中不能提到紧扣主题。

在批改完成后，你还需要进一步对作文与生成的批改结果进行对比，检查并修改错误的地方。
"""

        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        messages = [{"role": "user", "content": prompt}]
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True,
            temperature=0.6,
            max_tokens=6000
        )

        full_response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_response += delta.content

        return full_response.strip()

    except Exception as e:
        return f"异常：{str(e)}"
# ===================== 可配置变量 =====================
PARALLEL_NUM = 10  # 并行处理数量，可根据需求调整（如5、10、15）
START_NUM = 1     # 起始作文编号
END_NUM = 50   # 结束作文编号
# ======================================================

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
        docs = processESIndex_Knn(search_query, index_name_knn, es_client)


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
        problem_docs=problem_docs[:10]
        # 4. 写入KNN结果（使用专用锁）
        knn_output_path_1 = rf"D:\zuowen\output\KNNS\knn_1_{input_num}.txt"

        
        with knn_file_lock:
            with open(knn_output_path_1, 'w', encoding='utf-8') as output_knn:
                print(f"作文 {input_num} - ---------------------------------------")
                count = 0
                tcnt=0
                for i in docs:
                    count += 1
                    if count == 1 or i['_source']['id'] == input_num:
                        continue
                    tcnt+=1
                    if tcnt>10:
                        break
                    print(f"作文 {input_num} - {i['_source']['id']}    {i['_score']}")
                    output_knn.write('标题：'+str(i['_source']['title']) + '\n' + '内容：'+str(i['_source']['text']) + '\n' + '标准评论：'+str(i['_source']['comment']))
                    output_knn.write("----------------------------" * 20 + "\n")
                    count = 0
                for i in problem_docs:
                    count += 1
                    if count == 1 or i['_source']['id'] == input_num:
                        continue
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

        result_none = grade_essay_no_change(title, content, first_samples,DASHSCOPE_API_KEY,rule,None,"deepseek-v3.2")



        
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
    root_dir = r'D:\zuowen\data\EDPO1'
    output_dir = r"D:\zuowen\data\eval\temp\deepseek"  # 输出目录
    
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