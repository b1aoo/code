import os
import requests
import json
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer 
import re
from datetime import date
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
import random
import math



# 获取阿里云灵积平台API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
def search_by_source_id(index_name, target_id,es_client):
    load_dotenv()
    try:
        # 构建查询：精确匹配_source.id字段
        query = {
            "query": {
                "term": {  # term查询用于精确匹配（不分词）
                    "id": target_id  # 字段名为"id"（对应_source中的id）
                }
            }
        }

        # 执行查询
        response = es_client.search(
            index=index_name,
            body=query
        )

        # 提取结果
        hits = response["hits"]["hits"]
        if hits:
            # 返回匹配的文档（包含_id和_source等信息）
            return {
                "found": True,
                "documents": hits
            }
        else:
            return {"found": False, "message": f"未找到id为{target_id}的文档"}

    except RequestError as e:
        return {"found": False, "message": f"查询错误：{str(e)}"}
    except Exception as e:
        return {"found": False, "message": f"查询失败：{str(e)}"}
    
def processESIndex_Knn(search_query,index_name,es_client):
    response = es_client.search(
        index=index_name,
        body=search_query,
        scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
        size=50  # Set the number of documents to retrieve per scroll
        )
    all_hits = response['hits']['hits']
    print(len(all_hits))
    docs_knn =[]
    for hit in all_hits:
        docs_knn.append(hit)

    """for ind in df_questions.index:
        print("Processsing -----",ind)
        question =df_questions['text'][ind]
        content_embedding =model.encode(question)
        ## content_embedding will be add into your query according the question
        flag = False
        for num, doc in enumerate(all_hits):
            if df_questions['answers'][ind] in  doc["_source"]['text']:
                flag = True
                break
        print ("DOC Score:", flag)
        df_questions['model_op_kNN'][ind] = flag"""
    return docs_knn

def processESIndex_Knn1(search_query,index_name,es_client):
    response = es_client.search(
        index=index_name,
        body=search_query,
        scroll='5m',  # Set the scroll timeout (e.g., 5 minutes)
        )
    all_hits = response['hits']['hits']
    print(len(all_hits))
    docs_knn =[]
    for hit in all_hits:
        docs_knn.append(hit)

    """for ind in df_questions.index:
        print("Processsing -----",ind)
        question =df_questions['text'][ind]
        content_embedding =model.encode(question)
        ## content_embedding will be add into your query according the question
        flag = False
        for num, doc in enumerate(all_hits):
            if df_questions['answers'][ind] in  doc["_source"]['text']:
                flag = True
                break
        print ("DOC Score:", flag)
        df_questions['model_op_kNN'][ind] = flag"""
    return docs_knn


def save_feedback_without(feedback):
    """将批改结果保存到文件"""
    try:
        os.chdir(r'D:\zuowen\data\eval')
        output_file = "without_RAG_feedback.txt"
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(feedback)
            
        return f"批改结果已保存到：{output_file}"
        
    except Exception as e:
        return f"保存批改结果时出错：{str(e)}"
    
def save_feedback(feedback):
    """将批改结果保存到文件"""
    try:
        os.chdir(r'D:\zuowen\data\eval')

        output_file = "RAG_feedback.txt"
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(feedback)
            
        return f"批改结果已保存到：{output_file}"
        
    except Exception as e:
        return f"保存批改结果时出错：{str(e)}"
    

def build_problem_knn_query(problem_types, summary, k=30):
    """
    构造基于“问题类型 + 判断摘要”的 ES 查询
    目标字段：comment（专家判断理由）
    """

    query_text = " ".join(problem_types) + " " + summary

    search_query = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "comment^3",   # 专家评语权重最高
                    "title",
                    "text"
                ],
                "type": "most_fields"
            }
        }
    }

    return search_query

def get_sorted_indices(
    question: str,
    answer: str,
    docs,
    w1: float = 1.0,
    w2: float = 2.0,
    model: SentenceTransformer = None
):
    """
    仅返回排序后的索引数组，不修改原数组
    
    参数:
        question: 题目要求
        answer: 作文内容
        docs: 原始文档列表
        w1, w2: 权重参数
        model: 模型实例
        
    返回:
        List[int]: 排序后的索引列表（降序）
    """
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    if not docs:
        return []
    
    # 编码输入
    emb_q = model.encode(question, normalize_embeddings=True)
    emb_a = model.encode(answer, normalize_embeddings=True)
    
    # 计算分数
    score_list = []
    for idx, doc in enumerate(docs):
        source = doc.get('_source', {})
        doc_question = source.get('title', '')
        doc_answer = source.get('text', '')
        
        emb_doc_q = model.encode(doc_question, normalize_embeddings=True)
        emb_doc_a = model.encode(doc_answer, normalize_embeddings=True)
        
        cos_q = np.dot(emb_q, emb_doc_q)
        cos_a = np.dot(emb_a, emb_doc_a)
        
        score = math.exp(w2 * cos_q) + math.exp(w1 * cos_a)
        score_list.append((idx, score))
    
    # 按分数降序排序，返回索引列表
    sorted_indices = [idx for idx, _ in sorted(score_list, key=lambda x: x[1], reverse=True)]
    
    return sorted_indices