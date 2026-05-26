import os
import re
from datetime import date
import pandas as pd
import json
from datetime import datetime
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from data_func import extract_dimension_content
es_client = Elasticsearch(
    "http://localhost:9200",
)
try:
    es_client.info()
    print("Elasticsearch连接成功")
except Exception as e:
    print(f"连接失败：{e}")
    exit()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def generate_text_embedding(text):
    """生成文本的向量表示"""
    embedding = model.encode(text)
    return embedding.tolist()
def get_all_files(folder_name):
    file_path_list = []
    for file in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file)
        if os.path.isfile(file_path) and file.endswith('.jsonl'):
            file_path_list.append(file_path)
            print(file_path)
    return file_path_list

def clean_text(text):
    """清理文本中的特殊字符和不可见字符"""
    if not isinstance(text, str):
        return ""
    # 移除控制字符和不可见字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 移除多余的空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_index(index_name, mapping_str):
    try:
        es_client.indices.delete(index=index_name, ignore_unavailable=True)
        es_client.indices.create(index=index_name, body=mapping_str, ignore=400)
        print(f"Index '{index_name}' created successfully.")
    except RequestError as e:
        if e.error == 'resource_already_exists_exception':
            print(f"Index '{index_name}' already exists.")
        else:
            print(f"创建索引错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

def index_data(df_docs, source, index_name, index_name_knn):
    i = 0
    for index, row in df_docs.iterrows():
        i += 1
        print(f"Processing {i}")
        
        # 清理文本字段
        id_ = str(i)
        title = clean_text(row.get('title', ''))
        text = clean_text(row.get('content', ''))
        comment = clean_text(row.get('comment', ''))
        
        # 生成正文的向量表示
        text_embedding = generate_text_embedding(text)
        
        # 适配bm25索引
        doc_bm25 = {
            "id": id_,
            "source": source,
            "text": text,
            "title": title,
            "comment": comment
        }
        
        # 适配knn索引
        doc_knn = {
            "id": id_,
            "source": source,
            "text": text,
            "title": title,
            "comment": comment,
            "text_embedding": text_embedding,
        }
        
        try:
            response = es_client.index(index=index_name, body=doc_bm25)
            print(f"BM25索引响应: {response}")
            
            response = es_client.index(index=index_name_knn, body=doc_knn)
            print(f"KNN索引响应: {response}")
        except Exception as e:
            print(f"索引文档 {i} 时出错: {e}")
            continue

# 主程序
index_name_knn = 'research_index_knn'
index_name = "research_index_bm25"

# 创建索引
with open(r"D:\zuowen\input\mapping\temp-bm25.txt", 'r', encoding='utf-8') as file:
    mapping = file.read().rstrip()
create_index(index_name, mapping)

with open(r"D:\zuowen\input\mapping\temp-knn.txt", 'r', encoding='utf-8') as file:
    mapping = file.read().rstrip()
create_index(index_name_knn, mapping)

# 读取数据
doc_folder = r"D:\zuowen\data"
files = get_all_files(doc_folder)
if files:
    df_corpus = pd.read_json(files[0], lines=True, encoding='utf-8')
    source = "mydoc"
    index_data(df_corpus, source, index_name, index_name_knn)
else:
    print("未找到JSONL文件")