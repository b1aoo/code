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
from data_func import get_essay_from_file,write_to_query_file,input_deal,extract_dimension_content,comment_upgrade
from search_func import search_by_source_id, processESIndex_Knn, save_feedback
from llm_func import handle_workflow_iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

LLM_MODEL = "RAG_problem_only"
#7595041545120792611  recall        7641890494687641609   precision
# 配置参数
CONFIG = {
    "coze_api_token": os.environ["COZE_API_TOKEN"],
    "coze_api_base": COZE_CN_BASE_URL,
    "workflow_id" : '7641890494687641609',
    "in_dir": r"D:\zuowen\data\eval\temp\\"+LLM_MODEL,
    "output_file": r"D:\zuowen\data\eval\result\RAG_problem_only_precision.txt",
    "error_log_file": r"D:\zuowen\data\eval\error_log.txt",
    "max_workers": 10,
    "start_num": 1,
    "end_num": 50
}

# 创建线程锁
file_lock = threading.Lock()
error_lock = threading.Lock()

def create_coze_client():
    """为每个线程创建独立的Coze客户端实例"""
    return Coze(
        auth=TokenAuth(token=CONFIG["coze_api_token"]), 
        base_url=CONFIG["coze_api_base"]
    )

def log_error(message):
    """线程安全的错误日志记录"""
    with error_lock:
        with open(CONFIG["error_log_file"], 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

def validate_files(input_num):
    """验证文件组是否完整存在"""
    required_files = [
        f"20250600{input_num}-at.txt",
        f"20250600{input_num}-q.txt", 
        f"20250600{input_num}-s.txt",
        f"20250600{input_num}-ta.txt"
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = CONFIG["in_dir"] + rf"\{filename}"
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        raise FileNotFoundError(f"缺失文件: {', '.join(missing_files)}")
    
    return True

def process_file(input_num):
    """处理单个编号的完整文件组"""
    try:
        # 验证文件完整性
        validate_files(input_num)
        
        # 为每个线程创建独立的Coze客户端
        coze = create_coze_client()
        
        # 读取文件内容（使用原始字符串路径）
        with open(CONFIG["in_dir"] + rf"\20250600{input_num}-at.txt", 'r', encoding='utf-8') as f:
            result_none = f.read()
        with open(CONFIG["in_dir"] + rf"\20250600{input_num}-q.txt", 'r', encoding='utf-8') as f:
            title = f.read()
        with open(CONFIG["in_dir"] + rf"\20250600{input_num}-s.txt", 'r', encoding='utf-8') as f:
            text = f.read()
        with open(CONFIG["in_dir"] + rf"\20250600{input_num}-ta.txt", 'r', encoding='utf-8') as f:
            standard = f.read()
        
        # 验证内容不为空
        if not all([title.strip(), text.strip(), standard.strip()]):
            raise ValueError("文件内容不能为空")
        
        # 调用workflow处理
        EVAL = handle_workflow_iterator(
            coze.workflows.runs.stream(
                workflow_id=CONFIG["workflow_id"],
                parameters={
                    "input0": title + "\n" + text,
                    "input1": standard,
                    "input2": result_none
                },   
            )
        )
        
        # 线程安全地写入结果
        with file_lock:
            with open(CONFIG["output_file"], 'a', encoding='utf-8') as f:
                f.write(f"=== 编号: {input_num} ===\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"评估结果:\n{EVAL}\n")
                f.write("="*50 + "\n\n")
        
        print(f"✅ 完成处理: {input_num}")
        return {
            "input_num": input_num,
            "status": "success",
            "error": None,
            "thread": threading.current_thread().name
        }
        
    except Exception as e:
        error_msg = f"❌ 处理 {input_num} 时出错: {str(e)}"
        print(error_msg)
        
        # 记录详细错误日志
        log_error(f"编号 {input_num} 错误详情:\n{traceback.format_exc()}")
        
        # 记录错误到输出文件
        with file_lock:
            with open(CONFIG["output_file"], 'a', encoding='utf-8') as f:
                f.write(f"=== 编号: {input_num} ===\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误: {str(e)}\n")
                f.write("="*50 + "\n\n")
        
        return {
            "input_num": input_num,
            "status": "failed",
            "error": str(e),
            "thread": threading.current_thread().name
        }

def initialize_files():
    """初始化输出文件"""
    # 清空输出文件
    with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
        f.write("作文评估处理记录\n")
        f.write("="*60 + "\n\n")
    
    # 清空错误日志
    with open(CONFIG["error_log_file"], 'w', encoding='utf-8') as f:
        f.write("错误日志\n")
        f.write("="*60 + "\n\n")

def main():
    """主处理函数"""
    print(f"开始处理任务，范围: {CONFIG['start_num']}-{CONFIG['end_num']}")
    print(f"最大并发线程数: {CONFIG['max_workers']}")
    print("-"*50)
    
    # 初始化文件
    initialize_files()
    
    # 使用线程池执行
    results = []
    with ThreadPoolExecutor(
        max_workers=CONFIG["max_workers"],
        thread_name_prefix="EssayProcessor"
    ) as executor:
        
        # 提交所有任务
        futures = {}
        for input_num in range(CONFIG["start_num"], CONFIG["end_num"] + 1):
            future = executor.submit(process_file, input_num)
            futures[future] = input_num
        
        # 处理完成的任务
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            input_num = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # 实时统计
                success_count = sum(1 for r in results if r["status"] == "success")
                fail_count = sum(1 for r in results if r["status"] == "failed")
                
                print(f"\r进度: {completed}/{total} | 成功: {success_count} | 失败: {fail_count}", end="")
                
            except Exception as e:
                error_msg = f"任务 {input_num} 获取结果异常: {str(e)}"
                print(error_msg)
                log_error(error_msg)
                results.append({
                    "input_num": input_num,
                    "status": "error",
                    "error": error_msg,
                    "thread": "Unknown"
                })
    
    # 生成最终报告
    print("\n\n" + "="*50)
    print("处理完成！生成报告:")
    print("="*50)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = sum(1 for r in results if r["status"] == "failed")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"总任务数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"异常: {error_count}")
    
    # 记录失败的编号
    failed_nums = [r["input_num"] for r in results if r["status"] in ["failed", "error"]]
    if failed_nums:
        print(f"失败编号: {', '.join(map(str, failed_nums))}")
    
    # 写入最终统计
    with file_lock:
        with open(CONFIG["output_file"], 'a', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("处理统计报告\n")
            f.write("="*60 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总任务数: {len(results)}\n")
            f.write(f"成功数量: {success_count}\n")
            f.write(f"失败数量: {fail_count}\n")
            f.write(f"异常数量: {error_count}\n")
            if failed_nums:
                f.write(f"失败编号: {', '.join(map(str, failed_nums))}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断！")
    except Exception as e:
        print(f"\n\n程序异常终止: {str(e)}")
        log_error(f"程序异常终止: {traceback.format_exc()}")
    finally:
        print(f"\n结果文件: {CONFIG['output_file']}")
        print(f"错误日志: {CONFIG['error_log_file']}")