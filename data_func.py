import json
import numpy as np
from sentence_transformers import SentenceTransformer  # 用于生成文本向量
import os

def get_essay_from_file(file_path):
    """从文件中读取作文内容"""
    try:
        if not os.path.exists(file_path):
            return None, f"错误：文件 '{file_path}' 不存在"
        
        if not os.path.isfile(file_path):
            return None, f"错误：'{file_path}' 不是一个文件"
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if not content.strip():
            return None, "错误：文件内容为空"
            
        return content, None
        
    except Exception as e:
        return None, f"读取文件时出错：{str(e)}"

def generate_text_embedding(text,model):
    """生成文本的向量表示"""
    embedding = model.encode(text)
    return embedding.tolist()  # 转换为列表格式，便于JSON序列化

def write_to_query_file(user_input,model,input_num):
    """
    将用户输入写入文件的query和text_embedding字段
    
    参数:
        user_input: 用户输入的字符串
        output_file: 输出的JSON文件路径
    """
    try:
        # 生成文本向量（维度与配置中的384保持一致）
        
        text_embedding = generate_text_embedding(user_input,model)
        # 构建包含query和knn的完整结构
        query_template = {
            "query": {
                "multi_match": {
                    "query": user_input,
                    "type": "best_fields",
                    "fields": ["text", "title"],
                    "tie_breaker": 0.3
                }
            },
            "knn": {
                "field": "text_embedding",
                "query_vector": text_embedding,
                "k": 10,
                "num_candidates": 1000,
                "boost": 100
            }
        }
        
        # 写入JSON文件
        with open(r'D:\zuowen\input\search_query\KNN\input\knn '+str(input_num)+'.txt', "w", encoding="utf-8") as f:
            json.dump(query_template, f, ensure_ascii=False, indent=2)
        print(f"查询已成功写入到文件")
        print(f"生成的向量维度: {len(text_embedding)}")
        
    except Exception as e:
        print(f"处理过程出错: {str(e)}")

def input_deal_last(input_num):
    indir = r'D:\zuowen\data\test'  # 使用原始字符串前缀
    file_before = '202506'
    output_dir = r"D:\zuowen\data\eval\temp\in"
    
    # 修复文件编号格式
    if input_num < 10:
        str_input_num = '000' + str(input_num)  # 改为4位数字
    elif 10 <= input_num < 100:
        str_input_num = '00' + str(input_num)   # 改为4位数字
    else:
        str_input_num = '0'+str(input_num)
    
    # 构建文件路径
    input_file_q = os.path.join(indir, f'{file_before}{str_input_num}-q.txt')
    input_file_s = os.path.join(indir, f'{file_before}{str_input_num}-s.txt')
    
    print(f"尝试读取文件: {input_file_q}")  # 调试信息
    
    try:
        with open(input_file_q, 'r', encoding='utf-8') as f:
            title = f.read().rstrip()

        with open(input_file_s, 'r', encoding='utf-8') as f:
            content = f.read().rstrip()


        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 写入处理后的文件
        output_title = os.path.join(output_dir, f'{input_num}title.txt')
        output_content = os.path.join(output_dir, f'{input_num}content.txt')

        
        with open(output_title, 'w', encoding='utf-8') as f:
            f.write(title + '\n')
        with open(output_content, 'w', encoding='utf-8') as f:
            f.write(content + '\n')

            
        print(f"成功处理作文 {input_num}")
        
        # 返回读取的内容，便于后续处理
        return title, content, None
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        raise
    except Exception as e:
        print(f"处理作文 {input_num} 时出错: {e}")
        raise
        
def input_deal(input_num):
    indir = r'D:\zuowen\data\test'  # 使用原始字符串前缀
    file_before = '202506'
    output_dir = r"D:\zuowen\data\eval\temp\in"
    
    # 修复文件编号格式
    if input_num < 10:
        str_input_num = '000' + str(input_num)  # 改为4位数字
    elif 10 <= input_num < 100:
        str_input_num = '00' + str(input_num)   # 改为4位数字
    else:
        str_input_num = '0'+str(input_num)
    
    # 构建文件路径
    input_file_q = os.path.join(indir, f'{file_before}{str_input_num}-q.txt')
    input_file_s = os.path.join(indir, f'{file_before}{str_input_num}-s.txt')
    input_file_ta = os.path.join(indir, f'{file_before}{str_input_num}-ta.txt')
    
    print(f"尝试读取文件: {input_file_q}")  # 调试信息
    
    try:
        with open(input_file_q, 'r', encoding='utf-8') as f:
            title = f.read().rstrip()

        with open(input_file_s, 'r', encoding='utf-8') as f:
            content = f.read().rstrip()

        with open(input_file_ta, 'r', encoding='utf-8') as f:
            comment = f.read().rstrip()

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 写入处理后的文件
        output_title = os.path.join(output_dir, f'{input_num}title.txt')
        output_content = os.path.join(output_dir, f'{input_num}content.txt')
        output_standard = os.path.join(output_dir, f'{input_num}standard.txt')
        
        with open(output_title, 'w', encoding='utf-8') as f:
            f.write(title + '\n')
        with open(output_content, 'w', encoding='utf-8') as f:
            f.write(content + '\n')
        with open(output_standard, 'w', encoding='utf-8') as f:
            f.write(comment + '\n')
            
        print(f"成功处理作文 {input_num}")
        
        # 返回读取的内容，便于后续处理
        return title, content, comment
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        raise
    except Exception as e:
        print(f"处理作文 {input_num} 时出错: {e}")
        raise

def extract_dimension_content(content, target_dimension):
    """
    根据输入的维度数字，提取文档中对应维度的评论内容
    
    参数:
        content (str): 文档文件路径（如 standard.txt）
        target_dimension (int): 目标维度数字（1-5，对应五个评价维度）
    
    返回:
        str: 提取到的对应维度评论内容，若参数无效或未找到内容则返回提示信息
    """
    # 定义所有维度的标识：key为维度数字，value为(维度起始标题, 下一个维度标题/结束标记)
    dimension_markers = {
        1: ("一、内容立意与主题表达", "二、结构安排与逻辑组织"),
        2: ("二、结构安排与逻辑组织", "三、语言表达与文采风格"),
        3: ("三、语言表达与文采风格", "四、思维深度与情感力量"),
        4: ("四、思维深度与情感力量", "五、书写规范与卷面整洁"),
        5: ("五、书写规范与卷面整洁", None)  # 最后一个维度无下一级标题，用None标识
    }
    
    # 1. 验证目标维度是否合法
    if target_dimension not in dimension_markers:
        return f"无效的维度数字！请输入1-5之间的整数（1对应内容立意，2对应结构安排，3对应语言表达，4对应思维情感，5对应书写规范）"
    
    # 2. 获取当前维度的起始标识和结束标识
    start_marker, end_marker = dimension_markers[target_dimension]
    
    # 3. 读取文件内容（处理常见编码问题）
    
    # 4. 定位当前维度的起始位置
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return f"未找到第{target_dimension}维度（{start_marker[:-1]}）的评论内容"
    
    # 5. 定位当前维度的结束位置
    if end_marker is None:
        # 最后一个维度，直接取到内容末尾
        target_content = content[start_idx + len(start_marker):].strip()
    else:
        # 非最后一个维度，取下一个维度标题作为结束边界
        end_idx = content.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            target_content = content[start_idx + len(start_marker):].strip()
        else:
            target_content = content[start_idx + len(start_marker):end_idx].strip()
    
    # 6. 处理提取内容为空的情况
    if not target_content:
        return f"第{target_dimension}维度（{start_marker[:-1]}）的评论内容为空"
    
    # 7. 格式化输出结果（保留原始结构）
    return f"第{target_dimension}维度「{start_marker[2:-1]}」评论内容：\n{target_content}"















def comment_upgrade(text,comment,model):
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        content=extract_dimension_content(comment,1)
        structure=extract_dimension_content(comment,2)
        style=extract_dimension_content(comment,3)
        content_embedding=generate_text_embedding(content,model)
        structure_embedding=generate_text_embedding(structure,model)
        style_embedding=generate_text_embedding(style,model)
        # 构建包含query和knn的完整结构
        query_content = {
            "query": {
                "multi_match": {
                    "query": content,
                    "type": "best_fields",
                    "fields": ["text", "title"],
                    "tie_breaker": 0.3
                }
            },
            "knn": {
                "field": "content_embedding",
                "query_vector": content_embedding,
                "k": 10,
                "num_candidates": 1000,
                "boost": 100
            }
        }
        query_structure = {
            "query": {
                "multi_match": {
                    "query": structure,
                    "type": "best_fields",
                    "fields": ["text", "title"],
                    "tie_breaker": 0.3
                }
            },
            "knn": {
                "field": "content_embedding",
                "query_vector": structure_embedding,
                "k": 10,
                "num_candidates": 1000,
                "boost": 100
            }
        }
        query_style = {
            "query": {
                "multi_match": {
                    "query": style,
                    "type": "best_fields",
                    "fields": ["text", "title"],
                    "tie_breaker": 0.3
                }
            },
            "knn": {
                "field": "content_embedding",
                "query_vector": style_embedding,
                "k": 10,
                "num_candidates": 1000,
                "boost": 100
            }
        }
        
        # 写入JSON文件
        with open(r'D:\zuowen\input\search_query\KNN\content.txt', "w", encoding="utf-8") as f:
            json.dump(query_content, f, ensure_ascii=False, indent=2)
        with open(r'D:\zuowen\input\search_query\KNN\structure.txt', "w", encoding="utf-8") as f:
            json.dump(query_structure, f, ensure_ascii=False, indent=2)
        with open(r'D:\zuowen\input\search_query\KNN\style.txt', "w", encoding="utf-8") as f:
            json.dump(query_style, f, ensure_ascii=False, indent=2)

        print(f"查询已成功写入到文件")
        print(f"生成的向量维度: {len(content_embedding)}")
        print(f"生成的向量维度: {len(structure_embedding)}")
        print(f"生成的向量维度: {len(style_embedding)}")
    except Exception as e:
        print(f"处理过程出错: {str(e)}")
