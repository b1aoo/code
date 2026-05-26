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
from cozepy import COZE_CN_BASE_URL
from cozepy import Coze, TokenAuth, Stream, WorkflowEvent, WorkflowEventType
from data_func import comment_upgrade,extract_dimension_content
from search_func import processESIndex_Knn
from concurrent.futures import ThreadPoolExecutor
coze_api_token = 'REMOVED_USE_ENVIRONMENT_VARIABLE'

# 默认访问地址为api.coze.com，如需访问api.coze.cn，
# 请使用base_url配置API端点
coze_api_base = COZE_CN_BASE_URL











# 通过access_token初始化Coze客户端。
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)

# 在Coze平台创建工作流实例后，从网页链接末尾获取数字作为工作流ID。
workflow_id = '7533075507467157539'

def join_result(result_1,result_2,result_3,content,DASHSCOPE_API_KEY,rule):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    try:
        # 构建提示词
        prompt =f"""你是一名经验丰富的初中语文老师，你的任务是判断不同的评价是否与作文是否相关，并将三个不同的评价结果取长补短，消除不正确的，整合正确的，最终合并为一个最终结果。
            总结规则：{rule}
            注意：批改规则仅供参考，需要结合作文内容及评论进行总结
            要求：不要对作文进行额外的评价，你的最终结果的评价只能来源于三个评价结果。
            格式要求：先分三个大类，一到三：
            一、内容和审题
            二、文章结构
            三、语言表达

            再分成优点，不足，建议，三种评价
            优点，不足，建议三种评价前不需要数字标号
            每个小点前要加上数字标号如"1."依次说明
            如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”
            三个评价结果：

            一、{result_1}
            二、{result_2}
            三、{result_3}
            作文内容：{content}
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
        print("API返回内容: ",response_data["output"]["choices"][0]["message"]["content"])
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                
                # 直接检查是否存在output.text字段
                final_result = response_data["output"]["choices"][0]["message"]["content"]
                return final_result
            except KeyError:
                return "错误：API响应格式错误，缺少output.text字段"
        else:
            return f"错误：API请求失败，状态码 {response.status_code}"
    
    except requests.exceptions.Timeout:
        return "错误：API请求超时，请检查网络"
    except requests.exceptions.ConnectionError:
        return "错误：网络连接失败，请检查网络设置"
    except Exception as e:
        return f"处理过程中出错：{str(e)}"

def join_result_new(result_1,result_2,result_3,content,DASHSCOPE_API_KEY,rule):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    try:
        # 构建提示词
        prompt =f"""你的任务是将三个维度的评论按照格式整理为一份最终评论
            格式要求：先分三个大类，一到三：
一、内容和审题
二、文章结构
三、语言表达

再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个小点前要加上数字标号如"1."依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”
三个评价结果：
内容和审题：{result_1}
文章结构：{result_2}
语言表达：{result_3}
作文内容：{content}
只要返回最终按照格式整理的评价，不要对作文进行额外的评价
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
        print("API返回内容: ",response_data["output"]["choices"][0]["message"]["content"])
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                
                # 直接检查是否存在output.text字段
                if "output" in response_data and "choices" in response_data["output"] and "message" in response_data["output"]["choices"][0] and "content" in response_data["output"]["choices"][0]["message"]:
                    return response_data["output"]["choices"][0]["message"]["content"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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

def select_top_3(essay_content,comment,DASHSCOPE_API_KEY,rule,model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        
        prompt = f"""
你会收到内容和审题，文章结构，语言表达三种维度之一的评价，你需要根据文章的内容，从每种评价里选出最优最相关的三条，并按照格式输出


评价要求：
1.  核心问题定位：挖本质，拒表层
先拆解文本主题的核心内涵，明确文本偏离本质的具体表现（例：若文本写 “雕版印刷” 却侧重 “文化传承”，需指出 “偏离‘低谷中的坚持’主线，主题摇摆”）；
识别逻辑断裂的根源（如 “科技与人文” 类文本若仅提 “15 秒便民” 未关联 “铁路网建设”，需点明 “案例与主题的因果链缺失，未体现科技对人文的赋能逻辑”）。

2.  主题聚焦：锚核心，补关联
明确文本需紧扣的核心主题，排查 “内容堆砌” 问题（如记叙文若同时写 “数学、豫剧、体测”，需建议 “聚焦 1 件事，如‘体测突破’，删除无关事件”）；
挖掘题目隐含的潜在关联（如 “传统文化类文本” 需关联 “传统与现代 / 科技”，“时代主题类文本” 需关联 “当代青年视角”，例：“剪纸文化” 需补充 “科技助力剪纸创新 / 避免科技导致的文化浮躁”）。

3.  细节指导：具象化，给方案
针对 “细节缺失”，提供可落地的场景 / 动作 / 心理描写方向
（例：写 “外婆做短视频” 需建议 “补充‘学剪辑时反复调整时间轴、向晚辈讨论文案措辞’的细节”；
写 “引体向上成长” 需建议 “添加‘肌肉酸痛时握杆的指节发白、想放弃时盯着操场标语的心理活动’”）；
针对 “结构问题”，给出具体逻辑框架 / 过渡示例（如议论文分论点需建议 “按‘精神层面→实践层面’递进，过渡句可写‘若说初心是坚守的内核，那么当代青年的行动便是内核的具象化’”；
记叙文过渡需建议 “从‘失败场景’到‘觉醒瞬间’可加‘书包里的错题本滑出来，扉页的 “再试一次” 突然扎进眼里’”）。

4.  思辨与升华：拓维度，强关联
引导社会意义 / 时代背景关联（如 “科研经历类文本” 需建议 “关联‘青年科技人才助力科技强国’，补充‘实验失败时想到 “卡脖子技术突破需坚持” 的时代号召’”；
“文化类文本” 需建议 “关联‘文化出海 / 文化自信’，如‘《黑神话》海外玩家讨论孙悟空形象，体现传统文化现代表达的国际影响力’”）；
加入正反思辨（如 “科技与文化” 需建议 “补充‘科技便捷性可能稀释文化深度（如短视频碎片化传播剪纸技艺），需平衡 “传播效率” 与 “文化内涵”’”）；
避免 “个人层面局限”：将 “个人学会书法” 升华为 “书法练习中领悟的‘静心坚持’，正是当代青年应对浮躁社会的重要品质”。

格式要求：

先是标题：一、内容与审题/二、文章结构/三、语言表达
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个小点前要加上数字标号如"1."依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”

注意：每种评价（优点，不足，建议）都只能返回最优的三条，即优点返回最好的三条，不足返回最好的三条，建议返回最好的三条。

作文内容：
{essay_content}
评论：
{comment}

注意：只要返回按格式要求的评价，不需要返回其他内容。
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

def grade_essay_origin_content(title,essay_content, samples,DASHSCOPE_API_KEY,rule,knowledge,model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        
        prompt = f"""
请批改作文，并按要求针对内容与审题维度给出评价，仅进行评价不修改作文、不创造作文，仅返回一遍评论无需总结，评论内容需丰富、详细、具体，全程基于作文文本本身独立诊断，核心提升批改的全面性、精准性与维度均衡性，助力提升总体匹配质量，要求如下：

评价要求：
1.  核心问题定位：挖本质，拒表层
先拆解文本主题的核心内涵，明确文本偏离本质的具体表现（例：若文本写 “雕版印刷” 却侧重 “文化传承”，需指出 “偏离‘低谷中的坚持’主线，主题摇摆”）；
识别逻辑断裂的根源（如 “科技与人文” 类文本若仅提 “15 秒便民” 未关联 “铁路网建设”，需点明 “案例与主题的因果链缺失，未体现科技对人文的赋能逻辑”）。

2.  主题聚焦：锚核心，补关联
明确文本需紧扣的核心主题，排查 “内容堆砌” 问题（如记叙文若同时写 “数学、豫剧、体测”，需建议 “聚焦 1 件事，如‘体测突破’，删除无关事件”）；
挖掘题目隐含的潜在关联（如 “传统文化类文本” 需关联 “传统与现代 / 科技”，“时代主题类文本” 需关联 “当代青年视角”，例：“剪纸文化” 需补充 “科技助力剪纸创新 / 避免科技导致的文化浮躁”）。

3.  细节指导：具象化，给方案
针对 “细节缺失”，提供可落地的场景 / 动作 / 心理描写方向
（例：写 “外婆做短视频” 需建议 “补充‘学剪辑时反复调整时间轴、向晚辈讨论文案措辞’的细节”；
写 “引体向上成长” 需建议 “添加‘肌肉酸痛时握杆的指节发白、想放弃时盯着操场标语的心理活动’”）；
针对 “结构问题”，给出具体逻辑框架 / 过渡示例（如议论文分论点需建议 “按‘精神层面→实践层面’递进，过渡句可写‘若说初心是坚守的内核，那么当代青年的行动便是内核的具象化’”；
记叙文过渡需建议 “从‘失败场景’到‘觉醒瞬间’可加‘书包里的错题本滑出来，扉页的 “再试一次” 突然扎进眼里’”）。

4.  思辨与升华：拓维度，强关联
引导社会意义 / 时代背景关联（如 “科研经历类文本” 需建议 “关联‘青年科技人才助力科技强国’，补充‘实验失败时想到 “卡脖子技术突破需坚持” 的时代号召’”；
“文化类文本” 需建议 “关联‘文化出海 / 文化自信’，如‘《黑神话》海外玩家讨论孙悟空形象，体现传统文化现代表达的国际影响力’”）；
加入正反思辨（如 “科技与文化” 需建议 “补充‘科技便捷性可能稀释文化深度（如短视频碎片化传播剪纸技艺），需平衡 “传播效率” 与 “文化内涵”’”）；
避免 “个人层面局限”：将 “个人学会书法” 升华为 “书法练习中领悟的‘静心坚持’，正是当代青年应对浮躁社会的重要品质”。

格式要求：

先是标题：一、内容和审题
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个小点前要加上数字标号如"1."依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”

注意：每种评价（优点，不足，建议）都需要返回10-20条评价。

作文内容：
{essay_content}
注意：以上的作文内容是唯一的评价对象，所有评论只能以它为基础，不能将下面的参考样例当作评价对象。

参考样例：
{samples}
注意：参考样例仅供学习如何评价相似作文与对应格式，其中的评论不能直接照搬，比如五篇样例均提到紧扣主题，但待批改的作文可能没有完全符合主题，所以评论中不能提到紧扣主题。

评价规则：{rule}
注意：评价规则仅供参考，须结合实际作文内容给出评价。

注意：只要给出内容和审题维度的评价，不需要给出文章结构和语言表达维度的评价。





在批改完成后，你还需要进一步对作文与生成的批改结果进行对比，检查并修改错误的地方。
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

def grade_essay_origin_structure(title,essay_content, samples,DASHSCOPE_API_KEY,rule,knowledge,model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        
        prompt = f"""
请批改作文，并按要求针对文章结构维度给出评价，仅进行评价不修改作文、不创造作文，仅返回一遍评论无需总结，评论内容需丰富、详细、具体，全程基于作文文本本身独立诊断，核心提升批改的全面性、精准性与维度均衡性，助力提升总体匹配质量，要求如下：

评价要求：
1.  核心问题定位：挖本质，拒表层
先拆解文本主题的核心内涵，明确文本偏离本质的具体表现（例：若文本写 “雕版印刷” 却侧重 “文化传承”，需指出 “偏离‘低谷中的坚持’主线，主题摇摆”）；
识别逻辑断裂的根源（如 “科技与人文” 类文本若仅提 “15 秒便民” 未关联 “铁路网建设”，需点明 “案例与主题的因果链缺失，未体现科技对人文的赋能逻辑”）。

2.  主题聚焦：锚核心，补关联
明确文本需紧扣的核心主题，排查 “内容堆砌” 问题（如记叙文若同时写 “数学、豫剧、体测”，需建议 “聚焦 1 件事，如‘体测突破’，删除无关事件”）；
挖掘题目隐含的潜在关联（如 “传统文化类文本” 需关联 “传统与现代 / 科技”，“时代主题类文本” 需关联 “当代青年视角”，例：“剪纸文化” 需补充 “科技助力剪纸创新 / 避免科技导致的文化浮躁”）。

3.  细节指导：具象化，给方案
针对 “细节缺失”，提供可落地的场景 / 动作 / 心理描写方向
（例：写 “外婆做短视频” 需建议 “补充‘学剪辑时反复调整时间轴、向晚辈讨论文案措辞’的细节”；
写 “引体向上成长” 需建议 “添加‘肌肉酸痛时握杆的指节发白、想放弃时盯着操场标语的心理活动’”）；
针对 “结构问题”，给出具体逻辑框架 / 过渡示例（如议论文分论点需建议 “按‘精神层面→实践层面’递进，过渡句可写‘若说初心是坚守的内核，那么当代青年的行动便是内核的具象化’”；
记叙文过渡需建议 “从‘失败场景’到‘觉醒瞬间’可加‘书包里的错题本滑出来，扉页的 “再试一次” 突然扎进眼里’”）。

4.  思辨与升华：拓维度，强关联
引导社会意义 / 时代背景关联（如 “科研经历类文本” 需建议 “关联‘青年科技人才助力科技强国’，补充‘实验失败时想到 “卡脖子技术突破需坚持” 的时代号召’”；
“文化类文本” 需建议 “关联‘文化出海 / 文化自信’，如‘《黑神话》海外玩家讨论孙悟空形象，体现传统文化现代表达的国际影响力’”）；
加入正反思辨（如 “科技与文化” 需建议 “补充‘科技便捷性可能稀释文化深度（如短视频碎片化传播剪纸技艺），需平衡 “传播效率” 与 “文化内涵”’”）；
避免 “个人层面局限”：将 “个人学会书法” 升华为 “书法练习中领悟的‘静心坚持’，正是当代青年应对浮躁社会的重要品质”。

格式要求：

先是标题：二、文章结构
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个小点前要加上数字标号如"1."依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”

注意：每种评价（优点，不足，建议）都需要返回10-20条评价。

作文内容：
{essay_content}
注意：以上的作文内容是唯一的评价对象，所有评论只能以它为基础，不能将下面的参考样例当作评价对象。

参考样例：
{samples}
注意：参考样例仅供学习如何评价相似作文与对应格式，其中的评论不能直接照搬，比如五篇样例均提到紧扣主题，但待批改的作文可能没有完全符合主题，所以评论中不能提到紧扣主题。

评价规则：{rule}
注意：评价规则仅供参考，须结合实际作文内容给出评价。

注意：只要给出文章结构维度的评价，不需要给出文章结构和内容与审题维度的评价。



在批改完成后，你还需要进一步对作文与生成的批改结果进行对比，检查并修改错误的地方。
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

def grade_essay_origin_language(title,essay_content, samples,DASHSCOPE_API_KEY,rule,knowledge,model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        
        prompt = f"""
请批改作文，并按要求针对语言表达维度给出评价，仅进行评价不修改作文、不创造作文，仅返回一遍评论无需总结，评论内容需丰富、详细、具体，全程基于作文文本本身独立诊断，核心提升批改的全面性、精准性与维度均衡性，助力提升总体匹配质量，要求如下：

评价要求：
1.  核心问题定位：挖本质，拒表层
先拆解文本主题的核心内涵，明确文本偏离本质的具体表现（例：若文本写 “雕版印刷” 却侧重 “文化传承”，需指出 “偏离‘低谷中的坚持’主线，主题摇摆”）；
识别逻辑断裂的根源（如 “科技与人文” 类文本若仅提 “15 秒便民” 未关联 “铁路网建设”，需点明 “案例与主题的因果链缺失，未体现科技对人文的赋能逻辑”）。

2.  主题聚焦：锚核心，补关联
明确文本需紧扣的核心主题，排查 “内容堆砌” 问题（如记叙文若同时写 “数学、豫剧、体测”，需建议 “聚焦 1 件事，如‘体测突破’，删除无关事件”）；
挖掘题目隐含的潜在关联（如 “传统文化类文本” 需关联 “传统与现代 / 科技”，“时代主题类文本” 需关联 “当代青年视角”，例：“剪纸文化” 需补充 “科技助力剪纸创新 / 避免科技导致的文化浮躁”）。

3.  细节指导：具象化，给方案
针对 “细节缺失”，提供可落地的场景 / 动作 / 心理描写方向
（例：写 “外婆做短视频” 需建议 “补充‘学剪辑时反复调整时间轴、向晚辈讨论文案措辞’的细节”；
写 “引体向上成长” 需建议 “添加‘肌肉酸痛时握杆的指节发白、想放弃时盯着操场标语的心理活动’”）；
针对 “结构问题”，给出具体逻辑框架 / 过渡示例（如议论文分论点需建议 “按‘精神层面→实践层面’递进，过渡句可写‘若说初心是坚守的内核，那么当代青年的行动便是内核的具象化’”；
记叙文过渡需建议 “从‘失败场景’到‘觉醒瞬间’可加‘书包里的错题本滑出来，扉页的 “再试一次” 突然扎进眼里’”）。

4.  思辨与升华：拓维度，强关联
引导社会意义 / 时代背景关联（如 “科研经历类文本” 需建议 “关联‘青年科技人才助力科技强国’，补充‘实验失败时想到 “卡脖子技术突破需坚持” 的时代号召’”；
“文化类文本” 需建议 “关联‘文化出海 / 文化自信’，如‘《黑神话》海外玩家讨论孙悟空形象，体现传统文化现代表达的国际影响力’”）；
加入正反思辨（如 “科技与文化” 需建议 “补充‘科技便捷性可能稀释文化深度（如短视频碎片化传播剪纸技艺），需平衡 “传播效率” 与 “文化内涵”’”）；
避免 “个人层面局限”：将 “个人学会书法” 升华为 “书法练习中领悟的‘静心坚持’，正是当代青年应对浮躁社会的重要品质”。

格式要求：

先是标题：三、语言表达
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个小点前要加上数字标号如"1."依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”

注意：每种评价（优点，不足，建议）都需要返回10-20条评价。

作文内容：
{essay_content}
注意：以上的作文内容是唯一的评价对象，所有评论只能以它为基础，不能将下面的参考样例当作评价对象。

参考样例：
{samples}
注意：参考样例仅供学习如何评价相似作文与对应格式，其中的评论不能直接照搬，比如五篇样例均提到紧扣主题，但待批改的作文可能没有完全符合主题，所以评论中不能提到紧扣主题。

评价规则：{rule}
注意：评价规则仅供参考，须结合实际作文内容给出评价。

注意：只要给出语言表达维度的评价，不需要给出文章结构和内容与审题维度的评价。



在批改完成后，你还需要进一步对作文与生成的批改结果进行对比，检查并修改错误的地方。
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
def similar_grade(essay_content, sample,DASHSCOPE_API_KEY):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    try:
        # 构建提示词
        prompt =f"""你是一名经验丰富的初中语文老师，你的任务是判断样例是否与作文是否相关
            相关的意思是在以下三个维度中：
            一、内容和主题；
            二、结构维度；
            三、语言维度。
            只要有一个维度相关，就判定相关。
            样例：{sample}
            作文内容：{essay_content}
            请判断这两个内容是否相关，相关仅返回“相关”，反之仅返回“无关”
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
        print("API返回内容: ",response_data["output"]["choices"][0]["message"]["content"])
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                
                # 直接检查是否存在output.text字段
                if "output" in response_data and "choices" in response_data["output"] and "message" in response_data["output"]["choices"][0] and "content" in response_data["output"]["choices"][0]["message"]:
                    return response_data["output"]["choices"][0]["message"]["content"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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


def similar_grade_content(essay_content, sample,DASHSCOPE_API_KEY):
    """调用Qwen-2.5 API批改作文，修正响应判断逻辑"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        prompt =f"""你是一名经验丰富的初中语文老师，你的任务是判断样例是否与作文是否在内容和审题维度相关
        相关的意思：
        核心主题 / 中心思想属同一范畴，无本质对立，核心内核一致（外延可不同）；命题作文需围绕同一命题核心，无偏 / 跑题差异。
        内容素材、论述 / 描写对象与主题的关联点高度重合，或一方素材是另一方的具体延伸 / 拓展。
        满足前两点则判定相关；任一核心主题范畴不同、素材与主题关联点无交集，判定不相关；仅部分素材关联但主题内核不同，仍为不相关。

        样例：{sample}
        作文内容：{essay_content}
        请判断这两个内容是否相关，相关仅返回“相关”，反之仅返回“无关”

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
        print("API返回内容: ",response_data["output"]["choices"][0]["message"]["content"])
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                
                # 直接检查是否存在output.text字段
                if "output" in response_data and "choices" in response_data["output"] and "message" in response_data["output"]["choices"][0] and "content" in response_data["output"]["choices"][0]["message"]:
                    return response_data["output"]["choices"][0]["message"]["content"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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
    
def similar_grade_structure(essay_content, sample,DASHSCOPE_API_KEY):
    """调用Qwen-2.5 API批改作文，修正响应判断逻辑"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        prompt =f"""你是一名经验丰富的初中语文老师，你的任务是判断样例是否与作文是否在结构维度相关
        相关的意思：
        整体框架匹配：二者均采用同类型行文结构（如总分总 / 并列 / 递进 / 因果 / 对比，记叙类的顺叙 / 倒叙 / 插叙等），核心结构逻辑一致。
        段落布局关联：主体段落的展开逻辑、层次划分（如分论点数量 / 描写顺序 / 论证步骤）呈对应性，无完全相悖的排布方式。
        结构要素契合：开篇（点题 / 引入）、主体（展开 / 论述）、结尾（升华 / 总结）的核心功能与衔接方式一致，关键结构节点匹配。
        以上三点至少有一点类似才可判定相关。
        样例：{sample}
        作文内容：{essay_content}
        请判断这两个内容是否相关，相关仅返回“相关”，反之仅返回“无关”
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
        print("API返回内容: ",response_data["output"]["choices"][0]["message"]["content"])
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                
                # 直接检查是否存在output.text字段
                if "output" in response_data and "choices" in response_data["output"] and "message" in response_data["output"]["choices"][0] and "content" in response_data["output"]["choices"][0]["message"]:
                    return response_data["output"]["choices"][0]["message"]["content"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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
    

def similar_grade_language(essay_content, sample,DASHSCOPE_API_KEY):
    """调用Qwen-2.5 API批改作文，修正响应判断逻辑"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        prompt =f"""你是一名经验丰富的初中语文老师，你的任务是判断样例是否与作文是否在语言表达维度相关
        相关的意思：
        表达风格匹配：二者整体语言风格属同一类型（如平实质朴 / 生动优美 / 严谨议论 / 抒情细腻 / 简洁明快等），无截然相反的风格倾向。
        表达手法关联：核心表达手法（修辞 / 句式 / 表现手法）高度重合或成延伸关系（如均用比喻 + 排比，或一方的手法是另一方的细化运用）。
        语言要素契合：句式特点（整散 / 长短句）、语体特征（书面 / 口语化）、情感基调（褒贬 / 悲喜 / 激昂 / 平和）一致，核心语言特征无冲突。
        以上三点至少有一点类似才可判定相关。


        样例：{sample}
        作文内容：{essay_content}
        请判断这两个内容是否相关，相关仅返回“相关”，反之仅返回“无关”
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
        print("API返回内容: ",response_data["output"]["choices"][0]["message"]["content"])
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                
                # 直接检查是否存在output.text字段
                if "output" in response_data and "choices" in response_data["output"] and "message" in response_data["output"]["choices"][0] and "content" in response_data["output"]["choices"][0]["message"]:
                    return response_data["output"]["choices"][0]["message"]["content"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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




def grade_essay_test(title,essay_content, content_samples,structure_samples,language_samples,DASHSCOPE_API_KEY,rule,knowledge,model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        
        prompt = f"""
请批改作文，并按要求给出评价，仅进行评价不修改作文、不创造作文，仅返回一遍评论无需总结，评论内容需丰富、详细、具体，全程基于作文文本本身独立诊断，核心提升批改的全面性、精准性与维度均衡性，助力提升总体匹配质量，要求如下：

评价要求：
1.  核心问题定位：挖本质，拒表层
先拆解文本主题的核心内涵，明确文本偏离本质的具体表现（例：若文本写 “雕版印刷” 却侧重 “文化传承”，需指出 “偏离‘低谷中的坚持’主线，主题摇摆”）；
识别逻辑断裂的根源（如 “科技与人文” 类文本若仅提 “15 秒便民” 未关联 “铁路网建设”，需点明 “案例与主题的因果链缺失，未体现科技对人文的赋能逻辑”）。

2.  主题聚焦：锚核心，补关联
明确文本需紧扣的核心主题，排查 “内容堆砌” 问题（如记叙文若同时写 “数学、豫剧、体测”，需建议 “聚焦 1 件事，如‘体测突破’，删除无关事件”）；
挖掘题目隐含的潜在关联（如 “传统文化类文本” 需关联 “传统与现代 / 科技”，“时代主题类文本” 需关联 “当代青年视角”，例：“剪纸文化” 需补充 “科技助力剪纸创新 / 避免科技导致的文化浮躁”）。

3.  细节指导：具象化，给方案
针对 “细节缺失”，提供可落地的场景 / 动作 / 心理描写方向
（例：写 “外婆做短视频” 需建议 “补充‘学剪辑时反复调整时间轴、向晚辈讨论文案措辞’的细节”；
写 “引体向上成长” 需建议 “添加‘肌肉酸痛时握杆的指节发白、想放弃时盯着操场标语的心理活动’”）；
针对 “结构问题”，给出具体逻辑框架 / 过渡示例（如议论文分论点需建议 “按‘精神层面→实践层面’递进，过渡句可写‘若说初心是坚守的内核，那么当代青年的行动便是内核的具象化’”；
记叙文过渡需建议 “从‘失败场景’到‘觉醒瞬间’可加‘书包里的错题本滑出来，扉页的 “再试一次” 突然扎进眼里’”）。

4.  思辨与升华：拓维度，强关联
引导社会意义 / 时代背景关联（如 “科研经历类文本” 需建议 “关联‘青年科技人才助力科技强国’，补充‘实验失败时想到 “卡脖子技术突破需坚持” 的时代号召’”；
“文化类文本” 需建议 “关联‘文化出海 / 文化自信’，如‘《黑神话》海外玩家讨论孙悟空形象，体现传统文化现代表达的国际影响力’”）；
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

内容和主题维度可参考样例：
{content_samples}
结构维度可参考样例：
{structure_samples}
语言表达维度可参考样例：
{language_samples}
注意：参考样例仅供学习如何评价相似作文与对应格式，其中的评论不能直接照搬，比如五篇样例均提到紧扣主题，但待批改的作文可能没有完全符合主题，所以评论中不能提到紧扣主题。

在批改完成后，你还需要进一步对作文与生成的批改结果进行对比，检查并修改错误的地方。
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
    
"""辅助评价规则：
{rule}
注意：评价规则仅供参考，须结合实际作文给出评价。"""

def grade_essay_origin(title,essay_content, samples,DASHSCOPE_API_KEY,rule,knowledge,model_name):
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        
        prompt = f"""
请你按照固定步骤独立批改下面这篇作文，只评价、不修改、不创作，严格按照「内容和审题 — 文章结构 — 语言表达」三部分输出，每部分包含：优点、不足、建议。
第一步：整体审题判断（先定方向，不跑偏）
先明确题目核心关键词是什么。
判断作文是否紧扣关键词、是否偏题、是否偏离题干主线。
检查是否满足题目要求：文体是否正确、格式是否规范、是否使用题干材料。
第二步：内容与审题评价（按顺序点评）
优点
点评是否扣题、立意是否明确。
点评素材是否真实、情感是否真挚。
点评事件是否完整、主题是否集中。
不足
先指出：主题理解是否偏差、是否偏离关键词主线。
再指出：事件与感悟是否脱节、事例是否不能支撑主题。
最后指出：是否内容空泛、是否停留在个人层面无合理升华。
建议
明确告诉学生：要聚焦哪一件事、删掉哪些无关内容。
补全 “事件→感悟” 的简单逻辑，不做复杂思辨。
给出一句可直接写进作文的细节或感悟示例。
第三步：文章结构评价（按顺序点评）
优点
点评结构是否完整、顺序是否合理。
点评是否首尾呼应、段落是否清晰。
不足
指出段落之间是否过渡生硬、跳转突兀。
指出详略是否不当：重点内容太短、次要内容太长。
指出结尾是否仓促、升华不足。
建议
给出一句具体过渡句，让上下文更连贯。
明确：压缩哪一段、扩充哪一段。
让结尾回扣开头，形成简单闭环。
第四步：语言表达评价（按顺序点评）
优点
点评语句是否通顺、用词是否准确。
点评修辞是否恰当、是否有画面感。
不足
指出是否有错别字、用词不当、搭配不妥。
指出是否句子重复、句式单调、表达平淡。
指出是否语言空泛、缺少真实细节。
建议
修正明显错误的词语或句子。
给出一处具体细节补充方向。
保持原文风格，不强行加文采。

评价规则：{rule}
注意：评价规则仅供参考，须结合实际作文内容给出评价。

 



在批改完成后，你还需要进一步对作文与生成的批改结果进行对比，检查并修改错误的地方。
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



    
"""    评价要求(需逐条落实):
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
避免 “个人层面局限”：将 “个人学会书法” 升华为 “书法练习中领悟的‘静心坚持’，正是当代青年应对浮躁社会的重要品质”。"""

def judge_essay_with_qwen(essay_content, input_comment,DASHSCOPE_API_KEY,rule,knowledge="none"):
    """调用Qwen-2.5 API批改作文，修正响应判断逻辑"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        prompt = f"""你是一名有着丰富批改经验的老师，你的任务是判断作文和给出的评论是否匹配，并对作文评价进行格式化：
评价规则：
{rule}
背景知识：
{knowledge}
要求：对于作文和评论匹配的的小点，删除评论的理由，
对于不匹配的小点，删去它
格式要求：先分三个大类，一到三：
一、内容和审题
二、文章结构
三、语言表达
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个维度的每种评价中的每个小点前要加上“1.” 依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”
输入的评论如下：
{input_comment}
作文内容：
{essay_content}
只返回格式化后的评价

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


    
def handle_workflow_iterator(stream: Stream[WorkflowEvent]):
    for event in stream:
        if event.event == WorkflowEventType.MESSAGE:
            print("收到消息", event.message.content)
            return event.message.content

        elif event.event == WorkflowEventType.ERROR:
            print("出现错误", event.error)
        elif event.event == WorkflowEventType.INTERRUPT:
            handle_workflow_iterator(
                coze.workflows.runs.resume(
                    workflow_id=workflow_id,
                    event_id=event.interrupt.interrupt_data.event_id,
                    resume_data="hey",
                    interrupt_type=event.interrupt.interrupt_data.type,
                )
            )








def extract_problem_types(essay_content, DASHSCOPE_API_KEY):
    """
    从作文中抽取“专家判题级”的问题类型标签 + 简短判断摘要
    """

    prompt = f"""
你是一名中高考作文阅卷教师。

请只做【判断】，不要给建议。

从下列作文中判断其主要存在的问题类型（可多选）：
- 审题偏差
- 主题分散
- 结构混乱
- 结构规范但表达一般
- 修辞过度
- 语言准确但不出彩
- 考场规范风险
- 内容不适龄
- 时代意识不足
- 已具备时代意识

输出格式必须严格为 JSON：
{{
  "problem_types": ["问题1", "问题2"],
  "summary": "用一句话概括专家会如何评价这篇作文的主要问题或优势"
}}

作文内容：
{essay_content}
"""

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
    }

    data = {
        "model": "deepseek-v3.1",
        "input": {"prompt": prompt},
        "parameters": {"temperature": 0.2, "max_tokens": 800}
    }

    response = requests.post(url, headers=headers, json=data, timeout=120)
    response_data = response.json()

    try:
        content = response_data["output"]["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"问题类型抽取失败: {e}")


























































































































def content_enhance(comment,text, samples,DASHSCOPE_API_KEY):
    """调用Qwen-2.5 API批改作文，修正响应判断逻辑"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        prompt = f"""请参考传入的作文和参考评论，提高内容维度的批改质量，并从以下的要求给出详细反馈：
你能且只能修改的评论维度为：一、内容立意与主题表达（内容维度）
格式要求：
首先，返回 一、内容立意与主题表达（内容维度）
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个维度的每种评价中的每个小点前要加上“1.” 依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”
参考样例如下：
{samples}
作文内容：
{text}
你需要修改的评论：
{comment}
仅需要返回你所修改维度的评论，不要遗漏该维度的评论，但其它一律不准返回
评论的格式要符合要求
"""
        # 调用Qwen-2.5 API
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
        print("API返回内容: ",response_data["output"]["text"])
        
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:

                # 直接检查是否存在output.text字段
                if "output" in response_data and "text" in response_data["output"]:
                    return response_data["output"]["text"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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
    
def style_enhance(comment,text, samples,DASHSCOPE_API_KEY):
    """调用Qwen-2.5 API批改作文，修正响应判断逻辑"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        prompt = f"""请参考传入的作文和参考评论，提高语言维度的批改质量，并从以下的要求给出详细反馈：
你能且只能修改的评论维度为：三、语言表达与文采风格（语言维度）

格式要求：
首先，返回 三、语言表达与文采风格（语言维度）
分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个维度的每种评价中的每个小点前要加上“1.” 依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”
参考样例如下：
{samples}
作文内容：
{text}
你需要修改的评论：
{comment}
仅需要返回你所修改维度的评论，不要遗漏该维度的评论，但其它一律不准返回
"""
        # 调用Qwen-2.5 API
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
        print("API返回内容: ",response_data["output"]["text"])
        
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                # 直接检查是否存在output.text字段
                if "output" in response_data and "text" in response_data["output"]:
                    return response_data["output"]["text"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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

def structure_enhance(comment,text, samples,DASHSCOPE_API_KEY):
    """调用Qwen-2.5 API批改作文，修正响应判断逻辑"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置DASHSCOPE_API_KEY，请在.env文件中设置"
    
    try:
        # 构建提示词
        prompt = f"""请参考传入的作文和参考评论，提高结构维度的批改质量，并从以下的要求给出详细反馈：
你能且只能修改的评论维度为：二、结构安排与逻辑组织（结构维度）
格式要求：
首先，返回 二、结构安排与逻辑组织（结构维度）
再分成优点，不足，建议，三种评价
优点，不足，建议三种评价前不需要数字标号
每个维度的每种评价中的每个小点前要加上“1.” 依次说明
如果某一大类中没有相关的点评，就写一个“无。”；如果某一个大类下面没有某个小类（如优点、不足、建议），不用写该小类的名字及“无”
参考样例如下：
{samples}
作文内容：
{text}
你需要修改的评论：
{comment}
仅需要返回你所修改维度的评论，不要遗漏该维度的评论，但其它一律不准返回
"""

        # 调用Qwen-2.5 API
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
        print("API返回内容: ",response_data["output"]["text"])
        
        # 解析响应（忽略code字段，直接基于HTTP状态码和output字段判断）
        if response.status_code == 200:
            try:
                # 直接检查是否存在output.text字段
                if "output" in response_data and "text" in response_data["output"]:
                    return response_data["output"]["text"]
                else:
                    return f"API返回格式异常：缺少output.text字段，响应内容：{response.text}"
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

def enhance_comment(text,comment,es_client,DASHSCOPE_API_KEY,num,model):
    comment_upgrade(text,comment,model)
    with open(r'D:\zuowen\input\search_query\KNN\content.txt', "r", encoding="utf-8") as f:
        content = f.read().rstrip()
    with open(r'D:\zuowen\input\search_query\KNN\structure.txt', "r", encoding="utf-8") as f:
        structure = f.read().rstrip()
    with open(r'D:\zuowen\input\search_query\KNN\style.txt', "r", encoding="utf-8") as f:
        style = f.read().rstrip()
    def search_single_dimension(query):
        return processESIndex_Knn(query, "research_index_knn", es_client)
    
    # 2. 并行执行3个维度的检索
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交3个检索任务，获取未来对象
        future_content = executor.submit(search_single_dimension, content)
        future_structure = executor.submit(search_single_dimension, structure)
        future_style = executor.submit(search_single_dimension, style)
        # 等待所有任务完成，获取结果
        docs_content = future_content.result()
        docs_structure = future_structure.result()
        docs_style = future_style.result()

    f1=open(r"D:\zuowen\output\content.txt",'w',encoding='utf-8')
    f2=open(r"D:\zuowen\output\structure.txt",'w',encoding='utf-8')
    f3=open(r"D:\zuowen\output\style.txt",'w',encoding='utf-8')
    count=0
    for i in docs_content:
        count+=1
        if i['_source']['id']==num:
            print(i['_source']['id'],end='    ')
            print(i['_score'])
            continue
        if count>5:
            break
        f1.write(str(i['_source']['title'])+'\n'+str(i['_source']['text'])+'\n'+str(i['_source']['comment']))
        f1.write("----------------------------"*20)
        print(i['_source']['id'],end='    ')
        print(i['_score'])
    count=0
    for i in docs_structure:
        count+=1
        if i['_source']['id']==num:
            print(i['_source']['id'],end='    ')
            print(i['_score'])
            continue
        if count>5:
            break
        f2.write(str(i['_source']['title'])+'\n'+str(i['_source']['text'])+'\n'+str(i['_source']['comment']))
        f2.write("----------------------------"*20)
        print(i['_source']['id'],end='    ')
        print(i['_score'])
    count=0
    for i in docs_style:
        count+=1
        if i['_source']['id']==num:
            print(i['_source']['id'],end='    ')
            print(i['_score'])
            continue
        if count>5:
            break
        f3.write(str(i['_source']['title'])+'\n'+str(i['_source']['text'])+'\n'+str(i['_source']['comment']))
        f3.write("----------------------------"*20)
        print(i['_source']['id'],end='    ')
        print(i['_score'])
    f1.close()
    f2.close()
    f3.close()
    with open(r"D:\zuowen\output\content.txt", "r", encoding="utf-8") as f:
        content = f.read().rstrip()
    with open(r"D:\zuowen\output\structure.txt", "r", encoding="utf-8") as f:
        structure = f.read().rstrip()
    with open(r"D:\zuowen\output\style.txt", "r", encoding="utf-8") as f:
        style = f.read().rstrip()
    temp_content=extract_dimension_content(comment,1)
    temp_structure=extract_dimension_content(comment,2)
    temp_style=extract_dimension_content(comment,3)
    with open(r"D:\zuowen\output\temp.txt", 'w',encoding='utf-8') as file:
        file.write(temp_content+'\n'+temp_structure+'\n'+temp_style)
    result_content=content_enhance(temp_content,text,content,DASHSCOPE_API_KEY)
    print("content_enhance done")
    result_structure=structure_enhance(temp_structure,text,structure,DASHSCOPE_API_KEY)
    print("structure_enhance done")
    result_style=style_enhance(temp_style,text,style,DASHSCOPE_API_KEY)
    print("style_enhance done")
    rest=extract_dimension_content(comment,4)+'\n'+extract_dimension_content(comment,5)
    return result_content+'\n'+result_structure+'\n'+result_style+'\n'+rest




