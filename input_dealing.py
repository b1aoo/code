import os
import json
os.chdir(r"D:\zuowen\data\test")
jsonl_file_path=r"D:\zuowen\data\1.jsonl"

def num_change(num):
    if num < 10:
        return '000'+str(num)
    elif num < 100:
        return '00'+str(num)
    else:
        return '0'+str(num)

for i in range(101,654):
    with open("202506"+num_change(i)+"-q.txt",'r',encoding='utf-8') as f:
        title=f.read().strip()
    with open("202506"+num_change(i)+"-s.txt",'r',encoding='utf-8') as f:
        content=f.read().strip()
    with open("202506"+num_change(i)+"-ta.txt",'r',encoding='utf-8') as f:
        comment=f.read().strip()
    json_data = {
                "id": i,
                "title": title,  
                "content": content,  
                "comment": comment  
            }
    with open(jsonl_file_path,'a',encoding='utf-8') as f:
        f.write(json.dumps(json_data, ensure_ascii=False) + '\n')