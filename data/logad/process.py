import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def generate_alpaca_format():
    window_size = 50
    print('Loading...')
    df = pd.read_csv('data/logad/BGL.log_structured.csv')
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    content_list = df['Content'].to_list()
    label_list = df['Label'].to_list() 
    
    assert len(content_list) == len(label_list)
    
    total_logs = len(content_list)
    total_window= int(total_logs / 50)
    
    question = "Please determine if the given log messages are an anomaly or not. If is an anomaly, output 'yes', otherwise, output 'no'. Do not output addition words."
    
    formatted_data = []
    for i in tqdm(range(total_window)):
        start_index = i*window_size
        end_index = (i+1)*window_size
        content = content_list[start_index:end_index]
        label = max(label_list[start_index:end_index])

        new_record = {'conversation': [{
                    'input':  'Question:\n' + question + '\n###Logs:' + '\n'.join(content),
                    'output': 'yes' if label == 1 else 'no'
                }]
            }
        formatted_data.append(new_record)
        
    formatted_data = json.dumps(formatted_data, ensure_ascii=False, indent=4)
    with open('./data/logad/bgl_ad_alpaca.json', 'w') as w:
        w.write(formatted_data)

def split():
    with open('data/logad/bgl_ad_alpaca.json', 'r') as f:
        data = json.load(f)
        
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=22)
    
    print(len(X_train))
    print(len(X_test))
    X_train = json.dumps(X_train, ensure_ascii=False, indent=4)
    X_test = json.dumps(X_test, ensure_ascii=False, indent=4)
    with open('./data/logad/bgl_ad_alpaca_train.json', 'w') as w:
        w.write(X_train)
    with open('./data/logad/bgl_ad_alpaca_test.json', 'w') as w:
        w.write(X_test)

import json
import os

def split_json_file(input_file, output_directory, num_splits):
    with open(input_file, 'r') as f:
        data = json.load(f)

    num_items = len(data)
    items_per_split = num_items // num_splits
    remainder = num_items % num_splits
    print(num_items)
    start = 0
    for i in range(num_splits):
        end = start + items_per_split + (1 if i < remainder else 0)
        output_data = data[start:end]

        output_file = os.path.join(output_directory, f'split_{i+1}.json')
        with open(output_file, 'w') as out_f:
            json.dump(output_data, out_f, indent=2)

        start = end


if __name__ == '__main__':
    # format2alpaca('hdfs_qa_alpaca.json')
    
    # generate_alpaca_format()
    # split()
    ############
    input_file = "data/logad/bgl_ad_alpaca_train.json"  # 请替换为实际的文件路径
    output_directory = "data/logad/"  # 请替换为实际的输出目录

    # 指定要分割成的子文件数量
    num_splits = 3
    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)
    # 分割JSON文件
    split_json_file(input_file, output_directory, num_splits)
  