import json

data_path = 'data/logad/bgl_ad_alpaca_train.json'
with open(data_path, 'r') as f:
    data = json.load(f)

print(len(data))