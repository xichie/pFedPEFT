# Env
```bash
conda create --name xtuner-env python=3.10 -y
source activate xtuner-env
pip install -U 'xtuner[deepspeed]'
pip install protobuf
```
# Data
Copy the data into`/data/logad/`
# Model
Copy the model in any path, such as `~/llama2-7b/`

# Run
## Training
```bash
export NPROC_PER_NODE=4 # enable 4 GPUs
nohup xtuner train train/llama_7b_qlora_alpaca_e3.py --work-dir /data/qjx/logad/client_1 > client_1.log 2>&1 &
```
Parameters：
- `train/llama_7b_qlora_alpaca_e3.py`：Training script, parameters are changed in it
    - Just change two parameters
        - `pretrained_model_name_or_path`: model path
        - `alpaca_en_path`: data path
- `--work-dir`: LoRA module saved path

# Exp
The scenario we want to simulate is:
There are multiple clients, and the number of clients is currently set to 3.
- For each client, follow the above training steps to train the model with your own data
- Simulate your own data by dividing the training data into 3 parts: `data/logad/split_1/2/3.json`

- Because there is only one machine, you can only train one client at a time, train H=3 epochs, and get your own local LoRA parameters
- We also need to copy a copy of the local LoRA parameters as global LoRA parameters
- After training 3 clients for 3 epochs respectively, we get 3 different global LoRA parameters, and weight these parameters (simulating the role of the server) and distribute them to each client.
- Repeat the above steps T times


# LoRA load
1 line code：
```python
# Load the trained LoRA parameters, note that the base model parameters remain unchanged
peft_model='/data/qjx/logad/client_1/epoch_1.pth'  
```
