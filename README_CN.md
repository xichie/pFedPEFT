# FedLoRA

# 环境准备
```bash
conda create --name xtuner-env python=3.10 -y
source activate xtuner-env
pip install -U 'xtuner[deepspeed]'
pip install protobuf
```
# 数据准备
把数据放到`/data/logad/`
# 模型准备
把模型放到任意路径，例如`~/llama2-7b/`

# 运行步骤
## 训练
```bash
export NPROC_PER_NODE=4 # 启动四卡
nohup xtuner train train/llama_7b_qlora_alpaca_e3.py --work-dir /data/qjx/logad/client_1 > client_1.log 2>&1 &
```
参数解释：
- `train/llama_7b_qlora_alpaca_e3.py`：训练脚本，参数在里面改
    - 只需要改两个参数
        - `pretrained_model_name_or_path`: llama2模型路径
        - `alpaca_en_path`: 数据集路径
- `--work-dir`: LoRA模块输出的路径

# 实验

我们要模拟的场景是：
存在多个client，目前client的数量设置为3。
- 对于每个client，按照上述的训练步骤，用自己的数据训练模型
    - 自己的数据通过把训练数据分成3份来模拟：`data/logad/split_1/2/3.json`

- 因为只有一台机器，目前只能每次训练一个client，训练H=3个epoch，得到自己的本地LoRA参数
    - 我们这里还要copy一份本地LoRA参数，作为全局的LoRA参数
- 3个client分别训练3个epoch后，得到了3个不同的全局LoRA参数，将这些参数加权求和（模拟server的作用），分发给每个client。
- 重复上面的步骤T次


# LoRA 怎么加载
一行代码：
```python
# 加载训练好的LoRA参数，注意base model参数不变
peft_model='/data/qjx/logad/client_1/epoch_1.pth'  
```
