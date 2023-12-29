# FedLoRA

# 环境准备
```bash
conda create --name xtuner-env python=3.10 -y
source activate xtuner-env
pip install xtuner
# deepspeed 支持
pip install deepspeed
conda install mpi4py-mpich
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


