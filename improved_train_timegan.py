"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting - 修改为CPU优先
device = torch.device("cpu")  # 强制使用CPU，避免CUDA相关问题

# model parameter setting
batch_size = 32  # 减小batch size适配APT数据
max_len = 64     # APT序列长度通常较短
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# APT攻击检测特定参数
num_classes_binary = 2    # 二分类：正常/攻击
num_classes_multi = 5     # 多分类：正常/APT1/APT2/APT3/APT4
feature_dim = 3           # APT特征维度

# optimizer parameter setting
init_lr = 1e-4    # 稍微提高学习率
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 100       # 减少训练轮数
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

# 数据路径配置
binary_data_path = './attack_detection_binary'
multiclass_data_path = './attack_detection_multiclass'
