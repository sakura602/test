#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
条件式时间序列 GAN (Conditional TimeGAN) 训练和生成脚本

该脚本用于训练条件 TimeGAN 模型和生成带有类别标签的时间序列数据。
基于原始 TimeGAN 实现，增加了条件生成功能，支持多种优化和功能增强。

特性:
- 支持从检查点恢复训练
- GPU 加速和内存优化
- 高效数据加载和处理
- 序列质量评估
- TensorBoard 集成
- 多样化生成控制

作者: Your Name
创建日期: 2023-05-15
最后修改: 2023-05-30
版本: 1.0.0

参考文献:
- Yoon, J., Jarrett, D., & van der Schaar, M. (2019). 
  Time-series Generative Adversarial Networks. 
  In Advances in Neural Information Processing Systems (pp. 5508-5518).

使用示例:
$ python improved_train_timegan.py --mode train_and_generate --input_file atk_seqs.pkl
"""

import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import traceback
import sys
from sklearn.manifold import TSNE
from improved_timegan_model import TimeGAN
from conditional_timegan_model import ConditionalTimeGAN
from timegan_utils import (
    preprocess_apt_data,
    postprocess_generated_data,
    save_generated_sequences
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from itertools import islice
from tqdm import tqdm
import time
import json
import datetime
import random

# 全局异常处理
def global_exception_handler(exctype, value, tb):
    print(''.join(traceback.format_exception(exctype, value, tb)))
    sys.exit(1)

sys.excepthook = global_exception_handler

print("脚本开始执行...")
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"NumPy 版本: {np.__version__}")
print(f"当前工作目录: {os.getcwd()}")
print(f"脚本文件: {__file__}")
print(f"导入的模块: {sys.modules.keys()}")
try:
    print(f"TimeGAN 类: {TimeGAN}")
    print(f"ConditionalTimeGAN 类: {ConditionalTimeGAN}")
except Exception as e:
    print(f"导入类时出错: {e}")
    raise

# 定义Dataset类，用于处理APT序列数据
class TimeGANDataset(Dataset):
    """处理时间序列数据的Dataset类
    
    参数:
        data: 序列数据 [batch_size, seq_len, feature_dim]
        labels: 序列标签 [batch_size]
    """
    def __init__(self, data, labels=None):
        self.data = torch.FloatTensor(data)
        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

def load_apt_sequences(file_path=None):
    """加载APT攻击序列
    
    Args:
        file_path: 序列文件路径，如果为None则使用默认路径
        
    Returns:
        sequences: APT攻击序列列表
    """
    # 使用默认的APT攻击序列文件路径
    if file_path is None or not os.path.exists(file_path):
        default_file_path = "sequences_3tuple/apt_attack_sequences.pkl"
        if os.path.exists(default_file_path):
            file_path = default_file_path
            print(f"使用默认APT攻击序列文件路径: {file_path}")
        else:
            print(f"警告: 默认APT攻击序列文件 {default_file_path} 不存在，将尝试使用提供的路径: {file_path}")
    
    print(f"正在从 {file_path} 加载APT攻击序列...")
    
    try:
        with open(file_path, 'rb') as f:
            apt_sequences = pickle.load(f)
        
        print(f"共加载 {len(apt_sequences)} 条APT攻击序列")
        
        # 检查序列结构，适配为统一格式
        adapted_sequences = []
        for seq in apt_sequences:
            # 确定标签
            label = seq.get('label', 0)
            
            # 创建适配的序列字典
            adapted_seq = {
                'features': seq['features'],
                'label': label,
                'phases': seq.get('phases', []),
                'id': seq.get('id', ''),
                'data': seq.get('features', [])  # 添加data字段作为备用
            }
            
            # 添加到适配序列列表
            adapted_sequences.append(adapted_seq)
        
        # 统计标签分布
        labels = [seq['label'] for seq in adapted_sequences]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"APT攻击序列标签分布: {label_counts}")
        
        return adapted_sequences, labels
    except Exception as e:
        print(f"加载APT攻击序列时发生错误: {e}")
        print(f"将尝试加载并过滤 atk_seqs.pkl 文件...")
        
        # 如果加载失败，尝试从 atk_seqs.pkl 加载并过滤
        backup_file_path = "atk_seqs.pkl"
        if os.path.exists(backup_file_path):
            try:
                with open(backup_file_path, 'rb') as f:
                    all_sequences = pickle.load(f)
                
                print(f"从 {backup_file_path} 加载了 {len(all_sequences)} 条序列")
                
                # 区分APT和NAPT序列
                apt_sequences = []
                
                for seq in all_sequences:
                    # 确定序列标签
                    if 'apt_type' in seq:
                        if isinstance(seq['apt_type'], str) and seq['apt_type'].startswith('APT'):
                            label = int(seq['apt_type'].replace('APT', ''))
                        else:
                            label = seq['apt_type'] if isinstance(seq['apt_type'], int) else 0
                    else:
                        # 使用已有的label字段
                        label = seq.get('label', 0)
                    
                    # 创建适配的序列字典
                    adapted_seq = {
                        'features': seq['features'],
                        'label': label,
                        'phases': seq.get('phases', []),
                        'id': seq.get('session_id', ''),
                        'data': seq.get('features', [])  # 添加data字段作为备用
                    }
                    
                    # 仅保留APT攻击序列 (标签为1-4)
                    if 1 <= label <= 4:
                        apt_sequences.append(adapted_seq)
                
                # 统计APT标签分布
                apt_labels = [seq['label'] for seq in apt_sequences]
                apt_label_counts = {}
                for label in apt_labels:
                    apt_label_counts[label] = apt_label_counts.get(label, 0) + 1
                
                print(f"过滤后的APT序列数量: {len(apt_sequences)}")
                print(f"APT标签分布: {apt_label_counts}")
                
                return apt_sequences, apt_labels
            except Exception as backup_e:
                print(f"加载备用文件时发生错误: {backup_e}")
                print("无法加载任何APT攻击序列，返回空列表")
                return [], []
        else:
            print(f"备用文件 {backup_file_path} 不存在，返回空列表")
            return [], []

def balance_apt_sequences(sequences, target_nums):
    """平衡APT攻击序列
    
    Args:
        sequences: APT攻击序列列表
        target_nums: 目标数量字典，例如 {1: 50, 2: 50, 3: 50, 4: 50}
        
    Returns:
        balanced_sequences: 平衡后的序列列表
    """
    # 按标签分组
    grouped_sequences = {}
    for seq in sequences:
        label = seq['label']
        if label not in grouped_sequences:
            grouped_sequences[label] = []
        grouped_sequences[label].append(seq)
    
    # 平衡数据
    balanced_sequences = []
    for label, target_num in target_nums.items():
        if label not in grouped_sequences:
            print(f"警告: 找不到标签为 {label} 的序列")
            continue
            
        group = grouped_sequences[label]
        
        if len(group) >= target_num:
            # 如果原始数量足够，随机选取
            selected = np.random.choice(group, target_num, replace=False)
            balanced_sequences.extend(selected)
        else:
            # 如果原始数量不足，全部选取并复制到目标数量
            balanced_sequences.extend(group)
            
            # 复制序列
            num_copies = target_num - len(group)
            if num_copies > 0:
                copies = np.random.choice(group, num_copies, replace=True)
                balanced_sequences.extend(copies)
    
    print(f"平衡后的序列数量: {len(balanced_sequences)}")
    
    # 统计标签分布
    labels = [seq['label'] for seq in balanced_sequences]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"平衡后的标签分布: {label_counts}")
    
    return balanced_sequences

def visualize_tsne(original_data, generated_data, labels, output_dir):
    """使用t-SNE可视化原始和生成的数据
    
    Args:
        original_data: 原始数据 [no, seq_len, dim]
        generated_data: 生成数据 [no, seq_len, dim]
        labels: 标签 [no]
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 将3D数据展平为2D数据
    original_data_reshaped = original_data.reshape(original_data.shape[0], -1)
    generated_data_reshaped = generated_data.reshape(generated_data.shape[0], -1)
    
    # 合并数据
    combined_data = np.vstack([original_data_reshaped, generated_data_reshaped])
    
    print("正在进行t-SNE降维，这可能需要几分钟...")
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(original_data)-1))
    tsne_results = tsne.fit_transform(combined_data)
    
    # 分离原始和生成的t-SNE结果
    tsne_original = tsne_results[:len(original_data)]
    tsne_generated = tsne_results[len(original_data):]
    
    # 绘制t-SNE图
    plt.figure(figsize=(12, 10))
    
    # 原始数据
    plt.scatter(tsne_original[:, 0], tsne_original[:, 1], c='blue', label='原始数据', alpha=0.7)
    
    # 生成数据
    plt.scatter(tsne_generated[:, 0], tsne_generated[:, 1], c='red', label='生成数据', alpha=0.7)
    
    plt.title('t-SNE: 原始数据 vs 生成数据')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=300)
    plt.close()
    
    # 按标签绘制t-SNE图
    plt.figure(figsize=(12, 10))
    
    # 设置颜色映射
    colors = ['blue', 'green', 'orange', 'purple']
    
    # 确保标签的长度和原始数据一致
    if len(labels) > len(original_data):
        orig_labels = labels[:len(original_data)]
    else:
        orig_labels = labels
    
    # 生成数据的标签（假设与原始标签长度相同）
    gen_labels = np.array([i % 4 for i in range(len(generated_data))])  # 简单循环分配标签
    
    # 获取唯一标签
    labels_set = sorted(set(orig_labels))
    
    # 原始数据，按标签着色
    for i, label in enumerate(labels_set):
        indices = np.where(orig_labels == label)[0]
        if len(indices) > 0:
            plt.scatter(
                tsne_original[indices, 0], 
                tsne_original[indices, 1], 
                c=colors[i % len(colors)], 
                label=f'原始-标签{label}', 
                alpha=0.7
            )
    
    # 生成数据，按标签着色
    for i, label in enumerate(labels_set):
        gen_indices = np.where(gen_labels == label)[0]
        if len(gen_indices) > 0:
            plt.scatter(
                tsne_generated[gen_indices, 0], 
                tsne_generated[gen_indices, 1], 
                c=colors[i % len(colors)], 
                label=f'生成-标签{label}', 
                marker='x',
                alpha=0.7
            )
    
    plt.title('t-SNE: 按标签分类的原始数据 vs 生成数据')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization_by_label.png'), dpi=300)
    plt.close()
    
    print("t-SNE可视化已保存")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Improved TimeGAN for APT Attack Sequence Generation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'], help='训练或生成模式')
    parser.add_argument('--input_file', type=str, default='sequences_3tuple/apt_attack_sequences.pkl', help='输入文件路径')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--model_path', type=str, default='output/improved_timegan_model.pth', help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default='logs/improved_timegan', help='TensorBoard日志目录')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='用于恢复训练的检查点路径')
    parser.add_argument('--max_seq_len', type=int, default=10, help='最大序列长度')
    parser.add_argument('--feature_dim', type=int, default=79, help='特征维度')
    parser.add_argument('--hidden_dim', type=int, default=24, help='隐藏层维度')
    parser.add_argument('--z_dim', type=int, default=24, help='噪声维度')
    parser.add_argument('--num_layers', type=int, default=3, help='RNN层数')
    parser.add_argument('--gamma', type=float, default=1.0, help='判别器损失权重')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--pre_train_epochs', type=int, default=10, help='预训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--balance', action='store_true', help='是否平衡训练数据')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--generate_nums', type=str, default='100,100,100,200', help='各类型生成数量')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='生成时噪声缩放因子')
    parser.add_argument('--device', type=str, default=None, help='使用的设备')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载器的工作线程数')
    parser.add_argument('--profile', action='store_true', help='是否进行性能分析')
    args = parser.parse_args()
    
    print("开始执行主函数...")
    print(f"解析命令行参数完成: {args}")
    
    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    print(f"随机种子已设置为: {seed}")
    
    # 设置计算设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载APT攻击序列
    try:
        apt_sequences, labels = load_apt_sequences(args.input_file)
        print(f"加载完成: {len(apt_sequences)} 条APT攻击序列")
        
        # 检查并展示第一条序列的信息，处理不同的序列格式
        if len(apt_sequences) > 0:
            first_seq = apt_sequences[0]
            if isinstance(first_seq, dict):
                # 字典格式序列处理
                if 'features' in first_seq:
                    features = first_seq['features']
                    if isinstance(features, np.ndarray):
                        print(f"序列示例 - 第一条: 标签={labels[0]}, 特征形状={features.shape}")
                    else:
                        print(f"序列示例 - 第一条: 标签={labels[0]}, 特征类型={type(features)}")
                else:
                    print(f"序列示例 - 第一条: 标签={labels[0]}, 字典键={first_seq.keys()}")
            elif isinstance(first_seq, np.ndarray):
                # 直接是numpy数组
                print(f"序列示例 - 第一条: 标签={labels[0]}, 特征形状={first_seq.shape}")
            else:
                print(f"序列示例 - 第一条: 标签={labels[0]}, 类型={type(first_seq)}")
        
        # 分析标签分布情况
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"原始标签分布: {label_counts}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        traceback.print_exc()
        return
    
    # 设置生成目标数量
    generate_nums = {}
    for i, num in enumerate(args.generate_nums.split(',')):
        generate_nums[i+1] = int(num)
    print(f"生成目标数量: {generate_nums}")
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        print("进入训练模式...")
        
        # 预处理数据
        print("开始预处理数据...")
        data, lengths, preprocessed_labels, _, scaler = preprocess_apt_data(
            apt_sequences, args.max_seq_len, args.feature_dim
        )
        print(f"预处理后的数据形状: {data.shape}")
        print(f"标签形状: {preprocessed_labels.shape}, 类型: {preprocessed_labels.dtype}")
        
        # 确保标签是正数，并检查分布
        # 将负数标签映射到正数域
        if np.any(preprocessed_labels < 0):
            print("检测到负标签值，将其映射到非负整数")
            # 将标签映射到从0开始的索引
            unique_labels = np.unique(preprocessed_labels)
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            preprocessed_labels = np.array([label_map[lbl] for lbl in preprocessed_labels])
            
        print(f"处理后的标签分布: {np.bincount(preprocessed_labels.astype(int))}")
        
        # 分割训练集和验证集
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, preprocessed_labels, test_size=0.2, random_state=seed, stratify=preprocessed_labels
        )
        
        print(f"训练集形状: {train_data.shape}, 验证集形状: {val_data.shape}")
        print(f"训练集标签分布: {np.bincount(train_labels.astype(int))}")
        print(f"验证集标签分布: {np.bincount(val_labels.astype(int))}")
        
        # 平衡训练数据
        if args.balance:
            print("平衡训练数据...")
            train_data, train_labels = balance_training_data(train_data, train_labels)
            print(f"平衡后训练集形状: {train_data.shape}")
            print(f"平衡后训练集标签分布: {np.bincount(train_labels.astype(int))}")
        
        # 准备数据加载器
        train_dataset = TimeGANDataset(train_data, train_labels)
        val_dataset = TimeGANDataset(val_data, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers if args.num_workers else 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers if args.num_workers else 0
        )
        
        # 确定分类数
        num_categories = len(np.unique(preprocessed_labels))
        print(f"分类数量: {num_categories}")
        
        # 创建和训练模型
        model = ConditionalTimeGAN(
            args.feature_dim,
            args.hidden_dim,
            args.z_dim,
            num_categories,
            args.num_layers,
            args.gamma,
            args.learning_rate
        )
        
        model.to(device)
        
        # 如果需要恢复训练
        start_epoch = 0
        if args.resume and args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"从检查点恢复训练，起始轮数: {start_epoch}")
            else:
                print(f"检查点文件 {args.checkpoint_path} 不存在，从头开始训练")
        
        # 训练模型
        model.fit(
            train_loader,
            val_loader,
            args.epochs,
            args.pre_train_epochs,
            None,  # steps_per_epoch参数，设为None表示使用整个数据集
            True,  # verbose参数，设为True表示打印训练进度
            args.log_dir,
            start_epoch
        )
        
        # 训练完成后保存模型
        model.save(args.model_path)
        print(f"模型已保存到 {args.model_path}")
        
        # 保存归一化器和预处理信息
        with open(os.path.join(args.output_dir, 'improved_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        preprocess_info = {
            'max_seq_len': args.max_seq_len,
            'feature_dim': args.feature_dim
        }
        
        with open(os.path.join(args.output_dir, 'improved_preprocess_info.pkl'), 'wb') as f:
            pickle.dump(preprocess_info, f)
        
        print("训练和预处理信息保存完成")
        
    elif args.mode == 'generate':
        print("进入生成模式...")
        
        # 加载模型
        if not os.path.exists(args.model_path):
            print(f"模型文件 {args.model_path} 不存在")
            return
        
        print(f"从 {args.model_path} 加载模型")
        
        # 加载预处理信息
        scaler_path = os.path.join(args.output_dir, 'improved_scaler.pkl')
        preprocess_info_path = os.path.join(args.output_dir, 'improved_preprocess_info.pkl')
        
        if not os.path.exists(scaler_path) or not os.path.exists(preprocess_info_path):
            print("缺少归一化器或预处理信息文件")
            return
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(preprocess_info_path, 'rb') as f:
            preprocess_info = pickle.load(f)
        
        # 加载模型
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 确定分类数
        num_categories = len(generate_nums)
        
        model = ConditionalTimeGAN(
            args.feature_dim,
            args.hidden_dim,
            args.z_dim,
            num_categories,
            args.num_layers,
            args.gamma,
            args.learning_rate
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("模型加载成功")
        
        # 生成序列
        generated_data = []
        generated_labels = []
        
        for category, num_samples in generate_nums.items():
            print(f"生成APT类型 {category} 的样本，数量: {num_samples}")
            
            # 生成指定类别的样本
            category_id = category - 1  # 转换为从0开始的索引
            start_time = time.time()
            gen_data = model.generate(
                num_samples,
                args.max_seq_len,
                category_id,
                args.noise_scale,
                device
            )
            end_time = time.time()
            
            print(f"生成完成，耗时: {end_time - start_time:.2f}秒")
            print(f"生成数据形状: {gen_data.shape}")
            
            # 显示生成数据的统计信息
            print(f"生成数据统计: 最小值={gen_data.min():.4f}, 最大值={gen_data.max():.4f}, 均值={gen_data.mean():.4f}, 标准差={gen_data.std():.4f}")
            
            # 反归一化
            # 这里注意保持数据格式与预处理时一致
            gen_data_reshaped = gen_data.reshape(-1, args.feature_dim)
            gen_data_inverse = scaler.inverse_transform(gen_data_reshaped)
            gen_data = gen_data_inverse.reshape(-1, args.max_seq_len, args.feature_dim)
            
            # 将生成的数据和标签添加到列表中
            generated_data.extend(gen_data)
            generated_labels.extend([category] * num_samples)
        
        # 转换为numpy数组
        generated_data = np.array(generated_data)
        generated_labels = np.array(generated_labels)
        
        print(f"最终生成数据形状: {generated_data.shape}")
        print(f"标签分布: {dict(zip(*np.unique(generated_labels, return_counts=True)))}")
        
        # 保存生成的序列
        generated_sequences = []
        for i in range(len(generated_data)):
            seq = generated_data[i]
            label = int(generated_labels[i])
            generated_sequences.append({
                'sequence': seq.tolist(),
                'label': label,
                'apt_type': f"APT{label}"
            })
        
        # 保存为JSON格式
        with open(os.path.join(args.output_dir, 'generated_sequences.json'), 'w') as f:
            json.dump(generated_sequences, f, indent=2)
        
        # 保存生成元数据
        generation_metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': args.model_path,
            'noise_scale': args.noise_scale,
            'max_seq_len': args.max_seq_len,
            'feature_dim': args.feature_dim,
            'num_generated': len(generated_sequences),
            'label_distribution': {int(k): int(v) for k, v in zip(*np.unique(generated_labels, return_counts=True))}
        }
        
        with open(os.path.join(args.output_dir, 'generation_metadata.json'), 'w') as f:
            json.dump(generation_metadata, f, indent=2)
        
        print(f"生成的序列已保存到 {os.path.join(args.output_dir, 'generated_sequences.json')}")
        print(f"生成元数据已保存到 {os.path.join(args.output_dir, 'generation_metadata.json')}")
    
    else:
        print(f"不支持的模式: {args.mode}")

def plot_training_history(history, output_dir):
    """绘制训练历史
    
    Args:
        history: 训练历史字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['embedding_loss'], label='Train')
    if history['val_embedding_loss']:
        plt.plot(history['val_embedding_loss'], label='Validation')
    plt.title('Embedding Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['generator_loss'])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['discriminator_loss'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()

def extract_features_labels_phases(sequences):
    """提取特征、标签和阶段信息"""
    X = []  # 特征
    y = []  # 标签(APT类型)
    phases = []  # 阶段序列
    
    for seq in sequences:
        X.append(seq['features'])
        y.append(seq['label'] - 1)  # 转为0-3的标签
        phases.append([phase_to_id(p) for p in seq['phases']])
    
    return np.array(X), np.array(y), np.array(phases)

# 阶段映射
phase_mapping = {'SN': 0, 'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4}
def phase_to_id(phase):
    return phase_mapping.get(phase, 0)

# 用于检查生成的训练集和测试集的函数
def check_generated_datasets():
    # 加载训练集
    with open('generated_sequences/train_apt_sequences.pkl', 'rb') as f:
        train_sequences = pickle.load(f)

    # 加载测试集
    with open('generated_sequences/test_apt_sequences.pkl', 'rb') as f:
        test_sequences = pickle.load(f)

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"测试集样本数: {len(test_sequences)}")

    # 统计标签分布
    train_labels = [seq['label'] for seq in train_sequences]
    test_labels = [seq['label'] for seq in test_sequences]

    for label in range(1, 5):
        train_count = train_labels.count(label)
        test_count = test_labels.count(label)
        print(f"APT{label} - 训练集: {train_count}样本, 测试集: {test_count}样本")

def evaluate_sequence_quality(real_data, generated_data, real_labels=None, generated_labels=None, labels=None, prefix=''):
    """评估生成序列的质量
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        real_labels: 真实数据的标签，形状为 [n_samples] (可选)
        generated_labels: 生成数据的标签，形状为 [n_samples] (可选)
        labels: 兼容旧版本的标签参数 (将被弃用)
        prefix: 评估指标前缀
        
    Returns:
        metrics: 评估指标字典
    """
    print(f"\n{prefix}评估生成序列质量...")
    
    # 将数据转换为numpy数组
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()
    
    # 确保labels兼容性（如果同时提供了real_labels和labels，优先使用real_labels）
    if real_labels is not None:
        if isinstance(real_labels, torch.Tensor):
            real_labels = real_labels.cpu().numpy()
    elif labels is not None:
        real_labels = labels
        if isinstance(real_labels, torch.Tensor):
            real_labels = real_labels.cpu().numpy()
    
    if generated_labels is not None and isinstance(generated_labels, torch.Tensor):
        generated_labels = generated_labels.cpu().numpy()
    
    # 打印标签分布（如果有）
    if real_labels is not None:
        unique_real, counts_real = np.unique(real_labels, return_counts=True)
        print(f"{prefix}真实数据标签分布: {dict(zip(unique_real, counts_real))}")
    
    if generated_labels is not None:
        unique_gen, counts_gen = np.unique(generated_labels, return_counts=True)
        print(f"{prefix}生成数据标签分布: {dict(zip(unique_gen, counts_gen))}")
    
    # 1. 判别性度量（训练分类器区分真实和生成数据）
    discriminative_score = calculate_discriminative_score(real_data, generated_data)
    print(f"{prefix}判别性分数 (AUC)：{discriminative_score:.4f} (理想值: 0.5)")
    
    # 2. 预测性度量（用前n-1步预测第n步）
    predictive_score = calculate_predictive_score(real_data, generated_data)
    print(f"{prefix}预测性分数 (MAE比率)：{predictive_score:.4f} (理想值: 1.0)")
    
    # 返回评估指标
    return {
        'discriminative_score': discriminative_score,
        'predictive_score': predictive_score
    }

def calculate_discriminative_score(real_data, generated_data):
    """计算判别性分数（使用分类器区分真实和生成数据）
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        
    Returns:
        score: 判别性分数（AUC），理想值为0.5
    """
    # 重塑数据为2D
    real_data_2d = real_data.reshape(real_data.shape[0], -1)
    generated_data_2d = generated_data.reshape(generated_data.shape[0], -1)
    
    # 平衡数据集大小
    min_samples = min(len(real_data_2d), len(generated_data_2d))
    if len(real_data_2d) > min_samples:
        indices = np.random.choice(len(real_data_2d), min_samples, replace=False)
        real_data_2d = real_data_2d[indices]
    if len(generated_data_2d) > min_samples:
        indices = np.random.choice(len(generated_data_2d), min_samples, replace=False)
        generated_data_2d = generated_data_2d[indices]
    
    # 合并数据并创建标签
    combined_data = np.vstack([real_data_2d, generated_data_2d])
    labels = np.concatenate([np.ones(len(real_data_2d)), np.zeros(len(generated_data_2d))])
    
    # 训练分类器
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 使用交叉验证
    try:
        auc_scores = cross_val_score(clf, combined_data, labels, cv=5, scoring='roc_auc')
        score = np.mean(auc_scores)
    except Exception as e:
        print(f"计算判别性分数时出错: {e}")
        score = 0.5  # 默认为0.5
    
    return score

def calculate_predictive_score(real_data, generated_data):
    """计算预测性分数（用前面的时间步预测后面的时间步）
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        
    Returns:
        score: 预测性分数（MAE比率），理想值为1.0
    """
    # 对真实数据和生成数据分别计算自预测
    real_mae = calculate_self_prediction_error(real_data)
    generated_mae = calculate_self_prediction_error(generated_data)
    
    # 返回MAE比率（理想情况下接近1）
    if generated_mae > 0:
        return real_mae / generated_mae
    else:
        return float('inf')

def calculate_self_prediction_error(data):
    """计算自预测误差（用前面的时间步预测后面的时间步）
    
    Args:
        data: 数据，形状为 [n_samples, seq_len, feature_dim]
        
    Returns:
        mae: 平均绝对误差
    """
    # 如果序列长度小于2，无法进行预测
    if data.shape[1] < 2:
        return 0
    
    # 用前n-1步预测第n步
    X = data[:, :-1, :]  # 所有时间步，除了最后一步
    y = data[:, 1:, :]   # 所有时间步，除了第一步
    
    # 重塑为2D
    X_2d = X.reshape(-1, X.shape[-1])
    y_2d = y.reshape(-1, y.shape[-1])
    
    # 简单线性回归
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    
    model = LinearRegression()
    try:
        model.fit(X_2d, y_2d)
        
        # 预测并计算MAE
        y_pred = model.predict(X_2d)
        mae = mean_absolute_error(y_2d, y_pred)
    except Exception as e:
        print(f"计算自预测误差时出错: {e}")
        mae = 0  # 默认为0
    
    return mae

def main():
    """主函数，解析命令行参数并执行相应操作"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Improved TimeGAN for APT Attack Sequence Generation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'], help='训练或生成模式')
    parser.add_argument('--input_file', type=str, default='sequences_3tuple/apt_attack_sequences.pkl', help='输入文件路径')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--model_path', type=str, default='output/improved_timegan_model.pth', help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default='logs/improved_timegan', help='TensorBoard日志目录')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='用于恢复训练的检查点路径')
    parser.add_argument('--max_seq_len', type=int, default=10, help='最大序列长度')
    parser.add_argument('--feature_dim', type=int, default=79, help='特征维度')
    parser.add_argument('--hidden_dim', type=int, default=24, help='隐藏层维度')
    parser.add_argument('--z_dim', type=int, default=24, help='噪声维度')
    parser.add_argument('--num_layers', type=int, default=3, help='RNN层数')
    parser.add_argument('--gamma', type=float, default=1.0, help='判别器损失权重')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--pre_train_epochs', type=int, default=10, help='预训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--balance', action='store_true', help='是否平衡训练数据')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--generate_nums', type=str, default='100,100,100,200', help='各类型生成数量')
    parser.add_argument('--noise_scale', type=float, default=1.0, help='生成时噪声缩放因子')
    parser.add_argument('--device', type=str, default=None, help='使用的设备')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载器的工作线程数')
    parser.add_argument('--profile', action='store_true', help='是否进行性能分析')
    args = parser.parse_args()
    
    print("开始执行主函数...")
    print(f"解析命令行参数完成: {args}")
    
    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    print(f"随机种子已设置为: {seed}")
    
    # 设置计算设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载APT攻击序列
    try:
        apt_sequences, labels = load_apt_sequences(args.input_file)
        print(f"加载完成: {len(apt_sequences)} 条APT攻击序列")
        
        # 检查并展示第一条序列的信息，处理不同的序列格式
        if len(apt_sequences) > 0:
            first_seq = apt_sequences[0]
            if isinstance(first_seq, dict):
                # 字典格式序列处理
                if 'features' in first_seq:
                    features = first_seq['features']
                    if isinstance(features, np.ndarray):
                        print(f"序列示例 - 第一条: 标签={labels[0]}, 特征形状={features.shape}")
                    else:
                        print(f"序列示例 - 第一条: 标签={labels[0]}, 特征类型={type(features)}")
                else:
                    print(f"序列示例 - 第一条: 标签={labels[0]}, 字典键={first_seq.keys()}")
            elif isinstance(first_seq, np.ndarray):
                # 直接是numpy数组
                print(f"序列示例 - 第一条: 标签={labels[0]}, 特征形状={first_seq.shape}")
            else:
                print(f"序列示例 - 第一条: 标签={labels[0]}, 类型={type(first_seq)}")
        
        # 分析标签分布情况
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"原始标签分布: {label_counts}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        traceback.print_exc()
        return
    
    # 设置生成目标数量
    generate_nums = {}
    for i, num in enumerate(args.generate_nums.split(',')):
        generate_nums[i+1] = int(num)
    print(f"生成目标数量: {generate_nums}")
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        print("进入训练模式...")
        
        # 预处理数据
        print("开始预处理数据...")
        data, lengths, preprocessed_labels, _, scaler = preprocess_apt_data(
            apt_sequences, args.max_seq_len, args.feature_dim
        )
        print(f"预处理后的数据形状: {data.shape}")
        print(f"标签形状: {preprocessed_labels.shape}, 类型: {preprocessed_labels.dtype}")
        
        # 确保标签是正数，并检查分布
        # 将负数标签映射到正数域
        if np.any(preprocessed_labels < 0):
            print("检测到负标签值，将其映射到非负整数")
            # 将标签映射到从0开始的索引
            unique_labels = np.unique(preprocessed_labels)
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            preprocessed_labels = np.array([label_map[lbl] for lbl in preprocessed_labels])
            
        print(f"处理后的标签分布: {np.bincount(preprocessed_labels.astype(int))}")
        
        # 分割训练集和验证集
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, preprocessed_labels, test_size=0.2, random_state=seed, stratify=preprocessed_labels
        )
        
        print(f"训练集形状: {train_data.shape}, 验证集形状: {val_data.shape}")
        print(f"训练集标签分布: {np.bincount(train_labels.astype(int))}")
        print(f"验证集标签分布: {np.bincount(val_labels.astype(int))}")
        
        # 平衡训练数据
        if args.balance:
            print("平衡训练数据...")
            train_data, train_labels = balance_training_data(train_data, train_labels)
            print(f"平衡后训练集形状: {train_data.shape}")
            print(f"平衡后训练集标签分布: {np.bincount(train_labels.astype(int))}")
        
        # 准备数据加载器
        train_dataset = TimeGANDataset(train_data, train_labels)
        val_dataset = TimeGANDataset(val_data, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers if args.num_workers else 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers if args.num_workers else 0
        )
        
        # 确定分类数
        num_categories = len(np.unique(preprocessed_labels))
        print(f"分类数量: {num_categories}")
        
        # 创建和训练模型
        model = ConditionalTimeGAN(
            args.feature_dim,
            args.hidden_dim,
            args.z_dim,
            num_categories,
            args.num_layers,
            args.gamma,
            args.learning_rate
        )
        
        model.to(device)
        
        # 如果需要恢复训练
        start_epoch = 0
        if args.resume and args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"从检查点恢复训练，起始轮数: {start_epoch}")
            else:
                print(f"检查点文件 {args.checkpoint_path} 不存在，从头开始训练")
        
        # 训练模型
        model.fit(
            train_loader,
            val_loader,
            args.epochs,
            args.pre_train_epochs,
            None,  # steps_per_epoch参数，设为None表示使用整个数据集
            True,  # verbose参数，设为True表示打印训练进度
            args.log_dir,
            start_epoch
        )
        
        # 训练完成后保存模型
        model.save(args.model_path)
        print(f"模型已保存到 {args.model_path}")
        
        # 保存归一化器和预处理信息
        with open(os.path.join(args.output_dir, 'improved_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        preprocess_info = {
            'max_seq_len': args.max_seq_len,
            'feature_dim': args.feature_dim
        }
        
        with open(os.path.join(args.output_dir, 'improved_preprocess_info.pkl'), 'wb') as f:
            pickle.dump(preprocess_info, f)
        
        print("训练和预处理信息保存完成")
        
    elif args.mode == 'generate':
        print("进入生成模式...")
        
        # 加载模型
        if not os.path.exists(args.model_path):
            print(f"模型文件 {args.model_path} 不存在")
            return
        
        print(f"从 {args.model_path} 加载模型")
        
        # 加载预处理信息
        scaler_path = os.path.join(args.output_dir, 'improved_scaler.pkl')
        preprocess_info_path = os.path.join(args.output_dir, 'improved_preprocess_info.pkl')
        
        if not os.path.exists(scaler_path) or not os.path.exists(preprocess_info_path):
            print("缺少归一化器或预处理信息文件")
            return
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(preprocess_info_path, 'rb') as f:
            preprocess_info = pickle.load(f)
        
        # 加载模型
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 确定分类数
        num_categories = len(generate_nums)
        
        model = ConditionalTimeGAN(
            args.feature_dim,
            args.hidden_dim,
            args.z_dim,
            num_categories,
            args.num_layers,
            args.gamma,
            args.learning_rate
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("模型加载成功")
        
        # 生成序列
        generated_data = []
        generated_labels = []
        
        for category, num_samples in generate_nums.items():
            print(f"生成APT类型 {category} 的样本，数量: {num_samples}")
            
            # 生成指定类别的样本
            category_id = category - 1  # 转换为从0开始的索引
            start_time = time.time()
            gen_data = model.generate(
                num_samples,
                args.max_seq_len,
                category_id,
                args.noise_scale,
                device
            )
            end_time = time.time()
            
            print(f"生成完成，耗时: {end_time - start_time:.2f}秒")
            print(f"生成数据形状: {gen_data.shape}")
            
            # 显示生成数据的统计信息
            print(f"生成数据统计: 最小值={gen_data.min():.4f}, 最大值={gen_data.max():.4f}, 均值={gen_data.mean():.4f}, 标准差={gen_data.std():.4f}")
            
            # 反归一化
            # 这里注意保持数据格式与预处理时一致
            gen_data_reshaped = gen_data.reshape(-1, args.feature_dim)
            gen_data_inverse = scaler.inverse_transform(gen_data_reshaped)
            gen_data = gen_data_inverse.reshape(-1, args.max_seq_len, args.feature_dim)
            
            # 将生成的数据和标签添加到列表中
            generated_data.extend(gen_data)
            generated_labels.extend([category] * num_samples)
        
        # 转换为numpy数组
        generated_data = np.array(generated_data)
        generated_labels = np.array(generated_labels)
        
        print(f"最终生成数据形状: {generated_data.shape}")
        print(f"标签分布: {dict(zip(*np.unique(generated_labels, return_counts=True)))}")
        
        # 保存生成的序列
        generated_sequences = []
        for i in range(len(generated_data)):
            seq = generated_data[i]
            label = int(generated_labels[i])
            generated_sequences.append({
                'sequence': seq.tolist(),
                'label': label,
                'apt_type': f"APT{label}"
            })
        
        # 保存为JSON格式
        with open(os.path.join(args.output_dir, 'generated_sequences.json'), 'w') as f:
            json.dump(generated_sequences, f, indent=2)
        
        # 保存生成元数据
        generation_metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': args.model_path,
            'noise_scale': args.noise_scale,
            'max_seq_len': args.max_seq_len,
            'feature_dim': args.feature_dim,
            'num_generated': len(generated_sequences),
            'label_distribution': {int(k): int(v) for k, v in zip(*np.unique(generated_labels, return_counts=True))}
        }
        
        with open(os.path.join(args.output_dir, 'generation_metadata.json'), 'w') as f:
            json.dump(generation_metadata, f, indent=2)
        
        print(f"生成的序列已保存到 {os.path.join(args.output_dir, 'generated_sequences.json')}")
        print(f"生成元数据已保存到 {os.path.join(args.output_dir, 'generation_metadata.json')}")
    
    else:
        print(f"不支持的模式: {args.mode}")

if __name__ == "__main__":
    main() 
