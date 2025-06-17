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
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from itertools import islice
from tqdm import tqdm
import time
import json
import datetime

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

def load_apt_sequences(file_path):
    """加载APT攻击序列
    
    Args:
        file_path: 序列文件路径
        
    Returns:
        sequences: APT攻击序列列表
    """
    with open(file_path, 'rb') as f:
        sequences = pickle.load(f)
    
    print(f"已加载 {len(sequences)} 条APT攻击序列")
    
    # 检查序列格式并适配
    # atk_seqs.pkl中的键是：'features', 'labels', 'phases', 'session_id', 'window_type', 
    # 'start_idx', 'end_idx', 'apt_type', 'attack_phases', 'is_attack'
    
    adapted_sequences = []
    for seq in sequences:
        # 如果序列中有apt_type字段，将其作为label
        if 'apt_type' in seq:
            label = int(seq['apt_type'].replace('APT', '')) if isinstance(seq['apt_type'], str) else seq['apt_type']
        else:
            # 否则尝试使用已有的label字段
            label = seq.get('label', 1)  # 默认为1
        
        # 创建适配的序列字典
        adapted_seq = {
            'features': seq['features'],
            'label': label,
            'phases': seq.get('phases', []),
            'id': seq.get('session_id', ''),
            'data': seq.get('features', [])  # 添加data字段作为备用
        }
        adapted_sequences.append(adapted_seq)
    
    # 统计标签分布
    labels = [seq['label'] for seq in adapted_sequences]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"标签分布: {label_counts}")
    
    return adapted_sequences

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
    import argparse
    parser = argparse.ArgumentParser(description='训练和生成条件时间序列数据')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train_and_generate', 
                        choices=['train', 'generate', 'train_and_generate'],
                        help='运行模式: train, generate, train_and_generate')
    
    # 数据参数
    parser.add_argument('--input_file', type=str, default='atk_seqs.pkl',
                       help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='输出目录路径')
    parser.add_argument('--model_path', type=str, default='output/improved_timegan_model.pth',
                       help='模型保存/加载路径')
    parser.add_argument('--log_dir', type=str, default='logs/improved_timegan',
                       help='TensorBoard日志目录')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='检查点加载路径，用于恢复训练，默认为None（从头开始训练）')
    
    # 序列参数
    parser.add_argument('--max_seq_len', type=int, default=10,
                       help='最大序列长度')
    parser.add_argument('--feature_dim', type=int, default=79,
                       help='特征维度')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=24,
                       help='隐藏层维度')
    parser.add_argument('--z_dim', type=int, default=24,
                       help='噪声维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='RNN层数')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='信息不变系数，用于调整条件信息的影响')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--pre_train_epochs', type=int, default=10,
                       help='预训练嵌入器和恢复网络的轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--balance', action='store_true',
                       help='是否平衡训练数据')
    parser.add_argument('--resume', action='store_true',
                       help='是否从上一次的检查点恢复训练')
    
    # 生成参数
    parser.add_argument('--generate_nums', type=str, default='100,100,100,200',
                       help='每个类别生成的样本数量，逗号分隔，例如: 100,100,100,200')
    parser.add_argument('--noise_scale', type=float, default=1.0,
                       help='生成时的噪声比例，控制随机性')
    
    # 设备参数
    parser.add_argument('--device', type=str, default=None,
                       help='运行设备，例如: "cuda:0" 或 "cpu"，默认自动选择')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='数据加载器的工作进程数，默认自动选择')
    
    # 性能分析参数
    parser.add_argument('--profile', action='store_true',
                       help='是否进行性能分析')
    
    return parser.parse_args()

def set_random_seed(seed):
    """设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
        # 设置CUDA的确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

def create_data_loader(data, batch_size, shuffle=True, num_workers=None, pin_memory=None):
    """创建高效的数据加载器
    
    Args:
        data: 输入数据
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数，如果为None则自动选择
        pin_memory: 是否使用固定内存，如果为None则自动选择
        
    Returns:
        data_loader: PyTorch DataLoader
    """
    tensor_x = torch.Tensor(data)
    dataset = TensorDataset(tensor_x)
    
    # 自动选择工作进程数和内存固定
    if num_workers is None:
        num_workers = 4 if torch.cuda.is_available() else 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return data_loader

def get_device(device_str=None, verbose=True):
    """获取设备（CPU或GPU）
    
    Args:
        device_str: 设备字符串，例如"cuda:0"或"cpu"，如果为None则自动选择
        verbose: 是否打印设备信息
        
    Returns:
        device: PyTorch设备
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        if device.type == 'cuda':
            gpu_props = torch.cuda.get_device_properties(device)
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {gpu_props.total_memory / 1024**3:.2f} GB")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"当前GPU使用情况: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            print(f"使用CPU: {device}")
    
    return device

def main():
    """主函数"""
    print("开始执行主函数...")
    
    # 解析命令行参数
    args = parse_arguments()
    print(f"解析命令行参数完成: {args}")
    
    # 设置随机种子
    if hasattr(args, 'seed'):
        seed = args.seed
    else:
        seed = 42
    set_random_seed(seed)
    
    # 设置设备
    device = get_device(getattr(args, 'device', None))
    
    # 加载APT攻击序列
    sequences = load_apt_sequences(args.input_file)
    
    # 解析生成数量
    generate_nums = [int(num) for num in args.generate_nums.split(',')]
    generate_target = {i+1: generate_nums[i] for i in range(len(generate_nums)) if i < 4}
    
    if args.mode in ['train', 'train_and_generate']:
        print("进入训练模式...")
        # 平衡训练数据
        if args.balance:
            # 计算每个标签的目标数量
            target_nums = {}
            for i in range(1, 5):  # 假设标签为1-4
                target_nums[i] = max(60, generate_target.get(i, 60))  # 确保训练数据至少有60个样本
            sequences = balance_apt_sequences(sequences, target_nums)
        
        # 预处理数据
        processed_data, processed_time, labels, phases, scaler = preprocess_apt_data(
            sequences, 
            max_seq_len=args.max_seq_len,
            feature_dim=args.feature_dim
        )
        
        print(f"预处理后的数据形状: {processed_data.shape}")
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        train_data, val_data, train_labels, val_labels = train_test_split(
            processed_data, labels, test_size=0.2, random_state=seed
        )
        
        print(f"训练集形状: {train_data.shape}, 验证集形状: {val_data.shape}")
        print(f"训练集标签分布: {np.bincount(train_labels.astype(int))}")
        print(f"验证集标签分布: {np.bincount(val_labels.astype(int))}")
        
        # 创建数据加载器
        # 根据设备类型调整数据加载参数
        if device.type == 'cuda':
            # GPU上使用多个工作进程和固定内存
            num_workers = 4
            pin_memory = True
        else:
            # CPU上减少工作进程
            num_workers = 0
            pin_memory = False
        
        train_data_loader = create_data_loader(
            train_data, 
            args.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_data_loader = create_data_loader(
            val_data, 
            args.batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        print(f"训练数据加载器批次数: {len(train_data_loader)}")
        print(f"验证数据加载器批次数: {len(val_data_loader)}")
        
        # 计算实际的训练步数
        steps_per_epoch = len(train_data_loader)
        total_steps = args.epochs * steps_per_epoch
        
        print(f"每轮步数: {steps_per_epoch}, 总步数: {total_steps}")
        
        # 创建ConditionalTimeGAN模型
        print("创建ConditionalTimeGAN模型...")
        try:
            model = ConditionalTimeGAN(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                z_dim=args.z_dim,
                num_categories=4,  # APT类型数量
                embedding_dim=8,    # 类别嵌入维度
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                max_seq_len=args.max_seq_len,
                device=device,
                num_layers=args.num_layers
            )
            print("ConditionalTimeGAN模型创建成功")
            
            # 从检查点恢复（如果指定）
            start_epoch = 0
            if args.resume:
                # 如果未指定检查点路径，使用默认路径
                if args.checkpoint_path is None:
                    args.checkpoint_path = os.path.join(os.path.dirname(args.model_path), 'last_checkpoint.pth')
                
                if os.path.exists(args.checkpoint_path):
                    start_epoch = model.load_checkpoint(args.checkpoint_path)
                    print(f"从检查点恢复训练，起始轮数: {start_epoch}")
                    
                    # 调整剩余训练轮数
                    if start_epoch >= args.epochs:
                        print(f"检查点轮数 ({start_epoch}) 已达到或超过目标轮数 ({args.epochs})，无需继续训练")
                        args.epochs = start_epoch  # 设为相同值，跳过训练
                    else:
                        print(f"将继续训练 {args.epochs - start_epoch} 轮")
                else:
                    print(f"找不到检查点文件: {args.checkpoint_path}，将从头开始训练")
            
        except Exception as e:
            print(f"创建ConditionalTimeGAN模型时出错: {e}")
            raise
        
        # 确保日志目录存在
        os.makedirs(args.log_dir, exist_ok=True)
        
        # 训练模型
        print("开始训练模型...")
        try:
            # 如果使用GPU，监控GPU内存使用情况
            if device.type == 'cuda':
                print(f"训练前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # 如果从检查点恢复且已完成训练，则跳过
            if args.resume and start_epoch >= args.epochs:
                print("检查点已经达到目标训练轮数，跳过训练")
                history = None
            else:
                history = model.fit(
                    train_data_loader=train_data_loader,
                    test_data_loader=val_data_loader,
                    epochs=args.epochs,
                    pre_train_epochs=args.pre_train_epochs,
                    steps_per_epoch=None,  # 使用全部批次
                    verbose=True,
                    log_dir=args.log_dir,
                    start_epoch=start_epoch if args.resume else 0  # 如果恢复训练，则从检查点轮数开始
                )
            
            # 训练后再次检查GPU内存
            if device.type == 'cuda':
                print(f"训练后GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                # 清理GPU缓存
                torch.cuda.empty_cache()
                print(f"清理缓存后GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            print("模型训练完成")
            
            # 绘制训练历史
            if history:
                plot_training_history(history, os.path.dirname(args.model_path))
            
        except Exception as e:
            print(f"训练模型时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试保存检查点以恢复训练
            try:
                emergency_checkpoint_path = os.path.join(os.path.dirname(args.model_path), 'emergency_checkpoint.pth')
                model._save_checkpoint(emergency_checkpoint_path)
                print(f"发生错误，但已保存紧急检查点到: {emergency_checkpoint_path}")
                print(f"可以使用 --resume --checkpoint_path={emergency_checkpoint_path} 参数恢复训练")
            except Exception as e2:
                print(f"尝试保存紧急检查点时也出错: {e2}")
            
            raise
        
        # 保存模型
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        
        # 同时保存检查点，用于恢复训练
        checkpoint_path = os.path.join(os.path.dirname(args.model_path), 'last_checkpoint.pth')
        model._save_checkpoint(checkpoint_path, epoch=args.epochs)
        print(f"训练检查点已保存到: {checkpoint_path}")
        
        # 保存scaler用于后续生成
        scaler_path = os.path.join(os.path.dirname(args.model_path), 'improved_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler已保存到: {scaler_path}")
        
        # 保存预处理信息
        preprocess_info = {
            'labels': labels,
            'phases': phases,
            'feature_dim': args.feature_dim,
            'max_seq_len': args.max_seq_len
        }
        preprocess_info_path = os.path.join(os.path.dirname(args.model_path), 'improved_preprocess_info.pkl')
        with open(preprocess_info_path, 'wb') as f:
            pickle.dump(preprocess_info, f)
        print(f"预处理信息已保存到: {preprocess_info_path}")
    
    if args.mode in ['generate', 'train_and_generate']:
        print("进入生成模式...")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 如果在generate模式下，需要加载模型
        if args.mode == 'generate':
            print(f"从 {args.model_path} 加载模型...")
            try:
                # 加载模型
                model = ConditionalTimeGAN.load(args.model_path, device)
                print("模型加载成功")
                
                # 加载scaler
                scaler_path = os.path.join(os.path.dirname(args.model_path), 'improved_scaler.pkl')
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"Scaler从 {scaler_path} 加载成功")
                
                # 加载预处理信息
                preprocess_info_path = os.path.join(os.path.dirname(args.model_path), 'improved_preprocess_info.pkl')
                with open(preprocess_info_path, 'rb') as f:
                    preprocess_info = pickle.load(f)
                print(f"预处理信息从 {preprocess_info_path} 加载成功")
                
                # 提取预处理信息
                feature_dim = preprocess_info['feature_dim']
                max_seq_len = preprocess_info['max_seq_len']
                
            except Exception as e:
                print(f"加载模型时出错: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # 生成数据
        print("开始生成数据...")
        generated_data = []
        
        try:
            # 对每个APT类型生成指定数量的序列
            for apt_type, num_samples in generate_target.items():
                print(f"正在为APT类型 {apt_type} 生成 {num_samples} 个样本...")
                
                # 如果使用GPU，监控GPU内存使用情况
                if device.type == 'cuda':
                    print(f"生成前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                
                # 为当前APT类型生成数据
                start_time = time.time()
                generated = model.generate(
                    num_samples=num_samples, 
                    labels=[apt_type] * num_samples,
                    noise_scale=args.noise_scale
                )
                end_time = time.time()
                
                # 转换为numpy并记录
                if isinstance(generated, torch.Tensor):
                    generated = generated.cpu().numpy()
                
                # 显示生成时间和基本统计信息
                gen_time = end_time - start_time
                print(f"APT类型 {apt_type} 生成完成。时间: {gen_time:.2f}秒，每样本平均: {gen_time/num_samples:.4f}秒")
                print(f"生成数据形状: {generated.shape}")
                print(f"数据统计: 最小值={np.min(generated):.4f}, 最大值={np.max(generated):.4f}, 均值={np.mean(generated):.4f}, 标准差={np.std(generated):.4f}")
                
                # 反向转换数据（如果需要）
                if scaler:
                    # 重塑数据以适应scaler（从3D到2D）
                    reshaped_data = generated.reshape(-1, generated.shape[-1])
                    # 反向转换
                    inverse_data = scaler.inverse_transform(reshaped_data)
                    # 重塑回原始形状
                    generated = inverse_data.reshape(generated.shape)
                    print(f"反向转换后数据统计: 最小值={np.min(generated):.4f}, 最大值={np.max(generated):.4f}, 均值={np.mean(generated):.4f}, 标准差={np.std(generated):.4f}")
                
                # 为每个生成的序列分配标签和ID
                for i in range(num_samples):
                    sample_id = f"gen_{apt_type}_{i+1}"
                    generated_data.append({
                        'id': sample_id,
                        'label': apt_type,
                        'data': generated[i].tolist()  # 转换为Python列表以便JSON序列化
                    })
                
                # 如果使用GPU，清理缓存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    print(f"清理缓存后GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # 合并所有生成的数据，用于质量评估
            all_generated_data = np.array([item['data'] for item in generated_data])
            all_labels = np.array([item['label'] for item in generated_data])
            
            # 评估序列质量（如果有真实数据作为参考）
            if args.mode == 'train_and_generate' and 'processed_data' in locals():
                print("评估生成序列质量...")
                quality_metrics = evaluate_sequence_quality(
                    real_data=processed_data, 
                    generated_data=all_generated_data,
                    real_labels=labels,
                    generated_labels=all_labels,
                    prefix="全局"
                )
                
                # 按类别评估
                for apt_type in sorted(set(all_labels)):
                    real_idx = np.where(labels == apt_type)[0]
                    gen_idx = np.where(all_labels == apt_type)[0]
                    
                    if len(real_idx) > 0 and len(gen_idx) > 0:
                        print(f"\n评估APT类型 {apt_type} 的序列质量...")
                        apt_metrics = evaluate_sequence_quality(
                            real_data=processed_data[real_idx], 
                            generated_data=all_generated_data[gen_idx],
                            prefix=f"APT类型{apt_type}"
                        )
                        quality_metrics[f'apt_type_{apt_type}'] = apt_metrics
                
                # 保存质量指标
                metrics_path = os.path.join(args.output_dir, 'quality_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(quality_metrics, f, indent=2)
                print(f"质量指标已保存到: {metrics_path}")
            
            # 保存生成的数据
            output_file = os.path.join(args.output_dir, 'generated_sequences.json')
            with open(output_file, 'w') as f:
                json.dump(generated_data, f, indent=2)
            print(f"生成的序列已保存到: {output_file}")
            
            # 保存生成元数据
            metadata = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': args.model_path,
                'generation_params': {
                    'noise_scale': args.noise_scale,
                    'generate_target': generate_target
                },
                'counts': {apt_type: sum(1 for item in generated_data if item['label'] == apt_type) for apt_type in generate_target},
                'total_generated': len(generated_data)
            }
            metadata_file = os.path.join(args.output_dir, 'generation_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"生成元数据已保存到: {metadata_file}")
            
        except Exception as e:
            print(f"生成数据时出错: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print("任务完成!")

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

def evaluate_sequence_quality(real_data, generated_data, labels=None, prefix=''):
    """评估生成序列的质量
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        labels: 标签（可选）
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

if __name__ == "__main__":
    main() 
