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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

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
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--quiet', action='store_true', help='最小化输出，只显示关键信息')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    parser.add_argument('--detailed_eval', action='store_true', help='进行详细评估')
    args = parser.parse_args()
    
    return args

def main():
    """主函数，解析命令行参数并执行相应操作"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置全局变量VERBOSE，控制输出详细度
    global VERBOSE
    VERBOSE = args.verbose
    
    # 设置QUIET模式，最小化输出
    global QUIET
    QUIET = args.quiet
    
    if VERBOSE and QUIET:
        print("警告：同时设置了verbose和quiet参数，将优先使用quiet模式")
        VERBOSE = False
    
    if not QUIET:
        if VERBOSE:
            print("开始执行主函数...")
            print(f"解析命令行参数完成: {args}")
        else:
            # 即使在非详细模式下，也打印一些基本信息
            print(f"模式: {args.mode}, 设备: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if not QUIET:
        print(f"随机种子已设置为: {seed}")
    
    # 设置计算设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not QUIET:
        print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载APT攻击序列
    try:
        apt_sequences, labels = load_apt_sequences(args.input_file)
        if not QUIET:
            print(f"加载完成: {len(apt_sequences)} 条APT攻击序列")
        
        # 检查并展示第一条序列的信息，处理不同的序列格式
        if VERBOSE and len(apt_sequences) > 0 and not QUIET:
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
        if not QUIET:
            print(f"原始标签分布: {label_counts}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        traceback.print_exc()
        return
    
    # 设置生成目标数量
    generate_nums = {}
    for i, num in enumerate(args.generate_nums.split(',')):
        generate_nums[i+1] = int(num)
    if not QUIET:
        print(f"生成目标数量: {generate_nums}")
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        if not QUIET:
            print("进入训练模式...")
        
        # 预处理数据
        if not QUIET:
            print("开始预处理数据...")
        data, lengths, preprocessed_labels, phases, scaler = preprocess_apt_data(
            apt_sequences, args.max_seq_len, args.feature_dim
        )
        
        # 关键修改：将标签加1，抵消preprocess_apt_data中的减1操作
        # 说明：preprocess_apt_data函数内部将标签从1-4映射到0-3
        # 这里将其恢复为原始标签值1-4，确保在训练过程中使用正确的标签值
        preprocessed_labels = preprocessed_labels + 1  # 恢复原始标签值1-4
        
        if VERBOSE and not QUIET:
            print(f"预处理后的数据形状: {data.shape}")
            print(f"标签形状: {preprocessed_labels.shape}, 类型: {preprocessed_labels.dtype}")
        
        # 确保标签是正数，并检查分布
        # 将负数标签映射到正数域 - 这部分可能不再需要，因为我们恢复了原始值
        if np.any(preprocessed_labels <= 0) and not QUIET:
            print("警告：检测到非正标签值，确保所有标签都是1-4")
            
        if not QUIET:
            print(f"处理后的标签分布: {np.bincount(preprocessed_labels.astype(int))}")
        
        # 分割训练集和验证集
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, preprocessed_labels, test_size=0.2, random_state=seed, stratify=preprocessed_labels
        )
        
        if not QUIET:
            print(f"训练集形状: {train_data.shape}, 验证集形状: {val_data.shape}")
            print(f"训练集标签分布: {np.bincount(train_labels.astype(int))}")
            print(f"验证集标签分布: {np.bincount(val_labels.astype(int))}")
        
        # 平衡训练数据
        if args.balance:
            if not QUIET:
                print("平衡训练数据...")
            train_data, train_labels = balance_training_data(train_data, train_labels)
            if not QUIET:
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
        
        # 确定分类数 - 注意，分类数应该是标签的最大值，而不是唯一值的数量
        # 因为我们的标签是从1开始的，所以最大值就是分类数
        num_categories = int(np.max(preprocessed_labels))
        if not QUIET:
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
                if not QUIET:
                    print(f"从检查点恢复训练，起始轮数: {start_epoch}")
            else:
                if not QUIET:
                    print(f"检查点文件 {args.checkpoint_path} 不存在，从头开始训练")
        
        # 训练模型
        model.fit(
            train_loader,
            val_loader,
            args.epochs,
            args.pre_train_epochs,
            None,  # steps_per_epoch参数，设为None表示使用整个数据集
            VERBOSE and not QUIET,  # verbose参数，使用VERBOSE控制是否打印进度
            args.log_dir,
            start_epoch
        )
        
        # 训练完成后保存模型
        model.save(args.model_path)
        if not QUIET:
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
        
        if not QUIET:
            print("训练和预处理信息保存完成")
        
    elif args.mode == 'generate':
        if not QUIET:
            print("进入生成模式...")
        
        # 加载模型
        if not os.path.exists(args.model_path):
            print(f"模型文件 {args.model_path} 不存在")
            return
        
        if not QUIET:
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
        
        # 使用类方法加载模型
        try:
            # 使用ConditionalTimeGAN类的load方法加载模型
            model = ConditionalTimeGAN.load(args.model_path, device)
        except Exception as e:
            if not QUIET:
                print(f"无法使用类方法加载模型，尝试手动加载: {e}")
            # 手动加载模型状态
            checkpoint = torch.load(args.model_path, map_location=device)
            
            # 确定分类数
            num_categories = max(generate_nums.keys())  # 使用生成目标中的最大键值
            
            # 创建模型
            model = ConditionalTimeGAN(
                args.feature_dim,
                args.hidden_dim,
                args.z_dim,
                num_categories,
                num_layers=args.num_layers,
                gamma=args.gamma,
                learning_rate=args.learning_rate
            )
            
            # 尝试加载状态字典，处理不同的键名格式
            try:
                # 尝试直接加载模型状态字典
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                # 尝试加载各个组件的状态字典
                elif 'embedder_state_dict' in checkpoint:
                    model.embedder.load_state_dict(checkpoint['embedder_state_dict'])
                    model.recovery.load_state_dict(checkpoint['recovery_state_dict'])
                    model.generator.load_state_dict(checkpoint['generator_state_dict'])
                    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                    model.supervisor.load_state_dict(checkpoint['supervisor_state_dict'])
                else:
                    raise ValueError("模型文件格式不正确，无法找到状态字典")
            except Exception as load_error:
                print(f"加载模型状态时出错: {load_error}")
                if VERBOSE and not QUIET:
                    print("模型键名:", list(checkpoint.keys()))
                return
            
            model.to(device)
        
        if not QUIET:
            print("模型加载成功")
        
        # 生成序列
        generated_data = []
        generated_labels = []
        
        # 加载部分原始数据用于评估
        try:
            eval_data, eval_lengths, eval_labels, _, _ = preprocess_apt_data(
                apt_sequences[:100], args.max_seq_len, args.feature_dim
            )
            # 恢复原始标签
            # 说明：preprocess_apt_data函数内部将标签从1-4映射到0-3
            # 这里将其恢复为原始标签值1-4，确保评估时使用正确的标签值
            eval_labels = eval_labels + 1
        except Exception as e:
            if not QUIET:
                print(f"警告：无法加载原始数据用于评估，将跳过评估步骤: {e}")
            eval_data = None
            eval_labels = None
        
        for category, num_samples in generate_nums.items():
            if not QUIET:
                print(f"生成APT类型 {category} 的样本，数量: {num_samples}")
            
            # 生成指定类别的样本
            # 注意：model.generate方法内部会将标签减1，所以这里直接使用原始类别ID（1-4）
            # 这样在model.generate内部会正确转换为从0开始的类别ID（0-3）
            category_id = category  # 使用原始类别ID（1-4）
            start_time = time.time()
            gen_data = model.generate(
                num_samples,
                category_id,
                args.max_seq_len,
                args.noise_scale
            )
            end_time = time.time()
            
            if not QUIET:
                print(f"生成完成，耗时: {end_time - start_time:.2f}秒")
                print(f"生成数据形状: {gen_data.shape}")
            
            # 显示生成数据的统计信息
            if VERBOSE and not QUIET:
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
        
        if not QUIET:
            print(f"最终生成数据形状: {generated_data.shape}")
            print(f"标签分布: {dict(zip(*np.unique(generated_labels, return_counts=True)))}")
        
        # 显示生成序列的样例
        if not QUIET:
            print("\n生成序列样例:")
            for i in range(min(3, len(generated_data))):
                print(f"样例 {i+1} (标签: {generated_labels[i]}):")
                print(f"序列形状: {generated_data[i].shape}")
                print(f"前两个时间步的特征值:")
                for t in range(min(2, generated_data[i].shape[0])):
                    print(f"  时间步 {t+1}: {generated_data[i][t, :5]}...")  # 只显示前5个特征
                print("")
        
        # 评估生成序列的质量
        if eval_data is not None:
            if not QUIET:
                print("\n评估生成序列质量:")
            
            # 使用新的评估函数
            metrics = evaluate_generated_data(
                eval_data, 
                generated_data[:len(eval_data)], 
                eval_labels, 
                generated_labels[:len(eval_labels)],
                args.output_dir,
                visualize=args.visualize,
                detailed=args.detailed_eval
            )
        
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
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': args.model_path,
            'noise_scale': args.noise_scale,
            'max_seq_len': args.max_seq_len,
            'feature_dim': args.feature_dim,
            'num_generated': len(generated_sequences),
            'label_distribution': {int(k): int(v) for k, v in zip(*np.unique(generated_labels, return_counts=True))}
        }
        
        with open(os.path.join(args.output_dir, 'generation_metadata.json'), 'w') as f:
            json.dump(generation_metadata, f, indent=2)
        
        if not QUIET:
            print(f"生成的序列已保存到 {os.path.join(args.output_dir, 'generated_sequences.json')}")
            print(f"生成元数据已保存到 {os.path.join(args.output_dir, 'generation_metadata.json')}")
    
    else:
        print(f"不支持的模式: {args.mode}")

def calculate_discriminative_score_rf(real_data, generated_data):
    """根据用户提供的代码计算判别性分数（使用RandomForest区分真实和生成数据）
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        
    Returns:
        score: 判别性分数（Accuracy），理想值为0.5
    """
    # 重塑数据为2D
    real_data_2d = real_data.reshape(len(real_data), -1)
    generated_data_2d = generated_data.reshape(len(generated_data), -1)
    
    # 合并数据并创建标签（0=真实，1=生成）
    X = np.vstack([real_data_2d, generated_data_2d])
    y = np.array([0] * len(real_data) + [1] * len(generated_data))
    
    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # 使用RandomForest进行分类
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 计算准确率
    acc = clf.score(X_test, y_test)
    
    return acc

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

def calculate_feature_importance(real_data, generated_data):
    """计算特征重要性（哪些特征对区分真实和生成数据最重要）
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        
    Returns:
        feature_importance: 特征重要性数组，形状为 [feature_dim]
    """
    # 重塑数据为2D
    real_data_2d = real_data.reshape(len(real_data), -1)
    generated_data_2d = generated_data.reshape(len(generated_data), -1)
    
    # 合并数据并创建标签（0=真实，1=生成）
    X = np.vstack([real_data_2d, generated_data_2d])
    y = np.array([0] * len(real_data) + [1] * len(generated_data))
    
    # 训练RandomForest分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # 获取特征重要性
    # 注意：这里的特征是展平后的，需要重塑回原始形状
    feature_importance_flat = clf.feature_importances_
    
    # 重塑为 [seq_len, feature_dim] 并计算每个特征的平均重要性
    feature_dim = real_data.shape[2]
    seq_len = real_data.shape[1]
    feature_importance_reshaped = feature_importance_flat.reshape(-1, feature_dim)
    feature_importance = np.mean(feature_importance_reshaped, axis=0)
    
    return feature_importance

def calculate_temporal_consistency(real_data, generated_data):
    """计算时间一致性（生成数据的时间模式与真实数据的相似度）
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        
    Returns:
        temporal_consistency: 时间一致性分数（0-1，越高越好）
    """
    # 计算每个时间步的均值和标准差
    real_mean = np.mean(real_data, axis=0)  # [seq_len, feature_dim]
    real_std = np.std(real_data, axis=0)    # [seq_len, feature_dim]
    
    gen_mean = np.mean(generated_data, axis=0)  # [seq_len, feature_dim]
    gen_std = np.std(generated_data, axis=0)    # [seq_len, feature_dim]
    
    # 计算均值的相对误差
    mean_error = np.mean(np.abs(real_mean - gen_mean) / (np.abs(real_mean) + 1e-8))
    
    # 计算标准差的相对误差
    std_error = np.mean(np.abs(real_std - gen_std) / (np.abs(real_std) + 1e-8))
    
    # 计算时间一致性分数（1 - 平均相对误差）
    temporal_consistency = 1.0 - 0.5 * (mean_error + std_error)
    
    # 确保分数在0-1范围内
    temporal_consistency = max(0.0, min(1.0, temporal_consistency))
    
    return temporal_consistency

def calculate_label_consistency(real_labels, generated_labels):
    """计算标签一致性（生成数据的标签分布与真实数据的相似度）
    
    Args:
        real_labels: 真实数据的标签，形状为 [n_samples]
        generated_labels: 生成数据的标签，形状为 [n_samples]
        
    Returns:
        label_consistency: 标签一致性分数（0-1，越高越好）
    """
    # 计算真实数据的标签分布
    real_label_counts = {}
    for label in real_labels:
        real_label_counts[label] = real_label_counts.get(label, 0) + 1
    
    # 计算生成数据的标签分布
    gen_label_counts = {}
    for label in generated_labels:
        gen_label_counts[label] = gen_label_counts.get(label, 0) + 1
    
    # 确保所有标签都在两个字典中
    all_labels = set(real_label_counts.keys()) | set(gen_label_counts.keys())
    for label in all_labels:
        if label not in real_label_counts:
            real_label_counts[label] = 0
        if label not in gen_label_counts:
            gen_label_counts[label] = 0
    
    # 计算分布差异
    total_real = sum(real_label_counts.values())
    total_gen = sum(gen_label_counts.values())
    
    distribution_diff = 0.0
    for label in all_labels:
        real_prob = real_label_counts[label] / total_real if total_real > 0 else 0
        gen_prob = gen_label_counts[label] / total_gen if total_gen > 0 else 0
        distribution_diff += abs(real_prob - gen_prob)
    
    # 计算标签一致性分数（1 - 平均分布差异）
    label_consistency = 1.0 - distribution_diff / (2 * len(all_labels))
    
    return label_consistency

def visualize_feature_importance(feature_importance, output_dir):
    """可视化特征重要性
    
    Args:
        feature_importance: 特征重要性数组，形状为 [feature_dim]
        output_dir: 输出目录
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('特征索引')
    plt.ylabel('重要性')
    plt.title('特征重要性分布')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()

def visualize_temporal_patterns(real_data, generated_data, output_dir):
    """可视化时间模式
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        output_dir: 输出目录
    """
    # 计算每个时间步的均值
    real_mean = np.mean(real_data, axis=0)  # [seq_len, feature_dim]
    gen_mean = np.mean(generated_data, axis=0)  # [seq_len, feature_dim]
    
    # 选择前5个最重要的特征进行可视化
    feature_importance = calculate_feature_importance(real_data, generated_data)
    top_features = np.argsort(feature_importance)[-5:]
    
    plt.figure(figsize=(15, 10))
    for i, feature_idx in enumerate(top_features):
        plt.subplot(2, 3, i+1)
        plt.plot(real_mean[:, feature_idx], 'b-', label='真实数据')
        plt.plot(gen_mean[:, feature_idx], 'r--', label='生成数据')
        plt.xlabel('时间步')
        plt.ylabel(f'特征 {feature_idx} 均值')
        plt.title(f'特征 {feature_idx} 的时间模式')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_patterns.png'), dpi=300)
    plt.close()

def visualize_label_distribution(real_labels, generated_labels, output_dir):
    """可视化标签分布
    
    Args:
        real_labels: 真实数据的标签，形状为 [n_samples]
        generated_labels: 生成数据的标签，形状为 [n_samples]
        output_dir: 输出目录
    """
    # 计算标签分布
    real_label_counts = {}
    for label in real_labels:
        real_label_counts[label] = real_label_counts.get(label, 0) + 1
    
    gen_label_counts = {}
    for label in generated_labels:
        gen_label_counts[label] = gen_label_counts.get(label, 0) + 1
    
    # 确保所有标签都在两个字典中
    all_labels = sorted(set(real_label_counts.keys()) | set(gen_label_counts.keys()))
    
    # 准备绘图数据
    labels = [f"APT{label}" for label in all_labels]
    real_counts = [real_label_counts.get(label, 0) for label in all_labels]
    gen_counts = [gen_label_counts.get(label, 0) for label in all_labels]
    
    # 绘制条形图
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, real_counts, width, label='真实数据')
    ax.bar(x + width/2, gen_counts, width, label='生成数据')
    
    ax.set_xlabel('APT类型')
    ax.set_ylabel('样本数量')
    ax.set_title('真实数据与生成数据的标签分布')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'), dpi=300)
    plt.close()

def evaluate_generated_data(real_data, generated_data, real_labels, generated_labels, output_dir, visualize=False, detailed=False):
    """评估生成数据的质量
    
    Args:
        real_data: 真实数据，形状为 [n_samples, seq_len, feature_dim]
        generated_data: 生成数据，形状为 [n_samples, seq_len, feature_dim]
        real_labels: 真实数据的标签，形状为 [n_samples]
        generated_labels: 生成数据的标签，形状为 [n_samples]
        output_dir: 输出目录
        visualize: 是否生成可视化图表
        detailed: 是否进行详细评估
        
    Returns:
        metrics: 评估指标字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 基本评估指标
    metrics = {}
    
    # 判别性分数
    discriminative_score = calculate_discriminative_score_rf(real_data, generated_data[:len(real_data)])
    metrics['discriminative_score'] = discriminative_score
    print(f"判别性分数 (RF Accuracy)：{discriminative_score:.4f} (理想值: 0.5)")
    
    # 预测性分数
    predictive_score = calculate_predictive_score(real_data, generated_data[:len(real_data)])
    metrics['predictive_score'] = predictive_score
    print(f"预测性分数 (MAE比率)：{predictive_score:.4f} (理想值: 1.0)")
    
    # 详细评估
    if detailed:
        # 时间一致性
        temporal_consistency = calculate_temporal_consistency(real_data, generated_data[:len(real_data)])
        metrics['temporal_consistency'] = temporal_consistency
        print(f"时间一致性分数：{temporal_consistency:.4f} (理想值: 1.0)")
        
        # 标签一致性
        label_consistency = calculate_label_consistency(real_labels, generated_labels[:len(real_labels)])
        metrics['label_consistency'] = label_consistency
        print(f"标签一致性分数：{label_consistency:.4f} (理想值: 1.0)")
        
        # 特征重要性
        feature_importance = calculate_feature_importance(real_data, generated_data[:len(real_data)])
        top_features = np.argsort(feature_importance)[-5:]
        print(f"最重要的5个特征索引: {top_features}")
        print(f"对应的重要性值: {feature_importance[top_features]}")
        metrics['top_features'] = top_features.tolist()
        metrics['top_feature_importance'] = feature_importance[top_features].tolist()
    
    # 可视化
    if visualize:
        # t-SNE可视化
        visualize_tsne(real_data, generated_data[:len(real_data)], real_labels, output_dir)
        
        if detailed:
            # 特征重要性可视化
            visualize_feature_importance(feature_importance, output_dir)
            
            # 时间模式可视化
            visualize_temporal_patterns(real_data, generated_data[:len(real_data)], output_dir)
            
            # 标签分布可视化
            visualize_label_distribution(real_labels, generated_labels[:len(real_labels)], output_dir)
    
    # 保存评估指标
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    main() 
