"""
使用改进的TimeGAN生成APT攻击序列的主脚本
增加了训练轮数、模型容量和生成样本数量
添加了条件生成功能，使模型能够根据类别信息生成样本
特别适用于某些类别样本较少的情况，如APT4序列
"""

import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    
    # 统计标签分布
    labels = [seq['label'] for seq in sequences]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"标签分布: {label_counts}")
    
    return sequences

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
    parser = argparse.ArgumentParser(description='改进的TimeGAN生成APT攻击序列')
    
    # 数据参数
    parser.add_argument('--input_file', type=str, default='sequences_3tuple/apt_attack_sequences.pkl',
                        help='输入的APT攻击序列文件路径')
    parser.add_argument('--output_dir', type=str, default='improved_generated_sequences',
                        help='生成序列的输出目录')
    parser.add_argument('--log_dir', type=str, default='logs/improved_timegan',
                        help='TensorBoard日志目录')
    parser.add_argument('--model_path', type=str, default='models/improved_timegan_model.pt',
                        help='模型保存/加载路径')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度')
    parser.add_argument('--z_dim', type=int, default=32,
                        help='噪声维度')
    parser.add_argument('--max_seq_len', type=int, default=10,
                        help='最大序列长度')
    parser.add_argument('--feature_dim', type=int, default=40,
                        help='特征维度')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='RNN层数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='监督损失权重')
    parser.add_argument('--discriminator_iterations', type=int, default=3,
                        help='每次迭代中判别器训练的次数')
    parser.add_argument('--early_stopping', action='store_true',
                        help='是否启用早停机制')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值，验证损失多少轮未改善则停止训练')
    
    # 生成参数
    parser.add_argument('--generate_nums', type=str, default='100,100,100,200',
                        help='每个标签生成的样本数量，格式为逗号分隔的列表')
    
    # 操作模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'train_and_generate'],
                        help='操作模式: train, generate, train_and_generate')
    parser.add_argument('--balance', action='store_true', default=True,
                        help='是否平衡训练数据')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否可视化生成的数据')
    parser.add_argument('--verbose', action='store_true',
                        help='是否输出详细训练日志')
    
    return parser.parse_args()

def main():
    """主函数"""
    print("开始执行主函数...")
    
    # 解析命令行参数
    args = parse_arguments()
    print(f"解析命令行参数完成: {args}")
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载APT攻击序列
    sequences = load_apt_sequences(args.input_file)
    
    # 解析生成数量
    generate_nums = [int(num) for num in args.generate_nums.split(',')]
    generate_target = {i+1: generate_nums[i] for i in range(len(generate_nums)) if i < 4}
    
    if args.mode in ['train', 'train_and_generate']:
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
        
        # 创建ConditionalTimeGAN模型
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
        
        # 创建TensorDataset和DataLoader
        tensor_x = torch.Tensor(processed_data)
        train_dataset = TensorDataset(tensor_x)
        train_data_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        
        # 训练模型
        model.fit(
            train_data_loader=train_data_loader,
            epochs=args.epochs,
            verbose=args.verbose
        )
        
        # 保存模型
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        
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
        if args.mode == 'generate':
            # 加载模型
            model = ConditionalTimeGAN.load(args.model_path, device=device)
            
            # 加载scaler和预处理信息
            scaler_path = os.path.join(os.path.dirname(args.model_path), 'improved_scaler.pkl')
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(os.path.dirname(args.model_path), 'scaler.pkl')
                
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            preprocess_info_path = os.path.join(os.path.dirname(args.model_path), 'improved_preprocess_info.pkl')
            if not os.path.exists(preprocess_info_path):
                preprocess_info_path = os.path.join(os.path.dirname(args.model_path), 'preprocess_info.pkl')
                
            with open(preprocess_info_path, 'rb') as f:
                preprocess_info = pickle.load(f)
            
            labels = preprocess_info['labels']
            phases = preprocess_info['phases']
            feature_dim = preprocess_info['feature_dim']
            max_seq_len = preprocess_info['max_seq_len']
            
            # 预处理原始数据用于可视化（如果需要）
            processed_data = None
            if args.visualize:
                processed_data, _, _, _, _ = preprocess_apt_data(
                    sequences, 
                    max_seq_len=max_seq_len,
                    feature_dim=feature_dim
                )
        
        # 生成数据
        total_samples = sum(generate_target.values())
        print(f"开始生成 {total_samples} 个样本，目标数量: {generate_target}")
        
        # 准备标签
        generated_labels = []
        for label, num in generate_target.items():
            generated_labels.extend([label-1] * num)  # 标签从0开始
        
        generated_labels = np.array(generated_labels)
        
        # 分批生成数据，避免内存不足
        batch_size = 50
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        all_generated_data = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            batch_size_actual = end_idx - start_idx
            
            print(f"生成批次 {i+1}/{num_batches}, 大小: {batch_size_actual}")
            
            # 获取当前批次的类别ID
            batch_class_ids = generated_labels[start_idx:end_idx]
            
            # 生成数据
            batch_data = model.generate(n_samples=batch_size_actual, seq_len=args.max_seq_len, class_ids=batch_class_ids)
            all_generated_data.append(batch_data)
        
        # 合并所有生成的数据
        generated_data = np.concatenate(all_generated_data, axis=0)
        
        # 可视化
        if args.visualize:
            visualize_tsne(processed_data, generated_data, labels, args.output_dir)
        
        # 后处理生成的数据
        generated_sequences = postprocess_generated_data(
            generated_data,
            scaler,
            generated_labels,
            phases,
            feature_dim=args.feature_dim
        )
        
        # 保存生成的序列
        save_generated_sequences(
            generated_sequences,
            args.output_dir,
            filename='generated_apt_sequences.pkl'
        )
        
        # 保存混合数据集（原始 + 生成）
        if args.balance:
            # 混合序列
            mixed_sequences = sequences + generated_sequences
            save_generated_sequences(
                mixed_sequences,
                args.output_dir,
                filename='mixed_apt_sequences.pkl'
            )
            
            # 计算训练测试集
            # 随机打乱
            np.random.shuffle(mixed_sequences)
            
            # 划分为训练集(80%)和测试集(20%)
            split_idx = int(len(mixed_sequences) * 0.8)
            train_sequences = mixed_sequences[:split_idx]
            test_sequences = mixed_sequences[split_idx:]
            
            # 保存训练集和测试集
            save_generated_sequences(
                train_sequences,
                args.output_dir,
                filename='train_apt_sequences.pkl'
            )
            
            save_generated_sequences(
                test_sequences,
                args.output_dir,
                filename='test_apt_sequences.pkl'
            )

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

if __name__ == "__main__":
    main() 