import os
import pandas as pd
import numpy as np
import random
import json
from collections import defaultdict
import time
from sklearn.model_selection import train_test_split

class APTSequenceBuilder:
    def __init__(self, preprocessed_data_path):
        """
        APT攻击序列构建器
        
        Args:
            preprocessed_data_path: 预处理数据的路径
        """
        self.preprocessed_data_path = preprocessed_data_path
        self.stage_data = {}  # 存储各阶段的数据
        self.selected_features = []  # 选择的特征列表
        self.apt_sequences = []  # 构建的APT攻击序列
        self.normal_sequences = []  # 构建的正常序列
        
        # 攻击阶段映射
        self.stage_mapping = {
            'Reconnaissance': 'S1',
            'Establish Foothold': 'S2', 
            'Lateral Movement': 'S3',
            'Data Exfiltration': 'S4',
            '正常流量': 'SN'
        }
        
        # APT攻击类型定义
        self.apt_types = {
            'APT1': ['S1'],  # 只包含侦察阶段
            'APT2': ['S1', 'S2'],  # 侦察 + 建立立足点
            'APT3': ['S1', 'S2', 'S3'],  # 侦察 + 建立立足点 + 横向移动
            'APT4': ['S1', 'S2', 'S3', 'S4']  # 完整攻击链
        }
        
    def load_preprocessed_data(self):
        """加载预处理后的数据 - 语义特征编码"""
        print("--- 加载预处理数据（语义特征编码）---")

        # 加载各阶段的语义特征信息
        stage_features_path = os.path.join(self.preprocessed_data_path, 'stage_features.json')
        if os.path.exists(stage_features_path):
            with open(stage_features_path, 'r', encoding='utf-8') as f:
                self.stage_features_info = json.load(f)
            print(f"加载了各阶段语义特征配置")

            # 检查是否为语义特征
            for stage_name, stage_info in self.stage_features_info.items():
                if stage_info.get('feature_type') == 'semantic':
                    print(f"  {stage_name}: 语义特征模式")
                else:
                    print(f"  {stage_name}: 传统特征模式")
        else:
            # 兼容旧版本，尝试加载统一特征列表
            features_path = os.path.join(self.preprocessed_data_path, 'selected_features.json')
            if os.path.exists(features_path):
                with open(features_path, 'r', encoding='utf-8') as f:
                    self.selected_features = json.load(f)
                print(f"加载了统一特征列表 {len(self.selected_features)} 个特征（兼容模式）")
            else:
                raise FileNotFoundError("未找到特征配置文件")

        # 加载各阶段的预处理数据
        for stage_name in self.stage_mapping.keys():
            file_path = os.path.join(self.preprocessed_data_path, f'{stage_name}_preprocessed.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8')
                # 移除Stage列，只保留特征数据
                feature_data = df.drop('Stage', axis=1)
                self.stage_data[self.stage_mapping[stage_name]] = feature_data

                # 显示该阶段的特征信息
                stage_code = self.stage_mapping[stage_name]
                feature_count = len(feature_data.columns)
                print(f"加载 {stage_name} ({stage_code}): {len(feature_data)} 样本, {feature_count} 个语义特征")

                # 如果有阶段特征信息，显示详细信息
                if hasattr(self, 'stage_features_info') and stage_name in self.stage_features_info:
                    stage_info = self.stage_features_info[stage_name]
                    semantic_features = stage_info.get('features', [])
                    print(f"  语义特征: {semantic_features}")

                    # 显示语义映射信息
                    semantic_mappings = stage_info.get('semantic_mappings', {})
                    if semantic_mappings:
                        total_tokens = sum(len(mapping) for mapping in semantic_mappings.values())
                        print(f"  该阶段token数量: {total_tokens}")
            else:
                print(f"警告: 未找到 {stage_name} 的数据文件")

        print("-" * 50)
    
    def sample_from_stage(self, stage_code, num_samples=1):
        """
        从指定攻击阶段随机采样流量样本
        
        Args:
            stage_code: 阶段代码 (S1, S2, S3, S4, SN)
            num_samples: 采样数量
            
        Returns:
            采样的数据列表
        """
        if stage_code not in self.stage_data:
            raise ValueError(f"阶段 {stage_code} 的数据不存在")
        
        stage_df = self.stage_data[stage_code]
        if len(stage_df) < num_samples:
            # 如果样本不足，使用有放回采样
            sampled_data = stage_df.sample(n=num_samples, replace=True)
        else:
            sampled_data = stage_df.sample(n=num_samples, replace=False)
        
        # 转换为列表格式，每个样本是一个特征值列表
        return [row.tolist() for _, row in sampled_data.iterrows()]
    
    def insert_normal_traffic(self, attack_sequence, min_normal=1, max_normal=3):
        """
        在攻击序列中随机插入正常流量样本
        
        Args:
            attack_sequence: 原始攻击序列
            min_normal: 最少插入的正常流量数量
            max_normal: 最多插入的正常流量数量
            
        Returns:
            插入正常流量后的序列
        """
        result_sequence = []
        
        for i, attack_sample in enumerate(attack_sequence):
            # 添加当前攻击样本
            result_sequence.append(attack_sample)
            
            # 在攻击样本之间随机插入正常流量（除了最后一个攻击样本后）
            if i < len(attack_sequence) - 1:
                num_normal = random.randint(min_normal, max_normal)
                normal_samples = self.sample_from_stage('SN', num_normal)
                result_sequence.extend(normal_samples)
        
        # 在序列开始和结束时也可能插入正常流量
        if random.random() < 0.3:  # 30%概率在开始插入
            num_normal = random.randint(1, 2)
            normal_samples = self.sample_from_stage('SN', num_normal)
            result_sequence = normal_samples + result_sequence
            
        if random.random() < 0.3:  # 30%概率在结束插入
            num_normal = random.randint(1, 2)
            normal_samples = self.sample_from_stage('SN', num_normal)
            result_sequence.extend(normal_samples)
        
        return result_sequence
    
    def build_apt_sequence(self, apt_type):
        """
        构建指定类型的APT攻击序列
        
        Args:
            apt_type: APT类型 (APT1, APT2, APT3, APT4)
            
        Returns:
            构建的APT攻击序列
        """
        if apt_type not in self.apt_types:
            raise ValueError(f"不支持的APT类型: {apt_type}")
        
        stages = self.apt_types[apt_type]
        attack_sequence = []
        
        # 按照攻击阶段顺序采样并拼接
        for stage in stages:
            sample = self.sample_from_stage(stage, 1)[0]  # 每个阶段采样1个样本
            attack_sequence.append(sample)
        
        # 插入正常流量样本
        final_sequence = self.insert_normal_traffic(attack_sequence)
        
        return final_sequence
    
    def build_normal_sequence(self, length=None):
        """
        构建正常流量序列
        
        Args:
            length: 序列长度，如果为None则随机生成长度
            
        Returns:
            构建的正常流量序列
        """
        if length is None:
            length = random.randint(3, 10)  # 随机长度3-10
        
        normal_samples = self.sample_from_stage('SN', length)
        return normal_samples
    
    def generate_sequences(self, num_sequences_per_type=1000):
        """
        生成APT攻击序列和正常序列
        
        Args:
            num_sequences_per_type: 每种APT类型生成的序列数量
        """
        print("--- 开始生成APT攻击序列 ---")
        
        self.apt_sequences = []
        sequence_labels = []
        
        # 生成各种类型的APT攻击序列
        for apt_type in self.apt_types.keys():
            print(f"生成 {apt_type} 序列: {num_sequences_per_type} 个")
            
            for i in range(num_sequences_per_type):
                sequence = self.build_apt_sequence(apt_type)
                self.apt_sequences.append(sequence)
                sequence_labels.append(apt_type)
                
                # 进度显示
                if (i + 1) % 200 == 0:
                    print(f"  已生成 {i + 1}/{num_sequences_per_type} 个 {apt_type} 序列")
        
        print(f"总共生成 {len(self.apt_sequences)} 个APT攻击序列")
        
        # 生成正常序列（数量与APT序列相同）
        print(f"生成正常流量序列: {len(self.apt_sequences)} 个")
        self.normal_sequences = []
        
        for i in range(len(self.apt_sequences)):
            # 正常序列长度参考对应APT序列的长度
            apt_length = len(self.apt_sequences[i])
            normal_length = random.randint(max(3, apt_length-2), apt_length+2)
            
            sequence = self.build_normal_sequence(normal_length)
            self.normal_sequences.append(sequence)
            
            if (i + 1) % 1000 == 0:
                print(f"  已生成 {i + 1}/{len(self.apt_sequences)} 个正常序列")
        
        print(f"总共生成 {len(self.normal_sequences)} 个正常流量序列")
        print("-" * 50)
        
        return sequence_labels
    
    def analyze_sequences(self):
        """分析生成的序列特征"""
        print("--- 序列特征分析 ---")
        
        if not self.apt_sequences:
            print("没有生成的序列可供分析")
            return
        
        # APT序列分析
        apt_lengths = [len(seq) for seq in self.apt_sequences]
        print(f"APT攻击序列:")
        print(f"  数量: {len(self.apt_sequences)}")
        print(f"  长度范围: {min(apt_lengths)} - {max(apt_lengths)}")
        print(f"  平均长度: {np.mean(apt_lengths):.2f}")
        
        # 正常序列分析
        normal_lengths = [len(seq) for seq in self.normal_sequences]
        print(f"正常流量序列:")
        print(f"  数量: {len(self.normal_sequences)}")
        print(f"  长度范围: {min(normal_lengths)} - {max(normal_lengths)}")
        print(f"  平均长度: {np.mean(normal_lengths):.2f}")
        
        # 语义特征维度统计
        if self.apt_sequences:
            if hasattr(self, 'stage_features_info'):
                print(f"语义特征维度统计:")
                total_tokens = 0
                for stage_name, stage_info in self.stage_features_info.items():
                    if isinstance(stage_info, dict) and 'feature_count' in stage_info:
                        feature_count = stage_info['feature_count']
                        print(f"  {stage_name}: {feature_count} 个语义特征")

                        # 计算该阶段的token数量
                        semantic_mappings = stage_info.get('semantic_mappings', {})
                        stage_tokens = sum(len(mapping) for mapping in semantic_mappings.values())
                        total_tokens += stage_tokens
                        print(f"    token数量: {stage_tokens}")

                print(f"  预估总词汇表大小: {total_tokens} 个token")
            else:
                # 兼容模式
                feature_dim = len(self.apt_sequences[0][0])
                print(f"特征维度: {feature_dim} （传统模式）")
        
        print("-" * 50)
    
    def save_sequences(self, output_path):
        """保存生成的序列"""
        print(f"--- 保存序列到 {output_path} ---")
        
        os.makedirs(output_path, exist_ok=True)
        
        # 保存APT攻击序列
        apt_output_path = os.path.join(output_path, 'apt_sequences.json')
        with open(apt_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.apt_sequences, f, ensure_ascii=False, indent=2)
        print(f"已保存APT攻击序列到: {apt_output_path}")
        
        # 保存正常序列
        normal_output_path = os.path.join(output_path, 'normal_sequences.json')
        with open(normal_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.normal_sequences, f, ensure_ascii=False, indent=2)
        print(f"已保存正常序列到: {normal_output_path}")
        
        # 保存语义特征列表
        features_output_path = os.path.join(output_path, 'sequence_features.json')
        if hasattr(self, 'stage_features_info'):
            # 保存各阶段语义特征信息
            with open(features_output_path, 'w', encoding='utf-8') as f:
                json.dump(self.stage_features_info, f, ensure_ascii=False, indent=2)
            print(f"已保存各阶段语义特征信息到: {features_output_path}")
        elif hasattr(self, 'selected_features'):
            # 兼容模式：保存统一特征列表
            with open(features_output_path, 'w', encoding='utf-8') as f:
                json.dump(self.selected_features, f, ensure_ascii=False, indent=2)
            print(f"已保存统一特征列表到: {features_output_path}")
        else:
            print("警告: 没有特征信息可保存")
        
        print("序列保存完成")
        print("-" * 50)

    def _split_apt_sequences(self, test_size=0.2, random_state=42):
        """对APT序列进行分层划分，确保每种APT类型都有代表"""
        import random
        random.seed(random_state)

        # 每种APT类型1000个序列
        sequences_per_type = len(self.apt_sequences) // 4

        train_sequences = []
        test_sequences = []

        for i in range(4):  # APT1, APT2, APT3, APT4
            start_idx = i * sequences_per_type
            end_idx = (i + 1) * sequences_per_type if i < 3 else len(self.apt_sequences)

            type_sequences = self.apt_sequences[start_idx:end_idx]

            # 随机打乱
            random.shuffle(type_sequences)

            # 划分训练集和测试集
            split_point = int(len(type_sequences) * (1 - test_size))
            train_sequences.extend(type_sequences[:split_point])
            test_sequences.extend(type_sequences[split_point:])

        return train_sequences, test_sequences

    def _split_normal_sequences(self, test_size=0.2, random_state=42):
        """对正常序列进行划分"""
        import random
        random.seed(random_state)

        sequences = self.normal_sequences.copy()
        random.shuffle(sequences)

        split_point = int(len(sequences) * (1 - test_size))
        train_sequences = sequences[:split_point]
        test_sequences = sequences[split_point:]

        return train_sequences, test_sequences

    def save_sequences_with_split(self, output_path, test_size=0.2, random_state=42):
        """保存序列时同时进行训练/测试集划分"""
        print(f"--- 保存序列并划分训练/测试集到 {output_path} ---")

        os.makedirs(output_path, exist_ok=True)

        # 对APT序列进行分层划分
        train_apt_sequences, test_apt_sequences = self._split_apt_sequences(test_size, random_state)
        print(f"APT序列划分: 训练集 {len(train_apt_sequences)} 个, 测试集 {len(test_apt_sequences)} 个")

        # 对正常序列进行划分
        train_normal_sequences, test_normal_sequences = self._split_normal_sequences(test_size, random_state)
        print(f"正常序列划分: 训练集 {len(train_normal_sequences)} 个, 测试集 {len(test_normal_sequences)} 个")

        # 保存训练集
        train_apt_path = os.path.join(output_path, 'train_apt_sequences.json')
        with open(train_apt_path, 'w', encoding='utf-8') as f:
            json.dump(train_apt_sequences, f, ensure_ascii=False, indent=2)
        print(f"已保存训练集APT序列到: {train_apt_path}")

        train_normal_path = os.path.join(output_path, 'train_normal_sequences.json')
        with open(train_normal_path, 'w', encoding='utf-8') as f:
            json.dump(train_normal_sequences, f, ensure_ascii=False, indent=2)
        print(f"已保存训练集正常序列到: {train_normal_path}")

        # 保存测试集
        test_apt_path = os.path.join(output_path, 'test_apt_sequences.json')
        with open(test_apt_path, 'w', encoding='utf-8') as f:
            json.dump(test_apt_sequences, f, ensure_ascii=False, indent=2)
        print(f"已保存测试集APT序列到: {test_apt_path}")

        test_normal_path = os.path.join(output_path, 'test_normal_sequences.json')
        with open(test_normal_path, 'w', encoding='utf-8') as f:
            json.dump(test_normal_sequences, f, ensure_ascii=False, indent=2)
        print(f"已保存测试集正常序列到: {test_normal_path}")

        # 保存语义特征列表
        features_output_path = os.path.join(output_path, 'sequence_features.json')
        if hasattr(self, 'stage_features_info'):
            # 保存各阶段语义特征信息
            with open(features_output_path, 'w', encoding='utf-8') as f:
                json.dump(self.stage_features_info, f, ensure_ascii=False, indent=2)
            print(f"已保存各阶段语义特征信息到: {features_output_path}")
        elif hasattr(self, 'selected_features'):
            # 兼容模式：保存统一特征列表
            with open(features_output_path, 'w', encoding='utf-8') as f:
                json.dump(self.selected_features, f, ensure_ascii=False, indent=2)
            print(f"已保存统一特征列表到: {features_output_path}")
        else:
            print("警告: 没有特征信息可保存")

        print("序列保存和划分完成")
        print("-" * 50)

    def split_data_for_attack_detection(self, test_size=0.3, random_state=42, classification_type='binary'):
        """
        将生成的序列划分为训练集和测试集，用于攻击检测任务

        Args:
            test_size: 测试集比例 (默认0.3)
            random_state: 随机种子 (默认42)
            classification_type: 分类类型 ('binary' 或 'multiclass')

        Returns:
            dict: 包含训练集和测试集的字典
        """
        print(f"--- 数据划分 (测试集比例: {test_size}, 分类类型: {classification_type}) ---")

        if not self.apt_sequences or not self.normal_sequences:
            raise ValueError("请先生成序列数据")

        if classification_type == 'binary':
            return self._split_for_binary_classification(test_size, random_state)
        elif classification_type == 'multiclass':
            return self._split_for_multiclass_classification(test_size, random_state)
        else:
            raise ValueError("classification_type 必须是 'binary' 或 'multiclass'")

    def _split_for_binary_classification(self, test_size, random_state):
        """二分类数据划分：正常(0) vs 攻击(1)"""
        print("执行二分类数据划分...")

        # 准备数据和标签
        all_sequences = []
        all_labels = []

        # 添加攻击序列 (标签=1)
        all_sequences.extend(self.apt_sequences)
        all_labels.extend([1] * len(self.apt_sequences))

        # 添加正常序列 (标签=0)
        all_sequences.extend(self.normal_sequences)
        all_labels.extend([0] * len(self.normal_sequences))

        print(f"总数据量: {len(all_sequences)} 个序列")
        print(f"  - 攻击序列: {len(self.apt_sequences)} 个 (标签=1)")
        print(f"  - 正常序列: {len(self.normal_sequences)} 个 (标签=0)")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            all_sequences, all_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=all_labels  # 保持标签比例
        )

        # 统计划分结果
        train_attack_count = sum(y_train)
        train_normal_count = len(y_train) - train_attack_count
        test_attack_count = sum(y_test)
        test_normal_count = len(y_test) - test_attack_count

        print(f"\n训练集: {len(X_train)} 个序列")
        print(f"  - 攻击序列: {train_attack_count} 个")
        print(f"  - 正常序列: {train_normal_count} 个")

        print(f"\n测试集: {len(X_test)} 个序列")
        print(f"  - 攻击序列: {test_attack_count} 个")
        print(f"  - 正常序列: {test_normal_count} 个")

        # 返回划分结果
        # 计算特征维度
        if hasattr(self, 'stage_features_info'):
            # 各阶段独立特征模式 - 使用最大特征数作为参考
            max_feature_dim = max(
                stage_info.get('feature_count', 0)
                for stage_info in self.stage_features_info.values()
                if isinstance(stage_info, dict)
            )
            feature_dim = max_feature_dim
        elif hasattr(self, 'selected_features') and self.selected_features:
            feature_dim = len(self.selected_features)
        else:
            # 从实际数据推断
            feature_dim = len(X_train[0][0]) if X_train and X_train[0] else 0

        split_data = {
            'train_sequences': X_train,
            'train_labels': y_train,
            'test_sequences': X_test,
            'test_labels': y_test,
            'feature_dim': feature_dim,
            'classification_type': 'binary',
            'num_classes': 2,
            'label_mapping': {0: 'normal', 1: 'attack'},
            'split_info': {
                'test_size': test_size,
                'random_state': random_state,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_attack_count': train_attack_count,
                'train_normal_count': train_normal_count,
                'test_attack_count': test_attack_count,
                'test_normal_count': test_normal_count
            }
        }

        print("-" * 50)
        return split_data

    def _split_for_multiclass_classification(self, test_size, random_state):
        """多分类数据划分：正常(0) vs APT1(1) vs APT2(2) vs APT3(3) vs APT4(4)"""
        print("执行多分类数据划分...")

        # 准备数据和标签
        all_sequences = []
        all_labels = []

        # 每种APT类型的序列数量
        sequences_per_type = len(self.apt_sequences) // 4

        # 添加各类型APT攻击序列
        apt_type_counts = {}
        for i, sequence in enumerate(self.apt_sequences):
            # 根据序列索引确定APT类型和标签
            apt_type_index = i // sequences_per_type
            if apt_type_index >= 4:  # 防止索引越界
                apt_type_index = 3

            # 标签: APT1=1, APT2=2, APT3=3, APT4=4
            label = apt_type_index + 1

            all_sequences.append(sequence)
            all_labels.append(label)

            # 统计各类型数量
            apt_type = f"APT{label}"
            apt_type_counts[apt_type] = apt_type_counts.get(apt_type, 0) + 1

        # 添加正常序列 (标签=0)
        all_sequences.extend(self.normal_sequences)
        all_labels.extend([0] * len(self.normal_sequences))
        apt_type_counts["正常"] = len(self.normal_sequences)

        print(f"总数据量: {len(all_sequences)} 个序列")
        print(f"  - 正常序列: {apt_type_counts['正常']} 个 (标签=0)")
        for i in range(1, 5):
            apt_type = f"APT{i}"
            print(f"  - {apt_type}序列: {apt_type_counts.get(apt_type, 0)} 个 (标签={i})")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            all_sequences, all_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=all_labels  # 保持标签比例
        )

        # 统计划分结果
        train_label_counts = {}
        test_label_counts = {}

        for label in y_train:
            train_label_counts[label] = train_label_counts.get(label, 0) + 1

        for label in y_test:
            test_label_counts[label] = test_label_counts.get(label, 0) + 1

        print(f"\n训练集: {len(X_train)} 个序列")
        print(f"  - 正常序列 (标签=0): {train_label_counts.get(0, 0)} 个")
        for i in range(1, 5):
            print(f"  - APT{i}序列 (标签={i}): {train_label_counts.get(i, 0)} 个")

        print(f"\n测试集: {len(X_test)} 个序列")
        print(f"  - 正常序列 (标签=0): {test_label_counts.get(0, 0)} 个")
        for i in range(1, 5):
            print(f"  - APT{i}序列 (标签={i}): {test_label_counts.get(i, 0)} 个")

        # 返回划分结果
        # 计算特征维度
        if hasattr(self, 'stage_features_info'):
            # 各阶段独立特征模式 - 使用最大特征数作为参考
            max_feature_dim = max(
                stage_info.get('feature_count', 0)
                for stage_info in self.stage_features_info.values()
                if isinstance(stage_info, dict)
            )
            feature_dim = max_feature_dim
        elif hasattr(self, 'selected_features') and self.selected_features:
            feature_dim = len(self.selected_features)
        else:
            # 从实际数据推断
            feature_dim = len(X_train[0][0]) if X_train and X_train[0] else 0

        split_data = {
            'train_sequences': X_train,
            'train_labels': y_train,
            'test_sequences': X_test,
            'test_labels': y_test,
            'feature_dim': feature_dim,
            'classification_type': 'multiclass',
            'num_classes': 5,  # 0=正常, 1=APT1, 2=APT2, 3=APT3, 4=APT4
            'label_mapping': {
                0: 'normal',
                1: 'APT1',
                2: 'APT2',
                3: 'APT3',
                4: 'APT4'
            },
            'split_info': {
                'test_size': test_size,
                'random_state': random_state,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_label_counts': train_label_counts,
                'test_label_counts': test_label_counts
            }
        }

        print("-" * 50)
        return split_data

    def save_split_data_for_transformer(self, split_data, output_path):
        """
        保存划分后的数据，格式适合transformer攻击检测训练

        Args:
            split_data: split_data_for_attack_detection()的返回结果
            output_path: 输出路径
        """
        print(f"--- 保存划分数据到 {output_path} ---")

        os.makedirs(output_path, exist_ok=True)

        # 保存训练集
        train_data = {
            'sequences': split_data['train_sequences'],
            'labels': split_data['train_labels']
        }
        train_path = os.path.join(output_path, 'train_data.json')
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"已保存训练集到: {train_path}")

        # 保存测试集
        test_data = {
            'sequences': split_data['test_sequences'],
            'labels': split_data['test_labels']
        }
        test_path = os.path.join(output_path, 'test_data.json')
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"已保存测试集到: {test_path}")

        # 保存数据信息
        classification_type = split_data.get('classification_type', 'binary')
        num_classes = split_data.get('num_classes', 2)
        label_mapping = split_data.get('label_mapping', {0: 'normal', 1: 'attack'})

        # 构建标签描述
        if classification_type == 'binary':
            label_description = '二分类标签，0=正常，1=攻击'
        else:
            label_description = f'{num_classes}分类标签，0=正常，1=APT1，2=APT2，3=APT3，4=APT4'

        # 构建数据信息，适应各阶段独立特征
        if hasattr(self, 'stage_features_info'):
            features_info = self.stage_features_info
            feature_description = '各阶段使用独立特征集'
        elif hasattr(self, 'selected_features'):
            features_info = self.selected_features
            feature_description = '所有阶段使用统一特征集'
        else:
            features_info = {}
            feature_description = '特征信息不可用'

        data_info = {
            'feature_dim': split_data['feature_dim'],
            'features': features_info,
            'split_info': split_data['split_info'],
            'classification_type': classification_type,
            'num_classes': num_classes,
            'label_mapping': label_mapping,
            'data_format': {
                'sequences': '每个序列是一个二维数组，外层为时间步，内层为特征',
                'labels': label_description,
                'feature_description': feature_description,
                'feature_dim': f'{split_data["feature_dim"]}个特征（可能因阶段而异）'
            }
        }
        info_path = os.path.join(output_path, 'data_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=2)
        print(f"已保存数据信息到: {info_path}")

        print("划分数据保存完成")
        print("-" * 50)


if __name__ == '__main__':
    # 使用示例
    PREPROCESSED_DATA_PATH = './preprocessed_output'  # 预处理数据路径
    OUTPUT_PATH = './sequences_output'  # 序列输出路径
    NUM_SEQUENCES_PER_TYPE = 1000  # 每种APT类型生成的序列数量
    
    try:
        # 创建序列构建器
        builder = APTSequenceBuilder(PREPROCESSED_DATA_PATH)
        
        # 加载预处理数据
        builder.load_preprocessed_data()
        
        # 生成序列
        sequence_labels = builder.generate_sequences(num_sequences_per_type=NUM_SEQUENCES_PER_TYPE)
        
        # 分析序列
        builder.analyze_sequences()
        
        # 保存序列并划分训练/测试集
        builder.save_sequences_with_split(OUTPUT_PATH)

        # 数据划分用于二分类攻击检测
        print("\n--- 数据划分用于二分类攻击检测 ---")
        binary_split_data = builder.split_data_for_attack_detection(
            test_size=0.3,
            random_state=42,
            classification_type='binary'
        )

        # 保存二分类数据
        binary_output_path = './attack_detection_binary'
        builder.save_split_data_for_transformer(binary_split_data, binary_output_path)

        # 数据划分用于多分类攻击检测
        print("\n--- 数据划分用于多分类攻击检测 ---")
        multiclass_split_data = builder.split_data_for_attack_detection(
            test_size=0.3,
            random_state=42,
            classification_type='multiclass'
        )

        # 保存多分类数据
        multiclass_output_path = './attack_detection_multiclass'
        builder.save_split_data_for_transformer(multiclass_split_data, multiclass_output_path)

        print("\n--- 序列构建完成 ---")
        print(f"APT攻击序列: {len(builder.apt_sequences)} 个")
        print(f"正常流量序列: {len(builder.normal_sequences)} 个")
        # 显示语义特征维度信息
        if hasattr(builder, 'stage_features_info'):
            print("各阶段语义特征维度:")
            total_tokens = 0
            for stage_name, stage_info in builder.stage_features_info.items():
                if isinstance(stage_info, dict) and 'feature_count' in stage_info:
                    feature_count = stage_info['feature_count']
                    print(f"  {stage_name}: {feature_count} 个语义特征")

                    # 计算token数量
                    semantic_mappings = stage_info.get('semantic_mappings', {})
                    stage_tokens = sum(len(mapping) for mapping in semantic_mappings.values())
                    total_tokens += stage_tokens
                    print(f"    token数量: {stage_tokens}")

            print(f"  预估总词汇表大小: {total_tokens} 个token")
        elif hasattr(builder, 'selected_features') and builder.selected_features:
            print(f"特征维度: {len(builder.selected_features)} （传统模式）")
        else:
            print("特征维度: 信息不可用")
        print(f"二分类训练集大小: {binary_split_data['split_info']['train_size']} 个序列")
        print(f"二分类测试集大小: {binary_split_data['split_info']['test_size']} 个序列")
        print(f"多分类训练集大小: {multiclass_split_data['split_info']['train_size']} 个序列")
        print(f"多分类测试集大小: {multiclass_split_data['split_info']['test_size']} 个序列")
        
        # 显示各类型APT序列示例
        if builder.apt_sequences:
            print("\n--- APT序列示例展示 ---")

            # 显示每种APT类型的第一个序列
            apt_type_examples = {
                'APT1': 0,      # 第1个序列 (APT1)
                'APT2': 1000,   # 第1001个序列 (APT2)
                'APT3': 2000,   # 第2001个序列 (APT3)
                'APT4': 3000    # 第3001个序列 (APT4)
            }

            for apt_type, index in apt_type_examples.items():
                if index < len(builder.apt_sequences):
                    sequence = builder.apt_sequences[index]
                    print(f"\n{apt_type} 序列示例 (长度: {len(sequence)}):")
                    print(f"  攻击阶段数: {len(builder.apt_types[apt_type])}")
                    print(f"  序列构成: {builder.apt_types[apt_type]} + 正常流量")
                    print(f"  第一个样本前10个特征: {sequence[0][:10]}")
                    if len(sequence) > 1:
                        print(f"  最后一个样本前10个特征: {sequence[-1][:10]}")

            # 显示正常序列示例
            if builder.normal_sequences:
                normal_seq = builder.normal_sequences[0]
                print(f"\n正常序列示例 (长度: {len(normal_seq)}):")
                print(f"  第一个样本前10个特征: {normal_seq[0][:10]}")

        print("\n--- 序列文件说明 ---")
        print("原始序列文件 (./sequences_output/):")
        print("  - apt_sequences.json: APT攻击序列数据")
        print("  - normal_sequences.json: 正常流量序列数据")
        print("  - sequence_features.json: 特征名称列表")

        print("\n二分类攻击检测数据 (./attack_detection_binary/):")
        print("  - train_data.json: 训练集数据和标签")
        print("  - test_data.json: 测试集数据和标签")
        print("  - data_info.json: 数据信息和格式说明")
        print("  - 标签: 0=正常, 1=攻击")

        print("\n多分类攻击检测数据 (./attack_detection_multiclass/):")
        print("  - train_data.json: 训练集数据和标签")
        print("  - test_data.json: 测试集数据和标签")
        print("  - data_info.json: 数据信息和格式说明")
        print("  - 标签: 0=正常, 1=APT1, 2=APT2, 3=APT3, 4=APT4")

        print("\n数据格式说明:")
        print("  - 序列: 每个序列是一个二维数组 [时间步, 特征]")
        # 显示特征维度
        if hasattr(builder, 'stage_features_info'):
            print("  - 特征维度: 各阶段独立特征集")
        elif hasattr(builder, 'selected_features') and builder.selected_features:
            print(f"  - 特征维度: {len(builder.selected_features)} 个特征（统一）")
        else:
            print("  - 特征维度: 信息不可用")
        print("  - 适合直接用于transformer攻击检测训练")
            
    except Exception as e:
        print(f"序列构建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
