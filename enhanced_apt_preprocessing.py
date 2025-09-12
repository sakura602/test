import os
import json
import numpy as np
from collections import defaultdict
import time

class APTIDSequenceBuilder:
    def __init__(self, sequences_path):
        """
        APT攻击标识序列构建器 - 支持按阶段分别处理特征

        Args:
            sequences_path: 序列数据路径
        """
        self.sequences_path = sequences_path
        self.apt_sequences = []  # APT攻击序列
        self.normal_sequences = []  # 正常流量序列
        self.features = []  # 特征列表

        # 统一的attack2id映射表（所有阶段共享）
        self.attack2id = {}  # 统一映射表，所有阶段共享
        self.stage_feature_masks = {}  # 每个阶段的特征掩码，用于特征选择

        # APT攻击标识序列和标签
        self.apt_id_sequences = []  # APT攻击标识序列
        self.apt_labels = []  # APT攻击标签

        # 正常流量标识序列和标签
        self.normal_id_sequences = []  # 正常流量标识序列
        self.normal_labels = []  # 正常流量标签

        # APT类型到标签的映射
        self.apt_type_to_label = {
            'APT1': 1,  # 仅包含阶段1
            'APT2': 2,  # 包含前2个阶段
            'APT3': 3,  # 包含前3个阶段
            'APT4': 4,  # 包含前4个阶段
            'Normal': 0  # 正常流量
        }

        # 定义每个攻击阶段的语义特征（基于语义编码）
        self.stage_semantic_features = {
            'S1': {  # 侦察阶段：语义特征
                'feature_names': ['scan_intensity', 'probe_pattern', 'collection_level'],
                'feature_count': 3,
                'description': '扫描强度、探测模式、数据收集量'
            },
            'S2': {  # 建立立足点：语义特征
                'feature_names': ['target_service', 'connection_mode'],
                'feature_count': 2,
                'description': '目标服务类型、连接模式'
            },
            'S3': {  # 横向移动：语义特征
                'feature_names': ['movement_pattern', 'session_duration'],
                'feature_count': 2,
                'description': '移动模式、会话持续时间'
            },
            'S4': {  # 数据窃取：语义特征
                'feature_names': ['exfil_volume', 'transfer_mode'],
                'feature_count': 2,
                'description': '窃取数据量、传输模式'
            },
            'SN': {  # 正常流量：语义特征
                'feature_names': ['traffic_pattern', 'service_type', 'volume_level'],
                'feature_count': 3,
                'description': '流量模式、服务类型、流量量级'
            }
        }
        
    def load_sequences(self):
        """加载已划分的训练集和测试集序列数据"""
        print("--- 加载已划分的序列数据 ---")

        # 加载训练集APT攻击序列
        train_apt_path = os.path.join(self.sequences_path, 'train_apt_sequences.json')
        with open(train_apt_path, 'r', encoding='utf-8') as f:
            self.train_apt_sequences = json.load(f)
        print(f"加载训练集APT攻击序列: {len(self.train_apt_sequences)} 个")

        # 加载训练集正常流量序列
        train_normal_path = os.path.join(self.sequences_path, 'train_normal_sequences.json')
        with open(train_normal_path, 'r', encoding='utf-8') as f:
            self.train_normal_sequences = json.load(f)
        print(f"加载训练集正常流量序列: {len(self.train_normal_sequences)} 个")

        # 加载测试集APT攻击序列
        test_apt_path = os.path.join(self.sequences_path, 'test_apt_sequences.json')
        with open(test_apt_path, 'r', encoding='utf-8') as f:
            self.test_apt_sequences = json.load(f)
        print(f"加载测试集APT攻击序列: {len(self.test_apt_sequences)} 个")

        # 加载测试集正常流量序列
        test_normal_path = os.path.join(self.sequences_path, 'test_normal_sequences.json')
        with open(test_normal_path, 'r', encoding='utf-8') as f:
            self.test_normal_sequences = json.load(f)
        print(f"加载测试集正常流量序列: {len(self.test_normal_sequences)} 个")

        # 合并所有序列用于构建统一映射表
        self.apt_sequences = self.train_apt_sequences + self.test_apt_sequences
        self.normal_sequences = self.train_normal_sequences + self.test_normal_sequences
        print(f"总APT序列: {len(self.apt_sequences)} 个")
        print(f"总正常序列: {len(self.normal_sequences)} 个")

        # 加载特征列表（各阶段独立或统一）
        features_path = os.path.join(self.sequences_path, 'sequence_features.json')
        with open(features_path, 'r', encoding='utf-8') as f:
            features_data = json.load(f)

        if isinstance(features_data, dict) and any(isinstance(v, dict) for v in features_data.values()):
            # 新格式：各阶段语义特征
            self.stage_features_from_file = features_data
            print(f"加载各阶段语义特征配置:")
            for stage_name, stage_info in features_data.items():
                if isinstance(stage_info, dict) and 'features' in stage_info:
                    feature_type = stage_info.get('feature_type', 'unknown')
                    print(f"  {stage_name}: {stage_info['feature_count']} 个{feature_type}特征")

            # 从文件中的特征信息更新阶段特征配置
            self._update_semantic_features_from_file()
        else:
            # 旧格式：统一特征列表
            self.features = features_data
            print(f"加载统一特征列表: {len(self.features)} 个特征")

            # 初始化每个阶段的特征索引（兼容模式）
            self._initialize_stage_feature_indices()

        print("-" * 50)

    def _update_semantic_features_from_file(self):
        """从文件中的语义特征信息更新阶段特征配置"""
        print("--- 从文件更新语义特征配置 ---")

        # 映射文件中的阶段名称到内部阶段代码
        stage_name_mapping = {
            'Reconnaissance': 'S1',
            'Establish Foothold': 'S2',
            'Lateral Movement': 'S3',
            'Data Exfiltration': 'S4',
            '正常流量': 'SN'
        }

        for stage_name, stage_info in self.stage_features_from_file.items():
            if isinstance(stage_info, dict) and 'features' in stage_info:
                # 获取内部阶段代码
                stage_code = stage_name_mapping.get(stage_name, stage_name)

                if stage_code in self.stage_semantic_features:
                    # 使用文件中的语义特征列表
                    self.stage_semantic_features[stage_code]['feature_names'] = stage_info['features']
                    self.stage_semantic_features[stage_code]['feature_count'] = stage_info['feature_count']

                    # 保存语义映射信息
                    if 'semantic_mappings' in stage_info:
                        self.stage_semantic_features[stage_code]['semantic_mappings'] = stage_info['semantic_mappings']

                    print(f"{stage_code} ({stage_name}): {len(stage_info['features'])} 个语义特征")
                    print(f"  特征: {stage_info['features']}")
                else:
                    print(f"警告: 未知阶段 {stage_code} ({stage_name})")

        print("-" * 50)

    def _update_stage_features_from_file(self):
        """从文件中的特征信息更新阶段特征配置"""
        print("--- 从文件更新阶段特征配置 ---")

        # 映射文件中的阶段名称到内部阶段代码
        stage_name_mapping = {
            'Reconnaissance': 'S1',
            'Establish Foothold': 'S2',
            'Lateral Movement': 'S3',
            'Data Exfiltration': 'S4',
            '正常流量': 'SN'
        }

        for stage_name, stage_info in self.stage_features_from_file.items():
            if isinstance(stage_info, dict) and 'features' in stage_info:
                # 获取内部阶段代码
                stage_code = stage_name_mapping.get(stage_name, stage_name)

                if stage_code in self.stage_key_features:
                    # 使用文件中的实际特征列表
                    self.stage_key_features[stage_code]['feature_names'] = stage_info['features']
                    self.stage_key_features[stage_code]['indices'] = list(range(len(stage_info['features'])))

                    print(f"{stage_code} ({stage_name}): {len(stage_info['features'])} 个特征")
                else:
                    print(f"警告: 未知阶段 {stage_code} ({stage_name})")

        print("-" * 50)

    def _initialize_stage_feature_indices(self):
        """初始化每个攻击阶段的关键特征索引（兼容模式）"""
        print("--- 初始化阶段特征索引（兼容模式）---")

        if not hasattr(self, 'features') or not self.features:
            print("警告: 没有统一特征列表，跳过特征索引初始化")
            return

        for stage, stage_info in self.stage_key_features.items():
            if stage == 'SN':  # 正常流量使用所有特征
                stage_info['indices'] = list(range(len(self.features)))
                stage_info['feature_names'] = self.features.copy()
            else:
                # 找到每个关键特征在特征列表中的索引
                indices = []
                found_features = []
                for feature_name in stage_info['feature_names']:
                    if feature_name in self.features:
                        indices.append(self.features.index(feature_name))
                        found_features.append(feature_name)
                    else:
                        print(f"警告: 特征 '{feature_name}' 在特征列表中未找到")

                stage_info['indices'] = indices
                stage_info['feature_names'] = found_features

                print(f"{stage} 阶段关键特征: {len(indices)} 个")
                print(f"  特征名称: {found_features}")

        print("-" * 50)
    
    def extract_all_discrete_features(self):
        """
        提取所有阶段序列中的离散型编码特征 - 构建统一映射表

        Returns:
            unique_features: 所有阶段唯一的离散特征值集合
        """
        print("--- 提取所有阶段的离散型编码特征（统一映射表）---")

        # 收集所有阶段的所有特征值
        all_feature_values = set()
        total_samples = 0
        total_features = 0

        # 处理APT序列
        for sequence in self.apt_sequences:
            for sample in sequence:  # sample是包含特征的向量
                total_samples += 1
                for feature_value in sample:
                    all_feature_values.add(feature_value)
                    total_features += 1

        # 处理正常序列
        for sequence in self.normal_sequences:
            for sample in sequence:  # sample是包含特征的向量
                total_samples += 1
                for feature_value in sample:
                    all_feature_values.add(feature_value)
                    total_features += 1

        # 获取唯一的特征值并排序
        unique_features = sorted(list(all_feature_values))

        print(f"处理了 {len(self.apt_sequences + self.normal_sequences)} 个序列，{total_samples} 个样本，{total_features} 个特征值")
        print(f"提取到 {len(unique_features)} 个唯一的离散特征值")
        if unique_features:
            print(f"特征值范围: {min(unique_features):.3f} - {max(unique_features):.3f}")

        return unique_features
    
    def build_unified_attack2id_mapping(self):
        """构建统一的attack2id映射表（所有阶段共享）"""
        print("--- 构建统一的attack2id映射表 ---")

        # 提取所有阶段的离散特征值
        unique_feature_values = self.extract_all_discrete_features()

        # 构建统一映射表：每个唯一的特征值对应一个ID
        self.attack2id = {}
        for i, feature_value in enumerate(unique_feature_values):
            self.attack2id[feature_value] = i

        print(f"构建完成，统一映射表大小: {len(self.attack2id)}")
        print(f"ID范围: 0 - {len(self.attack2id) - 1}")

        # 显示映射示例
        print("映射示例 (前10个特征值):")
        for i, (feature_value, id_val) in enumerate(list(self.attack2id.items())[:10]):
            print(f"  特征值 {feature_value:.6f} -> ID {id_val}")

        # 显示一些统计信息
        feature_values = list(self.attack2id.keys())
        print(f"\n特征值统计:")
        print(f"  最小特征值: {min(feature_values):.6f}")
        print(f"  最大特征值: {max(feature_values):.6f}")
        print(f"  零值特征数量: {sum(1 for v in feature_values if v == 0.0)}")
        print(f"  正值特征数量: {sum(1 for v in feature_values if v > 0.0)}")
        print(f"  负值特征数量: {sum(1 for v in feature_values if v < 0.0)}")

        # 构建各阶段的特征掩码（用于特征选择）
        self._build_stage_feature_masks()

        print("-" * 50)
    
    def _build_stage_feature_masks(self):
        """构建各阶段的语义特征掩码，用于特征选择"""
        print("--- 构建各阶段语义特征掩码 ---")

        for stage_code, stage_info in self.stage_semantic_features.items():
            feature_names = stage_info['feature_names']
            feature_count = stage_info['feature_count']

            # 如果有从文件加载的语义特征信息，使用文件中的特征
            if hasattr(self, 'stage_features_from_file'):
                # 查找对应的阶段名称
                stage_name_mapping = {
                    'S1': 'Reconnaissance',
                    'S2': 'Establish Foothold',
                    'S3': 'Lateral Movement',
                    'S4': 'Data Exfiltration',
                    'SN': '正常流量'
                }

                stage_name = stage_name_mapping.get(stage_code, stage_code)
                if stage_name in self.stage_features_from_file:
                    file_stage_info = self.stage_features_from_file[stage_name]
                    if isinstance(file_stage_info, dict) and 'features' in file_stage_info:
                        feature_names = file_stage_info['features']
                        feature_count = file_stage_info['feature_count']

            # 构建该阶段的语义特征掩码（所有语义特征的索引列表）
            self.stage_feature_masks[stage_code] = list(range(feature_count))

            print(f"{stage_code}: {feature_count} 个语义特征")
            print(f"  语义特征: {feature_names}")

        print("-" * 50)

    def sequence_to_stage_id_sequence(self, sequence, stage_code):
        """
        将单个序列转换为该阶段的标识序列 - 使用统一映射表但只保留该阶段的特征

        Args:
            sequence: 输入序列，每个元素是包含该阶段特征的样本
            stage_code: 阶段代码 (S1, S2, S3, S4, SN)

        Returns:
            id_sequence: 标识序列，每个元素是包含该阶段特征ID的列表
        """
        id_sequence = []
        unknown_features = set()

        # 获取该阶段的特征掩码
        feature_mask = self.stage_feature_masks.get(stage_code, [])

        for sample in sequence:  # sample是包含该阶段所有特征的向量
            sample_ids = []
            # 只处理该阶段的特征（根据特征掩码）
            for idx in feature_mask:
                if idx < len(sample):
                    feature_value = sample[idx]
                    if feature_value in self.attack2id:
                        sample_ids.append(self.attack2id[feature_value])
                    else:
                        # 如果找不到对应的映射，使用-1表示未知
                        unknown_features.add(feature_value)
                        sample_ids.append(-1)
                else:
                    # 特征索引超出范围
                    sample_ids.append(-1)

            id_sequence.append(sample_ids)

        # 如果有未知特征，显示警告
        if unknown_features:
            print(f"警告: {stage_code} 阶段发现 {len(unknown_features)} 个未知特征值")
            if len(unknown_features) <= 5:
                print(f"未知特征值: {list(unknown_features)}")

        return id_sequence
    
    def _build_apt_id_sequences_for_dataset(self, apt_sequences, dataset_name):
        """为指定数据集构建APT攻击标识序列"""
        print(f"--- 构建{dataset_name}APT攻击标识序列 ---")

        apt_id_sequences = []
        apt_labels = []

        # 每种APT类型的序列数量（训练集和测试集可能不同）
        sequences_per_type = len(apt_sequences) // 4

        for i, sequence in enumerate(apt_sequences):
            # 根据序列索引确定APT类型
            apt_type_index = i // sequences_per_type
            if apt_type_index >= 4:  # 防止索引越界
                apt_type_index = 3

            apt_types = ['APT1', 'APT2', 'APT3', 'APT4']
            apt_type = apt_types[apt_type_index]
            label = self.apt_type_to_label[apt_type]

            # 根据APT类型确定包含的攻击阶段
            if apt_type == 'APT1':
                stages = ['S1']
            elif apt_type == 'APT2':
                stages = ['S1', 'S2']
            elif apt_type == 'APT3':
                stages = ['S1', 'S2', 'S3']
            elif apt_type == 'APT4':
                stages = ['S1', 'S2', 'S3', 'S4']

            # 为每个阶段分别转换序列
            combined_id_sequence = []
            samples_per_stage = len(sequence) // len(stages)  # 假设每个阶段的样本数相等

            for stage_idx, stage_code in enumerate(stages):
                # 计算该阶段的样本范围
                start_idx = stage_idx * samples_per_stage
                if stage_idx == len(stages) - 1:  # 最后一个阶段包含剩余所有样本
                    end_idx = len(sequence)
                else:
                    end_idx = (stage_idx + 1) * samples_per_stage

                stage_samples = sequence[start_idx:end_idx]

                # 转换该阶段的样本为ID序列（只使用该阶段的语义特征）
                stage_id_sequence = self.sequence_to_stage_id_sequence(stage_samples, stage_code)
                combined_id_sequence.extend(stage_id_sequence)

            apt_id_sequences.append(combined_id_sequence)
            apt_labels.append(label)

        print(f"{dataset_name}APT攻击标识序列构建完成: {len(apt_id_sequences)} 个")
        return apt_id_sequences, apt_labels

    def build_apt_id_sequences(self):
        """构建训练集和测试集的APT攻击标识序列"""
        print("--- 构建APT攻击标识序列（分别处理训练集和测试集）---")

        # 构建训练集APT标识序列
        self.train_apt_id_sequences, self.train_apt_labels = self._build_apt_id_sequences_for_dataset(
            self.train_apt_sequences, "训练集"
        )

        # 构建测试集APT标识序列
        self.test_apt_id_sequences, self.test_apt_labels = self._build_apt_id_sequences_for_dataset(
            self.test_apt_sequences, "测试集"
        )

        # 合并用于统计（兼容性）
        self.apt_id_sequences = self.train_apt_id_sequences + self.test_apt_id_sequences
        self.apt_labels = self.train_apt_labels + self.test_apt_labels

        # 统计各类型数量
        print("训练集APT标识序列统计:")
        train_label_counts = defaultdict(int)
        for label in self.train_apt_labels:
            train_label_counts[label] += 1
        for apt_type, label in self.apt_type_to_label.items():
            if apt_type != 'Normal':
                print(f"  {apt_type} (标签{label}): {train_label_counts[label]} 个")

        print("测试集APT标识序列统计:")
        test_label_counts = defaultdict(int)
        for label in self.test_apt_labels:
            test_label_counts[label] += 1
        for apt_type, label in self.apt_type_to_label.items():
            if apt_type != 'Normal':
                print(f"  {apt_type} (标签{label}): {test_label_counts[label]} 个")

        print("-" * 50)
    
    def _build_normal_id_sequences_for_dataset(self, normal_sequences, dataset_name):
        """为指定数据集构建正常流量标识序列"""
        print(f"--- 构建{dataset_name}正常流量标识序列 ---")

        normal_id_sequences = []
        normal_labels = []

        for i, sequence in enumerate(normal_sequences):
            # 转换为标识序列 - 正常流量使用所有语义特征
            id_sequence = self.sequence_to_stage_id_sequence(sequence, 'SN')
            normal_id_sequences.append(id_sequence)

            # 正常流量标签为0
            normal_labels.append(0)

        print(f"{dataset_name}正常流量标识序列构建完成: {len(normal_id_sequences)} 个")
        return normal_id_sequences, normal_labels

    def build_normal_id_sequences(self):
        """构建训练集和测试集的正常流量标识序列"""
        print("--- 构建正常流量标识序列（分别处理训练集和测试集）---")

        # 构建训练集正常标识序列
        self.train_normal_id_sequences, self.train_normal_labels = self._build_normal_id_sequences_for_dataset(
            self.train_normal_sequences, "训练集"
        )

        # 构建测试集正常标识序列
        self.test_normal_id_sequences, self.test_normal_labels = self._build_normal_id_sequences_for_dataset(
            self.test_normal_sequences, "测试集"
        )

        # 合并用于统计（兼容性）
        self.normal_id_sequences = self.train_normal_id_sequences + self.test_normal_id_sequences
        self.normal_labels = self.train_normal_labels + self.test_normal_labels

        print(f"训练集正常序列: {len(self.train_normal_id_sequences)} 个")
        print(f"测试集正常序列: {len(self.test_normal_id_sequences)} 个")
        print("-" * 50)
    
    def analyze_id_sequences(self):
        """分析标识序列特征 - 按单个特征值映射后的分析"""
        print("--- 标识序列特征分析（按单个特征值映射）---")

        # APT标识序列分析
        if self.apt_id_sequences:
            apt_lengths = [len(seq) for seq in self.apt_id_sequences]  # 每个序列的样本数
            apt_feature_counts = [len(seq[0]) if seq else 0 for seq in self.apt_id_sequences]  # 每个样本的特征数

            # 收集所有ID值
            apt_ids = []
            for seq in self.apt_id_sequences:
                for sample in seq:
                    for id_val in sample:
                        apt_ids.append(id_val)

            print(f"APT攻击标识序列:")
            print(f"  序列数量: {len(self.apt_id_sequences)}")
            print(f"  序列长度范围: {min(apt_lengths)} - {max(apt_lengths)} 个样本")
            print(f"  平均序列长度: {np.mean(apt_lengths):.2f} 个样本")
            print(f"  每样本特征数: {apt_feature_counts[0] if apt_feature_counts else 0}")
            print(f"  ID值范围: {min(apt_ids)} - {max(apt_ids)}")
            print(f"  唯一ID数量: {len(set(apt_ids))}")
            print(f"  总特征值数量: {len(apt_ids)}")

        # 正常标识序列分析
        if self.normal_id_sequences:
            normal_lengths = [len(seq) for seq in self.normal_id_sequences]
            normal_feature_counts = [len(seq[0]) if seq else 0 for seq in self.normal_id_sequences]

            # 收集所有ID值
            normal_ids = []
            for seq in self.normal_id_sequences:
                for sample in seq:
                    for id_val in sample:
                        normal_ids.append(id_val)

            print(f"正常流量标识序列:")
            print(f"  序列数量: {len(self.normal_id_sequences)}")
            print(f"  序列长度范围: {min(normal_lengths)} - {max(normal_lengths)} 个样本")
            print(f"  平均序列长度: {np.mean(normal_lengths):.2f} 个样本")
            print(f"  每样本特征数: {normal_feature_counts[0] if normal_feature_counts else 0}")
            print(f"  ID值范围: {min(normal_ids)} - {max(normal_ids)}")
            print(f"  唯一ID数量: {len(set(normal_ids))}")
            print(f"  总特征值数量: {len(normal_ids)}")

        # 映射表分析
        print(f"统一attack2id映射表:")
        print(f"  映射条目数: {len(self.attack2id)}")
        print(f"  ID范围: 0 - {len(self.attack2id) - 1}")

        # 各阶段语义特征掩码分析
        print(f"各阶段语义特征掩码:")
        for stage_code, mask in self.stage_feature_masks.items():
            stage_info = self.stage_semantic_features.get(stage_code, {})
            feature_names = stage_info.get('feature_names', [])
            print(f"  {stage_code}: {len(mask)} 个语义特征")
            print(f"    特征: {feature_names}")

        print("-" * 50)
    
    def save_id_sequences(self, output_path):
        """保存标识序列和映射表 - 统一映射表的结果"""
        print(f"--- 保存标识序列到 {output_path} ---")

        os.makedirs(output_path, exist_ok=True)

        # 保存统一的attack2id映射表
        attack2id_path = os.path.join(output_path, 'attack2id_mapping.json')
        attack2id_serializable = {str(k): v for k, v in self.attack2id.items()}

        with open(attack2id_path, 'w', encoding='utf-8') as f:
            json.dump(attack2id_serializable, f, ensure_ascii=False, indent=2)
        print(f"已保存统一attack2id映射表到: {attack2id_path}")

        # 保存阶段语义特征掩码配置
        stage_masks_path = os.path.join(output_path, 'stage_feature_masks.json')
        stage_masks_config = {}
        for stage_code, mask in self.stage_feature_masks.items():
            stage_info = self.stage_semantic_features[stage_code]
            stage_masks_config[stage_code] = {
                'feature_names': stage_info['feature_names'],
                'feature_mask': mask,
                'feature_count': len(mask),
                'feature_type': 'semantic',
                'description': stage_info.get('description', ''),
                'semantic_mappings': stage_info.get('semantic_mappings', {})
            }

        with open(stage_masks_path, 'w', encoding='utf-8') as f:
            json.dump(stage_masks_config, f, ensure_ascii=False, indent=2)
        print(f"已保存阶段语义特征掩码配置到: {stage_masks_path}")

        # 保存训练集APT攻击标识序列
        train_apt_id_path = os.path.join(output_path, 'train_apt_id_sequences.json')
        train_apt_data = {
            'sequences': self.train_apt_id_sequences,
            'labels': self.train_apt_labels,
            'data_format': {
                'description': '训练集APT攻击标识序列，每个序列包含多个样本，每个样本包含该阶段的语义特征ID',
                'structure': '[序列][样本][阶段语义特征ID]',
                'mapping_type': 'unified',
                'stage_features': {stage: info['feature_names'] for stage, info in self.stage_semantic_features.items()},
                'example': 'APT1序列只包含S1阶段语义特征，APT4序列包含S1-S4各阶段语义特征'
            }
        }
        with open(train_apt_id_path, 'w', encoding='utf-8') as f:
            json.dump(train_apt_data, f, ensure_ascii=False, indent=2)
        print(f"已保存训练集APT攻击标识序列到: {train_apt_id_path}")

        # 保存测试集APT攻击标识序列
        test_apt_id_path = os.path.join(output_path, 'test_apt_id_sequences.json')
        test_apt_data = {
            'sequences': self.test_apt_id_sequences,
            'labels': self.test_apt_labels,
            'data_format': {
                'description': '测试集APT攻击标识序列，每个序列包含多个样本，每个样本包含该阶段的语义特征ID',
                'structure': '[序列][样本][阶段语义特征ID]',
                'mapping_type': 'unified',
                'stage_features': {stage: info['feature_names'] for stage, info in self.stage_semantic_features.items()},
                'example': 'APT1序列只包含S1阶段语义特征，APT4序列包含S1-S4各阶段语义特征'
            }
        }
        with open(test_apt_id_path, 'w', encoding='utf-8') as f:
            json.dump(test_apt_data, f, ensure_ascii=False, indent=2)
        print(f"已保存测试集APT攻击标识序列到: {test_apt_id_path}")

        # 保存训练集正常流量标识序列
        train_normal_id_path = os.path.join(output_path, 'train_normal_id_sequences.json')
        train_normal_data = {
            'sequences': self.train_normal_id_sequences,
            'labels': self.train_normal_labels,
            'data_format': {
                'description': '训练集正常流量标识序列，每个序列包含多个样本，每个样本包含语义特征ID',
                'structure': '[序列][样本][语义特征ID]',
                'mapping_type': 'unified',
                'feature_count': len(self.stage_semantic_features['SN']['feature_names']),
                'example': '正常流量使用语义特征，使用统一的ID映射'
            }
        }
        with open(train_normal_id_path, 'w', encoding='utf-8') as f:
            json.dump(train_normal_data, f, ensure_ascii=False, indent=2)
        print(f"已保存训练集正常流量标识序列到: {train_normal_id_path}")

        # 保存测试集正常流量标识序列
        test_normal_id_path = os.path.join(output_path, 'test_normal_id_sequences.json')
        test_normal_data = {
            'sequences': self.test_normal_id_sequences,
            'labels': self.test_normal_labels,
            'data_format': {
                'description': '测试集正常流量标识序列，每个序列包含多个样本，每个样本包含语义特征ID',
                'structure': '[序列][样本][语义特征ID]',
                'mapping_type': 'unified',
                'feature_count': len(self.stage_semantic_features['SN']['feature_names']),
                'example': '正常流量使用语义特征，使用统一的ID映射'
            }
        }
        with open(test_normal_id_path, 'w', encoding='utf-8') as f:
            json.dump(test_normal_data, f, ensure_ascii=False, indent=2)
        print(f"已保存测试集正常流量标识序列到: {test_normal_id_path}")

        # 保存标签映射说明
        label_mapping_path = os.path.join(output_path, 'label_mapping.json')
        with open(label_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.apt_type_to_label, f, ensure_ascii=False, indent=2)
        print(f"已保存标签映射说明到: {label_mapping_path}")

        print("标识序列保存完成")
        print("-" * 50)


if __name__ == '__main__':
    # 使用示例
    SEQUENCES_PATH = './sequences_output'  # 序列数据路径
    OUTPUT_PATH = './id_sequences_output'  # 标识序列输出路径
    
    try:
        # 创建标识序列构建器
        builder = APTIDSequenceBuilder(SEQUENCES_PATH)
        
        # 加载序列数据
        builder.load_sequences()
        
        # 构建统一的attack2id映射表
        builder.build_unified_attack2id_mapping()
        
        # 构建APT攻击标识序列
        builder.build_apt_id_sequences()
        
        # 构建正常流量标识序列
        builder.build_normal_id_sequences()
        
        # 分析标识序列
        builder.analyze_id_sequences()
        
        # 保存标识序列
        builder.save_id_sequences(OUTPUT_PATH)
        
        print("\n--- 标识序列构建完成 ---")
        print(f"APT攻击标识序列: {len(builder.apt_id_sequences)} 个")
        print(f"正常流量标识序列: {len(builder.normal_id_sequences)} 个")
        print(f"统一attack2id映射表大小: {len(builder.attack2id)} 个token")
        print("各阶段语义特征掩码:")
        for stage_code, mask in builder.stage_feature_masks.items():
            stage_info = builder.stage_semantic_features.get(stage_code, {})
            feature_names = stage_info.get('feature_names', [])
            print(f"  {stage_code}: {len(mask)} 个语义特征")
            print(f"    特征: {feature_names}")
        
        # 显示各类型APT标识序列示例
        print("\n--- APT标识序列示例展示 ---")

        # 显示每种APT类型的第一个标识序列
        apt_type_examples = {
            'APT1': (0, 1),      # 第1个序列，标签1
            'APT2': (1000, 2),   # 第1001个序列，标签2
            'APT3': (2000, 3),   # 第2001个序列，标签3
            'APT4': (3000, 4)    # 第3001个序列，标签4
        }

        for apt_type, (index, expected_label) in apt_type_examples.items():
            if index < len(builder.apt_id_sequences):
                id_sequence = builder.apt_id_sequences[index]
                actual_label = builder.apt_labels[index]
                print(f"\n{apt_type} 标识序列示例:")
                print(f"  序列长度: {len(id_sequence)} 个样本")
                print(f"  每个样本特征数: {len(id_sequence[0]) if id_sequence and id_sequence[0] else 0}")
                print(f"  第一个样本的前10个特征ID: {id_sequence[0][:10] if id_sequence and id_sequence[0] else []}")
                print(f"  标签: {actual_label} (预期: {expected_label})")

                # 确定该APT类型包含的阶段
                if apt_type == 'APT1':
                    stages = ['S1']
                elif apt_type == 'APT2':
                    stages = ['S1', 'S2']
                elif apt_type == 'APT3':
                    stages = ['S1', 'S2', 'S3']
                elif apt_type == 'APT4':
                    stages = ['S1', 'S2', 'S3', 'S4']

                print(f"  包含阶段: {stages}")

                # 收集所有ID值
                all_ids = []
                for sample in id_sequence:
                    all_ids.extend(sample)
                if all_ids:
                    print(f"  ID值范围: {min(all_ids)} - {max(all_ids)}")
                    print(f"  唯一ID数量: {len(set(all_ids))}")

        # 显示正常流量标识序列示例
        if builder.normal_id_sequences:
            normal_seq = builder.normal_id_sequences[0]
            normal_label = builder.normal_labels[0]
            print(f"\n正常流量标识序列示例:")
            print(f"  序列长度: {len(normal_seq)} 个样本")
            print(f"  每个样本特征数: {len(normal_seq[0]) if normal_seq and normal_seq[0] else 0}")
            print(f"  第一个样本的前10个特征ID: {normal_seq[0][:10] if normal_seq and normal_seq[0] else []}")
            print(f"  标签: {normal_label}")

            # 收集所有ID值
            all_ids = []
            for sample in normal_seq:
                all_ids.extend(sample)
            if all_ids:
                print(f"  ID值范围: {min(all_ids)} - {max(all_ids)}")
                print(f"  唯一ID数量: {len(set(all_ids))}")

        print("\n--- 数据文件说明 ---")
        print("生成的文件:")
        print("  - attack2id_mapping.json: 特征样本到ID的映射表")
        print("  - apt_id_sequences.json: APT攻击标识序列和标签")
        print("  - normal_id_sequences.json: 正常流量标识序列和标签")
        print("  - label_mapping.json: APT类型到标签的映射说明")

        print("\n标识序列格式说明:")
        print("  - 每个序列包含多个样本，每个样本包含该阶段的语义特征ID")
        print("  - 数据结构: [序列][样本][阶段语义特征ID]")
        print(f"  - 统一ID值范围: 0 - {len(builder.attack2id) - 1}")
        print("  - 各阶段语义特征数量:")
        for stage_code, mask in builder.stage_feature_masks.items():
            stage_info = builder.stage_semantic_features.get(stage_code, {})
            feature_names = stage_info.get('feature_names', [])
            print(f"    - {stage_code}: {len(mask)} 个语义特征 {feature_names}")
        print("  - 标签: APT1=1, APT2=2, APT3=3, APT4=4, Normal=0")
        print("  - 每个阶段使用该阶段的语义特征，统一ID映射表保持语义一致性")
        print("  - 语义特征编码大幅降低了词汇表大小，适合SeqGAN训练")
            
    except Exception as e:
        print(f"标识序列构建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
