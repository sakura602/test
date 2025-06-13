import pandas as pd
import numpy as np
import os
import json
import joblib
from collections import defaultdict, Counter
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import hashlib
import logging
from datetime import datetime, timedelta
import seaborn as sns
import math

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("APTSequenceBuilder")

# 添加NumPy数据类型的JSON序列化支持
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class APTSequenceBuilder:
    """
    APT攻击序列构建器，用于将预处理数据转换为序列表示
    """
    
    # 1. 统一字段映射，解决 key 不一致的问题
    FIELD_MAP = {
        "session_col":         ("session_column",    "Session_ID"),
        "label_col":           ("apt_column",        "Label"),
        "timestamp_col":       ("timestamp_column",  "Timestamp"),
        "max_sequence_length": ("max_sequence_length",  100),
        "window_size":         ("window_size", 3),  # 调小默认window_size为3
        "overlap_rate":        ("overlap_rate",      0.3),  # 调整overlap_rate为0.3
        "padding_value":       ("padding_value",     0),
        "extract_mixed":       ("extract_mixed", True),
        "feature_columns":     ("feature_columns", []),
        "stage_mapping":       ("stage_mapping", {}),
        "entity_key":          ("entity_key", ["Src IP", "Dst IP"]),  # 新增实体键配置
        "time_gap_threshold":  ("time_aware.time_gap_threshold", 300),  # 新增时间阈值配置（5分钟）
        "min_len":             ("min_len", 2),  # 新增最小会话长度配置
        "min_benign_sessions": ("min_benign_sessions", 0),
        "benign_sequence":     ("benign_sequence", {}),
        "balance_classes":     ("balance_classes", False),
        "balance_strategy":    ("balance_strategy", ""),
        "target_ratio":        ("target_ratio", 0),
        # 可扩展更多参数
    }

    def __init__(self, cfg: dict):
        # 2. 统一读取所有配置
        for attr, (path, default) in self.FIELD_MAP.items():
            setattr(self, attr, self._get_cfg(cfg, path, default))
        # 3. 立刻打印，确保生效
        logger.info(
            f"CONFIG ▶ session_col={self.session_col}, "
            f"label_col={self.label_col}, "
            f"max_seq_len={self.max_sequence_length}, "
            f"window_size={self.window_size}, "
            f"overlap_rate={self.overlap_rate}, "
            f"pad={self.padding_value}, "
            f"extract_mixed={self.extract_mixed}, "
            f"entity_key={self.entity_key}, "
            f"time_gap_threshold={self.time_gap_threshold}s"
        )
        
        # 定义阶段顺序映射，用于排序
        self.stage_order = {1: 1, 2: 2, 3: 3, 4: 4}

    def _get_cfg(self, cfg: dict, path: str, default):
        # 支持 "a.b.c" 路径
        node = cfg
        for key in path.split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def get_available_features(self, df):
        """
        获取可用的特征列
        
        参数:
        df: 数据框
        
        返回:
        可用的特征列列表
        """
        # 获取配置的特征列
        feature_columns = self.feature_columns
        backup_columns = getattr(self, 'backup_feature_columns', [])
        
        # 检查是否启用了特征合成
        feature_synthesis_enabled = self.config.get("feature_synthesis", {}).get("enabled", False)
        
        # 检查数据框中是否有主要特征列
        available_primary = [col for col in feature_columns if col in df.columns]
        
        # 如果所有主要特征列都存在，直接返回这些列
        if len(available_primary) == len(feature_columns) and feature_columns:
            logger.info(f"所有主要特征列都存在于原始数据中，使用这些列: {available_primary}")
            # 分析特征列的质量
            self._analyze_feature_quality(df)
            return available_primary
        
        # 如果主要特征列缺失，并且启用了特征合成，尝试合成特征
        if len(available_primary) < len(feature_columns) and feature_synthesis_enabled:
            logger.info(f"缺少一些主要特征列，尝试通过特征合成获取: {[col for col in feature_columns if col not in available_primary]}")
            # 创建数据框的副本进行特征合成
            df_with_features = self.synthesize_features(df)
            
            # 重新检查可用的主要特征列
            available_primary = [col for col in feature_columns if col in df_with_features.columns]
            
            # 如果合成后有更多可用的主要特征列，使用增强后的数据框
            if len(available_primary) > 0:
                logger.info(f"通过特征合成得到 {len(available_primary)} 个主要特征列: {available_primary}")
                
                # 使用全局变量保存增强后的数据框，以便其他方法使用
                self.enhanced_df = df_with_features
                
                # 分析特征列的质量（使用增强后的数据框）
                self._analyze_feature_quality(df_with_features)
                
                return available_primary
        
        # 检查数据框中是否有备用特征列
        available_backup = [col for col in backup_columns if col in df.columns]
        
        # 分析特征列的质量
        self._analyze_feature_quality(df)
        
        # 如果主要特征列可用，使用它们
        if available_primary:
            logger.info(f"使用 {len(available_primary)} 个主要特征列: {available_primary}")
            return available_primary
        
        # 如果主要特征列不可用，但备用特征列可用，使用备用特征列
        if available_backup:
            logger.info(f"主要特征列不可用，使用 {len(available_backup)} 个备用特征列: {available_backup}")
            return available_backup
        
        # 如果主要和备用特征列都不可用，尝试增强特征提取
        if not available_primary and not available_backup:
            logger.info("\n主要和备用特征列都不可用，尝试增强特征提取...")
            
            # 考虑使用所有可能的特征列
            enhanced_columns = []
            
            # 考虑使用Is_Ordered_Sequence
            if 'Is_Ordered_Sequence' in df.columns:
                enhanced_columns.append('Is_Ordered_Sequence')
                logger.info("  添加Is_Ordered_Sequence作为特征")
            
            # 考虑使用Sequence_Stages
            if 'Sequence_Stages' in df.columns:
                enhanced_columns.append('Sequence_Stages')
                logger.info("  添加Sequence_Stages作为特征")
            
            # 从Timestamp提取时间特征（如果存在）
            if 'Timestamp' in df.columns:
                # 保留原始Timestamp
                enhanced_columns.append('Timestamp')
                logger.info("  添加Timestamp作为特征")
                
                # 尝试提取更多时间特征
                try:
                    # 暂时不添加衍生时间特征，避免复杂化
                    pass
                except Exception as e:
                    logger.warning(f"  警告: 提取时间特征时出错: {e}")
            
            # 从Session_ID提取特征（如果可能）
            session_column = self.config.get("session_column", "Session_ID")
            if session_column in df.columns and session_column not in enhanced_columns:
                enhanced_columns.append(session_column)
                logger.info(f"  添加{session_column}作为特征")
            
            if enhanced_columns:
                logger.info(f"\n使用 {len(enhanced_columns)} 个增强特征列: {enhanced_columns}")
                return enhanced_columns
        
        # 如果主要和备用特征列都不可用，尝试自动选择特征列
        logger.info("\n主要和备用特征列都不可用，尝试自动选择特征列...")
        
        # 根据非空率和唯一值数量选择特征列
        min_non_null = self.config.get("feature_extraction", {}).get("min_non_null_ratio", 0.5)
        max_unique = self.config.get("feature_extraction", {}).get("max_unique_values", 10000)
        
        auto_columns = []
        for col, stats in self.feature_stats.items():
            if stats['non_null_ratio'] >= min_non_null and (stats['unique_count'] <= max_unique or stats['unique_count'] == -1):
                auto_columns.append(col)
        
        if auto_columns:
            logger.info(f"\n自动选择了 {len(auto_columns)} 个特征列: {auto_columns}")
            return auto_columns
        
        # 如果无法自动选择特征列，使用所有非标签和非会话ID列
        logger.info("\n无法自动选择特征列，使用所有可用列...")
        
        session_column = self.config.get("session_column", "Session_ID")
        apt_column = self.config.get("apt_column", "Label")
        all_usable_columns = [col for col in df.columns if col not in [apt_column, session_column, 'Sequence_Stages']]
        
        if all_usable_columns:
            logger.info(f"\n使用 {len(all_usable_columns)} 个可用列: {all_usable_columns}")
            return all_usable_columns
        
        # 如果没有可用的特征列，至少使用会话ID和时间戳（如果存在）
        logger.info("\n警告: 没有找到常规可用特征列，尝试使用会话ID或时间戳")
        minimum_columns = []
        if session_column in df.columns:
            minimum_columns.append(session_column)
        if 'Timestamp' in df.columns:
            minimum_columns.append('Timestamp')
        
        if minimum_columns:
            logger.info(f"\n使用 {len(minimum_columns)} 个最小特征列: {minimum_columns}")
            return minimum_columns
            
        # 如果仍然没有列可用，返回空列表，后续处理将会失败
        logger.error("\n错误: 没有可用的特征列")
        return []
    
    def _analyze_feature_quality(self, df):
        """
        分析特征列的质量
        
        参数:
        df: 数据框
        """
        # 获取配置的特征列
        feature_columns = self.feature_columns
        backup_columns = getattr(self, 'backup_feature_columns', [])
        
        # 初始化特征统计信息
        self.feature_stats = {}
        
        # 检查所有列
        all_columns = list(df.columns)
        for col in all_columns:
            # 跳过标签和会话ID列
            session_column = self.config.get("session_column", "Session_ID")
            apt_column = self.config.get("apt_column", "Label")
            
            if col in [apt_column, session_column, 'Sequence_Stages']:
                continue
            
            # 计算非空值比例
            non_null_ratio = 1.0 - df[col].isna().mean()
            
            # 计算唯一值数量
            try:
                unique_count = df[col].nunique()
            except:
                unique_count = -1  # 无法计算唯一值
            
            # 存储特征统计信息
            self.feature_stats[col] = {
                'non_null_ratio': non_null_ratio,
                'unique_count': unique_count,
                'is_primary': col in feature_columns,
                'is_backup': col in backup_columns
            }
        
        # 打印特征统计信息
        logger.info("\n特征列质量分析:")
        for col, stats in self.feature_stats.items():
            status = "主要" if stats['is_primary'] else ("备用" if stats['is_backup'] else "其他")
            logger.info(f"  {col}: 非空率={stats['non_null_ratio']:.2f}, 唯一值={stats['unique_count']}, 类型={status}")
    
    def categorize_apt_sequences(self, df):
        """
        对APT攻击序列进行分类（基于整数标签）
        
        APT类别定义（基于test_nll.py的标签映射）：
        - APT1: 只包含侦察阶段(1)
        - APT2: 包含侦察(1)和立足(2)两个阶段
        - APT3: 包含侦察(1)、立足(2)和横向移动(3)三个阶段
        - APT4: 包含完整攻击链：侦察(1)、立足(2)、横向移动(3)和数据渗出(4)
        - APT5: 不符合上述任何一种情况的攻击序列（如缺少某些阶段或顺序不正确）
        
        参数:
        df: 数据框
        
        返回:
        分类后的APT序列字典
        """
        logger.info("对APT攻击序列进行分类...")
        
        # 获取配置中的列名
        session_column = self.config.get("session_column", "Session_ID")
        apt_column = self.config.get("apt_column", "Label")
        
        # 检查必要的列是否存在
        if session_column not in df.columns:
            logger.warning(f"警告: 数据中缺少必要的列: {session_column}")
            logger.info("尝试使用默认会话列名: 'Session_ID'")
            session_column = "Session_ID"
            if session_column not in df.columns:
                raise KeyError(f"数据中缺少必要的列: Session_ID")
        
        # 获取阶段映射和攻击阶段顺序
        stage_mapping = self.config.get("stage_mapping", {})
        attack_stage_order = self.config.get("attack_stage_order", {})
        
        # 确保stage_mapping的键是整数类型
        int_stage_mapping = {}
        for k, v in stage_mapping.items():
            try:
                if isinstance(k, str):
                    int_stage_mapping[int(k)] = v
                else:
                    int_stage_mapping[k] = v
            except (ValueError, TypeError):
                logger.warning(f"警告: 无法将标签 {k} 转换为整数，跳过")
        
        # 使用整数键的映射
        stage_mapping = int_stage_mapping
        
        # 尝试将标签列转换为整数类型
        try:
            df[apt_column] = df[apt_column].astype(int)
        except Exception as e:
            logger.warning(f"警告: 无法将标签列转换为整数: {e}")
            logger.info("将尝试在处理过程中逐个转换标签")
        
        # 统计每个会话包含的攻击阶段
        session_attack_stages = {}
        session_stage_sequences = {}  # 记录每个会话的攻击阶段序列
        
        # 检查是否有Sequence_Stages列
        if 'Sequence_Stages' in df.columns:
            logger.info("使用Sequence_Stages列进行分类")
            
            for session_id in df[session_column].unique():
                session_data = df[df[session_column] == session_id]
                
                # 获取第一行的Sequence_Stages（假设同一会话的所有行都有相同的Sequence_Stages）
                stages_str = session_data['Sequence_Stages'].iloc[0]
                
                # 如果是空字符串，跳过
                if not stages_str or pd.isna(stages_str):
                    continue
                    
                # 将字符串分割为阶段列表
                stages = set(stages_str.split(','))
                
                # 保存会话的攻击阶段
                session_attack_stages[session_id] = stages
        else:
            logger.info("使用Label字段进行分类")
            
            for session_id in df[session_column].unique():
                session_data = df[df[session_column] == session_id]
                
                # 获取攻击阶段标签（排除良性标签0）
                try:
                    # 尝试将标签转换为整数，并应用标签转换
                    attack_labels = []
                    for label in session_data[apt_column].unique():
                        try:
                            old_label_int = int(label)
                            if old_label_int != 0:  # 排除良性标签
                                new_label_int = self._convert_label(old_label_int)
                                if new_label_int != 0:  # 确保转换后也不是良性标签
                                    attack_labels.append(new_label_int)
                        except (ValueError, TypeError):
                            logger.warning(f"警告: 无法将标签 {label} 转换为整数，跳过")
                except Exception as e:
                    logger.warning(f"警告: 处理标签时出错: {e}")
                    continue
                
                # 如果没有攻击标签，跳过
                if len(attack_labels) == 0:
                    continue
                
                # 保存会话的攻击标签
                session_attack_stages[session_id] = sorted(attack_labels)
                
                # 如果启用了时间感知分析，提取时间序列中的攻击阶段序列
                if self.time_aware_enabled and self.timestamp_column in session_data.columns:
                    try:
                        # 按时间戳排序
                        sorted_data = session_data.sort_values(by=self.timestamp_column)
                        
                        # 提取攻击阶段序列
                        stage_sequence = []
                        for _, row in sorted_data.iterrows():
                            try:
                                label_int = int(row[apt_column])
                                if label_int != 0:  # 排除良性标签
                                    stage_sequence.append(label_int)
                            except (ValueError, TypeError):
                                continue
                        
                        # 保存会话的攻击阶段序列
                        if stage_sequence:
                            session_stage_sequences[session_id] = stage_sequence
                    except Exception as e:
                        logger.warning(f"警告: 提取时间序列中的攻击阶段失败: {e}")
        
        # 计算攻击阶段转移矩阵（如果启用了灵活的攻击阶段识别）
        flexible_recognition_enabled = self.config.get("flexible_stage_recognition", {}).get("enabled", False)
        use_transition_matrix = self.config.get("flexible_stage_recognition", {}).get("use_transition_matrix", False)
        
        if flexible_recognition_enabled and use_transition_matrix:
            self.stage_transition_matrix = self._calculate_stage_transition_matrix(session_stage_sequences)
            logger.info(f"计算了攻击阶段转移矩阵，包含 {len(self.stage_transition_matrix)} 个起始阶段")
        
        # 基于攻击阶段和灵活识别进行分类
        apt_categories = defaultdict(list)
        
        # 获取灵活识别配置
        allow_stage_repetition = self.config.get("flexible_stage_recognition", {}).get("allow_stage_repetition", True)
        allow_stage_skipping = self.config.get("flexible_stage_recognition", {}).get("allow_stage_skipping", True)
        
        # 打印会话阶段分布情况
        stage_counts = {}
        for session_id, attack_labels in session_attack_stages.items():
            for label in attack_labels:
                if label not in stage_counts:
                    stage_counts[label] = 0
                stage_counts[label] += 1
        
        logger.info("\n会话阶段分布情况:")
        for label, count in sorted(stage_counts.items()):
            # 获取阶段名称，避免显示"未知阶段"前缀
            stage_name = stage_mapping.get(label)
            if stage_name is None:
                # 尝试直接使用标签作为字符串键
                stage_name = stage_mapping.get(str(label))
            
            # 如果仍然找不到对应的阶段名称，显示为"未知标签"
            if stage_name is None:
                stage_name = f"未知标签 {label}"
                
            logger.info(f"  阶段 {label} ({stage_name}): {count} 个会话")
        
        # 根据新的分类逻辑对会话进行分类
        for session_id, attack_labels in session_attack_stages.items():
            # 如果是集合类型，转换为列表
            if isinstance(attack_labels, set):
                attack_labels = sorted(list(attack_labels))
            
            # 检查是否包含各个关键阶段
            has_reconnaissance = 1 in attack_labels  # 侦察阶段
            has_establish_foothold = 2 in attack_labels  # 立足阶段
            has_lateral_movement = 3 in attack_labels  # 横向移动
            has_data_exfiltration = 4 in attack_labels  # 数据渗出
            
            # 根据包含的阶段进行分类
            if has_data_exfiltration and has_lateral_movement and has_establish_foothold and has_reconnaissance:
                # 完整攻击链
                apt_categories["APT4"].append(session_id)
            elif has_lateral_movement and has_establish_foothold and has_reconnaissance:
                # 侦察+立足+横向移动
                apt_categories["APT3"].append(session_id)
            elif has_establish_foothold and has_reconnaissance:
                # 侦察+立足
                apt_categories["APT2"].append(session_id)
            elif has_reconnaissance:
                # 仅侦察
                apt_categories["APT1"].append(session_id)
            else:
                # 不规则攻击链或其他情况
                apt_categories["APT5"].append(session_id)
        
        # 打印分类结果
        logger.info("\nAPT攻击序列分类结果:")
        for category, sessions in apt_categories.items():
            logger.info(f"{category}: {len(sessions)} 个会话")
        
        return apt_categories
    
    def _select_target_label(self, non_zero_labels, session_id, stage_mapping):
        """
        根据配置的策略选择目标标签
        
        参数:
        non_zero_labels: 非零标签列表
        session_id: 会话ID
        stage_mapping: 阶段映射
        
        返回:
        选择的目标标签
        """
        if len(non_zero_labels) == 0:
            # 如果没有非零标签，使用默认标签1（Reconnaissance）
            if self.verbose:
                logger.warning(f"会话 {session_id} 没有非零标签，使用默认标签 1 (Reconnaissance)")
            return 1
        
        # 获取标签选择策略
        strategy = self.config.get("label_selection", {}).get("strategy", "max")
        
        if strategy == "max":
            # 使用最高的攻击阶段标签
            target_label = max(non_zero_labels)
            if self.verbose:
                logger.info(f"会话 {session_id} 使用最高攻击阶段标签: {target_label} ({stage_mapping.get(target_label, 'Unknown')})")
            return target_label
            
        elif strategy == "majority":
            # 使用出现次数最多的标签
            label_counts = Counter(non_zero_labels)
            target_label = label_counts.most_common(1)[0][0]
            if self.verbose:
                logger.info(f"会话 {session_id} 使用多数攻击阶段标签: {target_label} ({stage_mapping.get(target_label, 'Unknown')})")
            return target_label
            
        elif strategy == "weighted":
            # 使用加权策略，后期阶段权重更高
            weights = {1: 1, 2: 2, 3: 3, 4: 4}  # 权重与攻击阶段对应
            label_weights = {label: weights.get(label, 1) * count 
                            for label, count in Counter(non_zero_labels).items()}
            target_label = max(label_weights.items(), key=lambda x: x[1])[0]
            if self.verbose:
                logger.info(f"会话 {session_id} 使用加权攻击阶段标签: {target_label} ({stage_mapping.get(target_label, 'Unknown')})")
            return target_label
            
        else:
            # 默认使用最高阶段标签
            target_label = max(non_zero_labels)
            if self.verbose:
                logger.info(f"会话 {session_id} 使用默认最高攻击阶段标签: {target_label} ({stage_mapping.get(target_label, 'Unknown')})")
            return target_label

    def build_attack_sequences(self, df, apt_categories):
        """
        构建APT攻击标识序列
        
        参数:
        df: 数据框
        apt_categories: 分类后的APT序列字典
        
        返回:
        构建的序列列表
        """
        logger.info("\n构建APT攻击标识序列...")
        
        # 初始化序列列表
        sequences = []
        
        try:
            # 获取配置中的列名
            session_column = self.config.get("session_column", "Session_ID")
            apt_column = self.config.get("apt_column", "Label")
            
            # 检查必要的列是否存在
            if session_column not in df.columns:
                logger.error(f"错误: 会话列 '{session_column}' 不存在于数据集中")
                return []
                
            if apt_column not in df.columns:
                logger.error(f"错误: APT标签列 '{apt_column}' 不存在于数据集中")
                return []
            
            # 获取可用的特征列
            available_columns = self.get_available_features(df)
            
            if not available_columns:
                logger.error("错误: 没有可用的特征列，无法构建序列")
                return []
                
            # 检查APT类别是否为空
            if not apt_categories:
                logger.warning("警告: 没有APT类别，无法构建攻击序列")
                return []
            
            # 统计特征出现频率
            feature_frequency = {}
            
            # 初始化标签统计
            label_counts = Counter()
            
            # 初始化统计信息字典
            stats = {}
            
            # 预处理数据，减少重复计算
            logger.info("预处理数据...")
            # 创建一个字典，将会话ID映射到其数据
            session_data_dict = {}
            
            # 获取所有需要处理的会话ID
            all_apt_sessions = set()
            for sessions in apt_categories.values():
                all_apt_sessions.update(sessions)
                
            # 如果没有APT会话，返回空列表
            if not all_apt_sessions:
                logger.warning("警告: 没有APT会话，无法构建攻击序列")
                return []
            
            # 预处理每个会话的数据
            for session_id in all_apt_sessions:
                try:
                    # 获取会话数据
                    session_data = df[df[session_column] == session_id].copy()
                    
                    # 检查会话数据是否为空
                    if len(session_data) == 0:
                        logger.warning(f"警告: 会话 {session_id} 没有数据")
                        continue
                    
                    # 填充缺失值
                    for col in available_columns:
                        if col in session_data.columns and session_data[col].isna().any():
                            if col in ['Src IP', 'Dst IP']:
                                session_data[col] = session_data[col].fillna('0.0.0.0')
                            elif col in ['Src Port', 'Dst Port']:
                                session_data[col] = session_data[col].fillna(0)
                            elif col == 'Protocol':
                                session_data[col] = session_data[col].fillna('Unknown')
                            elif col == session_column:
                                session_data[col] = session_data[col].fillna('unknown_session')
                            elif col == 'Timestamp':
                                session_data[col] = session_data[col].fillna(pd.Timestamp.now())
                    
                    # 如果启用了时间感知，提取时间特征
                    if self.time_aware_enabled:
                        if self.extract_time_features_enabled:
                            try:
                                time_features = self.extract_time_features(session_data)
                                if time_features:
                                    # 检测时间间隙
                                    time_gaps = self.detect_time_gaps(session_data)
                                    if time_gaps and self.verbose:
                                        logger.info(f"会话 {session_id} 检测到 {len(time_gaps)} 个时间间隙")
                                    
                                    # 将时间特征添加到会话数据中
                                    session_data.loc[:, 'time_features'] = str(time_features)
                                    session_data.loc[:, 'time_gaps'] = str(time_gaps)
                            except Exception as e:
                                logger.warning(f"警告: 提取时间特征时出错: {e}")
                    
                    # 保存预处理后的会话数据
                    session_data_dict[session_id] = session_data
                except Exception as e:
                    logger.warning(f"警告: 预处理会话 {session_id} 时出错: {e}")
                    continue
            
            logger.info(f"预处理了 {len(session_data_dict)} 个会话")
            
            # 确保stage_mapping的键是整数类型
            stage_mapping = {}
            for k, v in self.config.get("stage_mapping", {}).items():
                try:
                    if isinstance(k, str):
                        stage_mapping[int(k)] = v
                    else:
                        stage_mapping[k] = v
                except (ValueError, TypeError):
                    if self.verbose:
                        logger.warning(f"警告: 无法将标签 {k} 转换为整数，跳过")
            
            # 初始化统计信息字典和标签计数器
            stats = {}
            label_counts = Counter()
            
            # 为每个APT类别构建序列
            for apt_category, session_ids in apt_categories.items():
                if self.verbose:
                    logger.info(f"\n处理 {apt_category} 类别...")
                
                # 处理每个会话
                for session_id in session_ids:
                    try:
                        # 获取预处理后的会话数据
                        if session_id not in session_data_dict:
                            continue
                        
                        session_data = session_data_dict[session_id]
                        
                        # 获取会话中的所有非零标签
                        try:
                            # 获取原始非零标签，并应用转换
                            non_zero_labels = []
                            for label in session_data[apt_column][session_data[apt_column] != 0]:
                                try:
                                    old_label = label
                                    new_label = self._convert_label(old_label)
                                    if new_label != 0:  # 确保转换后也不是良性标签
                                        non_zero_labels.append(new_label)
                                except Exception as e:
                                    logger.debug(f"转换标签 {label} 时出错: {e}")
                            
                            non_zero_labels = np.array(non_zero_labels)
                            
                            # 统计标签分布
                            for label in non_zero_labels:
                                label_counts[label] += 1
                            
                            if len(non_zero_labels) > 0:
                                # 获取会话中出现的所有非零标签
                                unique_labels = np.unique(non_zero_labels)
                                
                                # 打印会话中的标签分布
                                if self.verbose:
                                    label_str = ", ".join([f"{label}({stage_mapping.get(label, 'Unknown')})" for label in sorted(unique_labels)])
                                    logger.info(f"会话 {session_id} 包含的攻击阶段标签: {label_str}")
                                
                                # 使用配置的标签选择策略选择目标标签
                                target_label = self._select_target_label(non_zero_labels, session_id, stage_mapping)
                                logger.info(f"会话 {session_id} 使用最高攻击阶段标签: {target_label} ({stage_mapping.get(target_label, 'Unknown')})")
                            else:
                                # 如果没有非零标签，使用默认标签1（Reconnaissance）
                                target_label = 1
                                if self.verbose:
                                    logger.warning(f"警告: 会话 {session_id} 没有非零标签，使用默认标签 {target_label} (Reconnaissance)")
                        except Exception as e:
                            # 如果出错，使用默认标签
                            target_label = 1
                            logger.warning(f"警告: 获取会话 {session_id} 的攻击阶段标签时出错: {e}，使用默认标签 {target_label} (Reconnaissance)")
                        
                        # 检查是否有时间间隙，如果有，按时间间隙分割序列
                        time_gaps = []
                        if self.time_aware_enabled and 'time_gaps' in session_data.columns:
                            try:
                                time_gaps_str = session_data['time_gaps'].iloc[0]
                                if time_gaps_str.startswith('[') and time_gaps_str.endswith(']'):
                                    time_gaps = eval(time_gaps_str)
                            except Exception as e:
                                logger.warning(f"警告: 解析时间间隙失败: {e}")
                        
                        # 如果有时间间隙，按间隙分割序列
                        if time_gaps and self.time_aware_enabled:
                            try:
                                # 按时间戳排序
                                if self.timestamp_column in session_data.columns:
                                    session_data = session_data.sort_values(by=self.timestamp_column)
                                
                                # 分割点
                                split_points = [0] + time_gaps + [len(session_data)]
                                
                                # 为每个分段构建序列
                                for i in range(len(split_points) - 1):
                                    start_idx = split_points[i]
                                    end_idx = split_points[i + 1]
                                    
                                    # 获取分段数据
                                    segment_data = session_data.iloc[start_idx:end_idx]
                                    
                                    # 如果分段太小，跳过
                                    if len(segment_data) < 2:
                                        continue
                                    
                                    # 构建特征序列
                                    seq_features = self._build_sequence_features(segment_data, available_columns, apt_column, stage_mapping, feature_frequency)
                                    
                                    # 添加到序列列表，使用修改后的会话ID表示分段
                                    segment_id = f"{session_id}_segment_{i+1}"
                                    sequences.append((segment_id, target_label, seq_features))
                            except Exception as e:
                                logger.warning(f"警告: 处理会话 {session_id} 的时间间隙时出错: {e}")
                                # 如果处理时间间隙出错，尝试构建完整序列
                                seq_features = self._build_sequence_features(session_data, available_columns, apt_column, stage_mapping, feature_frequency)
                                sequences.append((session_id, target_label, seq_features))
                        else:
                            # 构建特征序列
                            seq_features = self._build_sequence_features(session_data, available_columns, apt_column, stage_mapping, feature_frequency)
                            
                            # 添加到序列列表
                            sequences.append((session_id, target_label, seq_features))
                    except Exception as e:
                        logger.warning(f"警告: 处理会话 {session_id} 时出错: {e}")
                        continue
            
            # 打印标签分布统计
            logger.info("\n标签分布统计:")
            for label, count in sorted(label_counts.items()):
                logger.info(f"  标签 {label} ({stage_mapping.get(label, 'Unknown')}): {count} 个事件")
            
            # 优化特征映射表
            if self.config.get("feature_extraction", {}).get("enable_feature_filtering", False) and feature_frequency:
                # 获取攻击特征的最小频率阈值
                attack_min_frequency = self.config.get("feature_extraction", {}).get("attack_min_frequency", 2)
                
                # 获取最少保留的特征数量
                min_features_to_keep = self.config.get("feature_extraction", {}).get("min_features_to_keep", 10)
                
                # 过滤低频特征
                filtered_features = {feature: freq for feature, freq in feature_frequency.items() if freq >= attack_min_frequency}
                
                # 确保至少保留一定数量的特征
                if len(filtered_features) < min_features_to_keep and feature_frequency:
                    # 如果过滤后的特征数量不足，保留出现频率最高的前N个特征
                    sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
                    top_features = dict(sorted_features[:min(min_features_to_keep, len(sorted_features))])
                    
                    # 将这些高频特征添加到过滤后的特征中
                    filtered_features.update(top_features)
                    logger.warning(f"警告: 过滤后的特征数量不足，保留出现频率最高的 {len(top_features)} 个特征")
                
                # 如果所有特征都被过滤掉，禁用过滤
                if not filtered_features and feature_frequency:
                    filtered_features = feature_frequency
                    logger.warning("警告: 所有特征都被过滤掉了，禁用过滤")
                
                # 重建映射表
                if len(filtered_features) < len(feature_frequency):
                    logger.info(f"\n过滤低频特征: 从 {len(feature_frequency)} 减少到 {len(filtered_features)}")
                    
                    # 创建新的映射表
                    new_attack2id = {}
                    new_id2attack = {}
                    new_next_id = 1
                    
                    # 更新序列中的特征ID
                    for i, (session_id, label, features) in enumerate(sequences):
                        new_features = []
                        for feature_id, event_label in features:
                            feature_str = self.id2attack[feature_id]
                            if feature_str in filtered_features:
                                if feature_str not in new_attack2id:
                                    new_attack2id[feature_str] = new_next_id
                                    new_id2attack[new_next_id] = feature_str
                                    new_next_id += 1
                                new_features.append((new_attack2id[feature_str], event_label))
                            else:
                                # 对于被过滤的特征，使用padding值
                                new_features.append((self.config.get('padding_value', 0), event_label))
                        
                        # 更新序列
                        sequences[i] = (session_id, label, new_features)
                    
                    # 更新映射表
                    self.attack2id = new_attack2id
                    self.id2attack = new_id2attack
                    self.next_id = new_next_id
            
            # 统计标签分布
            label_counts = Counter([seq[1] for seq in sequences])
            
            # 打印标签分布
            logger.info("\n序列标签分布:")
            for label, count in sorted(label_counts.items()):
                logger.info(f"  标签 {label} ({stage_mapping.get(label, 'Unknown')}): {count} 个序列")
            
            logger.info(f"\n构建了 {len(sequences)} 个APT攻击标识序列")
            logger.info(f"特征映射表大小: {len(self.attack2id)}")
            
            return sequences
            
        except Exception as e:
            logger.error(f"构建APT攻击标识序列时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _build_sequence_features(self, session_data, available_columns, apt_column, stage_mapping, feature_frequency, is_benign=False):
        """
        构建序列特征
        
        参数:
        session_data: 会话数据
        available_columns: 可用的特征列
        apt_column: APT标签列
        stage_mapping: 阶段映射
        feature_frequency: 特征频率字典
        is_benign: 是否为良性序列
        
        返回:
        构建的序列特征列表
        """
        seq_features = []
        
        # 使用适当的映射表
        feature2id = self.benign2id if is_benign else self.attack2id
        id2feature = self.id2benign if is_benign else self.id2attack
        
        for _, row in session_data.iterrows():
            # 使用通用特征提取函数
            feature_str = self._extract_features(row, available_columns)
            
            # 统计特征出现频率
            if feature_str not in feature_frequency:
                feature_frequency[feature_str] = 0
            feature_frequency[feature_str] += 1
            
            # 更新映射表
            if feature_str not in feature2id:
                next_id = self.next_benign_id if is_benign else self.next_id
                feature2id[feature_str] = next_id
                id2feature[next_id] = feature_str
                if is_benign:
                    self.next_benign_id += 1
                else:
                    self.next_id += 1
            
            # 确定事件标签
            try:
                if apt_column in row and pd.notna(row[apt_column]):
                    event_label = self._convert_label(row[apt_column])
                else:
                    # 如果没有标签，使用0（良性）
                    event_label = 0
            except Exception as e:
                # 如果转换失败，使用0（良性）
                if self.verbose:
                    logger.warning(f"警告: 转换标签时出错: {e}，使用默认标签0 (BENIGN)")
                event_label = 0
            
            # 添加到序列
            seq_features.append((feature2id[feature_str], event_label))
        
        return seq_features
    
    def build_benign_sequences(self, df, benign_sessions):
        """
        构建良性流量标识序列
        
        参数:
        df: 数据框
        benign_sessions: 良性会话ID列表
        
        返回:
        构建的序列列表
        """
        logger.info("\n构建良性流量标识序列...")
        
        # 初始化序列列表
        sequences = []
        
        try:
            # 检查良性会话列表是否为空
            if not benign_sessions:
                logger.warning("警告: 没有良性会话，无法构建良性序列")
                return []
                
            # 获取配置中的列名
            session_column = self.config.get("session_column", "Session_ID")
            apt_column = self.config.get("apt_column", "Label")
            
            # 检查必要的列是否存在
            if session_column not in df.columns:
                logger.error(f"错误: 会话列 '{session_column}' 不存在于数据集中")
                return []
                
            if apt_column not in df.columns:
                logger.error(f"错误: APT标签列 '{apt_column}' 不存在于数据集中")
                return []
            
            # 获取可用的特征列
            available_columns = self.get_available_features(df)
            
            if not available_columns:
                logger.error("错误: 没有可用的特征列，无法构建序列")
                return []
            
            # 统计特征出现频率
            feature_frequency = {}
            
            # 获取配置的最小和最大良性会话数
            min_benign_sessions = self.config.get("balancing", {}).get("min_benign_sessions", 100)
            max_benign_sessions = self.config.get("balancing", {}).get("max_benign_sessions", 1000)
            
            # 如果良性会话太多，随机采样
            if len(benign_sessions) > max_benign_sessions:
                logger.info(f"良性会话数量 ({len(benign_sessions)}) 超过最大限制 ({max_benign_sessions})，进行随机采样")
                random.seed(self.random_state)
                benign_sessions = random.sample(benign_sessions, max_benign_sessions)
            elif len(benign_sessions) < min_benign_sessions:
                logger.warning(f"警告: 良性会话数量 ({len(benign_sessions)}) 低于最小建议值 ({min_benign_sessions})")
            
            # 预处理数据，减少重复计算
            logger.info("预处理良性会话数据...")
            # 创建一个字典，将会话ID映射到其数据
            session_data_dict = {}
            
            # 预处理每个会话的数据
            for session_id in benign_sessions:
                try:
                    # 获取会话数据
                    session_data = df[df[session_column] == session_id].copy()
                    
                    # 只保留标签为0的事件（应用标签转换）
                    try:
                        benign_mask = session_data[apt_column].apply(
                            lambda x: self._convert_label(x) == 0 if pd.notna(x) else False
                        )
                        if not all(benign_mask):
                            session_data = session_data[benign_mask]
                            logger.info(f"会话 {session_id} 包含非零标签，只保留标签为0的事件，剩余 {len(session_data)} 个事件")
                    except Exception as e:
                        logger.warning(f"警告: 过滤会话 {session_id} 的非零标签时出错: {e}")
                    
                    # 检查会话数据是否为空
                    if len(session_data) == 0:
                        logger.warning(f"警告: 良性会话 {session_id} 没有数据")
                        continue
                    
                    # 填充缺失值
                    for col in available_columns:
                        if col in session_data.columns and session_data[col].isna().any():
                            if col in ['Src IP', 'Dst IP']:
                                session_data[col] = session_data[col].fillna('0.0.0.0')
                            elif col in ['Src Port', 'Dst Port']:
                                session_data[col] = session_data[col].fillna(0)
                            elif col == 'Protocol':
                                session_data[col] = session_data[col].fillna('Unknown')
                            elif col == session_column:
                                session_data[col] = session_data[col].fillna('unknown_session')
                            elif col == 'Timestamp':
                                session_data[col] = session_data[col].fillna(pd.Timestamp.now())
                    
                    # 如果启用了时间感知，提取时间特征
                    if self.time_aware_enabled:
                        if self.extract_time_features_enabled:
                            try:
                                time_features = self.extract_time_features(session_data)
                                if time_features:
                                    # 检测时间间隙
                                    time_gaps = self.detect_time_gaps(session_data)
                                    if time_gaps and self.verbose:
                                        logger.info(f"良性会话 {session_id} 检测到 {len(time_gaps)} 个时间间隙")
                                    
                                    # 将时间特征添加到会话数据中
                                    session_data.loc[:, 'time_features'] = str(time_features)
                                    session_data.loc[:, 'time_gaps'] = str(time_gaps)
                            except Exception as e:
                                logger.warning(f"警告: 提取时间特征时出错: {e}")
                    
                    # 保存预处理后的会话数据
                    session_data_dict[session_id] = session_data
                except Exception as e:
                    logger.warning(f"警告: 预处理良性会话 {session_id} 时出错: {e}")
                    continue
            
            logger.info(f"预处理了 {len(session_data_dict)} 个良性会话")
            
            # 处理每个良性会话
            for session_id in benign_sessions:
                try:
                    # 获取预处理后的会话数据
                    if session_id not in session_data_dict:
                        continue
                    
                    session_data = session_data_dict[session_id]
                    
                    # 检查是否有时间间隙，如果有，按时间间隙分割序列
                    time_gaps = []
                    if self.time_aware_enabled and 'time_gaps' in session_data.columns:
                        try:
                            time_gaps_str = session_data['time_gaps'].iloc[0]
                            if time_gaps_str.startswith('[') and time_gaps_str.endswith(']'):
                                time_gaps = eval(time_gaps_str)
                        except Exception as e:
                            logger.warning(f"警告: 解析时间间隙失败: {e}")
                    
                    # 如果有时间间隙，按间隙分割序列
                    if time_gaps and self.time_aware_enabled:
                        try:
                            # 按时间戳排序
                            if self.timestamp_column in session_data.columns:
                                session_data = session_data.sort_values(by=self.timestamp_column)
                            
                            # 分割点
                            split_points = [0] + time_gaps + [len(session_data)]
                            
                            # 为每个分段构建序列
                            for i in range(len(split_points) - 1):
                                start_idx = split_points[i]
                                end_idx = split_points[i + 1]
                                
                                # 获取分段数据
                                segment_data = session_data.iloc[start_idx:end_idx]
                                
                                # 如果分段太小，跳过
                                if len(segment_data) < 2:
                                    continue
                                
                                # 构建特征序列
                                seq_features = self._build_sequence_features(segment_data, available_columns, apt_column, {}, feature_frequency, is_benign=True)
                                
                                # 添加到序列列表，使用修改后的会话ID表示分段
                                segment_id = f"{session_id}_segment_{i+1}"
                                sequences.append((segment_id, 0, seq_features))  # 良性标签为0
                        except Exception as e:
                            logger.warning(f"警告: 处理良性会话 {session_id} 的时间间隙时出错: {e}")
                            # 如果处理时间间隙出错，尝试构建完整序列
                            seq_features = self._build_sequence_features(session_data, available_columns, apt_column, {}, feature_frequency, is_benign=True)
                            sequences.append((session_id, 0, seq_features))  # 良性标签为0
                    else:
                        # 构建特征序列
                        seq_features = self._build_sequence_features(session_data, available_columns, apt_column, {}, feature_frequency, is_benign=True)
                        
                        # 添加到序列列表
                        sequences.append((session_id, 0, seq_features))  # 良性标签为0
                except Exception as e:
                    logger.warning(f"警告: 处理良性会话 {session_id} 时出错: {e}")
                    continue
            
            # 优化特征映射表
            if self.config.get("feature_extraction", {}).get("enable_feature_filtering", False) and feature_frequency:
                # 获取良性特征的最小频率阈值
                benign_min_frequency = self.config.get("feature_extraction", {}).get("benign_min_frequency", 5)
                
                # 获取最少保留的特征数量
                min_features_to_keep = self.config.get("feature_extraction", {}).get("min_features_to_keep", 10)
                
                # 过滤低频特征
                filtered_features = {feature: freq for feature, freq in feature_frequency.items() if freq >= benign_min_frequency}
                
                # 确保至少保留一定数量的特征
                if len(filtered_features) < min_features_to_keep and feature_frequency:
                    # 如果过滤后的特征数量不足，保留出现频率最高的前N个特征
                    sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
                    top_features = dict(sorted_features[:min(min_features_to_keep, len(sorted_features))])
                    
                    # 将这些高频特征添加到过滤后的特征中
                    filtered_features.update(top_features)
                    logger.warning(f"警告: 过滤后的良性特征数量不足，保留出现频率最高的 {len(top_features)} 个特征")
                
                # 如果所有特征都被过滤掉，禁用过滤
                if not filtered_features and feature_frequency:
                    filtered_features = feature_frequency
                    logger.warning("警告: 所有良性特征都被过滤掉了，禁用过滤")
                
                # 重建映射表
                if len(filtered_features) < len(feature_frequency):
                    logger.info(f"\n过滤低频良性特征: 从 {len(feature_frequency)} 减少到 {len(filtered_features)}")
                    
                    # 创建新的映射表
                    new_benign2id = {}
                    new_id2benign = {}
                    new_next_id = 1
                    
                    # 更新序列中的特征ID
                    for i, (session_id, label, features) in enumerate(sequences):
                        new_features = []
                        for feature_id, event_label in features:
                            feature_str = self.id2benign[feature_id]
                            if feature_str in filtered_features:
                                if feature_str not in new_benign2id:
                                    new_benign2id[feature_str] = new_next_id
                                    new_id2benign[new_next_id] = feature_str
                                    new_next_id += 1
                                new_features.append((new_benign2id[feature_str], event_label))
                            else:
                                # 对于被过滤的特征，使用padding值
                                new_features.append((self.config.get('padding_value', 0), event_label))
                        
                        # 更新序列
                        sequences[i] = (session_id, label, new_features)
                    
                    # 更新映射表
                    self.benign2id = new_benign2id
                    self.id2benign = new_id2benign
                    self.next_benign_id = new_next_id
            
            logger.info(f"\n构建了 {len(sequences)} 个良性流量标识序列")
            logger.info(f"良性特征映射表大小: {len(self.benign2id)}")
            
            return sequences
            
        except Exception as e:
            logger.error(f"构建良性流量标识序列时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def create_synthetic_benign_sequences(self, num_sequences, df=None, available_columns=None):
        """
        创建合成正常流量序列
        
        参数:
        num_sequences: 要创建的序列数量
        df: 数据框，用于获取特征分布
        available_columns: 可用的特征列
        
        返回:
        合成序列列表，每个序列是一个元组 (session_id, label, features)
        """
        logger.info(f"创建 {num_sequences} 个合成正常流量序列...")
        
        # 获取配置
        min_length = self.config.get("benign_sequence", {}).get("min_length", 10)
        max_length = self.config.get("benign_sequence", {}).get("max_length", 50)
        
        # 确保最小长度至少为10，最大长度至少为20
        min_length = max(10, min_length)
        max_length = max(20, max_length)
        
        # 如果提供了数据框，尝试分析序列长度分布
        try:
            if df is not None and isinstance(df, pd.DataFrame) and self.config.get("session_column") in df.columns:
                session_column = self.config.get("session_column")
                session_lengths = df.groupby(session_column).size()
                if len(session_lengths) > 0:
                    # 使用10%分位数和90%分位数作为最小和最大长度
                    if len(session_lengths) >= 10:  # 确保有足够的样本计算分位数
                        min_length = max(min_length, int(session_lengths.quantile(0.1)))
                        max_length = max(20, min(max_length, int(session_lengths.quantile(0.9))))
                    else:
                        # 如果样本不足，使用平均长度作为参考
                        avg_length = session_lengths.mean()
                        min_length = max(min_length, max(1, int(avg_length * 0.5)))
                        max_length = max(20, min(max_length, int(avg_length * 1.5)))
        except Exception as e:
            logger.warning(f"警告: 分析序列长度分布时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 确保 min_length 小于 max_length
        if min_length >= max_length:
            logger.warning(f"警告: 最小长度 ({min_length}) 大于或等于最大长度 ({max_length})，调整长度范围")
            min_length = min(10, max_length - 10)
            max_length = max(20, min_length + 10)
        
        logger.info(f"合成序列长度范围: {min_length} - {max_length}")
        
        # 分析特征分布（如果可用）
        feature_distributions = {}
        try:
            if df is not None and available_columns:
                # 对每个可用列分析分布
                for col in available_columns:
                    if col in df.columns:
                        # 对于分类特征，记录唯一值及其频率
                        value_counts = df[col].value_counts(normalize=True)
                        if len(value_counts) > 0:
                            feature_distributions[col] = value_counts.to_dict()
                            top_values = value_counts.index[:3].tolist() if len(value_counts) >= 3 else value_counts.index.tolist()
                            logger.info(f"  分析了'{col}'的分布，发现 {len(value_counts)} 个唯一值，前三个常见值: {top_values}")
        except Exception as e:
            logger.warning(f"警告: 分析特征分布时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 为合成特征创建一个专用的ID空间，避免与真实特征冲突
        # 使用负数ID来区分合成特征
        synthetic_feature_base = -1000
        
        # 如果已经有合成特征映射表，使用它
        if not hasattr(self, 'synthetic_features'):
            self.synthetic_features = {}
        
        # 创建合成序列
        synthetic_sequences = []
        
        for i in range(num_sequences):
            # 确定序列长度
            seq_length = np.random.randint(min_length, max_length + 1)
            
            # 创建序列
            seq_features = []
            
            # 如果有特征分布信息，使用它来生成更真实的特征
            if feature_distributions and available_columns:
                for j in range(seq_length):
                    # 为每个特征列生成一个组合值
                    feature_str_parts = []
                    
                    for col in available_columns:
                        if col in feature_distributions and feature_distributions[col]:
                            # 从分布中随机选择一个值
                            values = list(feature_distributions[col].keys())
                            probabilities = list(feature_distributions[col].values())
                            
                            # 确保概率总和为1
                            prob_sum = sum(probabilities)
                            if prob_sum > 0 and abs(prob_sum - 1.0) > 1e-10:
                                probabilities = [p / prob_sum for p in probabilities]
                            
                            try:
                                value = np.random.choice(values, p=probabilities)
                                feature_str_parts.append(str(value))
                            except Exception as e:
                                logger.warning(f"警告: 生成特征值时出错: {e}，使用默认值")
                                feature_str_parts.append(f"synthetic_{j}")
                        else:
                            # 如果没有分布信息，生成一个占位符
                            feature_str_parts.append(f"synthetic_{j}_{i}")
                    
                    # 创建特征字符串
                    feature_str = "|".join(feature_str_parts)
                    
                    # 创建一个唯一的合成特征ID
                    feature_id = synthetic_feature_base - j - i * 100
                    
                    # 记录这个合成特征
                    self.synthetic_features[feature_id] = feature_str
                    
                    # 添加到序列
                    seq_features.append((feature_id, 0))  # 标签为0表示正常流量
            else:
                # 如果没有特征分布信息，生成简单的序列
                for j in range(seq_length):
                    feature_id = synthetic_feature_base - j - i * 100
                    self.synthetic_features[feature_id] = f"synthetic_feature_{-feature_id}"
                    seq_features.append((feature_id, 0))
            
            # 创建序列
            synthetic_sequences.append((f"synthetic_{i}", 0, seq_features))
        
        logger.info(f"创建了 {len(synthetic_sequences)} 个合成正常流量序列")
        return synthetic_sequences
    
    def standardize_sequence_length(self, sequences):
        """
        标准化序列长度
        
        参数:
        sequences: 序列列表
        
        返回:
        标准化后的序列列表
        """
        if not sequences:
            return []
        
        logger.info("\n标准化序列长度...")
        
        # 获取配置
        max_length = self.config.get("max_sequence_length", 50)
        min_length = self.config.get("min_sequence_length", 5)
        padding_value = self.config.get("padding_value", 0)
        padding_strategy = self.config.get("padding_strategy", "post")
        truncating_strategy = self.config.get("truncating_strategy", "post")
        
        # 统计序列长度
        lengths = [len(seq[2]) for seq in sequences]
        min_seq_length = min(lengths)
        max_seq_length = max(lengths)
        avg_seq_length = sum(lengths) / len(lengths)
        
        logger.info(f"序列长度统计: 最小={min_seq_length}, 最大={max_seq_length}, 平均={avg_seq_length:.2f}")
        
        # 标准化序列长度
        normalized_sequences = []
        
        for session_id, label, features in sequences:
            # 检查序列长度
            seq_length = len(features)
            
            # 如果序列太短，跳过
            if seq_length < min_length:
                logger.warning(f"警告: 序列 {session_id} 长度 {seq_length} 小于最小长度 {min_length}，跳过")
                continue
            
            # 如果序列长度等于目标长度，无需处理
            if seq_length == max_length:
                normalized_sequences.append((session_id, label, features))
                continue
            
            # 提取特征和标签
            feature_ids = [f[0] for f in features]
            event_labels = [f[1] for f in features]
            
            # 如果启用了时间感知，尝试保留时间信息
            if self.time_aware_enabled and "_segment_" in session_id:
                # 这是一个时间分段序列，需要特殊处理
                if seq_length > max_length:
                    # 如果序列太长，需要截断
                    if truncating_strategy == "pre":
                        feature_ids = feature_ids[-max_length:]
                        event_labels = event_labels[-max_length:]
                    else:  # post
                        feature_ids = feature_ids[:max_length]
                        event_labels = event_labels[:max_length]
                elif seq_length < max_length:
                    # 如果序列太短，需要填充
                    padding = [padding_value] * (max_length - seq_length)
                    if padding_strategy == "pre":
                        feature_ids = padding + feature_ids
                        event_labels = [0] * (max_length - seq_length) + event_labels
                    else:  # post
                        feature_ids = feature_ids + padding
                        event_labels = event_labels + [0] * (max_length - seq_length)
            else:
                # 普通序列处理
                if seq_length > max_length:
                    # 如果序列太长，需要截断
                    if truncating_strategy == "pre":
                        feature_ids = feature_ids[-max_length:]
                        event_labels = event_labels[-max_length:]
                    else:  # post
                        feature_ids = feature_ids[:max_length]
                        event_labels = event_labels[:max_length]
                elif seq_length < max_length:
                    # 如果序列太短，需要填充
                    padding = [padding_value] * (max_length - seq_length)
                    if padding_strategy == "pre":
                        feature_ids = padding + feature_ids
                        event_labels = [0] * (max_length - seq_length) + event_labels
                    else:  # post
                        feature_ids = feature_ids + padding
                        event_labels = event_labels + [0] * (max_length - seq_length)
            
            # 重新组合特征和标签
            normalized_features = list(zip(feature_ids, event_labels))
            
            # 添加到标准化序列列表
            normalized_sequences.append((session_id, label, normalized_features))
        
        # 打印标准化结果
        logger.info(f"标准化后的序列数量: {len(normalized_sequences)}")
        logger.info(f"标准化后的序列长度: {max_length}")
        
        return normalized_sequences
    
    def save_sequences(self, sequences, output_dir):
        """
        保存序列和映射表
        
        参数:
        sequences: 序列列表
        output_dir: 输出目录
        
        返回:
        统计信息字典
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存序列
        sequences_file = os.path.join(output_dir, "sequences.pkl")
        with open(sequences_file, "wb") as f:
            pickle.dump(sequences, f)
        
        # 保存映射表
        mapping_file = os.path.join(output_dir, "mapping.pkl")
        mapping = {
            "attack2id": self.attack2id,
            "id2attack": self.id2attack,
            "benign2id": self.benign2id,
            "id2benign": self.id2benign,
            "synthetic_features": getattr(self, "synthetic_features", {})
        }
        with open(mapping_file, "wb") as f:
            pickle.dump(mapping, f)
        
        # 计算统计信息
        stats = {}
        stats["total_sequences"] = len(sequences)
        
        # 计算攻击序列和正常序列的数量
        attack_sequences = [seq for seq in sequences if seq[1] != 0]
        benign_sequences = [seq for seq in sequences if seq[1] == 0]
        stats["attack_sequences"] = len(attack_sequences)
        stats["benign_sequences"] = len(benign_sequences)
        
        # 计算标签分布
        label_counts = Counter([int(seq[1]) for seq in sequences])
        label_dist = {}
        for label, count in sorted(label_counts.items()):
            label_dist[str(label)] = count
        stats["label_distribution"] = label_dist
        
        # 计算序列长度分布
        lengths = [len(seq[2]) for seq in sequences if isinstance(seq[2], list)]
        if lengths:
            stats["length_distribution"] = {
                "min": int(min(lengths)),
                "max": int(max(lengths)),
                "avg": float(sum(lengths) / len(lengths)),
                "median": int(sorted(lengths)[len(lengths) // 2])
            }
        else:
            stats["length_distribution"] = {"min": 0, "max": 0, "avg": 0, "median": 0}
        
        # 将统计信息转换为普通的Python字典（确保不包含NumPy类型）
        stats_dict = json.loads(json.dumps(stats, cls=NumpyEncoder))
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats_dict, f, indent=4)
        
        return stats
    
    def validate_dataset(self, df):
        """
        验证数据集结构和内容
        
        参数:
        df: 数据框
        
        返回:
        是否验证通过
        """
        logger.info("\n验证数据集...")
        
        # 获取配置中的列名
        session_column = self.config.get("session_column", "Session_ID")
        apt_column = self.config.get("apt_column", "Label")
        
        # 检查必要的列是否存在
        missing_columns = []
        if session_column not in df.columns:
            missing_columns.append(session_column)
            # 尝试使用默认会话列名
            if "Session_ID" in df.columns:
                logger.info(f"找到替代会话列: Session_ID")
                self.config["session_column"] = "Session_ID"
                session_column = "Session_ID"
                missing_columns.remove(session_column)
        
        if apt_column not in df.columns:
            missing_columns.append(apt_column)
            # 尝试使用默认标签列名
            if "Label" in df.columns:
                print(f"找到替代标签列: Label")
                self.config["apt_column"] = "Label"
                apt_column = "Label"
                missing_columns.remove(apt_column)
        
        if missing_columns:
            print(f"数据中缺少必要的列: {missing_columns}")
            return False
        
        # 检查会话数量
        session_count = df[session_column].nunique()
        print(f"唯一会话数: {session_count}")
        
        # 检查标签分布
        if apt_column in df.columns:
            label_counts = df[apt_column].value_counts()
            print("标签分布:")
            
            stage_mapping = self.config.get("stage_mapping", {})
            for label, count in sorted(label_counts.items()):
                # 获取标签名称，避免显示"未知标签"前缀
                label_str = str(label)
                label_name = stage_mapping.get(label_str)
                
                if label_name is None:
                    # 尝试使用整数键
                    try:
                        int_label = int(label)
                        label_name = stage_mapping.get(int_label)
                    except (ValueError, TypeError):
                        pass
                
                # 如果仍然找不到对应的名称，显示为"未知标签"
                if label_name is None:
                    label_name = f"未知标签 {label}"
                
                print(f"  标签 {label} ({label_name}): {count} 行 ({count/len(df):.2%})")
            
            # 统计每个会话的标签
            session_labels = {}
            for session_id in df[session_column].unique():
                session_data = df[df[session_column] == session_id]
                labels = session_data[apt_column].unique()
                for label in labels:
                    if label not in session_labels:
                        session_labels[label] = 0
                    session_labels[label] += 1
            
            print("会话标签分布:")
            for label, count in sorted(session_labels.items()):
                label_name = stage_mapping.get(str(label), f"未知标签 {label}")
                print(f"  标签 {label} ({label_name}): {count} 个会话 ({count/len(df[session_column].unique()):.2%})")
            
            # 检查良性流量比例
            benign_ratio = (df[apt_column] == 0).mean()
            print(f"良性流量比例: {benign_ratio:.2%}")
            if benign_ratio < 0.1:  # 如果良性流量少于10%
                print("警告: 良性流量比例过低，可能需要调整预处理步骤")
        
        return True
    
    def build_sequences(self, df, output_dir=None, need_mixed=True):
        """
        构建序列的主函数，支持两种方法：
        1. 原有的按攻击阶段组合分组+滑窗方法
        2. 新的实体+时间切分方法
        
        默认使用新方法
        """
        try:
            # 判断是否使用新方法
            use_entity_time_method = getattr(self, "use_entity_time_method", True)
            
            if use_entity_time_method:
                logger.info("使用实体+时间切分方法构建序列...")
                
                # 生成APT攻击序列
                apt_records = self.generate_apt_sequences(df)
                
                # 转换为与原方法兼容的返回格式
                all_seqs = []
                all_labels = []
                all_metadata = {"apttypes": [], "stage_groups": [], "orig_sessions": []}
                
                for apt_type, sess_df, sess_info in apt_records:
                    # 提取标签序列作为特征
                    flows = [(0, l) for l in sess_df[self.label_col]]
                    
                    # 确定序列标签（使用APT类型对应的数字）
                    if apt_type == "APT1":
                        label = 1
                    elif apt_type == "APT2":
                        label = 2
                    elif apt_type == "APT3":
                        label = 3
                    elif apt_type == "APT4":
                        label = 4
                    else:  # APT5
                        label = 5
                    
                    all_seqs.append(flows)
                    all_labels.append(label)
                    all_metadata["apttypes"].append(apt_type)
                    all_metadata["stage_groups"].append(sess_info["labels"])
                    all_metadata["orig_sessions"].append(f"Entity_Session_{len(all_seqs)}")
                
                # 打印窗口统计
                print("\n【实体+时间切分方法构建的序列统计】")
                print(f"总序列数: {len(all_seqs)}")
                from collections import Counter
                apt_counter = Counter(all_metadata["apttypes"])
                for apt_type, count in sorted(apt_counter.items()):
                    print(f"  {apt_type}: {count} 条序列")
                
                return all_seqs, all_labels, all_metadata
            
            else:
                logger.info("使用原有的攻击阶段组合+滑窗方法构建序列...")
                session_column = self.session_col
                apt_column = self.label_col
                timestamp_column = self.timestamp_col
                
                # 1. 按攻击阶段分组
                stage_groups = self.group_by_attack_stages(df)
                
                # 2. 合并同阶段组合的会话
                merged_groups = self.merge_sessions_by_stages(df, stage_groups)
                
                # 3. 对每个合并后的组应用滑窗
                all_seqs, all_labels, all_apttypes, all_orig_sessions, all_stage_groups = [], [], [], [], []
                window_counts_by_stage = {}
                
                for stages, group_data in merged_groups.items():
                    # 应用滑窗 - 注意这里不再跳过事件数小于window_size的会话
                    windows = self.event_level_sliding_window(group_data, need_mixed=need_mixed, min_len=self.window_size, overlap_rate=self.overlap_rate)
                    
                    # 记录窗口数量
                    window_counts_by_stage[stages] = len(windows)
                    
                    # 处理每个窗口
                    stage_str = "-".join(map(str, stages))
                    for window in windows:
                        # 检查是否是混合窗口
                        is_mixed = window[apt_column].nunique() > 1
                        
                        # 如果需要混合窗口但这不是混合窗口，跳过
                        if need_mixed and not is_mixed:
                            continue
                        
                        # 构建序列
                        flows = [(0, l) for l in window[apt_column]]
                        label = -1 if is_mixed else window[apt_column].iloc[0]
                        
                        # 收集结果
                        all_seqs.append(flows)
                        all_labels.append(label)
                        all_apttypes.append('MIXED' if is_mixed else label)
                        
                        # 使用阶段组合作为序列ID
                        all_orig_sessions.append(f"Stage_Group_{stage_str}")
                        all_stage_groups.append(stages)
                
                # 4. 终端输出窗口统计
                print("\n【最终事件级滑窗窗口统计】")
                print(f"总窗口数: {len(all_seqs)}")
                print(f"混合阶段窗口数: {sum(1 for l in all_labels if l == -1)}")
                print(f"纯阶段窗口数: {sum(1 for l in all_labels if l != -1)}")
                
                # 统计每个阶段组合的窗口数
                print("\n【按攻击阶段组合的窗口统计】")
                for stages, count in window_counts_by_stage.items():
                    stage_str = "->".join(map(str, stages))
                    print(f"  阶段组合 {stage_str}: {count} 个窗口")
                    
                    # 进一步统计该阶段组合中的混合窗口数
                    mixed_count = sum(1 for i, sg in enumerate(all_stage_groups) if sg == stages and all_labels[i] == -1)
                    pure_count = sum(1 for i, sg in enumerate(all_stage_groups) if sg == stages and all_labels[i] != -1)
                    
                    if count > 0:
                        print(f"    - 混合窗口: {mixed_count} ({mixed_count/count:.1%})")
                        print(f"    - 纯阶段窗口: {pure_count} ({pure_count/count:.1%})")
                
                # 进一步分析数据集中的Label分布情况
                print("\n【数据集Label分布分析】")
                unique_labels = df[apt_column].unique()
                label_counts = df[apt_column].value_counts()
                print(f"数据集中包含的Label: {sorted(unique_labels)}")
                for label, count in label_counts.items():
                    print(f"  Label {label}: {count} 条事件 ({count/len(df):.1%})")
                
                # 分析每个会话中包含的不同Label数量
                session_label_counts = df.groupby(session_column)[apt_column].nunique()
                multi_label_sessions = sum(session_label_counts > 1)
                print(f"\n包含多个不同Label的会话数: {multi_label_sessions} ({multi_label_sessions/len(session_label_counts):.1%})")
                
                # 返回结果
                return all_seqs, all_labels, {"apttypes": all_apttypes, "stage_groups": all_stage_groups}
                
        except Exception as e:
            import traceback
            print("[build_sequences] 捕获到异常：", e)
            traceback.print_exc()
            return None, None, None
    
    def detect_and_update_label_mapping(self, df):
        """
        检测并更新标签映射
        
        参数:
        df: 数据框
        
        返回:
        更新后的标签映射
        """
        apt_column = self.config.get("apt_column", "Label")
        apt_category_column = self.config.get("apt_category_column", "Label_Name")
        
        # 检查是否同时有标签列和标签名称列
        if apt_column in df.columns and apt_category_column in df.columns:
            logger.info("\n检测数据集中的标签映射...")
            
            # 获取唯一的标签和对应的名称
            label_mapping = {}
            unique_pairs = df[[apt_column, apt_category_column]].drop_duplicates()
            
            for _, row in unique_pairs.iterrows():
                label = str(row[apt_column])
                name = row[apt_category_column]
                label_mapping[label] = name
                logger.info(f"  检测到标签映射: {label} -> {name}")
            
            # 确保至少包含"0" -> "BENIGN"的映射
            if "0" not in label_mapping:
                label_mapping["0"] = "BENIGN"
                logger.info("  添加默认良性标签映射: 0 -> BENIGN")
                
            # 更新配置中的stage_mapping
            if label_mapping:
                logger.info("\n使用数据集中检测到的标签映射更新配置")
                self.config["stage_mapping"] = label_mapping
                
                # 同时更新attack_stage_order
                attack_stage_order = {}
                for label, name in label_mapping.items():
                    if label != "0" and name != "BENIGN":  # 跳过良性标签
                        attack_stage_order[name] = int(label)
                
                if attack_stage_order:
                    self.config["attack_stage_order"] = attack_stage_order
                    logger.info("  同时更新了攻击阶段顺序")
            
            return label_mapping
        
        return self.config.get("stage_mapping", {})
    
    def synthesize_features(self, df):
        """
        合成缺失的主要特征列
        
        如果数据集中缺少主要特征列（如Src IP, Dst IP等），此方法会基于现有数据
        合成这些特征列，使得可以进行更有效的序列构建。
        
        参数:
        df: 数据框
        
        返回:
        合成特征后的数据框副本
        """
        logger.info("\n检查并合成缺失的主要特征列...")
        
        # 创建数据框的副本，避免修改原始数据
        enhanced_df = df.copy()
        
        # 获取配置的主要特征列
        feature_columns = self.config["feature_columns"]
        
        # 获取会话列和标签列
        session_column = self.config.get("session_column", "Session_ID")
        apt_column = self.config.get("apt_column", "Label")
        
        # 检查哪些主要特征列缺失
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if not missing_features:
            logger.info("所有主要特征列都存在，无需合成")
            return enhanced_df
        
        logger.info(f"需要合成的特征列: {missing_features}")
        
        # 检查必要的基础列是否存在
        if session_column not in df.columns:
            logger.warning(f"警告: 缺少会话ID列 {session_column}，尝试使用默认会话列名")
            # 尝试使用默认会话列名
            if "Session_ID" in df.columns:
                session_column = "Session_ID"
                logger.info(f"找到替代会话列: {session_column}")
            else:
                logger.error(f"错误: 无法找到会话ID列，无法合成特征")
                return enhanced_df
        
        logger.info(f"使用会话列: {session_column}")
        
        # 为每个会话ID创建一个唯一标识符
        session_ids = df[session_column].unique()
        logger.info(f"找到 {len(session_ids)} 个唯一会话ID")
        session_to_id = {session: i for i, session in enumerate(session_ids)}
        
        # 合成源IP地址
        if "Src IP" in missing_features:
            logger.info("合成源IP地址...")
            enhanced_df["Src IP"] = df[session_column].apply(
                lambda x: self._generate_ip_from_session(session_to_id[x], "src")
            )
            logger.info(f"合成了 {len(enhanced_df['Src IP'].unique())} 个唯一源IP地址")
        
        # 合成目标IP地址
        if "Dst IP" in missing_features:
            logger.info("合成目标IP地址...")
            enhanced_df["Dst IP"] = df[session_column].apply(
                lambda x: self._generate_ip_from_session(session_to_id[x], "dst")
            )
            logger.info(f"合成了 {len(enhanced_df['Dst IP'].unique())} 个唯一目标IP地址")
        
        # 合成源端口
        if "Src Port" in missing_features:
            logger.info("合成源端口...")
            enhanced_df["Src Port"] = df[session_column].apply(
                lambda x: self._generate_port_from_session(session_to_id[x], "src")
            )
            logger.info(f"合成了 {len(enhanced_df['Src Port'].unique())} 个唯一源端口")
        
        # 合成目标端口
        if "Dst Port" in missing_features:
            logger.info("合成目标端口...")
            # 如果有标签列，基于标签生成更有意义的目标端口
            if apt_column in df.columns:
                enhanced_df["Dst Port"] = df.apply(
                    lambda row: self._generate_port_from_label(row[apt_column]), 
                    axis=1
                )
            else:
                enhanced_df["Dst Port"] = df[session_column].apply(
                    lambda x: self._generate_port_from_session(session_to_id[x], "dst")
                )
            logger.info(f"合成了 {len(enhanced_df['Dst Port'].unique())} 个唯一目标端口")
        
        # 合成协议
        if "Protocol" in missing_features:
            logger.info("合成协议...")
            # 如果有标签列，基于标签生成协议
            if apt_column in df.columns:
                enhanced_df["Protocol"] = df[apt_column].apply(
                    lambda x: self._generate_protocol_from_label(x)
                )
            else:
                # 否则，随机分配协议
                protocols = ["TCP", "UDP", "HTTP", "HTTPS", "DNS", "ICMP"]
                enhanced_df["Protocol"] = df[session_column].apply(
                    lambda x: protocols[session_to_id[x] % len(protocols)]
                )
            logger.info(f"合成了 {len(enhanced_df['Protocol'].unique())} 个唯一协议")
        
        # 验证合成结果
        still_missing = [col for col in feature_columns if col not in enhanced_df.columns]
        if still_missing:
            logger.warning(f"警告: 合成后仍然缺少以下特征列: {still_missing}")
        else:
            logger.info("所有缺失的特征列已成功合成")
        
        logger.info("特征合成完成，添加了以下列:")
        for col in missing_features:
            if col in enhanced_df.columns:
                logger.info(f"  - {col}: {len(enhanced_df[col].unique())} 个唯一值")
        
        return enhanced_df
    
    def _generate_ip_from_session(self, session_id, ip_type="src"):
        """
        基于会话ID生成IP地址
        
        参数:
        session_id: 会话ID的数值表示
        ip_type: "src"表示源IP，"dst"表示目标IP
        
        返回:
        生成的IP地址字符串
        """
        if ip_type == "src":
            # 源IP使用10.0.0.0/16私有地址段
            return f"10.0.{session_id//255 % 255}.{session_id % 255}"
        else:
            # 目标IP使用192.168.0.0/16私有地址段
            return f"192.168.{session_id//255 % 255}.{session_id % 255}"
    
    def _generate_port_from_session(self, session_id, port_type="src"):
        """
        基于会话ID生成端口号
        
        参数:
        session_id: 会话ID的数值表示
        port_type: "src"表示源端口，"dst"表示目标端口
        
        返回:
        生成的端口号
        """
        if port_type == "src":
            # 源端口使用动态端口范围(1024-65535)
            return 1024 + (session_id % 64512)
        else:
            # 目标端口使用常见服务端口
            common_ports = [80, 443, 22, 25, 53, 8080, 3389, 21, 23, 3306]
            return common_ports[session_id % len(common_ports)]
    
    def _generate_port_from_label(self, label):
        """
        基于标签生成目标端口号
        
        参数:
        label: 标签值
        
        返回:
        生成的目标端口号
        """
        # 确保应用标签转换
        new_label = self._convert_label(label)
        
        # 常见服务端口与APT攻击阶段的映射（基于用户要求的标签映射）
        label_to_port = {
            0: 80,    # BENIGN - HTTP
            1: 443,   # Reconnaissance - HTTPS
            2: 22,    # Establish Foothold - SSH
            3: 445,   # Lateral Movement - SMB
            4: 21     # Data Exfiltration - FTP
        }
        
        # 如果标签在映射表中，返回对应端口；否则返回默认端口
        return label_to_port.get(new_label, 8080)
    
    def _generate_protocol_from_label(self, label):
        """
        基于标签生成协议
        
        参数:
        label: 标签值
        
        返回:
        生成的协议字符串
        """
        # 确保应用标签转换
        new_label = self._convert_label(label)
        
        # 协议与APT攻击阶段的映射（基于用户要求的标签映射）
        label_to_protocol = {
            0: "HTTP",    # BENIGN
            1: "HTTPS",   # Reconnaissance
            2: "SSH",     # Establish Foothold
            3: "SMB",     # Lateral Movement
            4: "FTP"      # Data Exfiltration
        }
        
        # 如果标签在映射表中，返回对应协议；否则返回默认协议
        return label_to_protocol.get(new_label, "TCP")
    
    def extract_time_features(self, session_data):
        """
        提取时间特征
        
        参数:
        session_data: 会话数据框
        
        返回:
        时间特征字典
        """
        if not self.time_aware_enabled or not self.extract_time_features_enabled:
            return {}
        
        # 检查是否有时间戳列
        if self.timestamp_column not in session_data.columns:
            return {}
        
        try:
            # 确保时间戳是datetime类型
            if not pd.api.types.is_datetime64_dtype(session_data[self.timestamp_column]):
                try:
                    session_data[self.timestamp_column] = pd.to_datetime(session_data[self.timestamp_column])
                except Exception as e:
                    logger.warning(f"警告: 无法将时间戳列转换为datetime类型: {e}")
                    return {}
            
            # 按时间戳排序
            sorted_data = session_data.sort_values(by=self.timestamp_column)
            
            # 如果只有一条记录，无法提取时间特征
            if len(sorted_data) <= 1:
                return {}
            
            # 计算时间差
            timestamps = sorted_data[self.timestamp_column].tolist()
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            
            # 提取时间特征
            time_features = {
                'session_duration': (timestamps[-1] - timestamps[0]).total_seconds(),
                'avg_time_diff': sum(time_diffs) / len(time_diffs) if time_diffs else 0,
                'max_time_diff': max(time_diffs) if time_diffs else 0,
                'min_time_diff': min(time_diffs) if time_diffs else 0,
                'std_time_diff': np.std(time_diffs) if time_diffs else 0
            }
            
            # 检测是否有昼夜模式（工作时间vs非工作时间）
            hour_counts = sorted_data[self.timestamp_column].dt.hour.value_counts()
            working_hours = sum(hour_counts.get(h, 0) for h in range(9, 18))  # 9am-6pm
            non_working_hours = sum(hour_counts.get(h, 0) for h in range(24) if h < 9 or h >= 18)
            time_features['working_hours_ratio'] = working_hours / (working_hours + non_working_hours) if (working_hours + non_working_hours) > 0 else 0
            
            return time_features
        
        except Exception as e:
            logger.warning(f"警告: 提取时间特征时出错: {e}")
            return {}
    
    def detect_time_gaps(self, session_data):
        """
        检测时间间隙
        
        参数:
        session_data: 会话数据框
        
        返回:
        时间间隙索引列表
        """
        if not self.time_aware_enabled:
            return []
        
        # 检查是否有时间戳列
        if self.timestamp_column not in session_data.columns:
            return []
        
        try:
            # 确保时间戳是datetime类型
            if not pd.api.types.is_datetime64_dtype(session_data[self.timestamp_column]):
                try:
                    session_data[self.timestamp_column] = pd.to_datetime(session_data[self.timestamp_column])
                except Exception as e:
                    logger.warning(f"警告: 无法将时间戳列转换为datetime类型: {e}")
                    return []
            
            # 按时间戳排序
            sorted_data = session_data.sort_values(by=self.timestamp_column).reset_index(drop=True)
            
            # 如果只有一条记录，无法检测时间间隙
            if len(sorted_data) <= 1:
                return []
            
            # 计算时间差
            timestamps = sorted_data[self.timestamp_column].tolist()
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            
            # 检测时间间隙
            gap_threshold = self.time_gap_threshold  # 默认1小时
            time_gaps = []
            
            for i, diff in enumerate(time_diffs):
                if diff > gap_threshold:
                    # 记录间隙位置（索引）
                    time_gaps.append(i + 1)
            
            return time_gaps
        
        except Exception as e:
            logger.warning(f"警告: 检测时间间隙时出错: {e}")
            return []
    
    def _calculate_stage_transition_matrix(self, session_stage_sequences):
       
        # 获取所有可能的攻击阶段
        all_stages = set()
        for sequence in session_stage_sequences.values():
            all_stages.update(sequence)
        
        # 如果没有攻击阶段，返回空矩阵
        if not all_stages:
            return {}
        
        # 初始化转移矩阵
        max_stage = max(all_stages)
        transition_matrix = np.zeros((max_stage + 1, max_stage + 1))
        
        # 计算转移次数
        for sequence in session_stage_sequences.values():
            for i in range(len(sequence) - 1):
                from_stage = sequence[i]
                to_stage = sequence[i + 1]
                transition_matrix[from_stage, to_stage] += 1
        
        # 计算转移概率（行归一化）
        row_sums = transition_matrix.sum(axis=1)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                transition_matrix[i, :] /= row_sums[i]
        
        # 应用平滑因子，避免零概率
        smoothing = self.config.get("flexible_stage_recognition", {}).get("transition_smoothing", 0.01)
        if smoothing > 0:
            transition_matrix = (1 - smoothing) * transition_matrix + smoothing / transition_matrix.shape[1]
        
        # 转换为字典格式，便于使用
        transition_dict = {}
        for i in range(transition_matrix.shape[0]):
            transition_dict[i] = {}
            for j in range(transition_matrix.shape[1]):
                if transition_matrix[i, j] > 0:
                    transition_dict[i][j] = transition_matrix[i, j]
        
        return transition_dict
    
    def _categorize_with_flexible_recognition(self, stage_sequence, allow_repetition=True, allow_skipping=True):
        """
        使用灵活的攻击阶段识别进行分类
        
        参数:
        stage_sequence: 攻击阶段序列
        allow_repetition: 是否允许阶段重复
        allow_skipping: 是否允许阶段跳跃
        
        返回:
        APT类别
        """
        # 如果序列为空，返回APT5
        if not stage_sequence:
            return "APT5"
        
        # 去除重复阶段（如果不允许重复）
        if not allow_repetition:
            cleaned_sequence = []
            prev_stage = None
            for stage in stage_sequence:
                if stage != prev_stage:
                    cleaned_sequence.append(stage)
                    prev_stage = stage
            stage_sequence = cleaned_sequence
        
        # 获取序列中的唯一阶段
        unique_stages = set(stage_sequence)
        
        # 检查是否包含所有必要的阶段（基于标准标签映射）
        has_reconnaissance = 1 in unique_stages      # 侦察阶段 (标签1)
        has_establish_foothold = 2 in unique_stages  # 立足阶段 (标签2)
        has_lateral_movement = 3 in unique_stages    # 横向移动 (标签3)
        has_data_exfiltration = 4 in unique_stages   # 数据渗出 (标签4)
        
        # 检查阶段顺序（如果不允许跳跃）
        if not allow_skipping:
            # 定义期望的阶段顺序（基于标准标签映射）
            expected_order = [1, 2, 3, 4]  # 侦察 -> 立足 -> 横向移动 -> 数据渗出
            
            # 检查序列是否按照正确的顺序进行
            is_ordered = True
            current_phase_idx = 0
            
            for stage in stage_sequence:
                if stage not in expected_order:
                    continue  # 忽略不在预期顺序中的阶段
                
                # 找到当前阶段在预期顺序中的位置
                try:
                    stage_idx = expected_order.index(stage)
                except ValueError:
                    continue  # 如果阶段不在预期顺序中，跳过
                
                # 如果当前阶段的索引小于之前的索引，说明顺序不对
                if stage_idx < current_phase_idx:
                    is_ordered = False
                    break
                
                # 更新当前阶段索引
                current_phase_idx = max(current_phase_idx, stage_idx)
            
            if not is_ordered:
                return "APT5"  # 不规则攻击链
        
        # 根据包含的阶段进行分类（基于新的标签映射）
        if has_data_exfiltration and has_lateral_movement and has_establish_foothold and has_reconnaissance:
            return "APT4"  # 完整攻击链
        elif has_lateral_movement and has_establish_foothold and has_reconnaissance:
            return "APT3"  # 侦察+立足+横向移动
        elif has_establish_foothold and has_reconnaissance:
            return "APT2"  # 侦察+立足
        elif has_reconnaissance:
            return "APT1"  # 只有侦察阶段
        else:
            return "APT5"  # 不规则攻击链
    
    def _extract_behavioral_features(self, session_data):
        """
        提取行为特征
        
        参数:
        session_data: 会话数据框
        
        返回:
        None（直接修改session_data）
        """
        if not self.extract_behavioral_features:
            return
        
        # 检查是否有足够的数据
        if len(session_data) <= 1:
            return
        
        try:
            # 按时间戳排序（如果有）
            if self.timestamp_column in session_data.columns:
                session_data = session_data.sort_values(by=self.timestamp_column)
            
            # 提取流量方向变化
            if 'Src IP' in session_data.columns and 'Dst IP' in session_data.columns:
                direction_changes = []
                prev_src = None
                prev_dst = None
                
                for _, row in session_data.iterrows():
                    src = row['Src IP']
                    dst = row['Dst IP']
                    
                    if prev_src is not None and prev_dst is not None:
                        # 检查方向是否变化
                        if src == prev_dst and dst == prev_src:
                            direction_changes.append(1)  # 方向反转
                        else:
                            direction_changes.append(0)  # 方向不变或新方向
                    else:
                        direction_changes.append(0)  # 首个记录
                    
                    prev_src = src
                    prev_dst = dst
                
                # 添加方向变化特征
                session_data.loc[:, 'direction_change'] = [0] + direction_changes[:-1]
            
            # 提取协议变化
            if 'Protocol' in session_data.columns:
                protocol_changes = []
                prev_protocol = None
                
                for _, row in session_data.iterrows():
                    protocol = row['Protocol']
                    
                    if prev_protocol is not None:
                        # 检查协议是否变化
                        if protocol != prev_protocol:
                            protocol_changes.append(1)  # 协议变化
                        else:
                            protocol_changes.append(0)  # 协议不变
                    else:
                        protocol_changes.append(0)  # 首个记录
                    
                    prev_protocol = protocol
                
                # 添加协议变化特征
                session_data.loc[:, 'protocol_change'] = [0] + protocol_changes[:-1]
            
            # 提取端口扫描行为
            if 'Src Port' in session_data.columns and 'Dst Port' in session_data.columns:
                # 检查是否有端口扫描特征（同一源IP短时间内连接多个目标端口）
                if 'Src IP' in session_data.columns and self.timestamp_column in session_data.columns:
                    port_scan_features = []
                    
                    # 按源IP分组
                    for src_ip, group in session_data.groupby('Src IP'):
                        # 如果组内记录数小于3，不太可能是端口扫描
                        if len(group) < 3:
                            for _ in range(len(group)):
                                port_scan_features.append(0)
                            continue
                        
                        # 按时间戳排序
                        group = group.sort_values(by=self.timestamp_column)
                        
                        # 检查是否在短时间内访问多个不同端口
                        dst_ports = group['Dst Port'].tolist()
                        timestamps = group[self.timestamp_column].tolist()
                        
                        for i in range(len(group)):
                            if i < 2:  # 前两条记录无法判断
                                port_scan_features.append(0)
                                continue
                            
                            # 检查前3条记录是否在短时间内访问不同端口
                            recent_ports = set(dst_ports[max(0, i-2):i+1])
                            
                            if len(recent_ports) >= 3:
                                # 检查时间间隔
                                time_span = (timestamps[i] - timestamps[max(0, i-2)]).total_seconds()
                                if time_span < 60:  # 1分钟内
                                    port_scan_features.append(1)  # 可能是端口扫描
                                    continue
                            
                            port_scan_features.append(0)
                    
                    # 确保特征长度与会话数据长度一致
                    if len(port_scan_features) == len(session_data):
                        session_data.loc[:, 'port_scan_behavior'] = port_scan_features
        
        except Exception as e:
            logger.warning(f"警告: 提取行为特征失败: {e}")
    
    def _extract_statistical_features(self, session_data):
        """
        提取统计特征
        
        参数:
        session_data: 会话数据框
        
        返回:
        None（直接修改session_data）
        """
        if not self.extract_statistical_features:
            return
        
        try:
            # 计算滑动窗口特征
            window_size = self.window_size
            
            if len(session_data) <= window_size:
                return
            
            # 按时间戳排序（如果有）
            if self.timestamp_column in session_data.columns:
                session_data = session_data.sort_values(by=self.timestamp_column)
            
            # 初始化滑动窗口特征
            session_data.loc[:, 'window_unique_dst_ips'] = 0
            session_data.loc[:, 'window_unique_dst_ports'] = 0
            session_data.loc[:, 'window_protocol_entropy'] = 0
            
            # 计算滑动窗口特征
            for i in range(len(session_data) - window_size + 1):
                window = session_data.iloc[i:i+window_size]
                
                # 计算窗口内唯一目标IP数量
                if 'Dst IP' in window.columns:
                    unique_dst_ips = window['Dst IP'].nunique()
                    session_data.loc[session_data.index[i+window_size-1], 'window_unique_dst_ips'] = unique_dst_ips
                
                # 计算窗口内唯一目标端口数量
                if 'Dst Port' in window.columns:
                    unique_dst_ports = window['Dst Port'].nunique()
                    session_data.loc[session_data.index[i+window_size-1], 'window_unique_dst_ports'] = unique_dst_ports
                
                # 计算窗口内协议熵
                if 'Protocol' in window.columns:
                    protocol_counts = window['Protocol'].value_counts(normalize=True)
                    entropy = -np.sum(protocol_counts * np.log2(protocol_counts))
                    session_data.loc[session_data.index[i+window_size-1], 'window_protocol_entropy'] = entropy
        
        except Exception as e:
            logger.warning(f"警告: 提取统计特征失败: {e}")
    
    def _detect_disguised_behaviors(self, session_data_dict):
        """
        检测伪装行为
        
        参数:
        session_data_dict: 会话数据字典
        
        返回:
        None（直接修改session_data_dict）
        """
        # 暂时禁用此功能，因为缺少必要的配置
        return
    
    def _calculate_anomaly_score(self, window):
        """
        计算窗口内的异常分数
        
        参数:
        window: 数据窗口
        
        返回:
        异常分数 (0-1)
        """
        # 这里使用简单的启发式方法计算异常分数
        # 在实际应用中，可以使用更复杂的异常检测算法
        
        anomaly_score = 0.0
        
        # 检查协议变化
        if 'protocol_change' in window.columns:
            protocol_change_ratio = window['protocol_change'].mean()
            anomaly_score += protocol_change_ratio * 0.3
        
        # 检查方向变化
        if 'direction_change' in window.columns:
            direction_change_ratio = window['direction_change'].mean()
            anomaly_score += direction_change_ratio * 0.3
        
        # 检查端口扫描行为
        if 'port_scan_behavior' in window.columns:
            port_scan_ratio = window['port_scan_behavior'].mean()
            anomaly_score += port_scan_ratio * 0.4
        
        # 检查窗口特征
        if 'window_unique_dst_ips' in window.columns and window['window_unique_dst_ips'].max() > 0:
            dst_ip_ratio = window['window_unique_dst_ips'].iloc[-1] / window['window_unique_dst_ips'].max()
            anomaly_score += dst_ip_ratio * 0.2
        
        if 'window_protocol_entropy' in window.columns and window['window_protocol_entropy'].max() > 0:
            entropy_ratio = window['window_protocol_entropy'].iloc[-1] / window['window_protocol_entropy'].max()
            anomaly_score += entropy_ratio * 0.2
        
        # 归一化分数到0-1范围
        return min(1.0, anomaly_score)
    
    def visualize_sequence(self, sequence, output_file=None):
        """
        可视化序列，用于调试和分析
        
        参数:
        sequence: 序列元组 (session_id, label, features)
        output_file: 输出文件路径（可选）
        
        返回:
        None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
            
            session_id, label, features = sequence
            
            # 提取特征ID和事件标签
            feature_ids = [f[0] for f in features]
            event_labels = [f[1] for f in features]
            
            # 创建图形
            plt.figure(figsize=(15, 8))
            
            # 绘制特征热图
            plt.subplot(2, 1, 1)
            feature_array = np.array(feature_ids).reshape(1, -1)
            plt.imshow(feature_array, aspect='auto', cmap='viridis')
            plt.colorbar(label='Feature ID')
            plt.title(f'Sequence Visualization - Session: {session_id}, Label: {label}')
            plt.ylabel('Features')
            
            # 绘制事件标签
            plt.subplot(2, 1, 2)
            
            # 创建自定义颜色映射，不同的标签使用不同的颜色
            unique_labels = sorted(set(event_labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            cmap = LinearSegmentedColormap.from_list('apt_stages', colors, N=len(unique_labels))
            
            label_array = np.array(event_labels).reshape(1, -1)
            plt.imshow(label_array, aspect='auto', cmap=cmap)
            plt.colorbar(label='Event Label')
            plt.xlabel('Sequence Position')
            plt.ylabel('Labels')
            
            # 添加标签说明
            stage_mapping = self.config.get("stage_mapping", {})
            legend_elements = []
            for i, label_value in enumerate(unique_labels):
                if label_value in stage_mapping:
                    label_name = stage_mapping[label_value]
                else:
                    label_name = f"Unknown ({label_value})"
                legend_elements.append(patches.Patch(color=colors[i], label=f"{label_value}: {label_name}"))
            
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # 如果启用了时间感知，添加时间间隙标记
            if self.time_aware_enabled and "_segment_" in session_id:
                # 这是一个时间分段序列，标记起始点
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存或显示图形
            if output_file:
                plt.savefig(output_file)
                logger.info(f"序列可视化已保存到 {output_file}")
            else:
                plt.show()
        
        except Exception as e:
            logger.warning(f"警告: 可视化序列失败: {e}")
    
    def visualize_apt_patterns(self, sequences, output_dir=None, max_samples=5):
        """
        可视化APT攻击模式
        
        参数:
        sequences: 序列列表
        output_dir: 输出目录（可选）
        max_samples: 每个类别的最大样本数
        
        返回:
        None
        """
        try:
            # 按标签分组
            label_groups = {}
            for seq in sequences:
                session_id, label, features = seq
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(seq)
            
            # 为每个标签可视化样本
            for label, seqs in label_groups.items():
                logger.info(f"可视化标签 {label} 的样本...")
                
                # 限制样本数量
                samples = seqs[:min(max_samples, len(seqs))]
                
                for i, seq in enumerate(samples):
                    if output_dir:
                        output_file = os.path.join(output_dir, f"label_{label}_sample_{i+1}.png")
                    else:
                        output_file = None
                    
                    self.visualize_sequence(seq, output_file)
        
        except Exception as e:
            logger.warning(f"警告: 可视化APT模式失败: {e}")
    
    def _convert_label(self, label):
        """
        将旧标签转换为新标签
        
        参数:
        label: 原始标签值
        
        返回:
        转换后的标签值
        """
        # 处理None和NaN值
        if label is None or (isinstance(label, float) and np.isnan(label)):
            return 0  # 默认为BENIGN
        
        try:
            # 尝试将标签转换为整数
            if isinstance(label, str):
                # 如果是空字符串，返回0
                if not label.strip():
                    return 0
                
                # 尝试将字符串转换为整数
                label_int = int(label)
            else:
                label_int = int(label)
            
            # 使用配置的映射进行转换
            old_to_new_label = self.config.get("old_to_new_label", {})
            
            # 如果标签在映射中，返回映射的值
            if label_int in old_to_new_label:
                return old_to_new_label[label_int]
            else:
                # 如果标签不在映射中，记录一个调试消息并返回0
                if self.verbose:
                    logger.debug(f"标签 {label_int} 不在映射中，默认为0 (BENIGN)")
                return 0
        except (ValueError, TypeError) as e:
            # 如果转换失败，尝试查找阶段名称
            if isinstance(label, str):
                # 检查是否是阶段名称
                stage_mapping = self.config.get("stage_mapping", {})
                for key, value in stage_mapping.items():
                    if label.lower() == value.lower():
                        try:
                            return int(key)
                        except (ValueError, TypeError):
                            pass
            
            # 如果所有尝试都失败，返回0并记录警告
            if self.verbose:
                logger.warning(f"无法转换标签 {label}，默认为0 (BENIGN): {e}")
            return 0
    
    def _extract_features(self, row, available_columns):
        """
        从行数据中提取特征
        
        参数:
        row: 数据行
        available_columns: 可用的特征列
        
        返回:
        特征字符串
        """
        # 基本特征
        basic_features = []
        for col in available_columns:
            if col in row:
                basic_features.append(str(row[col]))
            else:
                basic_features.append("")
        
        # 行为特征
        behavior_features = []
        if 'direction_change' in row:
            behavior_features.append(f"direction_change:{row['direction_change']}")
        if 'protocol_change' in row:
            behavior_features.append(f"protocol_change:{row['protocol_change']}")
        if 'port_scan_behavior' in row:
            behavior_features.append(f"port_scan:{row['port_scan_behavior']}")
        
        # 窗口特征
        window_features = []
        if 'window_unique_dst_ips' in row:
            window_features.append(f"unique_dst_ips:{row['window_unique_dst_ips']}")
        if 'window_unique_dst_ports' in row:
            window_features.append(f"unique_dst_ports:{row['window_unique_dst_ports']}")
        if 'window_protocol_entropy' in row:
            window_features.append(f"protocol_entropy:{row['window_protocol_entropy']}")
        
        # 伪装特征
        disguise_features = []
        if 'is_disguised' in row:
            disguise_features.append(f"disguised:{row['is_disguised']}")
        
        # 时间特征
        time_features = []
        if 'time_features' in row:
            time_feature_str = row['time_features']
            if isinstance(time_feature_str, str) and time_feature_str.startswith('{') and time_feature_str.endswith('}'):
                try:
                    time_feature_dict = eval(time_feature_str)
                    for k, v in time_feature_dict.items():
                        time_features.append(f"{k}:{v}")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"解析时间特征时出错: {e}")
        
        # 组合所有特征
        all_features = []
        if basic_features:
            all_features.append("||".join(basic_features))
        if behavior_features:
            all_features.append("||".join(behavior_features))
        if window_features:
            all_features.append("||".join(window_features))
        if disguise_features:
            all_features.append("||".join(disguise_features))
        if time_features:
            all_features.append("||".join(time_features))
        
        # 返回最终特征字符串
        return "||".join(all_features)
    
    def balance_sequences(self, sequences):
        """
        平衡序列数据
        
        参数:
        sequences: 序列列表
        
        返回:
        平衡后的序列列表
        """
        logger.info("\n进行序列平衡...")
        
        # 获取平衡配置
        balance_method = self.balancing_method
        target_ratio = self.target_ratio
        random_state = self.random_state
        
        # 按标签分组序列
        label_groups = {}
        for seq in sequences:
            label = seq[1]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(seq)
        
        # 计算每个标签的序列数量
        label_counts = {label: len(seqs) for label, seqs in label_groups.items()}
        
        # 打印当前分布
        logger.info("当前标签分布:")
        for label, count in sorted(label_counts.items()):
            stage_name = self.config.get("stage_mapping", {}).get(label, f"未知标签 {label}")
            logger.info(f"  标签 {label} ({stage_name}): {count} 个序列")
        
        # 如果只有一种标签，无需平衡
        if len(label_groups) <= 1:
            logger.warning("警告: 只有一种标签，无需平衡")
            return sequences
        
        # 设置随机种子
        random.seed(random_state)
        
        # 平衡序列
        balanced_sequences = []
        
        if balance_method == "undersample":
            # 欠采样：减少多数类
            logger.info("使用欠采样策略平衡序列...")
            
            # 找出最少的非零标签序列数量
            non_zero_labels = [label for label in label_counts if label != 0]
            if non_zero_labels:
                min_count = min(label_counts[label] for label in non_zero_labels)
                
                # 计算良性序列的目标数量
                benign_target = int(min_count * target_ratio)
                
                # 对每个标签进行欠采样
                for label, seqs in label_groups.items():
                    if label == 0:  # 良性序列
                        if len(seqs) > benign_target:
                            # 随机采样
                            balanced_sequences.extend(random.sample(seqs, benign_target))
                            logger.info(f"  标签 {label} (BENIGN): 从 {len(seqs)} 减少到 {benign_target}")
                        else:
                            # 全部保留
                            balanced_sequences.extend(seqs)
                            logger.info(f"  标签 {label} (BENIGN): 保留全部 {len(seqs)} 个序列")
                    else:  # 攻击序列
                        if len(seqs) > min_count:
                            # 随机采样
                            balanced_sequences.extend(random.sample(seqs, min_count))
                            logger.info(f"  标签 {label}: 从 {len(seqs)} 减少到 {min_count}")
                        else:
                            # 全部保留
                            balanced_sequences.extend(seqs)
                            logger.info(f"  标签 {label}: 保留全部 {len(seqs)} 个序列")
            else:
                # 如果没有非零标签，直接返回原始序列
                logger.warning("警告: 没有非零标签序列，无法进行欠采样")
                return sequences
                
        elif balance_method == "oversample":
            # 过采样：增加少数类
            logger.info("使用过采样策略平衡序列...")
            
            # 找出最多的非零标签序列数量
            non_zero_labels = [label for label in label_counts if label != 0]
            if non_zero_labels:
                max_count = max(label_counts[label] for label in non_zero_labels)
                
                # 计算良性序列的目标数量
                benign_target = int(max_count * target_ratio)
                
                # 对每个标签进行过采样
                for label, seqs in label_groups.items():
                    if label == 0:  # 良性序列
                        if len(seqs) < benign_target:
                            # 需要过采样
                            additional = []
                            while len(additional) + len(seqs) < benign_target:
                                additional.extend(random.choices(seqs, k=min(benign_target - len(additional) - len(seqs), len(seqs))))
                            
                            balanced_sequences.extend(seqs)
                            balanced_sequences.extend(additional)
                            logger.info(f"  标签 {label} (BENIGN): 从 {len(seqs)} 增加到 {len(seqs) + len(additional)}")
                        else:
                            # 随机采样到目标数量
                            balanced_sequences.extend(random.sample(seqs, benign_target))
                            logger.info(f"  标签 {label} (BENIGN): 从 {len(seqs)} 减少到 {benign_target}")
                    else:  # 攻击序列
                        if len(seqs) < max_count:
                            # 需要过采样
                            additional = []
                            while len(additional) + len(seqs) < max_count:
                                additional.extend(random.choices(seqs, k=min(max_count - len(additional) - len(seqs), len(seqs))))
                            
                            balanced_sequences.extend(seqs)
                            balanced_sequences.extend(additional)
                            logger.info(f"  标签 {label}: 从 {len(seqs)} 增加到 {len(seqs) + len(additional)}")
                        else:
                            # 全部保留
                            balanced_sequences.extend(seqs)
                            logger.info(f"  标签 {label}: 保留全部 {len(seqs)} 个序列")
            else:
                # 如果没有非零标签，直接返回原始序列
                logger.warning("警告: 没有非零标签序列，无法进行过采样")
                return sequences
                
        elif balance_method == "synthetic":
            # 合成：使用合成数据平衡
            logger.info("使用合成数据策略平衡序列...")
            
            # 找出最多的非零标签序列数量
            non_zero_labels = [label for label in label_counts if label != 0]
            if non_zero_labels:
                max_count = max(label_counts[label] for label in non_zero_labels)
                
                # 计算良性序列的目标数量
                benign_target = int(max_count * target_ratio)
                
                # 对良性序列进行处理
                if 0 in label_groups:
                    benign_seqs = label_groups[0]
                    if len(benign_seqs) < benign_target:
                        # 需要合成
                        synthetic_count = benign_target - len(benign_seqs)
                        logger.info(f"  合成 {synthetic_count} 个良性序列")
                        
                        # 合成良性序列
                        synthetic_benign = self.create_synthetic_sequences(synthetic_count)
                        
                        balanced_sequences.extend(benign_seqs)
                        balanced_sequences.extend(synthetic_benign)
                        logger.info(f"  标签 0 (BENIGN): 从 {len(benign_seqs)} 增加到 {len(benign_seqs) + len(synthetic_benign)}")
                    else:
                        # 随机采样到目标数量
                        balanced_sequences.extend(random.sample(benign_seqs, benign_target))
                        logger.info(f"  标签 0 (BENIGN): 从 {len(benign_seqs)} 减少到 {benign_target}")
                
                # 对非零标签序列进行处理
                for label in non_zero_labels:
                    seqs = label_groups[label]
                    if len(seqs) < max_count:
                        # 需要过采样（对于攻击序列，使用过采样而不是合成）
                        additional = []
                        while len(additional) + len(seqs) < max_count:
                            additional.extend(random.choices(seqs, k=min(max_count - len(additional) - len(seqs), len(seqs))))
                        
                        balanced_sequences.extend(seqs)
                        balanced_sequences.extend(additional)
                        logger.info(f"  标签 {label}: 从 {len(seqs)} 增加到 {len(seqs) + len(additional)}")
                    else:
                        # 全部保留
                        balanced_sequences.extend(seqs)
                        logger.info(f"  标签 {label}: 保留全部 {len(seqs)} 个序列")
            else:
                # 如果没有非零标签，直接返回原始序列
                logger.warning("警告: 没有非零标签序列，无法进行合成平衡")
                return sequences
        else:
            # 未知的平衡方法，直接返回原始序列
            logger.warning(f"警告: 未知的平衡方法 '{balance_method}'，不进行平衡")
            return sequences
        
        # 打印平衡后的分布
        balanced_label_counts = Counter([seq[1] for seq in balanced_sequences])
        logger.info("平衡后的标签分布:")
        for label, count in sorted(balanced_label_counts.items()):
            stage_name = self.config.get("stage_mapping", {}).get(label, f"未知标签 {label}")
            logger.info(f"  标签 {label} ({stage_name}): {count} 个序列")
        
        return balanced_sequences
    
    def pad_and_mask(self, seq):
        """
        将 seq pad 到 window_size（或至少 min_len），并返回 (pad_seq, mask)
        mask 中 1 表示真实数据，0 表示 pad
        """
        real_len = len(seq)
        target_len = max(real_len, self.min_len)
        target_len = min(target_len, self.window_size)
        seq = seq[:target_len]
        mask = [1] * len(seq)
        if len(seq) < target_len:
            pad_len = target_len - len(seq)
            seq = seq + [self.pad_val] * pad_len
            mask = mask + [0] * pad_len
        if target_len < self.window_size:
            extra_pad = self.window_size - target_len
            seq = seq + [self.pad_val] * extra_pad
            mask = mask + [0] * extra_pad
        return seq, mask
    
    def sliding_window_sequences(self, flows, label_func=None):
        """
        flows: 按时间排序的事件列表（如特征ID或向量）
        label_func: 决定标签的函数，默认为窗口内最多标签
        返回: list of (seq, mask, label)
        """
        sequences, masks, labels = [], [], []
        if label_func is None:
            def label_func(window):
                from collections import Counter
                cnt = Counter(window)
                return cnt.most_common(1)[0][0]
        for start in range(0, len(flows), self.stride):
            window = flows[start : start + self.window_size]
            if not window:
                break
            seq, mask = self.pad_and_mask(window)
            label = label_func(window)
            sequences.append(seq)
            masks.append(mask)
            labels.append(label)
        return sequences, masks, labels
    
    def event_level_sliding_window(self, events, need_mixed=False, min_len=None, overlap_rate=None):
        """
        事件级滑窗生成器。可选只保留窗口内有多个不同阶段的窗口。
        events: DataFrame，按时间排序的事件
        need_mixed: True时只保留窗口内有多个不同阶段的窗口
        min_len: 最小窗口长度，默认self.window_size
        overlap_rate: 滑窗重叠率，默认self.overlap_rate
        返回：窗口列表，每个窗口为DataFrame
        """
        if min_len is None:
            min_len = self.window_size
        if overlap_rate is None:
            overlap_rate = getattr(self, 'overlap_rate', 0.2)
        
        return self.build_event_windows(events)
    
    def build_event_windows(self, df: pd.DataFrame):
        """
        对单个会话按事件滑窗：
         - 如果事件数 >= window_size：正常滑窗（可 overlap）
         - 如果事件数 <  window_size：补齐一次，保证至少产出一个窗口
        """
        events = df.sort_values(self.timestamp_col).reset_index(drop=True)
        N = len(events)
        
        # 1) 对短序列，做一次 padding / repeat
        if N < self.window_size:
            # 复制原事件 + pad 行 = window_size
            pad_cnt = self.window_size - N
            
            # 如果你希望用最后一条事件 label 继续填充，可先复制：
            tail = events.iloc[[-1]].copy()
            tail[self.session_col] = events[self.session_col].iloc[0]
            # 这里复制 tail pad_cnt 次
            pads = pd.concat([tail]*pad_cnt, ignore_index=True)
            
            padded = pd.concat([events, pads], ignore_index=True)
            return [padded]  # 保证至少有一个窗口

        # 2) 对足够长的序列，按 overlap_rate 做滑窗
        step = max(1, int(self.window_size * (1 - self.overlap_rate)))
        windows = []
        for start in range(0, N - self.window_size + 1, step):
            win = events.iloc[start : start + self.window_size]
            windows.append(win)
        return windows
    
    def group_by_attack_stages(self, df):
        """
        按攻击阶段组合分组
        
        参数:
        df: 数据框
        
        返回:
        {阶段组合: 会话ID列表}字典
        """
        logger.info("按攻击阶段组合分组...")
        session_stages = {}
        stage_groups = {}
        
        # 获取每个会话的攻击阶段
        for session_id in df[self.session_col].unique():
            session_data = df[df[self.session_col] == session_id]
            stages = sorted(session_data[self.label_col].unique())
            # 转为元组作为字典键
            stage_key = tuple(stages)
            session_stages[session_id] = stages
            
            if stage_key not in stage_groups:
                stage_groups[stage_key] = []
            stage_groups[stage_key].append(session_id)
        
        # 打印分组统计
        logger.info(f"找到 {len(stage_groups)} 种不同的攻击阶段组合:")
        for stages, sessions in stage_groups.items():
            stage_names = []
            for stage in stages:
                if stage == 0:
                    stage_names.append("BENIGN")
                else:
                    stage_mapping = self.stage_mapping
                    stage_names.append(stage_mapping.get(stage, f"Stage-{stage}"))
            
            stage_str = "->".join(map(str, stages))
            stage_name_str = "+".join(stage_names)
            logger.info(f"  {stage_str} ({stage_name_str}): {len(sessions)} 个会话")
        
        return stage_groups
    
    def merge_sessions_by_stages(self, df, stage_groups):
        """
        合并同阶段组合的会话
        
        参数:
        df: 数据框
        stage_groups: {阶段组合: 会话ID列表}字典
        
        返回:
        {阶段组合: 合并DataFrame}字典
        """
        logger.info("合并同阶段组合的会话...")
        merged_groups = {}
        
        for stages, sessions in stage_groups.items():
            # 筛选这些会话的数据
            group_data = df[df[self.session_col].isin(sessions)].copy()
            
            # 按时间排序
            if self.timestamp_col in group_data.columns:
                try:
                    group_data = group_data.sort_values(by=self.timestamp_col)
                except Exception as e:
                    logger.warning(f"按时间排序失败: {e}，将使用原始顺序")
            
            # 保存合并后的数据
            merged_groups[stages] = group_data
            logger.info(f"  阶段组合 {stages}: 合并了 {len(sessions)} 个会话，共 {len(group_data)} 条事件")
        
        return merged_groups

    def split_time_sessions(self, grp: pd.DataFrame):
        """
        对同一实体（IP对）按时间阈值切分出"攻击会话"
        
        参数:
        grp: 同一实体的事件组
        
        返回:
        切分后的会话列表，每个会话是一个DataFrame
        """
        # 确保时间戳列为日期时间类型
        try:
            if not pd.api.types.is_datetime64_dtype(grp[self.timestamp_col]):
                grp[self.timestamp_col] = pd.to_datetime(grp[self.timestamp_col])
        except Exception as e:
            logger.warning(f"转换时间戳类型失败: {e}")
            return [grp]  # 如果转换失败，直接返回原始组
        
        # 按时间排序
        df = grp.sort_values(self.timestamp_col).reset_index(drop=True)
        sessions = []
        last_ts = df[self.timestamp_col].iloc[0]
        buf = [df.iloc[0]]
        
        for i in range(1, len(df)):
            now = df[self.timestamp_col].iloc[i]
            time_diff = (now - last_ts).total_seconds()
            
            if time_diff > self.time_gap_threshold:
                sessions.append(pd.DataFrame(buf))
                buf = []
            
            buf.append(df.iloc[i])
            last_ts = now
        
        # 添加最后一个会话
        if buf:
            sessions.append(pd.DataFrame(buf))
        
        return sessions

    def build_time_based_sessions(self, data: pd.DataFrame):
        """
        遍历所有实体，先切时间，再返回一列表的DataFrame
        
        参数:
        data: 原始数据框
        
        返回:
        切分后的会话列表，每个会话是一个DataFrame
        """
        all_chunks = []
        entity_keys = self.entity_key
        
        # 确保entity_keys是列表
        if isinstance(entity_keys, str):
            entity_keys = [entity_keys]
        
        # 检查所有实体键是否在数据框中
        missing_keys = [key for key in entity_keys if key not in data.columns]
        if missing_keys:
            logger.warning(f"以下实体键不在数据框中: {missing_keys}")
            available_keys = [key for key in entity_keys if key in data.columns]
            if not available_keys:
                logger.error("没有有效的实体键，无法按实体分组")
                return [data]  # 如果没有有效的实体键，直接返回原始数据作为一个会话
            entity_keys = available_keys
            logger.info(f"将使用有效的实体键: {entity_keys}")
        
        # 按实体分组并切分时间会话
        for _, grp in data.groupby(entity_keys):
            chunks = self.split_time_sessions(grp)
            all_chunks.extend(chunks)
        
        logger.info(f"按实体和时间切分出 {len(all_chunks)} 个会话")
        
        # 打印会话长度分布
        session_lengths = [len(chunk) for chunk in all_chunks]
        if session_lengths:
            min_len = min(session_lengths)
            max_len = max(session_lengths)
            avg_len = sum(session_lengths) / len(session_lengths)
            logger.info(f"会话长度统计: 最小={min_len}, 最大={max_len}, 平均={avg_len:.2f}")
            
            # 统计不同长度的会话数量
            length_bins = {}
            for length in session_lengths:
                bin_key = (length // 5) * 5  # 按5个为一组进行分组
                length_bins[bin_key] = length_bins.get(bin_key, 0) + 1
            
            logger.info("会话长度分布:")
            for bin_key, count in sorted(length_bins.items()):
                logger.info(f"  长度 {bin_key}-{bin_key+4}: {count} 个会话")
        
        return all_chunks

    def categorize_chain(self, df: pd.DataFrame):
        """
        根据标签序列分类APT攻击链
        
        参数:
        df: 会话数据框
        
        返回:
        APT类别（APT1-APT5）
        """
        # 获取会话中的唯一标签
        labs = df[self.label_col].unique().tolist()
        
        # 如果全是良性(0)，直接返回BENIGN
        if set(labs) == {0}:
            return "BENIGN"
        
        # 按阶段顺序排序标签
        labs_sorted = sorted([lab for lab in labs if lab != 0], key=lambda x: self.stage_order.get(x, 999))
        
        # 如果没有攻击标签，返回BENIGN
        if not labs_sorted:
            return "BENIGN"
        
        # 按照攻击阶段组合分类
        if labs_sorted == [1, 2, 3, 4]:
            return "APT4"  # 完整攻击链
        elif labs_sorted == [1, 2, 3]:
            return "APT3"  # 侦察+立足+横向移动
        elif labs_sorted == [1, 2]:
            return "APT2"  # 侦察+立足
        elif labs_sorted == [1]:
            return "APT1"  # 只有侦察
        else:
            return "APT5"  # 其他（缺阶段、顺序乱、只有2/3/4等）

    def pad_or_crop(self, df: pd.DataFrame):
        """
        对会话进行填充或截断，使其达到指定长度
        
        参数:
        df: 会话数据框
        
        返回:
        处理后的会话数据框
        """
        max_len = self.max_sequence_length
        
        # 如果长度已经符合要求，直接返回
        if len(df) == max_len:
            return df
        
        # 如果需要截断
        if len(df) > max_len:
            # 默认保留前max_len行
            return df.iloc[:max_len].copy()
        
        # 如果需要填充
        if len(df) < max_len:
            # 复制最后一行进行填充
            pad_count = max_len - len(df)
            tail = df.iloc[[-1]].copy()
            pads = pd.concat([tail] * pad_count, ignore_index=True)
            return pd.concat([df, pads], ignore_index=True)

    def generate_apt_sequences(self, data: pd.DataFrame):
        """
        生成APT攻击序列
        
        参数:
        data: 原始数据框
        
        返回:
        (apt_type, sequence)元组列表
        """
        logger.info("使用实体+时间切分方法生成APT攻击序列...")
        
        # 1) 切出所有按实体＋时间分出的"会话片段"
        sessions = self.build_time_based_sessions(data)
        
        # 2) 按每段事件数 >= min_len 才继续
        sessions = [s for s in sessions if len(s) >= self.min_len]
        logger.info(f"过滤后剩余 {len(sessions)} 个长度≥{self.min_len}的会话")
        
        # 3) 给每段打上APT类别
        records = []
        for sess in sessions:
            apt_type = self.categorize_chain(sess)
            # 记录原始会话的信息
            sess_info = {
                "length": len(sess),
                "labels": sorted(sess[self.label_col].unique().tolist()),
                "apt_type": apt_type
            }
            
            # 只保留攻击序列（APT1-5），排除BENIGN
            if apt_type.startswith("APT"):
                # pad/truncate到max_sequence_length
                processed_sess = self.pad_or_crop(sess)
                records.append((apt_type, processed_sess, sess_info))
        
        # 4) 输出统计
        from collections import Counter
        cnt = Counter([t for t, _, _ in records])
        logger.info("==== APT类型分布 ====")
        for k, v in sorted(cnt.items()):
            logger.info(f"  {k}: {v} 条序列")
        
        # 计算每种APT类型的平均长度
        apt_lengths = {}
        for apt_type, _, info in records:
            if apt_type not in apt_lengths:
                apt_lengths[apt_type] = []
            apt_lengths[apt_type].append(info["length"])
        
        logger.info("==== 各APT类型的平均长度 ====")
        for apt_type, lengths in sorted(apt_lengths.items()):
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            logger.info(f"  {apt_type}: 平均 {avg_len:.2f} 事件/序列")
        
        return records


# 添加主函数，使得当这个文件被直接运行时，能够执行一些测试操作
def main():
    logger.info("=" * 80)
    logger.info("APT序列构建器测试")
    logger.info("=" * 80)
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # 获取工作空间根目录（C:\Users\123\.cursor-tutor）
    workspace_root = os.path.abspath(os.path.join(project_root, '..'))
    
    # 打印路径信息
    logger.info(f"当前目录: {current_dir}")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"工作空间根目录: {workspace_root}")
    
    # 尝试加载配置文件
    config_path = os.path.join(project_root, 'configs', 'config.json')
    logger.info(f"配置文件路径: {config_path}")
    
    if os.path.exists(config_path):
        logger.info(f"找到配置文件: {config_path}")
        # 直接加载配置文件内容作为字典
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 添加关键列名检查
            logger.info("\n配置中的关键列名:")
            logger.info(f"会话ID列名: {config.get('session_column', '未指定')}")
            logger.info(f"APT标识列名: {config.get('apt_column', '未指定')}")
            logger.info(f"APT类别列名: {config.get('apt_category_column', '未指定')}")
            
            builder = APTSequenceBuilder(config)
            logger.info("已使用配置文件初始化APT序列构建器")
        except Exception as e:
            logger.error(f"加载配置文件时出错: {e}")
            logger.info("使用默认配置初始化APT序列构建器")
            builder = APTSequenceBuilder({})
    else:
        logger.info(f"未找到配置文件: {config_path}")
        logger.info("使用默认配置初始化APT序列构建器")
        builder = APTSequenceBuilder({})
    
    # 打印基本信息
    logger.info("\n基本配置信息:")
    logger.info(f"特征列: {builder.feature_columns}")
    logger.info(f"最大序列长度: {builder.max_sequence_length}")
    logger.info(f"阶段映射: {builder.stage_mapping}")
    logger.info(f"最小良性会话数: {builder.min_benign_sessions}")
    logger.info(f"合成良性序列数量: {builder.benign_sequence.get('synthetic_count', 0)}")
    logger.info(f"类别平衡: {builder.balance_classes}")
    logger.info(f"平衡策略: {builder.balance_strategy}")
    logger.info(f"目标比例: {builder.target_ratio}")
    
    # 尝试加载示例数据（从工作空间根目录）
    apt_sequences_path = os.path.join(workspace_root, 'apt_sequences.csv')
    clean_csv_path = os.path.join(workspace_root, 'clean_dapt.csv')
    if os.path.exists(clean_csv_path):
        logger.info(f"优先加载clean_dapt.csv: {clean_csv_path}")
        df = pd.read_csv(clean_csv_path)
    elif os.path.exists(apt_sequences_path):
        logger.info(f"加载默认示例数据: {apt_sequences_path}")
        df = pd.read_csv(apt_sequences_path)
    else:
        logger.warning(f"\n未找到示例数据: {apt_sequences_path} 也未找到 {clean_csv_path}")
        logger.info("无法构建序列，但APT序列构建器已成功初始化")
        return
    
    # 检查必要的列是否存在
    session_column = builder.session_col
    apt_column = builder.label_col
    
    # === 自动调试：打印每个会话的事件数分布 ===
    session_counts = df.groupby(session_column).size()
    print("每个会话的事件数分布（前20个）:", session_counts.sort_values(ascending=False).head(20).to_dict())
    print("事件数小于window_size的会话数:", (session_counts < builder.window_size).sum())
    print("事件数大于等于window_size的会话数:", (session_counts >= builder.window_size).sum())
    
    missing_columns = []
    if session_column not in df.columns:
        missing_columns.append(session_column)
    if apt_column not in df.columns:
        missing_columns.append(apt_column)
        
    if missing_columns:
        logger.warning(f"警告: 数据中缺少必要的列: {missing_columns}")
        logger.info("无法构建序列")
        return
    else:
        logger.info("数据包含必要的列，可以构建序列")
        # 创建输出目录
        output_dir = os.path.join(project_root, 'data', 'sequences')
        os.makedirs(output_dir, exist_ok=True)
        # 构建序列
        logger.info("\n开始构建序列...")
        try:
            all_seqs, all_labels, stats = builder.build_sequences(df, output_dir, need_mixed=True)
            logger.info("\n序列构建完成")
            # 彻底删除所有"=== 分类统计结果 ==="、BENIGN/APT1等相关print输出
            # 只保留APT阶段组合统计和良性会话数输出（已在build_sequences内部输出）
        except Exception as e:
            logger.error(f"构建序列时出错: {e}")
            import traceback
            traceback.print_exc()
    logger.info("\n测试完成")

if __name__ == "__main__":
    main() 