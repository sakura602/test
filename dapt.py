import os
import pandas as pd
import numpy as np
import random
# import re # No longer needed for filename extraction
import time
from collections import defaultdict
# from sklearn.preprocessing import LabelEncoder # Might be useful later if encoding features
from sklearn.feature_selection import mutual_info_classif
import warnings
import json  # For potentially saving map

# Ignore SettingWithCopyWarning for cleaner output during per-stage processing
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Define standard column names based on user provided list
USER_PROVIDED_COLUMNS = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration',
    'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
    'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
    'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Activity', 'Stage'  # User confirmed 'Stage' is the label column
]

# Mapping from original stage names to internal keys used for sequence generation
# And reverse mapping
INTERNAL_STAGE_MAP = {
    'Reconnaissance': 'S1',
    'Establish Foothold': 'S2',
    'Lateral Movement': 'S3',
    'Data Exfiltration': 'S4',  # 对应 "窃取信息或破坏系统"
    # 'Persistence'/'Cleanup' or similar for S5? - Not found in this run's data
    '正常流量': 'SN'  # 统一后的正常流量
}
REVERSE_STAGE_MAP = {v: k for k, v in INTERNAL_STAGE_MAP.items()}


class DAPTPreprocessor:
    def __init__(self, csv_dir_path, output_path):
        if not os.path.isdir(csv_dir_path):
            raise ValueError(f"Provided path is not a valid directory: {csv_dir_path}")
        self.csv_dir_path = csv_dir_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)  # Ensure output dir exists
        self.data_frames = {}  # Stores DataFrames keyed by unified stage name (e.g., '侦察', '正常流量')
        self.selected_features = None  # List of feature names after GLOBAL cleaning/selection
        self.dimension_tracking = {}  # Tracks dimensions per stage: {'StageName': {'Original': X, 'Cleaned': Y, 'Selected': Z}}

        # Sequence related attributes
        self.apt_sequences_labels = []  # List of lists containing stage labels (e.g., ['S1', 'SN', 'S2', ...])
        self.normal_sequences_labels = []  # List of lists containing stage labels (e.g., ['SN', 'SN', ...])
        self.apt_sequences_data = []  # List of sequence data (e.g., list of numpy arrays or dataframes)
        self.normal_sequences_data = []  # List of sequence data
        self.attack2id = {}  # Mapping from internal stage label ('S1'-SN') to integer ID
        self.apt_sequences_ids = []  # List of lists containing integer IDs for APT seqs
        self.normal_sequences_ids = []  # List of lists containing integer IDs for Normal seqs
        self.apt_labels = []  # Final labels (1-5) for APT sequences
        self.normal_labels = []  # Final labels (0) for Normal sequences

        # Define key features to exclude from correlation-based removal
        # Updated based on user request
        self.key_features = [
            'Dst Port', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
            'Src IP', 'Dst IP', 'Total Length of Fwd Packet', 'Timestamp'
        ]

        # 定义语义特征编码映射
        self.semantic_mappings = {
            'scan_intensity': {'low': 0, 'medium': 1, 'high': 2},
            'probe_pattern': {'service': 0, 'random': 1, 'system': 2},
            'collection_level': {'low': 0, 'medium': 1, 'high': 2},
            'target_service': {'web': 0, 'apt_specific': 1, 'system': 2, 'high_port': 3},
            'connection_mode': {'single': 0, 'burst': 1, 'persistent': 2},
            'movement_pattern': {'direct': 0, 'indirect': 1, 'complex': 2},
            'session_duration': {'short': 0, 'medium': 1, 'long': 2, 'very_long': 3},
            'exfil_volume': {'small': 0, 'medium': 1, 'large': 2, 'massive': 3},
            'transfer_mode': {'continuous': 0, 'burst': 1, 'stealth': 2}
        }  # Note: IP addresses and Timestamp might be non-numeric or require special handling
        self.stage_column = 'Stage'  # Define the column containing stage labels
        self.unified_normal_label = '正常流量'

    # REMOVED _extract_stage_from_filename as partitioning is now based on 'Stage' column

    def _load_data_and_partition(self):
        """Loads all CSVs, concatenates, unifies normal labels, and partitions by 'Stage'."""
        print("--- Loading and Partitioning Data ---")
        all_data_list = []
        found_files = 0
        for filename in os.listdir(self.csv_dir_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.csv_dir_path, filename)
                print(f"Loading data from {filename}...")
                try:
                    # Attempt to load with header=0 first
                    df_chunk = pd.read_csv(file_path, header=0, low_memory=False)
                    df_chunk.columns = df_chunk.columns.str.strip()  # Clean column names

                    # Simple check: Does it have roughly the expected number of columns?
                    # Or check if 'Stage' column exists?
                    if self.stage_column not in df_chunk.columns:
                        print(
                            f"  Warning: File {filename} might be missing headers or the '{self.stage_column}' column. Attempting load with predefined columns.")
                        df_chunk = pd.read_csv(file_path, header=None, names=USER_PROVIDED_COLUMNS, low_memory=False)
                        df_chunk.columns = df_chunk.columns.str.strip()  # Clean again

                    if self.stage_column not in df_chunk.columns:
                        print(
                            f"  Error: Could not find '{self.stage_column}' column in {filename} even with predefined names. Skipping file.")
                        continue

                    # Special handling for preprocessed data - convert numeric labels to text if needed
                    if df_chunk[self.stage_column].dtype in [np.int64, np.float64]:
                        print(f"  Detected numeric '{self.stage_column}' values. Converting to text labels...")
                        # Create reverse mapping from numeric to text labels
                        label_map = {
                            0: "正常流量",  # Normal traffic
                            1: "Reconnaissance",
                            2: "Establish Foothold",
                            3: "Lateral Movement",
                            4: "Data Exfiltration"
                        }
                        df_chunk[self.stage_column] = df_chunk[self.stage_column].map(label_map)
                        print("  Mapped numeric labels back to text labels.")

                    all_data_list.append(df_chunk)
                    found_files += 1
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")

        if not all_data_list:
            raise RuntimeError("No valid CSV data loaded. Check directory path and files.")

        # Concatenate all loaded data
        print("Concatenating loaded data...")
        combined_df = pd.concat(all_data_list, ignore_index=True)
        print(f"Total samples loaded: {len(combined_df)}")

        # Pre-partitioning check for stage column
        if self.stage_column not in combined_df.columns:
            raise ValueError(f"The required '{self.stage_column}' column was not found in the combined data.")

        # --- Unify Normal Labels --- #
        normal_labels_to_unify = ['benign', 'BENIGN', 'Benign', 0]  # Add others if needed, including numeric 0
        original_stages = combined_df[self.stage_column].unique()
        print(f"Original stages found: {original_stages}")
        replace_map = {label: self.unified_normal_label for label in normal_labels_to_unify if label in original_stages}
        if replace_map:
            print(f"Unifying normal labels: Mapping {list(replace_map.keys())} to '{self.unified_normal_label}'")
            combined_df[self.stage_column] = combined_df[self.stage_column].replace(replace_map)

        # Replace infinite values with NaN globally
        combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Partition based on the 'Stage' column
        stages_found = combined_df[self.stage_column].unique()
        print(f"Stages after unification for partitioning: {stages_found}")

        for stage_name in stages_found:
            # Ensure stage_name is treated as string, handle potential NaN stage names if necessary
            if pd.isna(stage_name):
                print(f"  Warning: Found NaN value in '{self.stage_column}'. Skipping these rows.")
                continue
            stage_name_str = str(stage_name)

            stage_df = combined_df[combined_df[
                                       self.stage_column] == stage_name].copy()  # Use .copy() to avoid SettingWithCopyWarning later
            if not stage_df.empty:
                print(f"  Partitioning data for stage: '{stage_name_str}' ({len(stage_df)} samples)")
                self.data_frames[stage_name_str] = stage_df
                # Initialize dimension tracking for this stage
                self.dimension_tracking[stage_name_str] = {
                    'Original': stage_df.shape[1],  # Record original dimension
                    'Cleaned': 0,
                    'Selected': 0
                }
            else:
                print(f"  Warning: No data found for stage '{stage_name_str}' after filtering.")

        if not self.data_frames:
            raise RuntimeError("No dataframes were created after partitioning by 'Stage'. Check 'Stage' column values.")
        print(f"Successfully partitioned data into stages: {list(self.data_frames.keys())}")
        print("-" * 30)

    def _clean_data(self):
        """Basic data cleaning: remove non-numeric columns and handle missing values."""
        print("--- Data Cleaning ---")

        for stage_name, df in self.data_frames.items():
            original_shape = df.shape

            # Convert all columns to numeric except Stage
            for col in df.columns:
                if col != 'Stage':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill missing values with 0
            df = df.fillna(0)

            # Remove non-numeric columns except Stage
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Stage' in df.columns:
                numeric_cols.append('Stage')
            df = df[numeric_cols]

            self.data_frames[stage_name] = df
            print(f"{stage_name}: {original_shape} -> {df.shape}")

        print("-" * 30)






    def _advanced_feature_selection(self):
        """
        Advanced feature selection based on variance, sparsity, and correlation analysis.
        """
        print("--- Advanced Feature Selection ---")

        # Combine all data for global analysis
        combined_data = pd.concat([df for df in self.data_frames.values()], ignore_index=True)
        combined_data = combined_data.drop('Stage', axis=1)

        # Convert to numeric
        for col in combined_data.columns:
            combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
        combined_data = combined_data.fillna(0)

        original_features = len(combined_data.columns)
        print(f"Original features: {original_features}")

        # 1. Remove zero variance features
        zero_var_features = combined_data.columns[combined_data.var() == 0].tolist()
        combined_data = combined_data.drop(zero_var_features, axis=1)
        print(f"Removed {len(zero_var_features)} zero variance features")

        # 2. Remove low variance features (< 0.01)
        low_var_features = combined_data.columns[combined_data.var() < 0.01].tolist()
        combined_data = combined_data.drop(low_var_features, axis=1)
        print(f"Removed {len(low_var_features)} low variance features (< 0.01)")

        # 3. Remove high sparsity features (> 0.5)
        sparsity = (combined_data == 0).mean()
        high_sparse_features = sparsity[sparsity > 0.5].index.tolist()
        combined_data = combined_data.drop(high_sparse_features, axis=1)
        print(f"Removed {len(high_sparse_features)} high sparsity features (> 0.5)")

        # 4. Remove highly correlated features (> 0.9)
        if len(combined_data.columns) > 1:
            corr_matrix = combined_data.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
            combined_data = combined_data.drop(high_corr_features, axis=1)
            print(f"Removed {len(high_corr_features)} highly correlated features (> 0.9)")

        self.selected_features = combined_data.columns.tolist()
        final_features = len(self.selected_features)
        print(f"Final features: {final_features} (reduced by {original_features - final_features})")

        # Apply to all stage dataframes
        for stage_name, df in self.data_frames.items():
            cols_to_keep = [col for col in self.selected_features if col in df.columns] + ['Stage']
            self.data_frames[stage_name] = df[cols_to_keep]

    def _print_dimension_summary(self):
        """Print simplified dimension summary."""
        print("--- Feature Dimension Summary ---")
        for stage_name, df in self.data_frames.items():
            feature_count = len([col for col in df.columns if col != 'Stage'])
            print(f"{stage_name}: {feature_count} features")

        # 检查特征集类型
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            print(f"Selected features: {len(self.selected_features)} (unified)")
        elif hasattr(self, 'stage_selected_features'):
            total_features = sum(len(features) for features in self.stage_selected_features.values())
            print(f"Stage-specific features: {total_features} (independent)")
        else:
            print("Feature selection: Not available")
        print("-" * 30)

    def _generate_dimension_table(self):
        """生成并打印按阶段特征选择的维度变化汇总表"""
        print("--- 按阶段特征选择维度变化汇总 ---")
        if not self.dimension_tracking:
            print("没有可用的维度跟踪信息。")
            return

        # 按照攻击阶段顺序排列
        stage_order_keys = list(INTERNAL_STAGE_MAP.keys())
        found_stages = list(self.dimension_tracking.keys())
        ordered_stages = [s for s in stage_order_keys if s in found_stages] + \
                         [s for s in found_stages if s not in stage_order_keys]

        print("各阶段特征维度变化:")
        for stage_name in ordered_stages:
            dims = self.dimension_tracking.get(stage_name, {})
            print(f"\n{stage_name}:")
            print(f"  原始特征: {dims.get('Original', 'N/A')}")
            print(f"  清洗后特征: {dims.get('Cleaned', 'N/A')}")
            print(f"  最终选择特征: {dims.get('Selected', 'N/A')}")

            # 显示该阶段独有的特征选择过程
            if 'Stage_Selection' in dims:
                stage_sel = dims['Stage_Selection']
                print(f"  该阶段特征选择过程:")
                print(f"    可选特征数: {stage_sel.get('available_for_selection', 'N/A')}")
                print(f"    零方差筛选后: {stage_sel.get('after_zero_var_removal', 'N/A')}")
                print(f"    低方差筛选后: {stage_sel.get('after_low_var_removal', 'N/A')}")
                print(f"    稀疏性筛选后: {stage_sel.get('after_sparse_removal', 'N/A')}")
                print(f"    相关性筛选后: {stage_sel.get('after_corr_removal', 'N/A')}")
                print(f"    该阶段最终选择: {stage_sel.get('final_stage_selected', 'N/A')}")

            if 'Stage_Specific_Features' in dims:
                print(f"  该阶段独有特征数: {dims['Stage_Specific_Features']}")

        # 显示全局汇总
        print(f"\n全局特征选择汇总:")
        if hasattr(self, 'stage_selected_features'):
            all_stage_features = set()
            for features in self.stage_selected_features.values():
                all_stage_features.update(features)
            print(f"  各阶段选择特征并集: {len(all_stage_features)} 个")

        key_features_count = len([f for f in self.key_features if f in list(self.data_frames.values())[0].columns])
        print(f"  关键特征数: {key_features_count} 个")

        # 检查是否使用统一特征集或各阶段独立特征集
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            # 统一特征集模式
            print(f"  最终特征总数: {len(self.selected_features)} 个（统一特征集）")
        elif hasattr(self, 'stage_selected_features'):
            # 各阶段独立特征集模式
            total_features = sum(len(features) for features in self.stage_selected_features.values())
            print(f"  各阶段特征总数: {total_features} 个（各阶段独立特征集）")

        print("-" * 30)

    def _analyze_dataset_characteristics(self):
        """分析数据集特征，包括数据分布、特征统计等"""
        print("--- 数据集特征分析 ---")

        # 分析每个攻击阶段的数据分布
        print("各攻击阶段数据分布:")
        for stage_name, df in self.data_frames.items():
            print(f"  {stage_name}: {len(df)} 样本")

        # 分析特征类型和分布
        print("\n特征类型分析:")
        if self.data_frames:
            sample_df = list(self.data_frames.values())[0]
            numeric_features = []
            categorical_features = []

            for col in sample_df.columns:
                if col != 'Stage':
                    if sample_df[col].dtype in ['int64', 'float64']:
                        numeric_features.append(col)
                    else:
                        categorical_features.append(col)

            print(f"  数值特征: {len(numeric_features)} 个")
            print(f"  分类特征: {len(categorical_features)} 个")

            # 显示前10个数值特征
            if numeric_features:
                print(f"  前10个数值特征: {numeric_features[:10]}")

        print("-" * 30)

    def _key_feature_analysis(self):
        """关键特征分析 - 分析目标端口号、传输时间、总发送包数和总接收包数等关键特征"""
        print("--- 关键特征分析 ---")

        # 定义关键特征指标
        key_feature_indicators = {
            'Dst Port': '目标端口号',
            'Flow Duration': '传输时间',
            'Total Fwd Packet': '总发送包数',
            'Total Bwd packets': '总接收包数',
            'Timestamp': '时间戳',
            'Src IP': '源IP地址',
            'Dst IP': '目标IP地址'
        }

        print("关键特征指标分析:")
        combined_data = pd.concat([df for df in self.data_frames.values()], ignore_index=True)

        for feature, description in key_feature_indicators.items():
            if feature in combined_data.columns:
                print(f"\n{description} ({feature}):")
                if combined_data[feature].dtype in ['int64', 'float64']:
                    # 数值特征统计
                    stats = combined_data[feature].describe()
                    print(f"  均值: {stats['mean']:.2f}")
                    print(f"  标准差: {stats['std']:.2f}")
                    print(f"  最小值: {stats['min']:.2f}")
                    print(f"  最大值: {stats['max']:.2f}")
                    print(f"  缺失值: {combined_data[feature].isnull().sum()}")
                else:
                    # 分类特征统计
                    unique_count = combined_data[feature].nunique()
                    print(f"  唯一值数量: {unique_count}")
                    print(f"  缺失值: {combined_data[feature].isnull().sum()}")
                    if unique_count <= 20:  # 只显示少量唯一值
                        print(f"  前10个值: {combined_data[feature].value_counts().head(10).to_dict()}")
            else:
                print(f"\n{description} ({feature}): 特征不存在")

        print("-" * 30)

    def _stage_wise_feature_selection(self):
        """按阶段进行特征选择 - 为每个攻击阶段分别选择最优特征"""
        print("--- 按阶段特征选择过程（排除关键特征指标）---")

        # 存储每个阶段选择的特征
        stage_selected_features = {}
        stage_feature_stats = {}

        # 排除关键特征指标和Stage列
        features_to_exclude = set(self.key_features + ['Stage'])

        print(f"原始特征数量: {len(list(self.data_frames.values())[0].columns) - 1}")  # 减去Stage列
        print(f"关键特征指标数量: {len([f for f in self.key_features if f in list(self.data_frames.values())[0].columns])}")

        # 为每个阶段分别进行特征选择
        for stage_name, df in self.data_frames.items():
            print(f"\n处理阶段: {stage_name} ({len(df)} 样本)")

            # 获取可用于特征选择的特征
            available_features = [col for col in df.columns if col not in features_to_exclude]
            print(f"  可用于特征选择的特征数量: {len(available_features)}")

            # 提取特征数据
            feature_data = df[available_features].copy()

            # 转换为数值类型
            for col in feature_data.columns:
                feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
            feature_data = feature_data.fillna(0)

            original_count = len(feature_data.columns)
            stats = {'original': original_count}

            # 1. 移除零方差特征
            zero_var_features = feature_data.columns[feature_data.var() == 0].tolist()
            feature_data = feature_data.drop(zero_var_features, axis=1)
            stats['zero_var_removed'] = len(zero_var_features)
            print(f"    移除零方差特征: {len(zero_var_features)} 个")

            # 2. 移除低方差特征 (< 0.01)
            low_var_features = feature_data.columns[feature_data.var() < 0.01].tolist()
            feature_data = feature_data.drop(low_var_features, axis=1)
            stats['low_var_removed'] = len(low_var_features)
            print(f"    移除低方差特征 (< 0.01): {len(low_var_features)} 个")

            # 3. 移除高稀疏性特征 (> 0.5)
            sparsity = (feature_data == 0).mean()
            high_sparse_features = sparsity[sparsity > 0.5].index.tolist()
            feature_data = feature_data.drop(high_sparse_features, axis=1)
            stats['high_sparse_removed'] = len(high_sparse_features)
            print(f"    移除高稀疏性特征 (> 0.5): {len(high_sparse_features)} 个")

            # 4. 移除高相关性特征 (> 0.8)
            high_corr_removed = 0
            if len(feature_data.columns) > 1:
                corr_matrix = feature_data.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
                feature_data = feature_data.drop(high_corr_features, axis=1)
                high_corr_removed = len(high_corr_features)
            stats['high_corr_removed'] = high_corr_removed
            print(f"    移除高相关性特征 (> 0.8): {high_corr_removed} 个")

            # 5. 基于方差的特征选择 (选择前30个最高方差的特征)
            if len(feature_data.columns) > 30:
                feature_variances = feature_data.var().sort_values(ascending=False)
                top_variance_features = feature_variances.head(30).index.tolist()
                feature_data = feature_data[top_variance_features]
                stats['variance_selected'] = 30
                print(f"    基于方差选择特征: {len(top_variance_features)} 个")
            else:
                stats['variance_selected'] = len(feature_data.columns)
                print(f"    保留所有剩余特征: {len(feature_data.columns)} 个")

            # 存储该阶段选择的特征
            stage_selected_features[stage_name] = feature_data.columns.tolist()
            stage_feature_stats[stage_name] = stats

            print(f"    {stage_name} 最终选择特征数: {len(feature_data.columns)}")

        # 使用语义特征编码替代传统特征选择
        print(f"\n语义特征编码汇总:")

        # 存储每个阶段的特征选择统计（用于兼容性）
        self.stage_feature_stats = stage_feature_stats
        self.stage_selected_features = {}

        # 为每个阶段提取语义特征
        for stage_name, df in self.data_frames.items():
            # 提取语义特征
            semantic_df = self.extract_semantic_features(df, stage_name)
            self.data_frames[stage_name] = semantic_df

            # 记录语义特征名称
            semantic_features = [col for col in semantic_df.columns if col != 'Stage']
            self.stage_selected_features[stage_name] = semantic_features

            print(f"  {stage_name}: {len(semantic_features)} 个语义特征")
            print(f"    语义特征: {semantic_features}")

        # 不设置统一的selected_features，因为每个阶段都有独立的语义特征集
        self.selected_features = None  # 标记为None，表示使用各阶段独立语义特征

        print("-" * 30)

    def _update_dimension_tracking(self):
        """更新维度跟踪信息 - 包含按阶段特征选择的详细信息"""
        print("--- 更新维度跟踪信息 ---")

        for stage_name, df in self.data_frames.items():
            if stage_name in self.dimension_tracking:
                # 更新清洗后的维度
                self.dimension_tracking[stage_name]['Cleaned'] = df.shape[1] - 1  # 减去Stage列
                # 更新最终选择后的维度
                self.dimension_tracking[stage_name]['Selected'] = len([col for col in df.columns if col != 'Stage'])

                # 添加按阶段特征选择的详细信息
                if hasattr(self, 'stage_feature_stats') and stage_name in self.stage_feature_stats:
                    stats = self.stage_feature_stats[stage_name]
                    self.dimension_tracking[stage_name]['Stage_Selection'] = {
                        'available_for_selection': stats['original'],
                        'after_zero_var_removal': stats['original'] - stats['zero_var_removed'],
                        'after_low_var_removal': stats['original'] - stats['zero_var_removed'] - stats['low_var_removed'],
                        'after_sparse_removal': stats['original'] - stats['zero_var_removed'] - stats['low_var_removed'] - stats['high_sparse_removed'],
                        'after_corr_removal': stats['original'] - stats['zero_var_removed'] - stats['low_var_removed'] - stats['high_sparse_removed'] - stats['high_corr_removed'],
                        'final_stage_selected': stats['variance_selected']
                    }

                # 添加该阶段独有的特征信息
                if hasattr(self, 'stage_selected_features') and stage_name in self.stage_selected_features:
                    stage_specific_features = len(self.stage_selected_features[stage_name])
                    self.dimension_tracking[stage_name]['Stage_Specific_Features'] = stage_specific_features

        print("维度跟踪信息已更新（包含按阶段特征选择详情）")
        print("-" * 30)

    def extract_semantic_features(self, df, stage_name):
        """
        从原始特征中提取攻击语义特征

        Args:
            df: 数据框
            stage_name: 阶段名称

        Returns:
            semantic_df: 包含语义特征的数据框
        """
        print(f"--- 提取 {stage_name} 阶段的语义特征 ---")

        semantic_data = []

        for _, row in df.iterrows():
            if stage_name == 'Reconnaissance':
                semantic_features = self._extract_reconnaissance_semantics(row)
            elif stage_name == 'Establish Foothold':
                semantic_features = self._extract_foothold_semantics(row)
            elif stage_name == 'Lateral Movement':
                semantic_features = self._extract_lateral_semantics(row)
            elif stage_name == 'Data Exfiltration':
                semantic_features = self._extract_exfiltration_semantics(row)
            else:  # Normal traffic
                semantic_features = self._extract_normal_semantics(row)

            semantic_data.append(semantic_features)

        # 创建语义特征数据框
        if stage_name == 'Reconnaissance':
            columns = ['scan_intensity', 'probe_pattern', 'collection_level']
        elif stage_name == 'Establish Foothold':
            columns = ['target_service', 'connection_mode']
        elif stage_name == 'Lateral Movement':
            columns = ['movement_pattern', 'session_duration']
        elif stage_name == 'Data Exfiltration':
            columns = ['exfil_volume', 'transfer_mode']
        else:  # Normal
            columns = ['traffic_pattern', 'service_type', 'volume_level']

        semantic_df = pd.DataFrame(semantic_data, columns=columns)
        semantic_df['Stage'] = stage_name

        print(f"语义特征提取完成: {len(semantic_df)} 个样本, {len(columns)} 个语义特征")
        return semantic_df

    def _extract_reconnaissance_semantics(self, row):
        """提取侦察阶段的语义特征"""
        # 扫描强度（基于数据包数量）
        total_packets = row.get('Total Fwd Packet', 0) + row.get('Total Bwd packets', 0)
        if total_packets > 100:
            scan_intensity = self.semantic_mappings['scan_intensity']['high']
        elif total_packets > 20:
            scan_intensity = self.semantic_mappings['scan_intensity']['medium']
        else:
            scan_intensity = self.semantic_mappings['scan_intensity']['low']

        # 探测模式（基于目标端口）
        port = row.get('Dst Port', 0)
        if port in [80, 443, 22, 21, 25, 53]:
            probe_pattern = self.semantic_mappings['probe_pattern']['service']
        elif port > 1024:
            probe_pattern = self.semantic_mappings['probe_pattern']['random']
        else:
            probe_pattern = self.semantic_mappings['probe_pattern']['system']

        # 数据收集量（基于传输数据量）
        data_volume = row.get('Total Length of Fwd Packet', 0)
        if data_volume > 10000:
            collection_level = self.semantic_mappings['collection_level']['high']
        elif data_volume > 1000:
            collection_level = self.semantic_mappings['collection_level']['medium']
        else:
            collection_level = self.semantic_mappings['collection_level']['low']

        return [scan_intensity, probe_pattern, collection_level]

    def _extract_foothold_semantics(self, row):
        """提取建立立足点阶段的语义特征"""
        # 目标服务类型（基于端口号）
        port = row.get('Dst Port', 0)
        if port in [80, 443]:
            target_service = self.semantic_mappings['target_service']['web']
        elif port == 9003:
            target_service = self.semantic_mappings['target_service']['apt_specific']
        elif port < 1024:
            target_service = self.semantic_mappings['target_service']['system']
        else:
            target_service = self.semantic_mappings['target_service']['high_port']

        # 连接模式（基于流持续时间和数据包数量）
        duration = row.get('Flow Duration', 0)
        packets = row.get('Total Fwd Packet', 0) + row.get('Total Bwd packets', 0)

        if duration > 60000 and packets > 50:  # 长时间多包
            connection_mode = self.semantic_mappings['connection_mode']['persistent']
        elif packets > 20:  # 短时间多包
            connection_mode = self.semantic_mappings['connection_mode']['burst']
        else:  # 单次连接
            connection_mode = self.semantic_mappings['connection_mode']['single']

        return [target_service, connection_mode]

    def _extract_lateral_semantics(self, row):
        """提取横向移动阶段的语义特征"""
        # 移动模式（基于流间到达时间和持续时间）
        iat_mean = row.get('Flow IAT Mean', 0)
        duration = row.get('Flow Duration', 0)

        if iat_mean > 1000 and duration > 30000:  # 间隔大，持续长
            movement_pattern = self.semantic_mappings['movement_pattern']['complex']
        elif iat_mean > 500:  # 中等间隔
            movement_pattern = self.semantic_mappings['movement_pattern']['indirect']
        else:  # 直接连接
            movement_pattern = self.semantic_mappings['movement_pattern']['direct']

        # 会话持续时间
        duration = row.get('Flow Duration', 0)
        if duration > 100000:  # 超长会话
            session_duration = self.semantic_mappings['session_duration']['very_long']
        elif duration > 30000:  # 长会话
            session_duration = self.semantic_mappings['session_duration']['long']
        elif duration > 5000:  # 中等会话
            session_duration = self.semantic_mappings['session_duration']['medium']
        else:  # 短会话
            session_duration = self.semantic_mappings['session_duration']['short']

        return [movement_pattern, session_duration]

    def _extract_exfiltration_semantics(self, row):
        """提取数据窃取阶段的语义特征"""
        # 窃取数据量（基于总传输数据量）
        total_data = row.get('Total Length of Fwd Packet', 0) + row.get('Total Length of Bwd Packet', 0)
        if total_data > 100000:  # 大量数据
            exfil_volume = self.semantic_mappings['exfil_volume']['massive']
        elif total_data > 10000:  # 较多数据
            exfil_volume = self.semantic_mappings['exfil_volume']['large']
        elif total_data > 1000:  # 中等数据
            exfil_volume = self.semantic_mappings['exfil_volume']['medium']
        else:  # 少量数据
            exfil_volume = self.semantic_mappings['exfil_volume']['small']

        # 传输模式（基于数据包大小和间隔）
        avg_packet_size = row.get('Average Packet Size', 0)
        iat_std = row.get('Flow IAT Std', 0)

        if avg_packet_size < 100 and iat_std > 1000:  # 小包，不规律间隔
            transfer_mode = self.semantic_mappings['transfer_mode']['stealth']
        elif iat_std > 500:  # 突发传输
            transfer_mode = self.semantic_mappings['transfer_mode']['burst']
        else:  # 连续传输
            transfer_mode = self.semantic_mappings['transfer_mode']['continuous']

        return [exfil_volume, transfer_mode]

    def _extract_normal_semantics(self, row):
        """提取正常流量的语义特征"""
        # 流量模式（基于数据包数量和持续时间）
        packets = row.get('Total Fwd Packet', 0) + row.get('Total Bwd packets', 0)
        duration = row.get('Flow Duration', 0)

        if packets > 50 and duration > 10000:
            traffic_pattern = 0  # 长连接
        elif packets > 20:
            traffic_pattern = 1  # 短连接多包
        else:
            traffic_pattern = 2  # 简单连接

        # 服务类型（基于端口）
        port = row.get('Dst Port', 0)
        if port in [80, 443]:
            service_type = 0  # Web服务
        elif port in [22, 21, 25, 53]:
            service_type = 1  # 系统服务
        else:
            service_type = 2  # 其他服务

        # 流量量级
        data_volume = row.get('Total Length of Fwd Packet', 0)
        if data_volume > 5000:
            volume_level = 0  # 大流量
        elif data_volume > 500:
            volume_level = 1  # 中流量
        else:
            volume_level = 2  # 小流量

        return [traffic_pattern, service_type, volume_level]

    def _save_results(self):
        """保存预处理结果 - 各阶段独立特征"""
        print(f"--- 保存预处理结果到 {self.output_path} ---")

        try:
            # 保存各阶段的语义特征信息
            stage_features_path = os.path.join(self.output_path, 'stage_features.json')
            stage_features_info = {}

            for stage_name, df in self.data_frames.items():
                # 获取该阶段的语义特征列表（排除Stage列）
                stage_features = [col for col in df.columns if col != 'Stage']
                stage_features_info[stage_name] = {
                    'features': stage_features,
                    'feature_count': len(stage_features),
                    'feature_type': 'semantic',
                    'semantic_mappings': {
                        feature: self.semantic_mappings.get(feature, {})
                        for feature in stage_features
                        if feature in self.semantic_mappings
                    },
                    'selected_features': self.stage_selected_features.get(stage_name, []) if hasattr(self, 'stage_selected_features') else []
                }

            with open(stage_features_path, 'w', encoding='utf-8') as f:
                json.dump(stage_features_info, f, indent=4, ensure_ascii=False)
            print(f"  已保存各阶段语义特征信息到: {stage_features_path}")

            # 保存每个阶段的预处理数据
            for stage_name, df in self.data_frames.items():
                stage_path = os.path.join(self.output_path, f'{stage_name}_preprocessed.csv')
                df.to_csv(stage_path, index=False, encoding='utf-8')
                print(f"  已保存 {stage_name} 预处理数据到: {stage_path} ({len([col for col in df.columns if col != 'Stage'])} 个特征)")

            # 保存维度跟踪信息
            dimension_path = os.path.join(self.output_path, 'dimension_tracking.json')
            with open(dimension_path, 'w', encoding='utf-8') as f:
                json.dump(self.dimension_tracking, f, indent=4, ensure_ascii=False)
            print(f"  已保存维度跟踪信息到: {dimension_path}")

            print("保存完成（各阶段保持独立特征集）。")

        except Exception as e:
            print(f"保存结果时出错: {e}")

        print("-" * 30)

    # --- 主要预处理方法 ---

    def preprocess(self):
        """数据预处理管道 - 专注于数据清洗和特征选择"""
        print("===== 开始 DAPT 数据预处理管道 =====")
        start_time = time.time()

        # 1. 加载和分区数据
        self._load_data_and_partition()

        # 2. 数据集特征分析
        self._analyze_dataset_characteristics()

        # 3. 数据清洗
        self._clean_data()

        # 4. 关键特征分析
        self._key_feature_analysis()

        # 5. 按阶段特征选择（排除关键特征）
        self._stage_wise_feature_selection()

        # 6. 更新维度跟踪
        self._update_dimension_tracking()

        # 7. 生成维度变化表
        self._generate_dimension_table()

        # 8. 保存结果
        self._save_results()

        end_time = time.time()
        print(f"===== 预处理管道完成 ({end_time - start_time:.2f} 秒) =====")

        # 返回各阶段独立的特征信息
        stage_features = {}
        if hasattr(self, 'stage_selected_features'):
            stage_features = self.stage_selected_features

        return self.data_frames, stage_features, self.dimension_tracking


if __name__ == '__main__':
    # 使用示例:
    # 重要: 请替换为您的 DAPT2020 CSV 目录的实际路径
    CSV_DIRECTORY = r'D:\PycharmProjects\DSRL-APT-2023'  # DAPT2020 CSV 文件路径
    OUTPUT_DIRECTORY = './preprocessed_output'  # 保存预处理文件的目录

    # 如果输出目录不存在则创建
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # 检查目录是否存在
    if not os.path.isdir(CSV_DIRECTORY):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 警告: 请检查 'CSV_DIRECTORY' 变量                        !!!")
        print("!!! 它应该指向包含您的 CSV 文件的目录。                        !!!")
        print(f"!!! 当前路径: {CSV_DIRECTORY}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("\n使用可能不正确的路径继续...")

    try:
        preprocessor = DAPTPreprocessor(CSV_DIRECTORY, OUTPUT_DIRECTORY)
        # 运行完整的预处理管道
        data_frames, stage_features, dimension_tracking = preprocessor.preprocess()

        print("\n--- 预处理结果摘要（语义特征编码）---")
        print(f"处理的攻击阶段数量: {len(data_frames)}")

        print("\n各阶段数据量和语义特征数:")
        for stage_name, df in data_frames.items():
            feature_count = len([col for col in df.columns if col != 'Stage'])
            print(f"  {stage_name}: {len(df)} 样本, {feature_count} 个语义特征")

        print(f"\n各阶段语义特征详情:")
        for stage_name, features in stage_features.items():
            if features:
                print(f"  {stage_name}: {len(features)} 个语义特征")
                print(f"    语义特征: {features}")

        print(f"\n语义特征映射统计:")
        total_tokens = 0
        for stage_name, features in stage_features.items():
            stage_tokens = 0
            for feature in features:
                if feature in preprocessor.semantic_mappings:
                    tokens = len(preprocessor.semantic_mappings[feature])
                    stage_tokens += tokens
                    print(f"    {feature}: {tokens} 个token")
            total_tokens += stage_tokens
            print(f"  {stage_name} 总计: {stage_tokens} 个token")

        print(f"\n总词汇表大小预估: ~{total_tokens} 个token")
        print("注意: 各攻击阶段现在使用语义特征编码，大幅降低了特征维度")

    except Exception as e:
        print(f"\n预处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
