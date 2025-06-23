import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import socket
import struct
import xgboost as xgb
import optuna
from sklearn.utils.class_weight import compute_class_weight
import warnings
import os
import time
import json
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import itertools
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

class EnhancedAPTPreprocessor:
    def __init__(self, input_path, output_path=None):
        self.input_path =input_path or r"D:\PycharmProjects\DSRL-APT-2023\DSRL-APT-2023.csv"
        self.df = None
        if output_path:
            self.output_path = output_path
            os.makedirs(output_path, exist_ok=True)
        else:
            self.output_path = os.path.dirname(input_path)
        
        self.df = None
        self.label_encoders = {}
        self.selected_features = []
        self.feature_importance = None
        self.cv_results = {}
        self.models_performance = {}
        
    def load_data(self):
        """仅加载原始 DSRL-APT-2023 数据集，不做任何列删除"""
        print("🔄 加载原始数据集...")
        start_time = time.time()

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"找不到数据文件: {self.input_path}")

        # 直接读取 CSV
        self.df = pd.read_csv(self.input_path)
        elapsed = time.time() - start_time
        print(f"✅ 数据加载完成，耗时 {elapsed:.2f} 秒")
        print(f"📊 数据集形状: {self.df.shape}")
        print(f"📋 数据列: {self.df.columns.tolist()}\n")
        return self
    
    def clean_data(self):
        """数据清洗"""
        print("\n🧹 数据清洗...")

        original_shape = self.df.shape

        if "Flow ID" in self.df.columns:
            self.df.drop(columns=["Flow ID"], inplace=True)
            print("🗑 已删除列: 'Flow ID'")
        else:
            print("ℹ️ 未发现 'Flow ID' 列，跳过删除")

        # 2. 处理IP地址列 - 转换为32位整数
        ip_cols = ['Src IP', 'Dst IP']
        for ip_col in ip_cols:
            if ip_col in self.df.columns:
                print(f"转换IP地址列 {ip_col} 为整数")
                self.df[f'{ip_col}_int'] = self.df[ip_col].apply(self._ip_to_int)
                # 删除原始IP列
                self.df.drop(columns=[ip_col], inplace=True)

        # 3. 处理时间戳列
        if 'Timestamp' in self.df.columns:
            print("处理时间戳列，提取时间特征")
            self._extract_time_features()

        # 4. 删除重复行
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            print(f"删除 {dup_count} 行重复数据")
            self.df.drop_duplicates(inplace=True)

        # 5. 处理缺失值
        null_count = self.df.isnull().sum().sum()
        if null_count > 0:
            print(f"处理 {null_count} 个缺失值")
            for col in self.df.select_dtypes(include=['number']).columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            for col in self.df.select_dtypes(exclude=['number']).columns:
                if not self.df[col].mode().empty:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # 6. 处理无穷值
        inf_count = np.isinf(self.df.select_dtypes(include=['number'])).sum().sum()
        if inf_count > 0:
            print(f"处理 {inf_count} 个无穷值")
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in self.df.select_dtypes(include=['number']).columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)

        print(f"✅ 清洗完成: {original_shape} → {self.df.shape}")
        return self

    def _ip_to_int(self, ip):
        """将IP地址转换为32位整数"""
        try:
            return struct.unpack("!I", socket.inet_aton(str(ip)))[0]
        except:
            return 0  # 对于无效IP返回0

    def _extract_time_features(self):
        """从时间戳提取时间特征"""
        try:
            # 尝试不同的时间格式
            self.df['datetime'] = pd.to_datetime(self.df['Timestamp'], errors='coerce')

            # 提取时间特征
            self.df['hour'] = self.df['datetime'].dt.hour
            self.df['minute'] = self.df['datetime'].dt.minute
            self.df['second'] = self.df['datetime'].dt.second
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek  # 0=Monday

            print("✅ 提取时间特征: hour, minute, second, day_of_week")
        except:
            return 0



    def create_statistical_features(self):
        """创建流量统计量特征（基于每条流）"""
        print("\n📊 创建统计量特征...")
        orig_cols = self.df.shape[1]

        # 1. 流量速率统计（Bytes/sec & Packets/sec）
        if 'Flow Duration' in self.df.columns:
            # 避免除零
            duration = self.df['Flow Duration'].replace(0, 1)

            # 前向字节速率 & 对数变换
            if 'Total Length of Fwd Packet' in self.df.columns:
                self.df['fwd_bytes_per_sec'] = (
                        self.df['Total Length of Fwd Packet'] / duration
                )
                self.df['fwd_bytes_per_sec_log'] = np.log1p(
                    self.df['fwd_bytes_per_sec']
                )

            # 反向字节速率 & 对数变换
            if 'Total Length of Bwd Packet' in self.df.columns:
                self.df['bwd_bytes_per_sec'] = (
                        self.df['Total Length of Bwd Packet'] / duration
                )
                self.df['bwd_bytes_per_sec_log'] = np.log1p(
                    self.df['bwd_bytes_per_sec']
                )

            # 前向包速率
            if 'Total Fwd Packet' in self.df.columns:
                self.df['fwd_packets_per_sec'] = (
                        self.df['Total Fwd Packet'] / duration
                )

            # 反向包速率
            if 'Total Bwd Packet' in self.df.columns:
                self.df['bwd_packets_per_sec'] = (
                        self.df['Total Bwd Packet'] / duration
                )

        # 2. 包长度比例
        if (
                'Fwd Packet Length Mean' in self.df.columns
                and 'Bwd Packet Length Mean' in self.df.columns
        ):
            bwd_mean = self.df['Bwd Packet Length Mean'].replace(0, 1)
            self.df['fwd_bwd_length_ratio'] = (
                    self.df['Fwd Packet Length Mean'] / bwd_mean
            )

        added = self.df.shape[1] - orig_cols
        print(f"✅ 统计量特征创建完成，新增 {added} 个特征")
        return self

    def encode_and_normalize(self):
        """编码和归一化"""
        print("\n🔢 编码和归一化...")

        # 1. 对所有分类特征进行Label Encoding
        categorical_cols = []

        # 检测分类列
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or col in ['Stage', 'Activity', 'Protocol']:
                if col in self.df.columns:
                    categorical_cols.append(col)

        print(f"检测到分类列: {categorical_cols}")

        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                # 处理缺失值
                self.df[col] = self.df[col].astype(str).fillna('Unknown')
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"✅ Label编码 {col} 列: {len(le.classes_)} 个类别")

                # 打印编码映射（前5个）
                mapping = dict(zip(le.classes_[:5], le.transform(le.classes_[:5])))
                print(f"   映射示例: {mapping}{'...' if len(le.classes_) > 5 else ''}")

        # 2. 对所有数值特征进行Min-Max归一化
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # 排除编码后的标签列（保持原始整数值）
        exclude_cols = [col for col in numeric_cols if col.endswith('_encoded')]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if feature_cols:
            print(f"对 {len(feature_cols)} 个数值特征进行Min-Max归一化...")
            scaler = MinMaxScaler()  # 使用Min-Max归一化
            self.df[feature_cols] = scaler.fit_transform(self.df[feature_cols])
            self.scaler = scaler  # 保存scaler以备后用
            print("✅ Min-Max归一化完成: X_new = (X - min) / (max - min)")

        print(f"✅ 编码和归一化完成，数据形状: {self.df.shape}")
        return self

    def domain_knowledge_feature_classification(self):
        """基于域知识的特征分类"""
        print(f"\n🏷️ 基于域知识的特征分类")

        # 获取所有数值特征
        numeric_cols = self.df.select_dtypes(include=[int, float]).columns.tolist()

        # 排除目标变量和活动标签
        exclude_cols = {'Label', 'Label_encoded', 'Stage', 'Stage_encoded', 'Activity', 'Activity_encoded'}
        available_features = [col for col in numeric_cols if col not in exclude_cols]

        # 定义特征类别
        self.feature_categories = {
            'temporal': [],      # 时序特征
            'flow_stats': [],    # 流量特征
            'packet_length': [], # 包长度统计
            'tcp_flags': [],     # TCP标志
            'network_meta': [],  # 网络元数据
            'iat_features': [],  # 时间间隔特征
            'other': []          # 其他特征
        }

        # 分类特征
        for feature in available_features:
            feature_lower = feature.lower()

            # 时序特征
            if any(keyword in feature_lower for keyword in ['hour', 'minute', 'second', 'day_of_week', 'duration']):
                self.feature_categories['temporal'].append(feature)

            # 流量特征
            elif any(keyword in feature_lower for keyword in ['bytes_s', 'packets_s', 'flow_bytes', 'flow_packets', 'total_fwd', 'total_bwd']):
                self.feature_categories['flow_stats'].append(feature)

            # 包长度统计
            elif any(keyword in feature_lower for keyword in ['packet_length', 'fwd_packet', 'bwd_packet', 'length_min', 'length_max', 'length_mean', 'length_std']):
                self.feature_categories['packet_length'].append(feature)

            # TCP标志
            elif any(keyword in feature_lower for keyword in ['flag', 'fin', 'syn', 'rst', 'psh', 'ack', 'urg', 'cwr', 'ece']):
                self.feature_categories['tcp_flags'].append(feature)

            # 时间间隔特征
            elif any(keyword in feature_lower for keyword in ['iat', 'inter_arrival']):
                self.feature_categories['iat_features'].append(feature)

            # 网络元数据
            elif any(keyword in feature_lower for keyword in ['protocol', 'port', 'ip']):
                self.feature_categories['network_meta'].append(feature)

            # 其他特征
            else:
                self.feature_categories['other'].append(feature)

        # 打印分类结果
        print(f"特征分类结果:")
        total_features = 0
        for category, features in self.feature_categories.items():
            print(f"  {category:15}: {len(features):3d} 个特征")
            if len(features) <= 5:
                print(f"    示例: {features}")
            else:
                print(f"    示例: {features[:5]} ...")
            total_features += len(features)

        print(f"  总计: {total_features} 个特征")

        # 确保每个大类至少保留1-2个代表性特征
        self.representative_features = {}
        for category, features in self.feature_categories.items():
            if features:
                # 计算每个特征的方差，选择方差较大的作为代表性特征
                feature_vars = {}
                for feat in features:
                    if feat in self.df.columns:
                        feature_vars[feat] = self.df[feat].var()

                # 按方差排序，选择前2个作为代表性特征
                sorted_features = sorted(feature_vars.items(), key=lambda x: x[1], reverse=True)
                self.representative_features[category] = [feat for feat, _ in sorted_features[:2]]

        print(f"\n代表性特征选择:")
        for category, features in self.representative_features.items():
            print(f"  {category:15}: {features}")

        return self

    def prepare_paper_aligned_features(self):
        # 1) 自动识别“唯一”目标列
        for col in ('Label_encoded', 'Label', 'Stage_encoded', 'Stage'):
            if col in self.df.columns:
                target = col
                break
        else:
            raise ValueError("找不到目标列：Label/Label_encoded/Stage/Stage_encoded")

        # 2) 取所有数值型列
        numeric_cols = self.df.select_dtypes(include=[int, float]).columns.tolist()

        # 3) 构造排除模式：标签、Activity + 后续衍生
        exclude = {target, 'Activity', 'Activity_encoded'}
        # 排除所有名称中含以下关键字的列
        bad_kw = ('_encoded', '_log', 'per_sec', 'bulk', 'ratio')
        for c in numeric_cols:
            low = c.lower()
            if any(kw in low for kw in bad_kw):
                exclude.add(c)

        # 4) 最终的候选池 = numeric_cols - exclude
        feature_cols = [c for c in numeric_cols if c not in exclude]

        # 5) 补充时间拆分（若存在的话）
        for tf in ('hour', 'minute', 'second', 'day_of_week'):
            if tf in self.df.columns and tf not in feature_cols:
                feature_cols.append(tf)

        # 检查论文中的关键特征（包含新增的3个重要特征）
        paper_critical_features = [
            'Protocol_encoded',  # 协议特征（TCP/UDP）
            'Flow Duration',     # 流持续时间
            'Fwd IAT Total',     # 正向IAT总时间
            'Fwd IAT Mean',      # 正向IAT平均时间
            'Bwd IAT Std',       # 反向IAT标准差
            'Bwd IAT Max',       # 反向IAT最大值
            'Bwd IAT Min',       # 反向IAT最小值
            'FIN Flag Count',    # FIN标志位计数
            'SYN Flag Count',    # SYN标志位计数
            'RST Flag Count',    # RST标志位计数
            'PSH Flag Count',    # PSH标志位计数
            'ACK Flag Count',    # ACK标志位计数
            'URG Flag Count',    # URG标志位计数
            # 新增的3个重要特征
            'Total Length of Fwd Packet',  # 前向包总长度
            'Fwd Packet Length Min',       # 前向包最小长度
            'Flow IAT Min'                 # 流间隔最小值
        ]

        # 检查关键特征的存在情况
        critical_found = []
        critical_missing = []

        for feat in paper_critical_features:
            if feat in self.df.columns:
                if feat not in feature_cols:
                    feature_cols.append(feat)  # 确保关键特征被包含
                critical_found.append(feat)
            else:
                critical_missing.append(feat)

        # 打印确认
        print(f"✅ 目标列（已排除）: {target}")
        print(f"✅ 排除了 {len(exclude)} 列，候选池共有 {len(feature_cols)} 列")
        print("候选特征示例:", feature_cols[:10], "...")

        # 详细分析论文关键特征
        print(f"\n📊 论文关键特征检查:")
        print(f"  论文关键特征找到: {len(critical_found)}/13")

        if critical_found:
            print(f"\n✅ 找到的论文关键特征:")
            for i, feat in enumerate(critical_found, 1):
                print(f"  {i:2d}. {feat}")

        if critical_missing:
            print(f"\n❌ 缺失的论文关键特征:")
            for i, feat in enumerate(critical_missing, 1):
                print(f"  {i:2d}. {feat}")

        # 检查特征类别分布
        protocol_features = [f for f in feature_cols if 'protocol' in f.lower()]
        duration_features = [f for f in feature_cols if 'duration' in f.lower()]
        iat_features = [f for f in feature_cols if 'iat' in f.lower()]
        flag_features = [f for f in feature_cols if 'flag' in f.lower()]
        time_features = ['hour', 'minute', 'day_of_week']
        time_in_candidates = [f for f in feature_cols if f in time_features]

        print(f"\n🔍 特征类别分布:")
        print(f"  协议特征: {len(protocol_features)} ({protocol_features})")
        print(f"  流持续时间: {len(duration_features)} ({duration_features})")
        print(f"  IAT时间间隔: {len(iat_features)}")
        print(f"  TCP标志位: {len(flag_features)}")
        print(f"  时间特征: {len(time_in_candidates)} ({time_in_candidates})")

        # 清理特征名称，避免LightGBM的特征名称问题
        cleaned_feature_cols = []
        for col in feature_cols:
            # 替换空格和特殊字符为下划线
            cleaned_col = col.replace(' ', '_').replace('/', '_').replace('-', '_')
            cleaned_feature_cols.append(cleaned_col)

        # 重命名DataFrame列
        rename_mapping = dict(zip(feature_cols, cleaned_feature_cols))
        self.df = self.df.rename(columns=rename_mapping)

        self.candidate_features = cleaned_feature_cols
        print(f"✅ 特征名称已清理，避免LightGBM兼容性问题")
        return self

    def initial_feature_screening(self, correlation_threshold=0.9, variance_threshold=0.01):
        """初筛：去相关性和高方差特征筛选"""
        print(f"\n🔍 初筛阶段：去相关性(>{correlation_threshold})和低方差(<{variance_threshold})特征")

        if not hasattr(self, 'candidate_features') or not self.candidate_features:
            raise ValueError("请先执行特征准备步骤")

        original_count = len(self.candidate_features)
        print(f"初始特征数量: {original_count}")

        # 1. 去除低方差特征（近乎常数的特征）
        print(f"\n1️⃣ 去除低方差特征（方差 < {variance_threshold}）")
        low_variance_features = []
        remaining_features = []

        for feature in self.candidate_features:
            if feature in self.df.columns:
                feature_var = self.df[feature].var()
                if feature_var < variance_threshold:
                    low_variance_features.append(feature)
                else:
                    remaining_features.append(feature)

        print(f"  移除低方差特征: {len(low_variance_features)} 个")
        if low_variance_features:
            print(f"  示例: {low_variance_features[:5]}")

        # 2. 计算相关性矩阵并去除高相关特征
        print(f"\n2️⃣ 去除高相关特征（相关系数 > {correlation_threshold}）")

        if len(remaining_features) > 1:
            # 计算相关性矩阵
            corr_matrix = self.df[remaining_features].corr().abs()

            # 找出高相关性的特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > correlation_threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))

            print(f"  发现 {len(high_corr_pairs)} 对高相关特征")

            # 选择要保留的特征（保留方差更大的）
            to_drop = set()
            for feat1, feat2, corr_val in high_corr_pairs:
                if feat1 not in to_drop and feat2 not in to_drop:
                    # 保留方差更大的特征
                    var1 = self.df[feat1].var()
                    var2 = self.df[feat2].var()

                    if var1 >= var2:
                        to_drop.add(feat2)
                        print(f"    移除 {feat2} (相关性={corr_val:.3f}, 保留方差更大的 {feat1})")
                    else:
                        to_drop.add(feat1)
                        print(f"    移除 {feat1} (相关性={corr_val:.3f}, 保留方差更大的 {feat2})")

            # 更新特征列表
            final_features = [f for f in remaining_features if f not in to_drop]
            print(f"  移除高相关特征: {len(to_drop)} 个")
        else:
            final_features = remaining_features
            print(f"  特征数量不足，跳过相关性分析")

        # 3. 保存筛选结果
        self.screened_features = final_features
        screened_count = len(final_features)

        print(f"\n✅ 初筛完成:")
        print(f"  原始特征: {original_count}")
        print(f"  筛选后特征: {screened_count}")
        print(f"  筛选比例: {(original_count - screened_count) / original_count * 100:.1f}%")
        print(f"  剩余特征预算: {screened_count} (目标: 50-70)")

        # 4. 按类别分析筛选结果
        if hasattr(self, 'feature_categories'):
            print(f"\n📊 按类别分析筛选结果:")
            category_stats = {}
            for category, original_features in self.feature_categories.items():
                remaining_in_category = [f for f in original_features if f in final_features]
                category_stats[category] = {
                    'original': len(original_features),
                    'remaining': len(remaining_in_category),
                    'features': remaining_in_category
                }
                print(f"  {category:15}: {len(original_features):2d} → {len(remaining_in_category):2d}")

            self.category_screening_stats = category_stats

        return self

    def find_optimal_feature_count_with_shap(self, max_features=None, cv_folds=5):
        """使用SHAP和交叉验证找到最优特征数量"""
        print(f"\n🔍 使用SHAP分析寻找最优特征数量")

        # 确定目标变量
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'

        # 准备数据
        X = self.df[self.candidate_features]
        y = self.df[target]

        if max_features is None:
            max_features = min(len(self.candidate_features), 60)  # 最多测试60个特征

        # 使用XGBoost进行SHAP分析
        print(f"  训练XGBoost模型进行SHAP分析...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )

        # 训练模型
        xgb_model.fit(X, y)

        # 使用XGBoost内置的特征重要性（更简单可靠）
        print(f"  使用XGBoost特征重要性...")
        feature_importance = xgb_model.feature_importances_

        # 按重要性排序特征
        feature_importance_df = pd.DataFrame({
            'feature': X.columns.tolist(),
            'importance': feature_importance.tolist()
        }).sort_values('importance', ascending=False)

        print(f"  XGBoost特征重要性Top 10:")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        # 测试不同特征数量的性能
        feature_counts = [10, 15, 20, 25, 30, 35, 40, 46, 50, min(max_features, len(self.candidate_features))]
        feature_counts = sorted(list(set(feature_counts)))  # 去重并排序

        print(f"\n  测试不同特征数量的性能:")

        best_score = 0
        best_feature_count = 46
        results = []

        for n_feat in feature_counts:
            if n_feat > len(feature_importance_df):
                continue

            # 选择Top N特征
            selected_features = feature_importance_df.head(n_feat)['feature'].tolist()
            X_selected = X[selected_features]

            # 交叉验证评估
            scores = cross_val_score(
                xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
                X_selected, y,
                cv=cv_folds,
                scoring='f1_macro',
                n_jobs=-1
            )

            mean_score = scores.mean()
            std_score = scores.std()

            results.append({
                'n_features': n_feat,
                'f1_score': mean_score,
                'f1_std': std_score
            })

            print(f"    {n_feat:2d}个特征: F1={mean_score:.4f}±{std_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_feature_count = n_feat

        # 保存结果
        self.feature_selection_results = {
            'xgboost_importance': feature_importance_df.to_dict('records'),
            'performance_by_count': results,
            'optimal_count': best_feature_count,
            'optimal_score': best_score
        }

        print(f"\n✅ 最优特征数量: {best_feature_count} (F1: {best_score:.4f})")

        # 更新候选特征为最优数量的特征
        self.optimal_features = feature_importance_df.head(best_feature_count)['feature'].tolist()

        return best_feature_count

    def automated_feature_scoring(self, use_screened_features=True):
        """自动化打分：XGBoost+SHAP、L1正则化、mRMR三种方法综合打分"""
        print(f"\n🎯 自动化特征打分：XGBoost+SHAP + L1正则化 + mRMR")

        # 选择要评分的特征
        if use_screened_features and hasattr(self, 'screened_features'):
            features_to_score = self.screened_features
            print(f"使用筛选后的特征: {len(features_to_score)} 个")
        else:
            features_to_score = self.candidate_features
            print(f"使用候选特征: {len(features_to_score)} 个")

        if not features_to_score:
            raise ValueError("没有可用于评分的特征")

        # 准备数据
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[features_to_score]
        y = self.df[target]

        print(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")

        # 初始化评分结果
        feature_scores = {feature: {'xgb_shap': 0, 'l1_reg': 0, 'mrmr': 0, 'combined': 0}
                         for feature in features_to_score}

        # 1. XGBoost + SHAP 评分
        print(f"\n1️⃣ XGBoost特征重要性评分")
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            xgb_model.fit(X, y)

            # 获取特征重要性
            xgb_importance = xgb_model.feature_importances_

            # 归一化到0-1
            xgb_importance_norm = xgb_importance / np.max(xgb_importance)

            for i, feature in enumerate(features_to_score):
                feature_scores[feature]['xgb_shap'] = xgb_importance_norm[i]

            print(f"  ✅ XGBoost评分完成")

        except Exception as e:
            print(f"  ❌ XGBoost评分失败: {e}")

        # 2. L1正则化评分
        print(f"\n2️⃣ L1正则化特征选择评分")
        try:
            # 标准化特征
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # L1正则化Logistic回归
            l1_model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=0.1,
                random_state=42,
                max_iter=1000
            )
            l1_model.fit(X_scaled, y)

            # 获取系数的绝对值作为重要性
            l1_importance = np.abs(l1_model.coef_).mean(axis=0)

            # 归一化到0-1
            if np.max(l1_importance) > 0:
                l1_importance_norm = l1_importance / np.max(l1_importance)
            else:
                l1_importance_norm = l1_importance

            for i, feature in enumerate(features_to_score):
                feature_scores[feature]['l1_reg'] = l1_importance_norm[i]

            print(f"  ✅ L1正则化评分完成")

        except Exception as e:
            print(f"  ❌ L1正则化评分失败: {e}")

        # 3. mRMR (最小冗余最大相关性) 评分
        print(f"\n3️⃣ 互信息(MI)评分")
        try:
            # 计算互信息
            mi_scores = mutual_info_classif(X, y, random_state=42)

            # 归一化到0-1
            if np.max(mi_scores) > 0:
                mi_scores_norm = mi_scores / np.max(mi_scores)
            else:
                mi_scores_norm = mi_scores

            for i, feature in enumerate(features_to_score):
                feature_scores[feature]['mrmr'] = mi_scores_norm[i]

            print(f"  ✅ 互信息评分完成")

        except Exception as e:
            print(f"  ❌ 互信息评分失败: {e}")

        # 4. 综合评分（加权平均）
        print(f"\n4️⃣ 综合评分计算")
        weights = {'xgb_shap': 0.4, 'l1_reg': 0.3, 'mrmr': 0.3}  # 可调整权重

        for feature in features_to_score:
            combined_score = (
                weights['xgb_shap'] * feature_scores[feature]['xgb_shap'] +
                weights['l1_reg'] * feature_scores[feature]['l1_reg'] +
                weights['mrmr'] * feature_scores[feature]['mrmr']
            )
            feature_scores[feature]['combined'] = combined_score

        # 按综合得分排序
        sorted_features = sorted(feature_scores.items(),
                               key=lambda x: x[1]['combined'],
                               reverse=True)

        # 保存评分结果
        self.feature_scores = feature_scores
        self.sorted_features_by_score = sorted_features

        # 打印Top 10特征
        print(f"\n🏆 Top 10 特征（按综合得分）:")
        print(f"{'特征名':<30} {'XGB':>8} {'L1':>8} {'MI':>8} {'综合':>8}")
        print("-" * 70)

        for i, (feature, scores) in enumerate(sorted_features[:10]):
            print(f"{feature:<30} {scores['xgb_shap']:>8.4f} {scores['l1_reg']:>8.4f} "
                  f"{scores['mrmr']:>8.4f} {scores['combined']:>8.4f}")

        print(f"\n✅ 自动化特征评分完成")
        print(f"  评分权重: XGBoost={weights['xgb_shap']}, L1={weights['l1_reg']}, MI={weights['mrmr']}")

        return self

    def filter_candidate_features(self, remove_overfitting_features=True, include_paper_features=True):
        """过滤候选特征，避免过拟合和信息泄露"""
        print(f"\n🔍 过滤候选特征，避免过拟合和信息泄露")

        original_count = len(self.candidate_features)
        filtered_features = self.candidate_features.copy()

        # 移除可能导致过拟合或信息泄露的特征
        if remove_overfitting_features:
            overfitting_features = [
                'Dst_IP_int',      # 目标IP可能导致过拟合
                'Src_IP_int',      # 源IP可能导致过拟合
                'Timestamp_encoded', # 时间戳编码可能泄露信息
                'Activity_encoded'   # 活动类型可能泄露标签信息
            ]

            removed_features = []
            for feature in overfitting_features:
                if feature in filtered_features:
                    filtered_features.remove(feature)
                    removed_features.append(feature)

            if removed_features:
                print(f"  移除可能过拟合的特征: {removed_features}")

        # 论文重要特征处理
        paper_features = [
            'Protocol_encoded', 'Flow_Duration', 'Fwd_IAT_Total', 'Fwd_IAT_Mean',
            'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'FIN_Flag_Count',
            'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count',
            'URG_Flag_Count', 'Total_Length_of_Fwd_Packet', 'Fwd_Packet_Length_Min',
            'Flow_IAT_Min'
        ]

        if include_paper_features:
            # 确保论文重要特征被包含
            for feature in paper_features:
                if feature not in filtered_features and feature in self.df.columns:
                    filtered_features.append(feature)
            print(f"  确保包含论文重要特征: {len([f for f in paper_features if f in filtered_features])} 个")
        else:
            # 移除论文特征，测试其他特征的效果
            removed_paper = []
            for feature in paper_features:
                if feature in filtered_features:
                    filtered_features.remove(feature)
                    removed_paper.append(feature)
            if removed_paper:
                print(f"  移除论文特征进行测试: {removed_paper}")

        # 添加稳定的网络流特征
        stable_features = [
            'Flow_Bytes_s', 'Flow_Packets_s', 'Fwd_Packets_s', 'Bwd_Packets_s',
            'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Max',
            'Flow_IAT_Mean', 'Flow_IAT_Std', 'Active_Mean', 'Idle_Mean',
            'hour', 'minute', 'day_of_week'  # 时间特征通常比较稳定
        ]

        for feature in stable_features:
            if feature not in filtered_features and feature in self.df.columns:
                filtered_features.append(feature)

        self.candidate_features = filtered_features
        filtered_count = len(filtered_features)

        print(f"✅ 特征过滤完成")
        print(f"  原始特征数: {original_count}")
        print(f"  过滤后特征数: {filtered_count}")
        print(f"  过滤比例: {(original_count - filtered_count) / original_count * 100:.1f}%")

        return self

    def stability_selection(self, n_runs=50, subsample_ratio=0.7, cv_folds=5, stability_threshold=0.6):
        """稳定性检验（Stability Selection）"""
        print(f"\n🔄 稳定性检验：{n_runs}次子采样 + {cv_folds}折交叉验证")
        print(f"子采样比例: {subsample_ratio}, 稳定性阈值: {stability_threshold}")

        # 使用评分后的特征
        if hasattr(self, 'sorted_features_by_score'):
            features_to_test = [feat for feat, _ in self.sorted_features_by_score]
            print(f"使用评分后的特征: {len(features_to_test)} 个")
        elif hasattr(self, 'screened_features'):
            features_to_test = self.screened_features
            print(f"使用筛选后的特征: {len(features_to_test)} 个")
        else:
            features_to_test = self.candidate_features
            print(f"使用候选特征: {len(features_to_test)} 个")

        if not features_to_test:
            raise ValueError("没有可用于稳定性检验的特征")

        # 准备数据
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[features_to_test]
        y = self.df[target]

        # 记录每个特征在每次运行中的选择情况
        feature_selection_counts = defaultdict(int)
        total_selections = 0

        print(f"\n开始 {n_runs} 次稳定性检验...")

        import random
        random.seed(42)
        np.random.seed(42)

        for run_idx in range(n_runs):
            if (run_idx + 1) % 10 == 0:
                print(f"  完成 {run_idx + 1}/{n_runs} 次运行")

            # 子采样
            n_samples = int(len(X) * subsample_ratio)
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sub = X.iloc[sample_indices]
            y_sub = y.iloc[sample_indices]

            # 交叉验证特征选择
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=run_idx)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_sub, y_sub)):
                X_train = X_sub.iloc[train_idx]
                y_train = y_sub.iloc[train_idx]

                try:
                    # 使用XGBoost进行特征选择
                    xgb_selector = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric='mlogloss'
                    )
                    xgb_selector.fit(X_train, y_train)

                    # 获取特征重要性
                    importance = xgb_selector.feature_importances_

                    # 选择重要性大于平均值的特征
                    mean_importance = np.mean(importance)
                    selected_features = [
                        features_to_test[i] for i, imp in enumerate(importance)
                        if imp > mean_importance
                    ]

                    # 记录选择的特征
                    for feature in selected_features:
                        feature_selection_counts[feature] += 1

                    total_selections += 1

                except Exception as e:
                    print(f"    运行 {run_idx+1} 折 {fold_idx+1} 失败: {e}")
                    continue

        # 计算稳定性分数
        print(f"\n📊 稳定性分析结果:")
        print(f"总选择次数: {total_selections}")

        stable_features = []
        feature_stability_scores = {}

        for feature in features_to_test:
            selection_frequency = feature_selection_counts[feature] / total_selections
            feature_stability_scores[feature] = selection_frequency

            if selection_frequency >= stability_threshold:
                stable_features.append(feature)

        # 按稳定性分数排序
        sorted_by_stability = sorted(feature_stability_scores.items(),
                                   key=lambda x: x[1],
                                   reverse=True)

        # 保存结果
        self.stability_scores = feature_stability_scores
        self.stable_features = stable_features
        self.sorted_features_by_stability = sorted_by_stability

        print(f"\n🏆 稳定性检验结果:")
        print(f"  稳定特征数量: {len(stable_features)} (阈值: {stability_threshold})")
        print(f"  原始特征数量: {len(features_to_test)}")
        print(f"  稳定性比例: {len(stable_features) / len(features_to_test) * 100:.1f}%")

        # 打印Top 15稳定特征
        print(f"\n📋 Top 15 最稳定特征:")
        print(f"{'特征名':<35} {'稳定性分数':>12} {'状态':>8}")
        print("-" * 60)

        for i, (feature, score) in enumerate(sorted_by_stability[:15]):
            status = "✅稳定" if score >= stability_threshold else "❌不稳定"
            print(f"{feature:<35} {score:>12.3f} {status:>8}")

        # 按类别分析稳定性
        if hasattr(self, 'feature_categories'):
            print(f"\n📊 按类别分析稳定性:")
            for category, original_features in self.feature_categories.items():
                stable_in_category = [f for f in original_features if f in stable_features]
                total_in_category = len([f for f in original_features if f in features_to_test])
                if total_in_category > 0:
                    stability_ratio = len(stable_in_category) / total_in_category
                    print(f"  {category:15}: {len(stable_in_category):2d}/{total_in_category:2d} "
                          f"({stability_ratio*100:5.1f}%) 稳定")

        print(f"\n✅ 稳定性检验完成")

        return self

    def optimize_xgboost_hyperparameters(self, n_trials=100, cv_folds=5):
        """使用Optuna优化XGBoost超参数"""
        print(f"\n🎯 使用Optuna优化XGBoost超参数 (试验次数: {n_trials})")

        # 确定目标变量
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[self.candidate_features]
        y = self.df[target]

        # 计算类别权重
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"  类别权重: {class_weight_dict}")

        def objective(trial):
            # 定义超参数搜索空间
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'eval_metric': 'mlogloss',
                'early_stopping_rounds': 50,
                'n_jobs': -1
            }

            # 交叉验证评估
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # 创建XGBoost模型
                model = xgb.XGBClassifier(**params)

                # 训练模型（带早停）
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                # 预测和评估
                y_pred = model.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='macro')
                scores.append(f1)

            return np.mean(scores)

        # 创建Optuna研究
        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=42))

        # 优化超参数
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # 保存最佳参数
        self.best_params = study.best_params
        self.best_score = study.best_value

        print(f"✅ 超参数优化完成")
        print(f"  最佳F1分数: {self.best_score:.4f}")
        print(f"  最佳参数: {self.best_params}")

        return self.best_params

    def stage_aware_feature_refinement(self, target_features=35):
        """精炼分层挑选：按攻击阶段分层选择特征"""
        print(f"\n🎯 精炼分层挑选：按攻击阶段分层选择特征（目标: {target_features}个）")

        # 使用稳定特征作为基础
        if hasattr(self, 'stable_features') and self.stable_features:
            base_features = self.stable_features
            print(f"使用稳定特征作为基础: {len(base_features)} 个")
        elif hasattr(self, 'sorted_features_by_score'):
            # 使用评分前50%的特征
            top_half = len(self.sorted_features_by_score) // 2
            base_features = [feat for feat, _ in self.sorted_features_by_score[:top_half]]
            print(f"使用评分前50%特征作为基础: {len(base_features)} 个")
        else:
            raise ValueError("请先执行特征评分或稳定性检验")

        # 准备数据
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[base_features]
        y = self.df[target]

        # 按攻击阶段分组数据
        stage_data = {}
        stage_names = {0: 'Normal', 1: 'Reconnaissance', 2: 'Establish_Foothold',
                      3: 'Lateral_Movement', 4: 'Data_Exfiltration'}

        for stage_id in y.unique():
            stage_mask = (y == stage_id)
            stage_data[stage_id] = {
                'name': stage_names.get(stage_id, f'Stage_{stage_id}'),
                'X': X[stage_mask],
                'y': y[stage_mask],
                'count': stage_mask.sum()
            }

        print(f"\n📊 攻击阶段数据分布:")
        for stage_id, data in stage_data.items():
            print(f"  {data['name']:<20}: {data['count']:>6} 样本")

        # 1. 阶段通用特征选择
        print(f"\n1️⃣ 选择阶段通用特征")
        universal_features = []

        # 对每个阶段进行特征重要性分析
        stage_feature_importance = {}

        for stage_id, data in stage_data.items():
            if data['count'] < 50:  # 样本太少，跳过
                print(f"  跳过 {data['name']} (样本数不足)")
                continue

            try:
                # 创建二分类问题：当前阶段 vs 其他阶段
                y_binary = (y == stage_id).astype(int)

                # 使用XGBoost评估特征重要性
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
                xgb_model.fit(X, y_binary)

                # 获取特征重要性
                importance = xgb_model.feature_importances_
                stage_feature_importance[stage_id] = dict(zip(base_features, importance))

                print(f"  {data['name']:<20}: 特征重要性分析完成")

            except Exception as e:
                print(f"  {data['name']:<20}: 分析失败 - {e}")
                continue

        # 计算每个特征在各阶段的平均重要性
        feature_universal_scores = {}
        for feature in base_features:
            scores = []
            for stage_id, importance_dict in stage_feature_importance.items():
                if feature in importance_dict:
                    scores.append(importance_dict[feature])

            if scores:
                # 使用平均值和最小值的组合（确保在所有阶段都有一定重要性）
                avg_score = np.mean(scores)
                min_score = np.min(scores)
                universal_score = 0.7 * avg_score + 0.3 * min_score
                feature_universal_scores[feature] = universal_score

        # 选择通用特征（按通用分数排序）
        sorted_universal = sorted(feature_universal_scores.items(),
                                key=lambda x: x[1], reverse=True)

        # 选择前60%作为通用特征
        n_universal = max(int(target_features * 0.6), 15)  # 至少15个通用特征
        universal_features = [feat for feat, _ in sorted_universal[:n_universal]]

        print(f"  选择通用特征: {len(universal_features)} 个")

        # 2. 阶段专用特征选择
        print(f"\n2️⃣ 选择阶段专用特征")
        stage_specific_features = []

        # 计算每个特征的阶段特异性
        feature_specificity_scores = {}
        for feature in base_features:
            if feature in universal_features:
                continue  # 跳过已选择的通用特征

            scores = []
            for stage_id, importance_dict in stage_feature_importance.items():
                if feature in importance_dict:
                    scores.append(importance_dict[feature])

            if scores and len(scores) > 1:
                # 计算特异性：最大值与平均值的比值
                max_score = np.max(scores)
                avg_score = np.mean(scores)
                specificity = max_score / (avg_score + 1e-8)  # 避免除零
                feature_specificity_scores[feature] = specificity

        # 选择特异性高的特征
        sorted_specific = sorted(feature_specificity_scores.items(),
                               key=lambda x: x[1], reverse=True)

        # 选择剩余的特征作为专用特征
        n_specific = target_features - len(universal_features)
        stage_specific_features = [feat for feat, _ in sorted_specific[:n_specific]]

        print(f"  选择专用特征: {len(stage_specific_features)} 个")

        # 3. 合并最终特征集
        final_features = universal_features + stage_specific_features

        # 保存结果
        self.refined_features = final_features
        self.universal_features = universal_features
        self.stage_specific_features = stage_specific_features
        self.stage_feature_importance = stage_feature_importance

        print(f"\n✅ 精炼分层挑选完成:")
        print(f"  通用特征: {len(universal_features)} 个")
        print(f"  专用特征: {len(stage_specific_features)} 个")
        print(f"  最终特征: {len(final_features)} 个")

        # 打印最终特征列表
        print(f"\n📋 最终特征列表:")
        print(f"通用特征 ({len(universal_features)} 个):")
        for i, feat in enumerate(universal_features, 1):
            print(f"  {i:2d}. {feat}")

        if stage_specific_features:
            print(f"\n专用特征 ({len(stage_specific_features)} 个):")
            for i, feat in enumerate(stage_specific_features, 1):
                print(f"  {i:2d}. {feat}")

        # 按类别分析最终特征
        if hasattr(self, 'feature_categories'):
            print(f"\n📊 最终特征类别分布:")
            for category, original_features in self.feature_categories.items():
                final_in_category = [f for f in original_features if f in final_features]
                if final_in_category:
                    print(f"  {category:15}: {len(final_in_category):2d} 个 - {final_in_category}")

        return self

    def evaluate_with_fold_internal_pipeline(self, n_features=None, cv_folds=10,
                                           optimize_hyperparams=True, n_trials=50):
        """按论文方法：每折内部做特征选择+分类的完整pipeline（优化版）"""
        print(f"\n🎯 XGBoost评估：特征过滤→超参数优化→特征选择→分类")

        # 1. 过滤候选特征，避免过拟合
        self.filter_candidate_features(remove_overfitting_features=True, include_paper_features=True)

        # 2. 超参数优化
        if optimize_hyperparams:
            best_params = self.optimize_xgboost_hyperparameters(n_trials=n_trials, cv_folds=5)
        else:
            # 使用默认优化参数
            best_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42,
                'eval_metric': 'mlogloss',
                'early_stopping_rounds': 50,
                'n_jobs': -1
            }

        # 3. 如果没有指定特征数量，使用XGBoost寻找最优数量
        if n_features is None:
            print(f"  未指定特征数量，使用XGBoost寻找最优特征数量...")
            n_features = self.find_optimal_feature_count_with_shap()
            # 使用XGBoost选择的最优特征
            if hasattr(self, 'optimal_features'):
                self.candidate_features = self.optimal_features
                print(f"  使用XGBoost选择的{len(self.optimal_features)}个最优特征")

        # 准备数据
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[self.candidate_features]
        y = self.df[target]

        print(f"候选特征数量: {X.shape[1]}")
        print(f"使用特征数量: {n_features}")
        print(f"样本数量: {X.shape[0]}")

        # 显示正确的类别分布
        class_counts = dict(y.value_counts().sort_index())
        print(f"类别分布: {class_counts}")
        print(f"  0(正常): {class_counts.get(0, 0)} 样本")
        print(f"  1(数据泄露): {class_counts.get(1, 0)} 样本")
        print(f"  2(建立立足点): {class_counts.get(2, 0)} 样本")
        print(f"  3(横向移动): {class_counts.get(3, 0)} 样本")
        print(f"  4(侦察): {class_counts.get(4, 0)} 样本")

        # 计算类别权重
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"类别权重: {class_weight_dict}")

        # 使用优化后的XGBoost模型
        xgb_model = xgb.XGBClassifier(**best_params)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        print(f"\n🚀 评估优化后的XGBoost模型")

        # 构建折内Pipeline：特征选择 → XGBoost分类器
        feature_selector = SelectFromModel(
            xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='mlogloss',
                early_stopping_rounds=10
            ),
            threshold=-np.inf,
            max_features=n_features,
            importance_getter='feature_importances_'
        )

        pipeline = Pipeline([
            ('feat_sel', feature_selector),
            ('clf', xgb_model)
        ])

        # 手动交叉验证以支持早停
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring_results = {
            'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': []
        }
        selected_features_per_fold = []

        print(f"开始{cv_folds}折交叉验证...")

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 特征选择
            feature_selector.fit(X_train, y_train)
            X_train_selected = feature_selector.transform(X_train)
            X_test_selected = feature_selector.transform(X_test)

            # 获取选择的特征名
            selected_mask = feature_selector.get_support()
            selected_feats = X.columns[selected_mask].tolist()
            selected_features_per_fold.append(selected_feats)

            # 分割训练集为训练和验证集（用于早停）
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_selected, y_train, test_size=0.2,
                random_state=42, stratify=y_train
            )

            # 训练XGBoost模型（带早停）
            model = xgb.XGBClassifier(**best_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 预测
            y_pred = model.predict(X_test_selected)

            # 计算指标
            scoring_results['accuracy'].append(accuracy_score(y_test, y_pred))
            scoring_results['precision_macro'].append(precision_score(y_test, y_pred, average='macro'))
            scoring_results['recall_macro'].append(recall_score(y_test, y_pred, average='macro'))
            scoring_results['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))

            print(f"Fold {fold_idx} 选出的 {len(selected_feats)} 个特征：")
            print(selected_feats)

        # 计算统计量
        results = {}
        for metric, scores in scoring_results.items():
            scores_array = np.array(scores)
            results[metric] = {
                'mean': scores_array.mean(),
                'std': scores_array.std(),
                'scores': scores
            }
            print(f"  {metric:<15}: {scores_array.mean():.4f} ± {scores_array.std():.4f}")

        self.cv_results = {
            'XGBoost': results,
            'selected_features_per_fold': selected_features_per_fold,
            'best_params': best_params,
            'class_weights': class_weight_dict
        }

        print(f"\n🏆 优化后XGBoost模型 F1: {results['f1_macro']['mean']:.4f}")
        print(f"📊 使用的最佳参数: {best_params}")
        print(f"⚖️ 类别权重: {class_weight_dict}")

        return self

    def evaluate_multiple_models(self, cv_folds=10):
        """按论文方法：用固定的46个特征，在10折CV中评估多模型"""
        print(f"\n🎯 使用固定的{len(self.selected_features)}个特征进行{cv_folds}折交叉验证")

        # 准备数据
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[self.selected_features]
        y = self.df[target]

        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"类别分布: {dict(y.value_counts().sort_index())}")

        # 按论文配置模型超参数
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                early_stopping=True,
                learning_rate_init=0.01,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                random_state=42
            )
        }

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        self.cv_results = {}

        for name, clf in models.items():
            print(f"\n🚀 评估模型: {name}")

            # 对MLP和SVM使用Pipeline进行折内标准化
            if name in ('MLP', 'SVM'):
                estimator = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', clf)
                ])
            else:
                estimator = clf

            # 使用cross_validate进行评估
            from sklearn.model_selection import cross_validate
            cv_results = cross_validate(
                estimator, X, y,
                cv=skf,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )

            # 计算统计量
            results = {}
            for metric in scoring:
                scores = cv_results[f'test_{metric}']
                results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
                print(f"  {metric:<15}: {scores.mean():.4f} ± {scores.std():.4f}")

            self.cv_results[name] = results

        # 找出最佳模型
        best_model = max(self.cv_results.keys(),
                        key=lambda x: self.cv_results[x]['f1_macro']['mean'])
        best_f1 = self.cv_results[best_model]['f1_macro']['mean']
        print(f"\n🏆 最佳模型: {best_model} (F1: {best_f1:.4f})")

        return self

    def detailed_model_analysis(self, cv_folds=10):
        """详细分析最佳模型：聚合10折CV的预测结果"""
        print(f"\n🔍 详细模型分析 (聚合{cv_folds}折结果)")

        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[self.selected_features]
        y = self.df[target]

        # 找出最佳模型
        best_model = max(self.cv_results.keys(),
                        key=lambda x: self.cv_results[x]['f1_macro']['mean'])
        print(f"🏆 分析最佳模型: {best_model}")

        # 重建最佳模型
        if best_model == 'RandomForest':
            clf = RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            estimator = clf
        elif best_model == 'MLP':
            clf = MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                early_stopping=True,
                learning_rate_init=1e-3,
                random_state=42
            )
            estimator = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        else:  # SVM
            clf = SVC(kernel='rbf', C=1.0, random_state=42)
            estimator = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

        # 10折交叉验证，收集所有预测
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        y_true_all, y_pred_all = [], []

        print("🔄 执行10折交叉验证...")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 训练和预测
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)

            y_true_all.extend(y_test.values)
            y_pred_all.extend(y_pred)

            print(f"  折 {fold:2d} 完成")

        # 聚合结果
        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)

        # 计算整体指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f"\n📊 聚合结果 ({best_model}):")
        print(f"  准确率 (Accuracy): {accuracy:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall): {recall:.4f}")
        print(f"  F1分数 (F1-Score): {f1:.4f}")

        print(f"\n📋 详细分类报告:")
        print(classification_report(y_true, y_pred))

        print(f"\n📊 混淆矩阵:")
        print(confusion_matrix(y_true, y_pred))

        # 添加每个攻击阶段的详细分析
        self._analyze_per_stage_performance(y_true, y_pred)

        # 保存详细结果
        self.models_performance = {
            'best_model': best_model,
            'aggregated_metrics': {
                'accuracy': accuracy,
                'precision_macro': precision,
                'recall_macro': recall,
                'f1_macro': f1
            },
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        return self

    def _analyze_per_stage_performance(self, y_true, y_pred):
        """分析每个APT攻击阶段的检测性能"""
        print(f"\n🎯 每个APT攻击阶段的检测性能分析:")
        print("="*60)

        # 定义阶段映射
        stage_names = {
            0: 'Benign (正常流量)',
            4: 'Data Exfiltration (数据渗透)',
            2: 'Establish Foothold (建立立足点)',
            3: 'Lateral Movement (横向移动)',
            1: 'Reconnaissance (侦察)'
        }

        # 计算每个类别的详细指标
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        print(f"{'阶段':<25} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<8} {'检测率':<10}")
        print("-" * 80)

        total_correct = 0
        total_samples = 0

        for i, (stage_id, stage_name) in enumerate(stage_names.items()):
            if i < len(precision):
                # 计算检测率 (该阶段被正确识别的比例)
                detection_rate = cm[i, i] / support[i] if support[i] > 0 else 0

                print(f"{stage_name:<25} {precision[i]:<10.4f} {recall[i]:<10.4f} "
                      f"{f1[i]:<10.4f} {support[i]:<8d} {detection_rate:<10.4f}")

                total_correct += cm[i, i]
                total_samples += support[i]

        print("-" * 80)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"{'总体准确率':<25} {'':<10} {'':<10} {'':<10} {total_samples:<8d} {overall_accuracy:<10.4f}")

        # 分析攻击阶段间的混淆情况
        print(f"\n🔍 攻击阶段间混淆分析:")
        print("="*50)

        for i, (true_stage_id, true_stage_name) in enumerate(stage_names.items()):
            if i < len(cm):
                print(f"\n真实阶段: {true_stage_name}")
                for j, (pred_stage_id, pred_stage_name) in enumerate(stage_names.items()):
                    if j < len(cm[i]) and cm[i, j] > 0:
                        confusion_rate = cm[i, j] / support[i] if support[i] > 0 else 0
                        if i != j:  # 只显示错误分类
                            print(f"  → 误分类为 {pred_stage_name}: {cm[i, j]} 样本 ({confusion_rate:.3f})")

        # 计算宏平均指标
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        print(f"\n📊 宏平均性能指标:")
        print(f"  宏平均精确率: {macro_precision:.4f}")
        print(f"  宏平均召回率: {macro_recall:.4f}")
        print(f"  宏平均F1分数: {macro_f1:.4f}")

        # 识别表现最好和最差的阶段
        best_stage_idx = np.argmax(f1)
        worst_stage_idx = np.argmin(f1)

        print(f"\n🏆 性能分析:")
        print(f"  最佳检测阶段: {list(stage_names.values())[best_stage_idx]} (F1: {f1[best_stage_idx]:.4f})")
        print(f"  最差检测阶段: {list(stage_names.values())[worst_stage_idx]} (F1: {f1[worst_stage_idx]:.4f})")

        # 保存每阶段性能
        self.per_stage_performance = {
            'stage_names': stage_names,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            }
        }

    def save_results(self):
        """保存所有结果"""
        print(f"\n💾 保存结果到: {self.output_path}")

        # 保存增强后的数据
        enhanced_data_path = os.path.join(self.output_path, 'enhanced_apt_data.csv')
        self.df.to_csv(enhanced_data_path, index=False)
        print(f"✅ 增强数据保存至: {enhanced_data_path}")

        # 保存选择的特征数据
        if self.selected_features:
            target_col = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
            selected_cols = self.selected_features + [target_col, 'Stage']
            selected_data_path = os.path.join(self.output_path, 'selected_features_data.csv')
            self.df[selected_cols].to_csv(selected_data_path, index=False)
            print(f"✅ 选择特征数据保存至: {selected_data_path}")

        # 保存特征重要性
        if self.feature_importance is not None:
            importance_path = os.path.join(self.output_path, 'feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
            print(f"✅ 特征重要性保存至: {importance_path}")

        # 保存交叉验证结果
        if self.cv_results:
            cv_path = os.path.join(self.output_path, 'cross_validation_results.json')
            with open(cv_path, 'w') as f:
                json.dump(self.cv_results, f, indent=4)
            print(f"✅ 交叉验证结果保存至: {cv_path}")

        # 保存详细模型性能
        if self.models_performance:
            models_path = os.path.join(self.output_path, 'models_performance.json')
            with open(models_path, 'w') as f:
                json.dump(self.models_performance, f, indent=4)
            print(f"✅ 模型性能结果保存至: {models_path}")

        # 保存每阶段性能分析
        if hasattr(self, 'per_stage_performance'):
            stage_path = os.path.join(self.output_path, 'per_stage_performance.json')
            with open(stage_path, 'w') as f:
                json.dump(self.per_stage_performance, f, indent=4)
            print(f"✅ 每阶段性能分析保存至: {stage_path}")

        return self

    def run_domain_knowledge_feature_selection(self, target_features=35,
                                              correlation_threshold=0.9,
                                              variance_threshold=0.01,
                                              stability_runs=50,
                                              stability_threshold=0.6):
        """运行基于域知识的完整特征选择流程"""
        print("🚀 开始基于域知识的特征选择流程")
        print("="*80)
        print("流程: 域知识分类 → 初筛 → 自动化打分 → 稳定性检验 → 精炼分层挑选")
        print("="*80)

        start_time = time.time()

        try:
            # 1. 域知识特征分类
            print(f"\n{'='*20} 步骤 1: 域知识特征分类 {'='*20}")
            self.domain_knowledge_feature_classification()

            # 2. 初筛：去相关性和低方差特征
            print(f"\n{'='*20} 步骤 2: 初筛阶段 {'='*20}")
            self.initial_feature_screening(
                correlation_threshold=correlation_threshold,
                variance_threshold=variance_threshold
            )

            # 3. 自动化打分
            print(f"\n{'='*20} 步骤 3: 自动化打分 {'='*20}")
            self.automated_feature_scoring(use_screened_features=True)

            # 4. 稳定性检验
            print(f"\n{'='*20} 步骤 4: 稳定性检验 {'='*20}")
            self.stability_selection(
                n_runs=stability_runs,
                stability_threshold=stability_threshold
            )

            # 5. 精炼分层挑选
            print(f"\n{'='*20} 步骤 5: 精炼分层挑选 {'='*20}")
            self.stage_aware_feature_refinement(target_features=target_features)

            # 6. 更新候选特征为最终选择的特征
            if hasattr(self, 'refined_features'):
                self.candidate_features = self.refined_features
                self.selected_features = self.refined_features
                print(f"\n✅ 更新候选特征为最终选择的 {len(self.refined_features)} 个特征")

            total_time = time.time() - start_time

            # 7. 生成特征选择报告
            self._generate_feature_selection_report(total_time)

        except Exception as e:
            print(f"❌ 特征选择过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

        return self

    def _generate_feature_selection_report(self, total_time):
        """生成特征选择报告"""
        print(f"\n{'='*20} 特征选择完成报告 {'='*20}")
        print(f"⏱️ 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

        # 统计各阶段的特征数量变化
        stages_stats = []

        if hasattr(self, 'feature_categories'):
            original_count = sum(len(features) for features in self.feature_categories.values())
            stages_stats.append(("原始特征", original_count))

        if hasattr(self, 'screened_features'):
            stages_stats.append(("初筛后", len(self.screened_features)))

        if hasattr(self, 'stable_features'):
            stages_stats.append(("稳定性检验后", len(self.stable_features)))

        if hasattr(self, 'refined_features'):
            stages_stats.append(("最终选择", len(self.refined_features)))

        print(f"\n📊 特征数量变化:")
        for stage_name, count in stages_stats:
            print(f"  {stage_name:<15}: {count:>4} 个特征")

        if len(stages_stats) >= 2:
            reduction_ratio = (stages_stats[0][1] - stages_stats[-1][1]) / stages_stats[0][1]
            print(f"  特征减少比例: {reduction_ratio*100:.1f}%")

        # 按类别分析最终特征分布
        if hasattr(self, 'feature_categories') and hasattr(self, 'refined_features'):
            print(f"\n📋 最终特征类别分布:")
            for category, original_features in self.feature_categories.items():
                final_in_category = [f for f in original_features if f in self.refined_features]
                if final_in_category:
                    print(f"  {category:15}: {len(final_in_category):2d}/{len(original_features):2d} "
                          f"({len(final_in_category)/len(original_features)*100:5.1f}%)")

        # 特征质量评估
        if hasattr(self, 'stability_scores') and hasattr(self, 'refined_features'):
            avg_stability = np.mean([self.stability_scores.get(f, 0) for f in self.refined_features])
            print(f"\n🎯 特征质量评估:")
            print(f"  平均稳定性分数: {avg_stability:.3f}")

            if hasattr(self, 'feature_scores'):
                avg_combined_score = np.mean([
                    self.feature_scores.get(f, {}).get('combined', 0)
                    for f in self.refined_features
                ])
                print(f"  平均综合评分: {avg_combined_score:.3f}")

        # 推荐的下一步
        print(f"\n🎯 推荐的下一步:")
        print(f"  1. 使用选择的 {len(self.refined_features)} 个特征进行模型训练")
        print(f"  2. 进行交叉验证评估模型性能")
        print(f"  3. 构建攻击序列用于序列生成模型")

        # 保存特征选择结果摘要
        self.feature_selection_summary = {
            'total_time': total_time,
            'stages_stats': stages_stats,
            'final_feature_count': len(self.refined_features) if hasattr(self, 'refined_features') else 0,
            'average_stability': avg_stability if 'avg_stability' in locals() else 0,
            'category_distribution': {}
        }

        if hasattr(self, 'feature_categories') and hasattr(self, 'refined_features'):
            for category, original_features in self.feature_categories.items():
                final_in_category = [f for f in original_features if f in self.refined_features]
                self.feature_selection_summary['category_distribution'][category] = {
                    'original': len(original_features),
                    'final': len(final_in_category),
                    'features': final_in_category
                }

    def run_complete_pipeline(self, n_features=46, cv_folds=10):
        """运行完整的预处理和评估流程"""
        print("🚀 开始增强版APT数据预处理和评估流程")
        print("="*80)

        start_time = time.time()

        try:
            # 执行完整流程（按论文方法修正）
            (self
             .load_data()
             .clean_data()
             .create_statistical_features()
             .encode_and_normalize()
             .prepare_paper_aligned_features()
             .evaluate_with_fold_internal_pipeline(n_features=n_features, cv_folds=cv_folds)
             .save_results())

            total_time = time.time() - start_time

            print(f"\n🎉 预处理和评估完成！")
            print(f"⏱️ 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
            print(f"📊 最终数据形状: {self.df.shape}")
            print(f"🎯 选择特征数量: {len(self.selected_features)}")

            # 打印最终结果摘要
            if self.cv_results:
                print(f"\n📈 论文方法模型性能摘要 ({cv_folds}折交叉验证):")
                for model_name, metrics in self.cv_results.items():
                    f1_mean = metrics['f1_macro']['mean']
                    f1_std = metrics['f1_macro']['std']
                    print(f"  {model_name}: F1={f1_mean:.4f}±{f1_std:.3f}")

            # 打印特征选择摘要
            if hasattr(self, 'candidate_features'):
                print(f"\n🎯 论文对齐特征摘要:")
                print(f"  候选特征数量: {len(self.candidate_features)}")

                # 检查时间特征
                time_features = ['hour', 'minute', 'day_of_week']
                time_in_candidates = [f for f in self.candidate_features if f in time_features]
                print(f"  包含时间特征: {time_in_candidates}")

                # 检查是否排除了Activity
                activity_excluded = not any('activity' in f.lower() for f in self.candidate_features)
                print(f"  已排除Activity标签: {'✅' if activity_excluded else '❌'}")

                # 显示特征类型分布
                network_features = len([f for f in self.candidate_features if f not in time_features + ['Protocol_encoded']])
                print(f"  网络流特征: {network_features}")
                print(f"  时间特征: {len(time_in_candidates)}")
                print(f"  协议特征: {1 if 'Protocol_encoded' in self.candidate_features else 0}")

            # 与论文结果对比
            if self.cv_results:
                best_f1 = max(metrics['f1_macro']['mean'] for metrics in self.cv_results.values())
                print(f"\n🎯 与论文对比:")
                print(f"  我们的最佳F1: {best_f1:.4f}")
                print(f"  论文报告F1: ~0.9800")
                print(f"  差距: {0.98 - best_f1:.4f}")
                if best_f1 >= 0.97:
                    print(f"  ✅ 接近论文水平！")
                elif best_f1 >= 0.95:
                    print(f"  🔶 良好水平，可进一步优化")
                else:
                    print(f"  ⚠️ 需要进一步调优")

        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

        return self

    def run_domain_knowledge_pipeline(self, target_features=35, cv_folds=10,
                                     correlation_threshold=0.9, variance_threshold=0.01,
                                     stability_runs=30, stability_threshold=0.6):
        """运行基于域知识的完整预处理和评估流程"""
        print("🚀 开始基于域知识的APT数据预处理和评估流程")
        print("="*80)
        print("特色: 域知识分类 → 初筛 → 自动化打分 → 稳定性检验 → 精炼分层挑选")
        print("="*80)

        start_time = time.time()

        try:
            # 1. 基础数据处理
            print(f"\n{'='*20} 阶段 1: 基础数据处理 {'='*20}")
            (self
             .load_data()
             .clean_data()
             .create_statistical_features()
             .encode_and_normalize())

            # 2. 域知识特征选择
            print(f"\n{'='*20} 阶段 2: 域知识特征选择 {'='*20}")
            self.run_domain_knowledge_feature_selection(
                target_features=target_features,
                correlation_threshold=correlation_threshold,
                variance_threshold=variance_threshold,
                stability_runs=stability_runs,
                stability_threshold=stability_threshold
            )

            # 3. 模型评估
            print(f"\n{'='*20} 阶段 3: 模型评估 {'='*20}")
            self.evaluate_with_fold_internal_pipeline(
                n_features=len(self.refined_features) if hasattr(self, 'refined_features') else target_features,
                cv_folds=cv_folds,
                optimize_hyperparams=True,
                n_trials=30
            )

            # 4. 保存结果
            print(f"\n{'='*20} 阶段 4: 保存结果 {'='*20}")
            self.save_results()

            total_time = time.time() - start_time

            # 5. 生成最终报告
            self._generate_final_pipeline_report(total_time, target_features, cv_folds)

        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

        return self

    def _generate_final_pipeline_report(self, total_time, target_features, cv_folds):
        """生成最终流程报告"""
        print(f"\n{'='*20} 🎉 流程完成报告 {'='*20}")
        print(f"⏱️ 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"📊 最终数据形状: {self.df.shape}")

        if hasattr(self, 'refined_features'):
            print(f"🎯 域知识选择特征数量: {len(self.refined_features)}")

        # 打印模型性能摘要
        if hasattr(self, 'cv_results') and self.cv_results:
            print(f"\n📈 域知识方法模型性能摘要 ({cv_folds}折交叉验证):")
            for model_name, metrics in self.cv_results.items():
                if isinstance(metrics, dict) and 'f1_macro' in metrics:
                    f1_mean = metrics['f1_macro']['mean']
                    f1_std = metrics['f1_macro']['std']
                    print(f"  {model_name}: F1={f1_mean:.4f}±{f1_std:.3f}")

        # 特征选择效果分析
        if hasattr(self, 'feature_selection_summary'):
            summary = self.feature_selection_summary
            print(f"\n🎯 域知识特征选择效果:")
            print(f"  目标特征数: {target_features}")
            print(f"  实际选择数: {summary.get('final_feature_count', 0)}")
            print(f"  平均稳定性: {summary.get('average_stability', 0):.3f}")

            if 'stages_stats' in summary:
                print(f"  特征筛选过程:")
                for stage_name, count in summary['stages_stats']:
                    print(f"    {stage_name}: {count} 个")

        # 与论文结果对比
        if hasattr(self, 'cv_results') and self.cv_results:
            best_f1 = 0
            for model_name, metrics in self.cv_results.items():
                if isinstance(metrics, dict) and 'f1_macro' in metrics:
                    f1_mean = metrics['f1_macro']['mean']
                    if f1_mean > best_f1:
                        best_f1 = f1_mean

            print(f"\n🎯 与论文对比:")
            print(f"  域知识方法最佳F1: {best_f1:.4f}")
            print(f"  论文报告F1: ~0.9800")
            print(f"  差距: {0.98 - best_f1:.4f}")

            if best_f1 >= 0.97:
                print(f"  ✅ 接近论文水平！域知识方法效果优秀")
            elif best_f1 >= 0.95:
                print(f"  🔶 良好水平，域知识方法有效")
            else:
                print(f"  ⚠️ 需要进一步调优域知识方法")

        # 推荐后续步骤
        print(f"\n🚀 推荐后续步骤:")
        print(f"  1. 使用选择的特征构建攻击序列")
        print(f"  2. 训练SeqGAN生成模型")
        print(f"  3. 进行数据增强和模型优化")

        # 保存域知识方法的配置
        if hasattr(self, 'refined_features'):
            domain_config = {
                'method': 'domain_knowledge_feature_selection',
                'target_features': target_features,
                'final_features': len(self.refined_features),
                'selected_features': self.refined_features,
                'performance': best_f1 if 'best_f1' in locals() else 0,
                'processing_time': total_time
            }

            # 保存配置到文件
            config_path = os.path.join(self.output_path, 'domain_knowledge_config.json')
            with open(config_path, 'w') as f:
                json.dump(domain_config, f, indent=4)
            print(f"✅ 域知识方法配置保存至: {config_path}")

    def build_attack_sequences(self, num_apt_sequences=1000, min_normal_insert=1, max_normal_insert=5):
        """构建攻击序列，参考dapt_preprocessing.py的方法"""
        print(f"\n🔗 构建攻击序列 (生成{num_apt_sequences}个APT序列)")

        # 确定目标变量
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'

        # 构建attack2id映射
        self._build_attack2id_mapping()

        # 按攻击阶段分组数据
        self._partition_data_by_stage(target)

        # 构建APT序列标签
        self._build_apt_sequence_labels(num_apt_sequences, min_normal_insert, max_normal_insert)

        # 构建正常序列标签
        self._build_normal_sequence_labels()

        # 为序列选择实际数据样本
        self._select_samples_for_sequences()

        # 转换标签序列为ID序列
        self._convert_labels_to_ids()

        # 分配最终序列标签
        self._assign_final_sequence_labels()

        # 保存序列数据
        self._save_sequence_results()

        print(f"✅ 攻击序列构建完成")
        print(f"  APT序列数量: {len(self.apt_sequences_data)}")
        print(f"  正常序列数量: {len(self.normal_sequences_data)}")
        print(f"  Attack2ID映射: {self.attack2id}")

        return self

    def build_session_based_attack_sequences(self, num_apt_sequences=1000, session_timeout=300):
        """基于会话的攻击序列构建方法"""
        print(f"\n🔗 构建基于会话的攻击序列 (生成{num_apt_sequences}个APT序列)")
        print("方法: 通过源IP、目标IP、协议和时间信息构建会话")

        # 确定目标变量
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'

        # 构建attack2id映射
        self._build_attack2id_mapping()

        # 基于会话构建攻击序列
        self._build_sessions_from_network_flows(session_timeout)

        # 从会话中提取攻击序列
        self._extract_attack_sequences_from_sessions(num_apt_sequences)

        # 构建正常序列
        self._build_normal_sequences_from_sessions(num_apt_sequences)

        # 转换标签序列为ID序列
        self._convert_session_labels_to_ids()

        # 分配最终序列标签
        self._assign_session_sequence_labels()

        # 保存序列数据
        self._save_session_sequence_results()

        print(f"✅ 基于会话的攻击序列构建完成")
        print(f"  APT序列数量: {len(self.session_apt_sequences_data)}")
        print(f"  正常序列数量: {len(self.session_normal_sequences_data)}")
        print(f"  Attack2ID映射: {self.attack2id}")

        return self

    def _build_sessions_from_network_flows(self, session_timeout):
        """基于网络流构建会话"""
        print("基于网络流构建会话...")

        # 确保有时间戳列
        if 'Timestamp' not in self.df.columns:
            print("  警告: 没有找到Timestamp列，使用索引作为时间顺序")
            self.df['Timestamp'] = pd.to_datetime(self.df.index, unit='s')
        else:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])

        # 按时间排序
        self.df = self.df.sort_values('Timestamp')

        # 定义会话键（源IP、目标IP、协议）
        session_keys = ['Src_IP_int', 'Dst_IP_int', 'Protocol_encoded']

        # 检查必要的列是否存在
        missing_cols = [col for col in session_keys if col not in self.df.columns]
        if missing_cols:
            print(f"  警告: 缺少会话键列: {missing_cols}")
            # 使用可用的列
            session_keys = [col for col in session_keys if col in self.df.columns]
            if not session_keys:
                print("  错误: 没有可用的会话键列")
                return

        print(f"  使用会话键: {session_keys}")

        # 构建会话
        self.sessions = {}
        session_id = 0

        # 按会话键分组
        for session_key, group in self.df.groupby(session_keys):
            # 按时间排序
            group = group.sort_values('Timestamp')

            # 根据时间间隔分割会话
            current_session = []
            last_time = None

            for idx, row in group.iterrows():
                current_time = row['Timestamp']

                # 如果时间间隔超过阈值，开始新会话
                if last_time is not None and (current_time - last_time).total_seconds() > session_timeout:
                    if len(current_session) > 1:  # 只保留有多个流的会话
                        self.sessions[session_id] = current_session
                        session_id += 1
                    current_session = []

                current_session.append(row.to_dict())
                last_time = current_time

            # 添加最后一个会话
            if len(current_session) > 1:
                self.sessions[session_id] = current_session
                session_id += 1

        print(f"  构建了 {len(self.sessions)} 个会话")

        # 分析会话中的攻击阶段分布
        self._analyze_session_attack_stages()

    def _analyze_session_attack_stages(self):
        """分析会话中的攻击阶段分布"""
        print("分析会话中的攻击阶段分布...")

        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'

        session_stage_stats = {
            'normal_only': 0,      # 只有正常流量
            'single_attack': 0,    # 单一攻击阶段
            'multi_attack': 0,     # 多个攻击阶段
            'complete_apt': 0      # 完整APT攻击链
        }

        self.attack_sessions = []  # 包含攻击的会话
        self.normal_sessions = []  # 只有正常流量的会话

        for session_id, flows in self.sessions.items():
            # 提取会话中的攻击阶段
            stages = [flow[target] for flow in flows]
            unique_stages = set(stages)

            # 分类会话
            if unique_stages == {0}:  # 只有正常流量
                session_stage_stats['normal_only'] += 1
                self.normal_sessions.append((session_id, flows))
            else:
                attack_stages = unique_stages - {0}  # 去除正常流量
                if len(attack_stages) == 1:
                    session_stage_stats['single_attack'] += 1
                else:
                    session_stage_stats['multi_attack'] += 1
                    # 检查是否是完整的APT攻击链
                    if {1, 2, 3, 4}.issubset(unique_stages):
                        session_stage_stats['complete_apt'] += 1

                self.attack_sessions.append((session_id, flows, list(attack_stages)))

        print(f"  会话统计:")
        for stat_name, count in session_stage_stats.items():
            print(f"    {stat_name}: {count}")

        print(f"  攻击会话: {len(self.attack_sessions)}")
        print(f"  正常会话: {len(self.normal_sessions)}")

    def _extract_attack_sequences_from_sessions(self, num_apt_sequences):
        """从会话中提取攻击序列"""
        print("从会话中提取攻击序列...")

        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'

        self.session_apt_sequences_data = []
        self.session_apt_sequences_labels = []

        import random
        random.seed(42)

        # 如果攻击会话不够，重复使用
        available_sessions = self.attack_sessions.copy()

        for i in range(num_apt_sequences):
            if not available_sessions:
                available_sessions = self.attack_sessions.copy()
                random.shuffle(available_sessions)

            session_id, flows, attack_stages = available_sessions.pop()

            # 按时间顺序提取攻击序列
            sequence_data = []
            sequence_labels = []

            for flow in flows:
                stage = flow[target]
                stage_label = self.stage_to_internal[stage]

                # 只保留选择的特征
                if hasattr(self, 'candidate_features'):
                    flow_data = {k: v for k, v in flow.items() if k in self.candidate_features}
                else:
                    flow_data = flow

                sequence_data.append(flow_data)
                sequence_labels.append(stage_label)

            self.session_apt_sequences_data.append(sequence_data)
            self.session_apt_sequences_labels.append(sequence_labels)

        print(f"  提取了 {len(self.session_apt_sequences_data)} 个APT序列")

    def _build_normal_sequences_from_sessions(self, num_sequences):
        """从会话中构建正常序列"""
        print("从会话中构建正常序列...")

        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'

        self.session_normal_sequences_data = []
        self.session_normal_sequences_labels = []

        import random
        random.seed(42)

        # 如果正常会话不够，重复使用
        available_sessions = self.normal_sessions.copy()

        for i in range(num_sequences):
            if not available_sessions:
                available_sessions = self.normal_sessions.copy()
                random.shuffle(available_sessions)

            session_id, flows = available_sessions.pop()

            # 提取正常序列
            sequence_data = []
            sequence_labels = []

            for flow in flows:
                stage = flow[target]
                stage_label = self.stage_to_internal[stage]

                # 只保留选择的特征
                if hasattr(self, 'candidate_features'):
                    flow_data = {k: v for k, v in flow.items() if k in self.candidate_features}
                else:
                    flow_data = flow

                sequence_data.append(flow_data)
                sequence_labels.append(stage_label)

            self.session_normal_sequences_data.append(sequence_data)
            self.session_normal_sequences_labels.append(sequence_labels)

        print(f"  构建了 {len(self.session_normal_sequences_data)} 个正常序列")

    def _convert_session_labels_to_ids(self):
        """转换会话标签序列为ID序列"""
        print("转换会话标签序列为ID序列...")

        try:
            self.session_apt_sequences_ids = [[self.attack2id[label] for label in seq] for seq in self.session_apt_sequences_labels]
            self.session_normal_sequences_ids = [[self.attack2id[label] for label in seq] for seq in self.session_normal_sequences_labels]
            print(f"  APT序列ID: {len(self.session_apt_sequences_ids)}")
            print(f"  正常序列ID: {len(self.session_normal_sequences_ids)}")
        except KeyError as e:
            print(f"错误: 标签 {e} 不在attack2id映射中")
            raise

    def _assign_session_sequence_labels(self):
        """分配会话序列标签"""
        print("分配会话序列标签...")

        # APT序列标签基于最高攻击阶段
        stage_to_final_label = {'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4, 'SN': 0}

        self.session_apt_labels = []
        for seq_labels in self.session_apt_sequences_labels:
            max_stage_num = 0
            for label in seq_labels:
                stage_num = stage_to_final_label.get(label, 0)
                max_stage_num = max(max_stage_num, stage_num)

            # 根据最高攻击阶段确定APT类型
            if max_stage_num == 1:
                apt_type = 1  # APT1: 仅侦察
            elif max_stage_num == 2:
                apt_type = 2  # APT2: 侦察+立足点
            elif max_stage_num == 3:
                apt_type = 3  # APT3: 前三阶段
            elif max_stage_num == 4:
                apt_type = 4  # APT4: 完整攻击链
            else:
                apt_type = 1  # 默认为APT1

            self.session_apt_labels.append(apt_type)

        # 正常序列标签都是0 (NAPT)
        self.session_normal_labels = [0] * len(self.session_normal_sequences_ids)

        print(f"  APT标签: {len(self.session_apt_labels)}")
        print(f"  正常标签: {len(self.session_normal_labels)}")

    def _save_session_sequence_results(self):
        """保存会话序列结果"""
        print("保存会话序列结果...")

        # 创建会话专用输出目录
        session_output_path = os.path.join(self.output_path, 'session_based')
        os.makedirs(session_output_path, exist_ok=True)

        # 保存attack2id映射
        attack2id_path = os.path.join(session_output_path, 'attack2id.json')
        with open(attack2id_path, 'w') as f:
            json.dump(self.attack2id, f, indent=4)
        print(f"  Attack2ID映射保存至: {attack2id_path}")

        # 保存APT序列数据
        apt_data_path = os.path.join(session_output_path, 'apt_sequences_data.json')
        with open(apt_data_path, 'w') as f:
            serializable_apt_data = [
                [{k: (v.item() if hasattr(v, 'item') else v) for k, v in step.items()} for step in seq]
                for seq in self.session_apt_sequences_data
            ]
            json.dump(serializable_apt_data, f, indent=2)
        print(f"  APT序列数据保存至: {apt_data_path}")

        # 保存正常序列数据
        normal_data_path = os.path.join(session_output_path, 'normal_sequences_data.json')
        with open(normal_data_path, 'w') as f:
            serializable_normal_data = [
                [{k: (v.item() if hasattr(v, 'item') else v) for k, v in step.items()} for step in seq]
                for seq in self.session_normal_sequences_data
            ]
            json.dump(serializable_normal_data, f, indent=2)
        print(f"  正常序列数据保存至: {normal_data_path}")

        # 保存序列标签
        apt_labels_path = os.path.join(session_output_path, 'apt_labels.npy')
        np.save(apt_labels_path, np.array(self.session_apt_labels))
        print(f"  APT标签保存至: {apt_labels_path}")

        normal_labels_path = os.path.join(session_output_path, 'normal_labels.npy')
        np.save(normal_labels_path, np.array(self.session_normal_labels))
        print(f"  正常标签保存至: {normal_labels_path}")

        # 保存序列ID
        apt_ids_path = os.path.join(session_output_path, 'apt_sequences_ids.npy')
        np.save(apt_ids_path, np.array(self.session_apt_sequences_ids, dtype=object), allow_pickle=True)
        print(f"  APT序列ID保存至: {apt_ids_path}")

        normal_ids_path = os.path.join(session_output_path, 'normal_sequences_ids.npy')
        np.save(normal_ids_path, np.array(self.session_normal_sequences_ids, dtype=object), allow_pickle=True)
        print(f"  正常序列ID保存至: {normal_ids_path}")

    def _build_attack2id_mapping(self):
        """构建attack2id映射"""
        print("构建attack2id映射...")

        # 基于Stage_encoded的值构建映射
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        unique_stages = sorted(self.df[target].unique())

        # 创建内部标签映射（修正版本）
        # 根据数据集实际分布：{0: 10000, 1: 5002, 2: 22968, 3: 12664, 4: 14366}
        # 其中1对应的是数据泄露阶段（数量5002），需要重新映射
        self.stage_to_internal = {
            0: 'SN',  # 正常流量 (Normal) - 10000样本
            4: 'S1',  # 侦察阶段 (Reconnaissance) - 14366样本
            2: 'S2',  # 建立立足点 (Establish Foothold) - 22968样本
            3: 'S3',  # 横向移动 (Lateral Movement) - 12664样本
            1: 'S4'   # 数据泄露 (Data Exfiltration) - 5002样本
        }

        # 构建attack2id映射
        self.attack2id = {
            'SN': 0,  # 正常流量
            'S1': 1,  # 侦察阶段
            'S2': 2,  # 建立立足点
            'S3': 3,  # 横向移动
            'S4': 4   # 数据泄露
        }

        print(f"  Attack2ID映射: {self.attack2id}")

    def _partition_data_by_stage(self, target):
        """按攻击阶段分组数据"""
        print("按攻击阶段分组数据...")

        self.stage_dataframes = {}

        for stage_encoded, internal_label in self.stage_to_internal.items():
            stage_data = self.df[self.df[target] == stage_encoded]
            if not stage_data.empty:
                # 只保留选择的特征
                if hasattr(self, 'candidate_features'):
                    stage_data = stage_data[self.candidate_features]
                self.stage_dataframes[internal_label] = stage_data
                print(f"  {internal_label}: {len(stage_data)} 样本")
            else:
                print(f"  {internal_label}: 0 样本 (跳过)")

    def _build_apt_sequence_labels(self, num_apt_sequences, min_normal_insert, max_normal_insert):
        """构建APT序列标签"""
        print("构建APT序列标签...")

        import random
        random.seed(42)

        # 可用的攻击阶段（按顺序）
        available_attack_stages = ['S1', 'S2', 'S3', 'S4']
        normal_stage = 'SN'

        # 序列类型数量（1-4个攻击阶段）
        num_sequence_types = len(available_attack_stages)
        num_per_type = num_apt_sequences // num_sequence_types
        remainder = num_apt_sequences % num_sequence_types

        self.apt_sequences_labels = []

        for i in range(num_sequence_types):
            # 定义当前序列类型的基础攻击阶段
            current_base_stages = available_attack_stages[:i+1]
            print(f"  生成类型{i+1} (最大阶段: {current_base_stages[-1]}): {current_base_stages}")

            num_for_this_type = num_per_type + (1 if i < remainder else 0)

            for _ in range(num_for_this_type):
                # 基础序列
                current_seq = list(current_base_stages)

                # 插入随机数量的正常流量
                num_sn_to_insert = random.randint(min_normal_insert, max_normal_insert)
                for _ in range(num_sn_to_insert):
                    insert_pos = random.randint(0, len(current_seq))
                    current_seq.insert(insert_pos, normal_stage)

                self.apt_sequences_labels.append(current_seq)

        print(f"  构建了{len(self.apt_sequences_labels)}个APT序列标签")

    def _build_normal_sequence_labels(self):
        """构建正常序列标签"""
        print("构建正常序列标签...")

        self.normal_sequences_labels = []

        # 为每个APT序列生成对应长度的正常序列
        for apt_seq in self.apt_sequences_labels:
            normal_len = len(apt_seq)
            self.normal_sequences_labels.append(['SN'] * normal_len)

        print(f"  构建了{len(self.normal_sequences_labels)}个正常序列标签")

    def _select_samples_for_sequences(self):
        """为序列选择实际数据样本"""
        print("为序列选择实际数据样本...")

        import random
        random.seed(42)

        self.apt_sequences_data = []
        self.normal_sequences_data = []

        # 为APT序列选择样本
        for seq_labels in self.apt_sequences_labels:
            sequence_data = []
            for stage_label in seq_labels:
                if stage_label in self.stage_dataframes:
                    stage_df = self.stage_dataframes[stage_label]
                    if not stage_df.empty:
                        # 随机选择一个样本
                        sample = stage_df.sample(1).iloc[0].to_dict()
                        sequence_data.append(sample)
                    else:
                        print(f"    警告: {stage_label} 阶段数据为空")
                else:
                    print(f"    警告: 找不到 {stage_label} 阶段数据")

            if sequence_data:  # 只添加非空序列
                self.apt_sequences_data.append(sequence_data)

        # 为正常序列选择样本
        for seq_labels in self.normal_sequences_labels:
            sequence_data = []
            for stage_label in seq_labels:  # 都是'SN'
                if stage_label in self.stage_dataframes:
                    stage_df = self.stage_dataframes[stage_label]
                    if not stage_df.empty:
                        sample = stage_df.sample(1).iloc[0].to_dict()
                        sequence_data.append(sample)

            if sequence_data:
                self.normal_sequences_data.append(sequence_data)

        print(f"  APT序列数据: {len(self.apt_sequences_data)}")
        print(f"  正常序列数据: {len(self.normal_sequences_data)}")

    def _convert_labels_to_ids(self):
        """转换标签序列为ID序列"""
        print("转换标签序列为ID序列...")

        try:
            self.apt_sequences_ids = [[self.attack2id[label] for label in seq] for seq in self.apt_sequences_labels]
            self.normal_sequences_ids = [[self.attack2id[label] for label in seq] for seq in self.normal_sequences_labels]
            print(f"  APT序列ID: {len(self.apt_sequences_ids)}")
            print(f"  正常序列ID: {len(self.normal_sequences_ids)}")
        except KeyError as e:
            print(f"错误: 标签 {e} 不在attack2id映射中")
            raise

    def _assign_final_sequence_labels(self):
        """分配最终序列标签"""
        print("分配最终序列标签...")

        # APT序列标签基于最高攻击阶段
        stage_to_final_label = {'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4, 'SN': 0}

        self.apt_labels = []
        for seq_labels in self.apt_sequences_labels:
            max_stage_num = 0
            for label in seq_labels:
                stage_num = stage_to_final_label.get(label, 0)
                max_stage_num = max(max_stage_num, stage_num)
            self.apt_labels.append(max_stage_num if max_stage_num > 0 else 1)

        # 正常序列标签都是0
        self.normal_labels = [0] * len(self.normal_sequences_ids)

        print(f"  APT标签: {len(self.apt_labels)}")
        print(f"  正常标签: {len(self.normal_labels)}")

    def _save_sequence_results(self):
        """保存序列结果"""
        print("保存序列结果...")

        # 保存attack2id映射
        attack2id_path = os.path.join(self.output_path, 'attack2id.json')
        with open(attack2id_path, 'w') as f:
            json.dump(self.attack2id, f, indent=4)
        print(f"  Attack2ID映射保存至: {attack2id_path}")

        # 保存APT序列数据
        apt_data_path = os.path.join(self.output_path, 'apt_sequences_data.json')
        with open(apt_data_path, 'w') as f:
            # 转换numpy类型为Python类型
            serializable_apt_data = [
                [{k: (v.item() if hasattr(v, 'item') else v) for k, v in step.items()} for step in seq]
                for seq in self.apt_sequences_data
            ]
            json.dump(serializable_apt_data, f, indent=2)
        print(f"  APT序列数据保存至: {apt_data_path}")

        # 保存正常序列数据
        normal_data_path = os.path.join(self.output_path, 'normal_sequences_data.json')
        with open(normal_data_path, 'w') as f:
            serializable_normal_data = [
                [{k: (v.item() if hasattr(v, 'item') else v) for k, v in step.items()} for step in seq]
                for seq in self.normal_sequences_data
            ]
            json.dump(serializable_normal_data, f, indent=2)
        print(f"  正常序列数据保存至: {normal_data_path}")

        # 保存序列标签
        apt_labels_path = os.path.join(self.output_path, 'apt_labels.npy')
        np.save(apt_labels_path, np.array(self.apt_labels))
        print(f"  APT标签保存至: {apt_labels_path}")

        normal_labels_path = os.path.join(self.output_path, 'normal_labels.npy')
        np.save(normal_labels_path, np.array(self.normal_labels))
        print(f"  正常标签保存至: {normal_labels_path}")

        # 保存序列ID
        apt_ids_path = os.path.join(self.output_path, 'apt_sequences_ids.npy')
        np.save(apt_ids_path, np.array(self.apt_sequences_ids, dtype=object), allow_pickle=True)
        print(f"  APT序列ID保存至: {apt_ids_path}")

        normal_ids_path = os.path.join(self.output_path, 'normal_sequences_ids.npy')
        np.save(normal_ids_path, np.array(self.normal_sequences_ids, dtype=object), allow_pickle=True)
        print(f"  正常序列ID保存至: {normal_ids_path}")




def main():
    """主函数"""
    print("🎯 增强版APT数据预处理器")
    print("特点: XGBoost特征选择 + 动态特征数量优化 + 两种攻击序列构建方法")
    print("="*80)

    # 设置路径
    input_path = r"D:\PycharmProjects\DSRL-APT-2023\DSRL-APT-2023.csv"
    output_path = "enhanced_apt_output"

    # 创建处理器并运行
    processor = EnhancedAPTPreprocessor(input_path, output_path)

    # 运行完整的预处理和评估流程
    processor.load_data().clean_data().create_statistical_features().encode_and_normalize().prepare_paper_aligned_features()
    available_features = len(processor.candidate_features)

    print(f"\n📊 特征数量分析:")
    print(f"  可用特征数量: {available_features}")
    print(f"  将使用XGBoost分析寻找最优特征数量")

    # 测试不同的特征选择策略
    print(f"\n🔬 测试不同特征选择策略")
    print("="*60)

    # 策略1: 包含论文特征 + 移除过拟合特征
    print(f"\n📋 策略1: 包含论文特征 + 移除过拟合特征")
    processor1 = EnhancedAPTPreprocessor(input_path, output_path + "_strategy1")
    processor1.load_data().clean_data().create_statistical_features().encode_and_normalize().prepare_paper_aligned_features()
    processor1.evaluate_with_fold_internal_pipeline(n_features=None, cv_folds=5, optimize_hyperparams=True, n_trials=10)
    f1_strategy1 = processor1.cv_results['XGBoost']['f1_macro']['mean']
    print(f"策略1 F1分数: {f1_strategy1:.4f}")

    # 策略2: 不包含论文特征，测试其他特征
    print(f"\n📋 策略2: 不包含论文特征，测试其他特征")
    processor2 = EnhancedAPTPreprocessor(input_path, output_path + "_strategy2")
    processor2.load_data().clean_data().create_statistical_features().encode_and_normalize().prepare_paper_aligned_features()
    # 修改过滤策略
    processor2.filter_candidate_features(remove_overfitting_features=True, include_paper_features=False)
    processor2.evaluate_with_fold_internal_pipeline(n_features=None, cv_folds=5, optimize_hyperparams=True, n_trials=10)
    f1_strategy2 = processor2.cv_results['XGBoost']['f1_macro']['mean']
    print(f"策略2 F1分数: {f1_strategy2:.4f}")

    # 策略3: 包含所有特征（包括可能过拟合的）
    print(f"\n📋 策略3: 包含所有特征（包括可能过拟合的）")
    processor3 = EnhancedAPTPreprocessor(input_path, output_path + "_strategy3")
    processor3.load_data().clean_data().create_statistical_features().encode_and_normalize().prepare_paper_aligned_features()
    # 不过滤任何特征
    processor3.filter_candidate_features(remove_overfitting_features=False, include_paper_features=True)
    processor3.evaluate_with_fold_internal_pipeline(n_features=None, cv_folds=5, optimize_hyperparams=True, n_trials=10)
    f1_strategy3 = processor3.cv_results['XGBoost']['f1_macro']['mean']
    print(f"策略3 F1分数: {f1_strategy3:.4f}")

    # 比较结果
    print(f"\n📊 特征选择策略对比:")
    print("="*60)
    strategies = [
        ("策略1 (论文特征+过滤)", f1_strategy1, processor1),
        ("策略2 (无论文特征)", f1_strategy2, processor2),
        ("策略3 (所有特征)", f1_strategy3, processor3)
    ]

    best_strategy = max(strategies, key=lambda x: x[1])

    for name, f1, processor in strategies:
        status = "✅ 最佳" if (name, f1, processor) == best_strategy else ""
        print(f"  {name}: {f1:.4f} {status}")

    print(f"\n🏆 最佳策略: {best_strategy[0]} (F1: {best_strategy[1]:.4f})")

    # 保存最佳结果
    best_processor = best_strategy[2]
    best_processor.save_results()

    # 比较两种攻击序列构建方法
    print(f"\n🔬 比较两种攻击序列构建方法")
    print("="*60)

    # 方法1: 基于AttackID的随机构建
    print(f"\n📋 方法1: 基于AttackID的随机构建")
    processor_method1 = EnhancedAPTPreprocessor(input_path, output_path + "_method1_attackid")
    processor_method1.load_data().clean_data().create_statistical_features().encode_and_normalize().prepare_paper_aligned_features()
    processor_method1.build_attack_sequences(num_apt_sequences=1000, min_normal_insert=1, max_normal_insert=5)

    # 方法2: 基于会话的构建
    print(f"\n📋 方法2: 基于会话的构建")
    processor_method2 = EnhancedAPTPreprocessor(input_path, output_path + "_method2_session")
    processor_method2.load_data().clean_data().create_statistical_features().encode_and_normalize().prepare_paper_aligned_features()
    processor_method2.build_session_based_attack_sequences(num_apt_sequences=1000, session_timeout=300)

    # 分析两种方法的差异
    print(f"\n📊 两种方法对比分析:")
    print("="*60)

    # 方法1统计
    print(f"\n方法1 (AttackID随机构建):")
    print(f"  APT序列数量: {len(processor_method1.apt_sequences_data)}")
    print(f"  正常序列数量: {len(processor_method1.normal_sequences_data)}")

    # 分析方法1的序列长度分布
    method1_lengths = [len(seq) for seq in processor_method1.apt_sequences_data]
    print(f"  APT序列长度: 平均={np.mean(method1_lengths):.1f}, 最小={min(method1_lengths)}, 最大={max(method1_lengths)}")

    # 分析方法1的APT类型分布
    from collections import Counter
    method1_apt_types = Counter(processor_method1.apt_labels)
    print(f"  APT类型分布: {dict(method1_apt_types)}")

    # 方法2统计
    print(f"\n方法2 (会话构建):")
    print(f"  APT序列数量: {len(processor_method2.session_apt_sequences_data)}")
    print(f"  正常序列数量: {len(processor_method2.session_normal_sequences_data)}")

    # 分析方法2的序列长度分布
    method2_lengths = [len(seq) for seq in processor_method2.session_apt_sequences_data]
    print(f"  APT序列长度: 平均={np.mean(method2_lengths):.1f}, 最小={min(method2_lengths)}, 最大={max(method2_lengths)}")

    # 分析方法2的APT类型分布
    method2_apt_types = Counter(processor_method2.session_apt_labels)
    print(f"  APT类型分布: {dict(method2_apt_types)}")

    # 比较分析
    print(f"\n🔍 方法对比总结:")
    print(f"  序列长度差异:")
    print(f"    方法1平均长度: {np.mean(method1_lengths):.1f}")
    print(f"    方法2平均长度: {np.mean(method2_lengths):.1f}")
    print(f"    长度差异: {np.mean(method2_lengths) - np.mean(method1_lengths):+.1f}")

    print(f"\n  真实性分析:")
    print(f"    方法1: 随机构建，可能不符合真实攻击时序")
    print(f"    方法2: 基于真实网络会话，保持时间和网络关系的连续性")

    print(f"\n  推荐使用:")
    if np.mean(method2_lengths) > np.mean(method1_lengths):
        print(f"    ✅ 推荐方法2 (会话构建): 序列更长，更符合真实攻击场景")
    else:
        print(f"    ⚠️ 两种方法各有优势，建议根据具体需求选择")

    print(f"\n✅ 攻击序列构建方法比较完成")
    print(f"  方法1结果保存在: {output_path}_method1_attackid")
    print(f"  方法2结果保存在: {output_path}_method2_session")


if __name__ == "__main__":
    main()
