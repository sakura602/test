"""
增强版APT数据预处理器
- 丰富的统计量特征
- 详细的时间属性
- 随机森林特征选择
- 多模型交叉验证 (RF, MLP, SVM)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import socket
import struct
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
import os
import time
import json
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

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

            # 删除原始时间戳和临时datetime列
            self.df.drop(columns=['Timestamp', 'datetime'], inplace=True, errors='ignore')

        except Exception as e:
            print(f"⚠️ 时间戳处理失败: {e}")
            # 如果处理失败，删除时间戳列
            self.df.drop(columns=['Timestamp'], inplace=True, errors='ignore')

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

    def prepare_paper_aligned_features(self):
        """按论文方法准备特征：只用网络流特征+时间特征，排除Activity等标签信息"""
        print(f"\n🎯 按论文方法准备特征（排除Activity等标签信息）")

        # 确定目标变量
        if 'Stage_encoded' in self.df.columns:
            target = 'Stage_encoded'
        elif 'Label' in self.df.columns:
            target = 'Label'
        else:
            raise ValueError("找不到目标变量列")

        # 获取所有数值特征
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # 排除目标变量和Activity相关的编码列（避免标签泄露）
        exclude_cols = [target, 'Activity_encoded', 'Stage_encoded']
        if target == 'Stage_encoded':
            exclude_cols.remove('Stage_encoded')  # 如果Stage_encoded是目标，就不排除它

        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # 确保时间特征被包含（论文Table 3中的关键特征）
        time_features = ['hour', 'minute', 'day_of_week']  # 移除second，按论文只用这3个
        for tf in time_features:
            if tf in self.df.columns and tf not in feature_cols:
                feature_cols.append(tf)

        # 确保Protocol编码被包含（网络特征的一部分）
        if 'Protocol_encoded' in self.df.columns and 'Protocol_encoded' not in feature_cols:
            feature_cols.append('Protocol_encoded')

        print(f"网络流特征数量: {len([f for f in feature_cols if f not in time_features + ['Protocol_encoded']])}")
        print(f"时间特征数量: {len([f for f in feature_cols if f in time_features])}")
        print(f"协议特征数量: {len([f for f in feature_cols if f == 'Protocol_encoded'])}")
        print(f"候选特征总数: {len(feature_cols)}")

        # 检查是否意外包含了Activity
        activity_features = [f for f in feature_cols if 'activity' in f.lower()]
        if activity_features:
            print(f"⚠️ 警告：发现Activity相关特征，将被移除: {activity_features}")
            feature_cols = [f for f in feature_cols if f not in activity_features]

        self.candidate_features = feature_cols
        return self

    def evaluate_with_fold_internal_pipeline(self, n_features=46, cv_folds=10):
        """按论文方法：每折内部做特征选择+标准化+分类的完整pipeline"""
        print(f"\n🎯 论文方法：每折内Pipeline(特征选择→标准化→分类)")

        # 准备数据
        target = 'Stage_encoded' if 'Stage_encoded' in self.df.columns else 'Label'
        X = self.df[self.candidate_features]
        y = self.df[target]

        print(f"候选特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"类别分布: {dict(y.value_counts().sort_index())}")

        # 按论文配置模型
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
                learning_rate_init=1e-3,
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

            # 构建折内Pipeline：特征选择 → 标准化 → 分类器
            if name == 'RandomForest':
                # RF不需要标准化，但需要特征选择
                pipeline = Pipeline([
                    ('feat_sel', SelectFromModel(
                        RandomForestClassifier(n_estimators=100, random_state=42),
                        max_features=n_features
                    )),
                    ('clf', clf)
                ])
            else:
                # MLP和SVM需要特征选择+标准化
                pipeline = Pipeline([
                    ('feat_sel', SelectFromModel(
                        RandomForestClassifier(n_estimators=100, random_state=42),
                        max_features=n_features
                    )),
                    ('scaler', StandardScaler()),
                    ('clf', clf)
                ])

            # 交叉验证评估
            cv_results = cross_validate(
                pipeline, X, y,
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

        return self

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


def main():
    """主函数"""
    print("🎯 增强版APT数据预处理器")
    print("特点: 丰富统计量 + 时间特征 + 随机森林特征选择 + 多模型评估")
    print("="*80)

    # 设置路径
    input_path = r"D:\PycharmProjects\DSRL-APT-2023\DSRL-APT-2023.csv"
    output_path = "enhanced_apt_output"

    # 创建处理器并运行
    processor = EnhancedAPTPreprocessor(input_path, output_path)
    processor.run_complete_pipeline(n_features=46, cv_folds=10)


if __name__ == "__main__":
    main()
