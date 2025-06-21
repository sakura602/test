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

    def select_features_with_rf_cv(self, n_features=46, n_estimators=100, cv_folds=10):
        """基于10折CV的鲁棒随机森林特征选择，只用数值特征"""
        print(f"\n🌲 用{cv_folds}折CV做鲁棒特征选择，目标Top {n_features}")

        # 确定目标变量
        if 'Stage_encoded' in self.df.columns:
            target = 'Stage_encoded'
        elif 'Label' in self.df.columns:
            target = 'Label'
        else:
            raise ValueError("找不到目标变量列")

        # 只保留数值列，并排除目标
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num_cols if c != target]

        X = self.df[feature_cols]
        y = self.df[target]

        print(f"数值特征总数: {len(feature_cols)}, 样本数: {len(X)}")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        freq = pd.Series(0, index=feature_cols)

        print("🔄 折内选特征中…")
        for fold, (tr, _) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=n_estimators,
                                       random_state=42,
                                       n_jobs=-1),
                max_features=n_features
            ).fit(X_tr, y_tr)
            chosen = X_tr.columns[selector.get_support()]
            freq[chosen] += 1
            print(f"  折 {fold:2d}: 选中 {len(chosen)} 个特征")

        # 汇总选频，取Top n_features
        self.selected_features = freq.sort_values(ascending=False).head(n_features).index.tolist()
        print("✅ 最终Top特征：")
        for i, f in enumerate(self.selected_features, 1):
            print(f"  {i:2d}. {f}")

        return self

    def evaluate_multiple_models(self, cv_folds=10):
        """在已选特征上，用10折CV对多模型做Macro指标评估"""
        print(f"\n🎯 用{cv_folds}折CV评估多模型…")
        # 准备数据
        target = 'Stage_encoded' if 'Stage_encoded' in self.df else 'Label'
        X = self.df[self.selected_features]
        y = self.df[target]

        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                early_stopping=True,
                random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, random_state=42)
        }

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        self.cv_results = {}
        for name, clf in models.items():
            print(f"\n🚀 评估模型: {name}")
            # MLP/SVM 前做标准化
            if name in ('MLP', 'SVM'):
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', clf)
                ])
            else:
                pipe = clf

            res = cross_validate(
                pipe, X, y,
                cv=skf,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )
            # 汇总
            stats = {m: (res[f'test_{m}'].mean(), res[f'test_{m}'].std())
                     for m in scoring}
            for m, (mu, sd) in stats.items():
                print(f"  {m:<15}: {mu:.4f} ± {sd:.4f}")
            self.cv_results[name] = stats

        return self

    def detailed_model_analysis(self, cv_folds=10):
        """在10折CV中，收集所有折的预测，输出整体Classification Report和Confusion Matrix"""
        print(f"\n🔍 详细模型分析 (聚合10折结果)…")
        target = 'Stage_encoded' if 'Stage_encoded' in self.df else 'Label'
        X = self.df[self.selected_features]
        y = self.df[target]

        # 只选表现最好的模型
        best = max(self.cv_results,
                   key=lambda nm: self.cv_results[nm]['f1_macro'][0])
        print(f"🏆 最佳模型: {best}")

        # 重建好管道
        if best in ('MLP', 'SVM'):
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', {
                    'MLP': self.models_performance['MLP'],
                    'SVM': self.models_performance['SVM']
                }[best])
            ])
        else:
            pipe = RandomForestClassifier(
                n_estimators=200, class_weight='balanced',
                random_state=42, n_jobs=-1)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        y_true_all, y_pred_all = [], []

        for tr, te in skf.split(X, y):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            y_pred = pipe.predict(X.iloc[te])
            y_true_all.append(y.iloc[te].values)
            y_pred_all.append(y_pred)

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)

        print("\n--- Aggregated Classification Report ---")
        print(classification_report(y_true, y_pred))
        print("--- Aggregated Confusion Matrix ---")
        print(confusion_matrix(y_true, y_pred))
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
            # 执行完整流程
            (self
             .load_data()
             .clean_data()
             .create_statistical_features()
             .encode_and_normalize()
             .select_features_with_rf_cv(n_features=n_features, cv_folds=cv_folds)
             .evaluate_multiple_models(cv_folds=cv_folds)
             .detailed_model_analysis()
             .save_results())

            total_time = time.time() - start_time

            print(f"\n🎉 预处理和评估完成！")
            print(f"⏱️ 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
            print(f"📊 最终数据形状: {self.df.shape}")
            print(f"🎯 选择特征数量: {len(self.selected_features)}")

            # 打印最终结果摘要
            if self.cv_results:
                print(f"\n📈 最终模型性能摘要 ({cv_folds}折交叉验证):")
                for model_name, metrics in self.cv_results.items():
                    print(f"  {model_name}: F1={metrics['f1_mean']:.4f}±{metrics['f1_std']:.3f}")

            # 打印特征选择摘要
            if hasattr(self, 'feature_importance'):
                print(f"\n🎯 特征选择摘要:")
                high_freq = (self.feature_importance['selection_frequency'] >= 0.8).sum()
                print(f"  高稳定性特征 (≥80%): {high_freq}/{n_features}")
                print(f"  平均重要性: {self.feature_importance.head(n_features)['importance_mean'].mean():.6f}")

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
