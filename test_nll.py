import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer as Imputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from math import ceil
import keras
from keras.models import Model, Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# import eli5
import joblib

# 在文件开头定义标签映射
# 更新标签映射，确保与apt_sequence_builder.py中的标签定义一致
label_d = {
    "BENIGN": 0,  
    "Reconnaissance": 1,
    "Establish Foothold": 2,
    "Lateral Movement": 3,
    "Data Exfiltration": 4
}

# 反向映射
label_id_to_name = {v: k for k, v in label_d.items()}

def datetime_to_timestamp(dt):
    try:
        return datetime.strptime(dt, '%m/%d/%Y %H:%M').weekday()
    except:
        return datetime.strptime(dt, '%Y-%m-%d %H:%M:%S').weekday()
    
def clean_dataset(df):
    """
    清理数据集，处理缺失值和异常值
    
    参数:
    df: 数据框
    
    返回:
    清理后的数据框
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    
    # 删除包含NaN的行
    df = df.dropna()
    
    # 识别数值列
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) > 0:
        # 只对数值列检查无穷值
        numeric_df = df[numeric_cols]
        indices_to_keep = ~numeric_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
        df = df.loc[indices_to_keep]
    
    return df

def train_test_dataset(df_train, df_test, deep=True):
    labelencoder = LabelEncoder()
    X_train = df_train.drop(columns=['Label']).copy()
    y_train = df_train.iloc[:, -1].values.reshape(-1,1).copy()
    y_train = np.ravel(y_train).copy()
    if deep:
        y_train = to_categorical(y_train)
    
    if df_test is not None:
        X_test = df_test.drop(columns=['Label']).copy()
        y_test = df_test.iloc[:, -1].values.reshape(-1,1).copy()
        y_test = np.ravel(y_test).copy()
        if deep:
            y_test = to_categorical(y_test)
        return X_train, X_test, y_train, y_test
    else:
        return train_test_split(X_train, y_train)
    
def show_cm(cm):
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()

def from_categorical(y_):
    return np.array([np.argmax(i) for i in y_])

def discretize(a):
    return 1 if a > 0.5 else 0

def labels_to_numbers(df: pd.DataFrame, name='Label'):
    labels = df[name].unique()
    d = {label: idx for idx, label in enumerate(labels)}
    return d

def number_to_label(df, number, name='Label'):
    labels = df[name].unique()
    d = {idx: label for idx, label in enumerate(labels)}
    return d[number]
        
def prepare_df(df: pd.DataFrame, dropcols=None, scaler=None, ldict=None):
    """
    准备数据集，包括删除列、标准化和标签编码
    
    参数:
    df: 数据框
    dropcols: 要删除的列列表
    scaler: 标准化方法，'minmax'或'standard'
    ldict: 标签字典
    
    返回:
    准备好的数据框
    """
    temp_df = df.copy()
    session_id_col = 'Session_ID'
    # 删除指定的列，但强制保留Session_ID
    if dropcols:
        dropcols_ = [col for col in dropcols if col != session_id_col]
        temp_df = temp_df.drop(columns=dropcols_, errors='ignore')
    # 如果Session_ID丢失且原始df有，则补回
    if session_id_col not in temp_df.columns and session_id_col in df.columns:
        temp_df[session_id_col] = df[session_id_col]
    # 将标签映射到数字
    if not ldict:
        ltn_dict = labels_to_numbers(temp_df)
    else:
        ltn_dict = ldict
    temp_df['Label'] = temp_df['Label'].map(ltn_dict)
    # 清理数据集
    temp_df = clean_dataset(temp_df)
    # 识别数值列（排除Label和非数值列）
    numeric_cols = temp_df.select_dtypes(include=['number']).columns.tolist()
    if 'Label' in numeric_cols:
        numeric_cols.remove('Label')
    # 标准化数值列
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    if scaler and numeric_cols:
        temp_df[numeric_cols] = scaler.fit_transform(temp_df[numeric_cols])
    return temp_df

def encode_categorical_features(df):
    """
    对非数值特征进行编码
    
    参数:
    df: 数据框
    
    返回:
    编码后的数据框和编码器字典
    """
    result_df = df.copy()
    encoders = {}
    
    # 识别非数值列
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    for col in non_numeric_cols:
        # 使用LabelEncoder对非数值特征进行编码
        le = LabelEncoder()
        result_df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return result_df, encoders

def binarize_label(y, label):
    idx = y == label
    y[idx] = 1
    y[~idx] = 0
    
def evaluate_model(model, df, binarize=None):
    X = df.drop(columns=['Label']).copy()
    y = df.iloc[:, -1].values.reshape(-1,1).copy()
    y = np.ravel(y).copy()
    if binarize is not None:
        binarize_label(y, binarize)
    
    model.evaluate(X, y)
    
    y_predicted = model.predict(X)
    cm = confusion_matrix(y, vdiscretize(y_predicted))
    show_cm(cm)
    
def unsup_compare_results(model, X, y_true):
    y_pred = model.predict(X)
    return f1_score(y_true, y_pred, average=None)

def plot_cm(model, X, y_true):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    show_cm(cm)
    
def make_binary_svm(train, test, l):
    """
    创建二分类SVM的数据集
    
    参数:
    train: 训练数据集
    test: 测试数据集
    l: 目标标签
    
    返回:
    训练特征、测试特征、训练标签、测试标签
    """
    _X_train, _X_test, _y_train, _y_test = train_test_split(train, test, deep=False)
    binarize_label(_y_train, l)
    binarize_label(_y_test, l)
    # _y_train[_y_train == 0] = -1
    # _y_train[_y_train == 0] = -1
    return _X_train, _X_test, _y_train, _y_test

def check_all_labels(train, test, model_constructor, l_name='Label', constructor_kwargs=None):
    labels = train[l_name].unique()
    for label in labels:
        print(f'============= Label {number_to_label(clean_test, label)} ==================')
        X_train, X_test, y_train, y_test = make_binary_svm(clean_train, clean_test, label)
        if constructor_kwargs:
            model = model_constructor(**constructor_kwargs)
        else:
            model = model_constructor()
        model.fit(X_train, y_train)
        print(unsup_compare_results(model, X_test, y_test))
        probs = model.predict_proba(X_test)[:, 1]
        plot_roc(y_test, probs)
        
def plot_roc(y, probs, label='Classifier'):
    ns_probs = [0 for _ in range(len(y))]
    
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, label=label)
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.05])
    plt.legend()
    plt.show()
        
vdiscretize = np.vectorize(discretize)

def display_class_metrics(y_true, y_pred, drev, name=""):
    """显示每个类别的评估指标"""
    print(f"\n=== {name}每个类别的F1分数 ===")
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(y_true, y_pred)
    
    # 获取所有可能的类别ID
    all_classes = sorted(set(y_true) | set(y_pred))
    
    # 创建结果字典，包含所有类别
    results = {}
    
    # 确保包含所有在字典中定义的类别
    for class_id in drev.keys():
        results[class_id] = {
            'name': drev.get(class_id, f"未知类别-{class_id}"),
            'f1': 0.0,
            'support': 0
        }
    
    # 填充实际计算的F1分数
    for i, class_id in enumerate(sorted(set(y_true) | set(y_pred))):
        if i < len(class_f1):
            class_name = drev.get(int(class_id), f"未知类别-{class_id}")
            results[int(class_id)] = {
                'name': class_name,
                'f1': class_f1[i],
                'support': class_support[i]
            }
    
    # 按类别ID排序并显示结果
    for class_id in sorted(results.keys()):
        result = results[class_id]
        print(f"类别 {class_id} ({result['name']}): F1={result['f1']:.4f}, 支持度={result['support']}")
    
    return results

def sort_by_timestamp(df, timestamp_col='Timestamp'):
    """
    按时间戳对数据进行排序
    
    参数:
    df: 数据框
    timestamp_col: 时间戳列名
    
    返回:
    排序后的数据框
    """
    if timestamp_col not in df.columns:
        print(f"警告: 数据集中没有'{timestamp_col}'列，无法按时间戳排序")
        return df
    
    # 尝试将时间戳转换为datetime格式
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        print(f"警告: 时间戳转换失败，将按原始格式排序。错误: {e}")
    
    # 按时间戳排序
    sorted_df = df.sort_values(by=timestamp_col)
    print(f"数据已按'{timestamp_col}'列排序")
    
    return sorted_df

def merge_by_identifiers(dataframes, identifiers=['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol'], time_window_seconds=60):
    """
    根据共同的事件标识符合并多个数据框，并考虑时间窗口
    
    参数:
    dataframes: 数据框列表
    identifiers: 用于合并的标识符列表
    time_window_seconds: 时间窗口（秒），超过此时间窗口的记录即使标识符相同也被视为不同事件
    
    返回:
    合并后的数据框
    """
    if not dataframes:
        print("没有数据框可合并")
        return None
    
    # 检查所有数据框是否都包含所有标识符
    for i, df in enumerate(dataframes):
        missing_cols = [col for col in identifiers if col not in df.columns]
        if missing_cols:
            print(f"警告: 数据框 {i} 缺少以下标识符列: {missing_cols}")
            print("将使用所有数据框共有的列进行合并")
            identifiers = [col for col in identifiers if col not in missing_cols]
    
    if not identifiers:
        print("没有共同的标识符列可用于合并")
        print("将直接连接所有数据框")
        return pd.concat(dataframes)
    
    print(f"使用以下标识符合并数据: {identifiers}")
    
    # 合并数据框
    merged_df = pd.concat(dataframes)
    
    # 确保有时间戳列
    timestamp_col = 'Timestamp'
    if timestamp_col in merged_df.columns:
        # 转换时间戳为datetime格式
        try:
            merged_df[timestamp_col] = pd.to_datetime(merged_df[timestamp_col])
            
            # 检查是否有重复的行（基于标识符）
            duplicates = merged_df.duplicated(subset=identifiers, keep=False)
            if duplicates.any():
                print(f"发现 {duplicates.sum()} 行潜在重复数据（基于标识符）")
                print(f"应用时间窗口 {time_window_seconds} 秒进行智能去重")
                
                # 按标识符和时间戳排序
                merged_df = merged_df.sort_values(by=identifiers + [timestamp_col])
                
                # 创建一个标志列，标记要保留的行
                merged_df['keep'] = True
                
                # 获取唯一的标识符组合
                grouped = merged_df.groupby(identifiers)
                
                # 对每个组应用时间窗口去重
                for name, group in grouped:
                    if len(group) > 1:
                        # 按时间戳排序
                        sorted_group = group.sort_values(by=timestamp_col)
                        
                        # 计算时间差
                        time_diffs = sorted_group[timestamp_col].diff().dt.total_seconds()
                        
                        # 第一行总是保留
                        keep_mask = [True] * len(sorted_group)
                        
                        # 检查每行与前一行的时间差
                        for i in range(1, len(sorted_group)):
                            # 如果时间差小于窗口且前一行被保留，则当前行不保留
                            if time_diffs.iloc[i] < time_window_seconds and keep_mask[i-1]:
                                keep_mask[i] = False
                                
                                # 检查是否有不同的Label值，如果有则保留
                                if 'Label' in sorted_group.columns:
                                    current_label = sorted_group['Label'].iloc[i]
                                    prev_label = sorted_group['Label'].iloc[i-1]
                                    if current_label != prev_label:
                                        keep_mask[i] = True
                                        
                                # 检查是否有Reconnaisance标签，如果有则保留
                                if 'Label' in sorted_group.columns:
                                    if sorted_group['Label'].iloc[i] == 'Reconnaisance':
                                        keep_mask[i] = True
                        
                        # 更新keep标志
                        sorted_group_indices = sorted_group.index
                        for i, idx in enumerate(sorted_group_indices):
                            merged_df.loc[idx, 'keep'] = keep_mask[i]
                
                # 只保留标记为keep=True的行
                before_count = len(merged_df)
                merged_df = merged_df[merged_df['keep']].drop(columns=['keep'])
                after_count = len(merged_df)
                print(f"智能去重后保留了 {after_count} 行，删除了 {before_count - after_count} 行")
                
                # 检查是否有Reconnaisance类别
                if 'Label' in merged_df.columns:
                    recon_count = (merged_df['Label'] == 'Reconnaisance').sum()
                    print(f"去重后保留了 {recon_count} 行Reconnaisance类别数据")
            else:
                print("没有发现重复数据")
        except Exception as e:
            print(f"应用时间窗口去重时出错: {e}")
            print("将使用标准去重方法")
            
            # 检查是否有重复的行（基于标识符）
            duplicates = merged_df.duplicated(subset=identifiers, keep=False)
            if duplicates.any():
                print(f"发现 {duplicates.sum()} 行重复数据（基于标识符）")
                print("保留第一次出现的行")
                merged_df = merged_df.drop_duplicates(subset=identifiers, keep='first')
    else:
        print(f"警告: 数据集中没有'{timestamp_col}'列，无法应用时间窗口去重")
        print("将使用标准去重方法")
        
        # 检查是否有重复的行（基于标识符）
        duplicates = merged_df.duplicated(subset=identifiers, keep=False)
        if duplicates.any():
            print(f"发现 {duplicates.sum()} 行重复数据（基于标识符）")
            print("保留第一次出现的行")
            merged_df = merged_df.drop_duplicates(subset=identifiers, keep='first')
    
    return merged_df

frames = []

def remove_correlated_features(df, threshold=0.9):
    """
    移除高度相关的特征（皮尔逊相关系数大于阈值）
    
    参数:
    df: 数据框
    threshold: 相关系数阈值，默认为0.9
    
    返回:
    移除高度相关特征后的数据框和被移除的特征列表
    """
    # 计算相关系数矩阵
    corr_matrix = df.corr().abs()
    
    # 创建上三角矩阵（不包括对角线）
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找出相关系数大于阈值的特征
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"发现 {len(to_drop)} 个高度相关的特征（相关系数 > {threshold}）:")
    for col in to_drop:
        # 找出与该特征高度相关的其他特征
        correlated_features = list(upper.index[upper[col] > threshold])
        if correlated_features:
            print(f"- {col} 与 {', '.join(correlated_features)} 高度相关")
    
    # 移除高度相关的特征
    df_reduced = df.drop(columns=to_drop)
    
    print(f"移除高度相关特征后的数据形状: {df_reduced.shape}")
    
    return df_reduced, to_drop

def select_features_by_mutual_info(X, y, k='all'):
    """
    使用互信息选择重要特征
    
    参数:
    X: 特征数据框
    y: 目标变量
    k: 要选择的特征数量，默认为'all'（根据特征重要性排序但不减少特征数量）
    
    返回:
    选择的特征数据框和特征重要性
    """
    # 创建SelectKBest对象
    if k == 'all' or k >= X.shape[1]:
        k = X.shape[1]
    
    # 使用互信息计算特征重要性
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    # 获取特征重要性分数
    scores = selector.scores_
    
    # 创建特征名称和重要性分数的映射
    feature_importance = dict(zip(X.columns, scores))
    
    # 按重要性排序
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"互信息特征选择结果（选择前 {k} 个特征）:")
    for i, (feature, score) in enumerate(sorted_features[:k]):
        print(f"{i+1}. {feature}: {score:.4f}")
    
    # 获取选择的特征名称
    if k < X.shape[1]:
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices]
        X_selected = X[selected_features]
    else:
        X_selected = X
    
    return X_selected, feature_importance

# 主程序
if __name__ == "__main__":
    # 检查数据集路径是否存在
    dataset_path = "D:/PycharmProjects/dapt2020/csv/"
    
    # 如果路径不存在，尝试其他可能的路径
    if not os.path.exists(dataset_path):
        print(f"警告: 找不到数据集路径 {dataset_path}")
        print("尝试使用当前目录下的csv文件夹...")
        
        # 尝试当前目录下的csv文件夹
        dataset_path = "./csv/"
        if not os.path.exists(dataset_path):
            print(f"警告: 找不到数据集路径 {dataset_path}")
            print("尝试使用上级目录下的csv文件夹...")
            
            # 尝试上级目录下的csv文件夹
            dataset_path = "../csv/"
            if not os.path.exists(dataset_path):
                print(f"错误: 无法找到有效的数据集路径")
                print("请确保路径正确或修改代码中的路径")
                exit(1)
    
    print(f"使用数据集路径: {dataset_path}")
    print("开始加载DAPT2020数据集...")
    frames = []
    
    # 更新数据集路径
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            path = os.path.join(dataset_path, file)
            print(f"加载文件: {file}")
            try:
                tmp = pd.read_csv(path)
                frames.append(tmp)
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")
    
    if not frames:
        print("没有找到有效的CSV文件")
        exit(1)
        
    print("合并数据集...")
    # 根据共同的事件标识符合并数据
    dapt2020 = merge_by_identifiers(frames)
    dapt2020 = dapt2020.rename(columns={"Stage": "Label"})
    
    # === 修复点：如无Session_ID则自动生成 ===
    if 'Session_ID' not in dapt2020.columns:
        dapt2020 = dapt2020.reset_index(drop=True)
        dapt2020['Session_ID'] = dapt2020.index.astype(str)
    
    # 按时间戳排序
    print("按时间戳排序...")
    dapt2020 = sort_by_timestamp(dapt2020)
    
    dapt_label_d = {
        "Benign": 0,
        'BENIGN': 0,
        'Reconnaissance': 1,
        'Establish Foothold': 2,
        'Lateral Movement': 3,
        'Data Exfiltration': 4
    }
    
    print("预处理数据集...")
    try:
        # 定义可以安全删除的特征列表（经过重新评估，进一步减少）
        safe_to_drop = [
            'Flow ID',  # 唯一标识符，可以通过IP、端口和时间戳来关联事件
            
            # 只删除确定冗余的特征
            'Packet Length Variance',  # 可以由Packet Length Std计算得出
            'Down/Up Ratio'  # 可以由其他流量特征计算得出
        ]
        
        # 检查要删除的列是否存在于数据集中
        existing_columns = dapt2020.columns
        dropcols = [col for col in safe_to_drop if col in existing_columns]
        
        # 打印将要删除的列
        print(f"将要删除的列（{len(dropcols)}个）:")
        for col in dropcols:
            print(f"- {col}")
        
        # 打印保留的列
        retained_cols = [col for col in existing_columns if col not in dropcols]
        print(f"\n保留的列（{len(retained_cols)}个）:")
        for col in retained_cols[:10]:  # 只打印前10个，避免输出过多
            print(f"- {col}")
        if len(retained_cols) > 10:
            print(f"- ... 以及其他 {len(retained_cols) - 10} 个列")
        
        clean_dapt = prepare_df(dapt2020, dropcols=dropcols, scaler='standard', ldict=dapt_label_d)
        drev = {val: key for key, val in dapt_label_d.items()}
        
        print("DAPT2020数据集预处理完成")
        print(f"数据形状: {clean_dapt.shape}")
        print(f"类别分布:\n{clean_dapt['Label'].value_counts()}")
        
        # 添加模型训练和评估部分
        print("\n=== 模型训练和评估 ===")
        
        # 分割数据集为训练集和测试集
        X = clean_dapt.drop(columns=['Label'])
        y = clean_dapt['Label']
        
        # 对非数值特征进行编码
        print("\n对非数值特征进行编码...")
        X_encoded, encoders = encode_categorical_features(X)
        print(f"编码后的特征形状: {X_encoded.shape}")
        
        # 使用皮尔逊相关性阈值0.9进行特征选择
        print("\n=== 使用皮尔逊相关性阈值0.9进行特征选择 ===")
        X_reduced, dropped_features = remove_correlated_features(X_encoded, threshold=0.9)
        print(f"移除的特征数量: {len(dropped_features)}")
        print(f"保留的特征数量: {X_reduced.shape[1]}")
        
        # 检查是否有Reconnaisance类别的样本
        recon_samples = (y == 2).sum()
        print(f"\n数据集中Reconnaisance类别的样本数量: {recon_samples}")
        
        # 如果有Reconnaisance样本，分析哪些特征对该类别最重要
        if recon_samples > 0:
            print("\n=== 分析Reconnaisance类别的重要特征 ===")
            # 创建二元分类问题：Reconnaisance vs 其他
            y_recon = (y == 2).astype(int)
            
            # 使用互信息计算每个特征对Reconnaisance分类的重要性
            selector = SelectKBest(mutual_info_classif, k='all')
            selector.fit(X_reduced, y_recon)
            
            # 获取特征重要性分数
            recon_scores = selector.scores_
            
            # 创建特征名称和重要性分数的映射
            recon_feature_importance = dict(zip(X_reduced.columns, recon_scores))
            
            # 按重要性排序
            sorted_recon_features = sorted(recon_feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"对Reconnaisance类别最重要的前10个特征:")
            for i, (feature, score) in enumerate(sorted_recon_features[:10]):
                print(f"{i+1}. {feature}: {score:.4f}")
            
            # 确保这些重要特征不会在后续特征选择中被过滤掉
            top_recon_features = [feature for feature, _ in sorted_recon_features[:10]]
            print(f"将确保保留这些对Reconnaisance类别重要的特征")
        else:
            print("\n警告: 数据集中没有Reconnaisance类别的样本")
            print("可能是由于数据预处理过程中的过滤导致，请检查数据处理步骤")
            top_recon_features = []
        
        # 应用互信息特征选择，但确保保留对Reconnaisance/Reconnaissance重要的特征
        print("\n=== 应用互信息特征选择 ===")
        # 选择前40个最重要的特征（增加数量以包含更多可能重要的特征）
        k = min(40, X_reduced.shape[1])
        
        # 使用互信息计算特征重要性
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X_reduced, y)
        
        # 获取特征重要性分数
        scores = selector.scores_
        
        # 创建特征名称和重要性分数的映射
        feature_importance = dict(zip(X_reduced.columns, scores))
        
        # 按重要性排序
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"互信息特征选择结果（选择前 {k} 个特征）:")
        for i, (feature, score) in enumerate(sorted_features[:k]):
            print(f"{i+1}. {feature}: {score:.4f}")
        
        # 获取选择的特征名称
        selected_indices = selector.get_support(indices=True)
        selected_features = X_reduced.columns[selected_indices]
        
        # 选择最终特征集
        X_selected = X_reduced[selected_features]
        print(f"选择的特征形状: {X_selected.shape}")
        
        # 可视化互信息特征重要性
        plt.figure(figsize=(10, 6))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:k]
        features, scores = zip(*sorted_features)
        plt.barh(range(len(features)), scores, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('互信息分数')
        plt.ylabel('特征')
        plt.title('特征重要性（互信息）')
        plt.tight_layout()
        plt.savefig('feature_importance_mutual_info.png')
        print("特征重要性图保存为 'feature_importance_mutual_info.png'")
        
        # 分割数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
        
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        print(f"训练集类别分布:\n{pd.Series(y_train).value_counts()}")
        print(f"测试集类别分布:\n{pd.Series(y_test).value_counts()}")
        
        # 训练随机森林分类器
        print("\n=== 训练随机森林分类器 ===")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print("\n=== 模型评估结果 ===")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 显示混淆矩阵
        print("\n=== 模型混淆矩阵 ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # 使用新的函数显示类别指标
        class_metrics = display_class_metrics(y_test, y_pred, drev, "")
        
        # 特征重要性分析
        print("\n=== 特征重要性分析 ===")
        feature_importances = rf_model.feature_importances_
        feature_names = X_selected.columns
        
        # 获取前20个重要特征
        indices = np.argsort(feature_importances)[::-1][:20]
        print("前20个重要特征:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {feature_importances[idx]:.4f}")
        
        # 保存模型（可选）
        # joblib.dump(rf_model, 'rf_model.joblib')
        # print("\n模型已保存为 'rf_model.joblib'")
        
    except Exception as e:
        print(f"预处理数据或训练模型时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n处理完成!")
    if 'Session_ID' not in clean_dapt.columns and 'Session_ID' in dapt2020.columns:
        clean_dapt['Session_ID'] = dapt2020['Session_ID'].values
    clean_dapt.to_csv("clean_dapt.csv", index=False)
    print("已保存clean_dapt.csv，供APT序列构建器使用")


