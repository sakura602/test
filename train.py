"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import APTClassifier
from util.epoch_timer import epoch_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def switch_classification_mode(mode='binary'):
    """切换分类模式并重新加载数据"""
    global train_iter, valid_iter, test_iter, num_classes, current_classification_type

    print(f"Switching to {mode} classification mode...")
    train_iter, valid_iter, test_iter, num_classes = load_apt_data(mode)
    current_classification_type = mode
    return num_classes


def create_model(num_classes):
    """创建APT分类模型"""
    model = APTClassifier(src_pad_idx=src_pad_idx,
                         feature_dim=feature_dim,
                         num_classes=num_classes,
                         d_model=d_model,
                         n_head=n_heads,
                         max_len=max_len,
                         ffn_hidden=ffn_hidden,
                         n_layers=n_layers,
                         drop_prob=drop_prob,
                         device=device).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)
    return model

# 创建初始模型（二分类）
model = create_model(num_classes)

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    all_predictions = []
    all_labels = []

    for i, batch in enumerate(iterator):
        src = batch['src']  # [batch_size, seq_len, feature_dim]
        labels = batch['labels']  # [batch_size]

        optimizer.zero_grad()
        output = model(src)  # [batch_size, num_classes]

        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        # 收集预测和标签用于计算准确率
        predictions = torch.argmax(output, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if i % 10 == 0:  # 每10个batch打印一次
            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    # 计算训练准确率
    train_acc = accuracy_score(all_labels, all_predictions)
    return epoch_loss / len(iterator), train_acc


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['src']  # [batch_size, seq_len, feature_dim]
            labels = batch['labels']  # [batch_size]

            output = model(src)  # [batch_size, num_classes]
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            # 收集预测和标签
            predictions = torch.argmax(output, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算整体评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    # 计算每个类别的详细指标
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )

    return epoch_loss / len(iterator), accuracy, precision, recall, f1, per_class_precision, per_class_recall, per_class_f1, support


def print_per_class_metrics(per_class_precision, per_class_recall, per_class_f1, support, classification_type):
    """打印每个类别的详细指标"""
    if classification_type == 'binary':
        class_names = ['Normal', 'Attack']
    else:
        class_names = ['Normal', 'APT1', 'APT2', 'APT3', 'APT4']

    print(f"\n{classification_type.upper()} Per-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)

    for i, class_name in enumerate(class_names):
        if i < len(per_class_precision):
            precision = per_class_precision[i]
            recall = per_class_recall[i]
            f1 = per_class_f1[i]
            sup = support[i]
            print(f"{class_name:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {sup:<10}")
    print("-" * 70)


def run(total_epoch, best_loss, classification_type='binary'):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_acc = 0.0

    print(f"Starting {classification_type} classification training...")

    for step in range(total_epoch):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iter, optimizer, criterion, clip)
        valid_results = evaluate(model, valid_iter, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = valid_results[:5]
        per_class_precision, per_class_recall, per_class_f1, support = valid_results[5:]
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        train_accs.append(train_acc)
        test_accs.append(valid_acc)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 保存最佳模型（基于准确率）
        if valid_acc > best_acc:
            best_acc = valid_acc
            model_name = f'saved/apt_classifier_{classification_type}_acc_{valid_acc:.4f}.pt'
            torch.save(model.state_dict(), model_name)
            print(f"New best model saved: {model_name}")

        # 保存训练记录
        import os
        os.makedirs('result', exist_ok=True)

        with open(f'result/train_loss_{classification_type}.txt', 'w') as f:
            f.write(str(train_losses))

        with open(f'result/train_acc_{classification_type}.txt', 'w') as f:
            f.write(str(train_accs))

        with open(f'result/test_loss_{classification_type}.txt', 'w') as f:
            f.write(str(test_losses))

        with open(f'result/test_acc_{classification_type}.txt', 'w') as f:
            f.write(str(test_accs))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'\tVal Loss: {valid_loss:.4f} | Val Acc: {valid_acc:.4f}')
        print(f'\tVal Precision: {valid_precision:.4f} | Val Recall: {valid_recall:.4f} | Val F1: {valid_f1:.4f}')

        # 打印每个类别的详细指标
        print_per_class_metrics(per_class_precision, per_class_recall, per_class_f1, support, classification_type)
        print('-' * 80)

    return best_acc


def train_both_modes():
    """训练二分类和多分类模型"""
    print("=" * 80)
    print("APT Attack Detection Training")
    print("=" * 80)

    # 1. 训练二分类模型
    print("\n1. Training Binary Classification Model...")
    binary_classes = switch_classification_mode('binary')
    global model, optimizer
    model = create_model(binary_classes)
    optimizer = Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)

    binary_best_acc = run(total_epoch=epoch, best_loss=inf, classification_type='binary')
    print(f"Binary classification best accuracy: {binary_best_acc:.4f}")

    # 2. 训练多分类模型
    print("\n2. Training Multiclass Classification Model...")
    multi_classes = switch_classification_mode('multiclass')
    model = create_model(multi_classes)
    optimizer = Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)

    multi_best_acc = run(total_epoch=epoch, best_loss=inf, classification_type='multiclass')
    print(f"Multiclass classification best accuracy: {multi_best_acc:.4f}")

    print("\n" + "=" * 80)
    print("Training Summary:")
    print(f"Binary Classification Best Accuracy: {binary_best_acc:.4f}")
    print(f"Multiclass Classification Best Accuracy: {multi_best_acc:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    train_both_modes()
