import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import os
from tqdm import tqdm
import json

# 导入自定义模块
from models.TimesNet import Model  # 修改：导入TimesNet模型
from provider_6_2_2 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from TimeMixer_t_v import train_epoch, val_epoch


class Config:
    """TimesNet分类器配置"""

    def __init__(self):
        # 任务配置
        self.task_name = 'classification'  # 任务类型

        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.label_len = 0  # 标签长度（分类任务不需要）
        self.pred_len = 0
        self.enc_in = 38  # 特征数量（important_features_90.txt）
        self.c_out = 38  # 输出通道数

        # TimesNet特有配置
        self.d_model = 128  # 模型嵌入维度
        self.d_ff = 256  # 前馈网络维度
        self.e_layers = 1  # TimesBlock层数
        self.embed = 'timeF'  # 时间特征嵌入类型
        self.freq = 'h'  # 频率标识
        self.dropout = 0.2  # Dropout率

        # TimesNet核心参数
        self.top_k = 5  # FFT中选择的主要周期数
        self.num_kernels = 6  # Inception块中的卷积核数量

        # 分类配置
        self.num_class = 2  # 二分类（正常/异常）

        # 训练配置
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.patience = 10  # 早停耐心值

        # 数据分割配置
        self.train_ratio = 0.6
        self.val_ratio = 0.2

        # 系统配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = 'checkpoints_timesnet'


def train_model(configs):
    """TimesNet主训练函数"""
    # 创建保存目录
    os.makedirs(configs.save_dir, exist_ok=True)

    print("=" * 60)
    print("TimesNet-Classifier Training for CICIDS2017")
    print("=" * 60)
    print(f"Device: {configs.device}")
    print(f"Window size: {configs.seq_len}, Features: {configs.enc_in}")
    print(f"Top-k periods: {configs.top_k}, Kernel count: {configs.num_kernels}")

    # 1. 加载数据
    print("\n1. Loading data...")
    data_dir = "/home/ubuntu/wyh/cicdis/cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 更新特征数量配置
    configs.enc_in = len(metadata['feature_names'])
    configs.c_out = configs.enc_in
    print(f"Updated feature count: {configs.enc_in}")

    # 2. 时序数据分割
    print("\n2. Splitting data chronologically...")
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )

    print(f"Train windows: {train_data[0].shape[0]}")
    print(f"Val windows: {val_data[0].shape[0]}")
    print(f"Test windows: {test_data[0].shape[0]}")

    # 3. 创建数据加载器
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    # 4. 创建TimesNet模型
    print("\n4. Creating TimesNet model...")
    model = Model(configs).to(configs.device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Model layers: {configs.e_layers} TimesBlocks")

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=configs.learning_rate,
        weight_decay=configs.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 6. 训练循环
    print("\n5. Starting training...")
    best_val_f1 = 0
    patience_counter = 0
    train_history = []

    for epoch in range(configs.epochs):
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, configs.device
        )

        # 验证阶段
        val_loss, val_acc, val_precision, val_recall, val_f1 = val_epoch(
            model, val_loader, criterion, configs.device
        )

        # 学习率调整
        scheduler.step(val_loss)

        # 记录训练历史
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_stats)

        # 打印训练进度
        print(f"Epoch {epoch + 1:3d}/{configs.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 模型保存和早停逻辑
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            # 保存最佳模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'configs': configs,
                'scaler': scaler,
                'best_val_f1': best_val_f1,
                'feature_names': metadata['feature_names'],
                'train_history': train_history
            }
            torch.save(checkpoint, os.path.join(configs.save_dir, 'best_model.pth'))
            print(f"  → New best F1: {best_val_f1:.4f}, model saved!")
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= configs.patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation F1: {best_val_f1:.4f}")
            break

    # 保存完整训练历史
    with open(os.path.join(configs.save_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)

    # 7. 测试最佳模型
    print("\n6. Testing best model...")
    best_checkpoint = torch.load(os.path.join(configs.save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_acc, test_precision, test_recall, test_f1 = val_epoch(
        model, test_loader, criterion, configs.device
    )

    # 8. 详细测试评估
    print("\n7. Generating detailed test report...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_X_mark, batch_y in test_loader:
            batch_X = batch_X.to(configs.device)
            batch_X_mark = batch_X_mark.to(configs.device)
            batch_y = batch_y.squeeze().to(configs.device)

            output = model(batch_X, batch_X_mark, None, None)
            preds = torch.argmax(output, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

    # 生成分类报告
    class_names = ['Benign', 'Malicious']
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # 保存测试结果
    test_results = {
        'best_val_f1': best_val_f1,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'classification_report': report,
        'model_config': {
            'top_k': configs.top_k,
            'num_kernels': configs.num_kernels,
            'd_model': configs.d_model,
            'e_layers': configs.e_layers,
            'total_params': total_params
        }
    }

    with open(os.path.join(configs.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2, default=str)

    # 最终结果展示
    print("\n" + "=" * 60)
    print("TIMESNET FINAL RESULTS")
    print("=" * 60)
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test Accuracy:      {test_acc:.4f}")
    print(f"Test Precision:     {test_precision:.4f}")
    print(f"Test Recall:        {test_recall:.4f}")
    print(f"Test F1:            {test_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    print("=" * 60)

    return model, train_history, test_results


def load_and_test_model(checkpoint_path, test_data_dir):
    """加载已训练模型进行测试"""
    print("Loading trained TimesNet model...")
    checkpoint = torch.load(checkpoint_path)
    configs = checkpoint['configs']

    # 重新创建模型
    model = Model(configs).to(configs.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Best validation F1: {checkpoint['best_val_f1']:.4f}")

    return model, configs


if __name__ == "__main__":
    # 创建TimesNet配置
    configs = Config()

    print("TimesNet Configuration:")
    print(f"- Task: {configs.task_name}")
    print(f"- Sequence length: {configs.seq_len}")
    print(f"- Top-k periods: {configs.top_k}")
    print(f"- Number of kernels: {configs.num_kernels}")
    print(f"- Model dimension: {configs.d_model}")
    print(f"- Encoder layers: {configs.e_layers}")
    print(f"- Batch size: {configs.batch_size}")
    print(f"- Learning rate: {configs.learning_rate}")

    # 开始训练
    print("\nStarting TimesNet training for supervised anomaly detection...")
    model, history, results = train_model(configs)

    print(f"\nTimesNet training completed!")
    print(f"Results saved to: {configs.save_dir}")