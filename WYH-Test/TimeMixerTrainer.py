import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import os
from tqdm import tqdm
import json

# 导入自定义模块
from models.TimeMixer import Model  # 修改：导入TimeMixer模型
from provider_6_2_2 import load_data, split_data_chronologically, create_data_loaders
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from trainer_valder import train_epoch, val_epoch


class Config:
    """TimeMixer分类器配置"""

    def __init__(self):
        # 任务配置
        self.task_name = 'classification'  # 新增：任务类型

        # 数据配置
        self.seq_len = CICIDS_WINDOW_SIZE  # 100
        self.label_len = 0  # 新增：标签长度（分类任务不需要）
        self.pred_len = 38  # 特征提取维度
        self.enc_in = 38  # 特征数量（important_features_90.txt）
        self.c_out = 38  # 新增：输出通道数

        # TimeMixer特有配置
        self.d_model = 512  # 新增：模型维度
        self.d_ff = 1024  # 新增：前馈网络维度
        self.e_layers = 2  # 新增：编码器层数
        self.embed = 'timeF'  # 新增：嵌入类型
        self.freq = 'h'  # 新增：频率
        self.dropout = 0.2  # 新增：dropout率

        # 分解相关配置
        self.decomp_method = 'moving_avg'  # 新增：分解方法
        self.moving_avg = 25  # 新增：移动平均窗口
        self.top_k = 5  # 新增：DFT分解的top-k

        # 多尺度配置
        self.down_sampling_window = 2  # 新增：下采样窗口
        self.down_sampling_layers = 1  # 新增：下采样层数
        self.down_sampling_method = 'avg'  # 新增：下采样方法
        self.channel_independence = False  # 新增：通道独立性
        self.use_norm = 1  # 新增：是否使用归一化

        # 分类配置
        self.num_class = 2  # 修改：二分类（原num_classes）

        # 训练配置
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        self.patience = 10  # 早停耐心值

        # 数据分割
        self.train_ratio = 0.6
        self.val_ratio = 0.2

        # 其他
        self.device = torch.device('cuda' )
        self.save_dir = 'checkpoints'


def train_model(configs):
    """主训练函数"""
    # 创建保存目录
    os.makedirs(configs.save_dir, exist_ok=True)

    print("=" * 60)
    print("TimeMixer-Classifier Training for CICIDS2017")  # 修改：更新标题
    print("=" * 60)
    print(f"Device: {configs.device}")
    print(f"Window size: {configs.seq_len}, Features: {configs.enc_in}")

    # 加载数据
    print("\n1. Loading data...")
    data_dir = "/home/ubuntu/wyh/cicdis/cicids2017/selected_features"
    X, y, metadata = load_data(data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 更新特征数量
    configs.enc_in = len(metadata['feature_names'])
    configs.c_out = configs.enc_in  # 新增：更新输出通道数
    print(f"Updated feature count: {configs.enc_in}")

    # 分割数据
    print("\n2. Splitting data...")
    train_data, val_data, test_data = split_data_chronologically(
        X, y, configs.train_ratio, configs.val_ratio
    )

    # 创建数据加载器
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, configs.batch_size
    )

    # 创建模型
    print("\n4. Creating model...")
    model = Model(configs).to(configs.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环
    print("\n5. Training...")
    best_val_f1 = 0
    patience_counter = 0
    train_history = []

    for epoch in range(configs.epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, configs.device)

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1 = val_epoch(
            model, val_loader, criterion, configs.device
        )

        # 学习率调整
        scheduler.step(val_loss)

        # 记录历史
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

        # 打印进度
        print(f"Epoch {epoch + 1:3d}/{configs.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'configs': configs,
                'scaler': scaler,
                'best_val_f1': best_val_f1,
                'feature_names': metadata['feature_names']
            }
            torch.save(checkpoint, os.path.join(configs.save_dir, 'best_model.pth'))
            print(f"  → New best F1: {best_val_f1:.4f}, model saved!")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= configs.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # 保存训练历史
    with open(os.path.join(configs.save_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)

    # 测试最佳模型
    print("\n6. Testing best model...")
    best_checkpoint = torch.load(os.path.join(configs.save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_acc, test_precision, test_recall, test_f1 = val_epoch(
        model, test_loader, criterion, configs.device
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test Accuracy:      {test_acc:.4f}")
    print(f"Test Precision:     {test_precision:.4f}")
    print(f"Test Recall:        {test_recall:.4f}")
    print(f"Test F1:            {test_f1:.4f}")

    return model, train_history


if __name__ == "__main__":
    # 创建配置
    configs = Config()

    # 开始训练
    model, history = train_model(configs)

    print("\nTraining completed!")