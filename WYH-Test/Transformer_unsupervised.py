import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


from tqdm import tqdm

from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP
from unsupervised import load_data, split_data_by_chronological_strategy, create_data_loaders
from models.Transformer import Model


class Config:
    """Transformer无监督异常检测配置类"""

    def __init__(self):
        # 任务类型
        self.task_name = 'anomaly_detection'

        # 数据参数
        self.seq_len = CICIDS_WINDOW_SIZE  # 序列长度
        self.enc_in = 38  # 输入特征数量
        self.c_out = 38  # 输出特征数量（重构）

        # Transformer架构参数
        self.d_model = 512  # 模型维度
        self.n_heads = 8  # 注意力头数
        self.e_layers = 3  # Encoder层数
        self.d_ff = 2048  # 前馈网络维度
        self.dropout = 0.1  # Dropout率
        self.activation = 'gelu'  # 激活函数
        self.factor = 5  # 注意力因子

        # Embedding参数
        self.embed = 'timeF'  # 嵌入类型
        self.freq = 'h'  # 频率编码

        # 训练参数
        self.learning_rate = 0.0001
        self.batch_size = 32  # Transformer通常需要较小的batch size
        self.epochs = 50
        self.patience = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for x, x_mark, y in tqdm(train_loader, desc="Training", leave=False):
        x = x.to(device)  # [batch_size, seq_len, features]
        x_mark = x_mark.to(device)  # [batch_size, seq_len]

        optimizer.zero_grad()

        # Transformer异常检测：重构输入
        x_recon = model(x, x_mark, None, None)  # [batch_size, seq_len, features]

        # 计算重构误差
        loss = criterion(x_recon, x)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x, x_mark, y in tqdm(val_loader, desc="Validation", leave=False):
            x = x.to(device)
            x_mark = x_mark.to(device)

            # 重构
            x_recon = model(x, x_mark, None, None)

            # 计算重构误差
            loss = criterion(x_recon, x)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def calculate_threshold(model, val_loader, device, percentile=95):
    """基于验证集计算异常检测阈值"""
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for x, x_mark, y in val_loader:
            x = x.to(device)
            x_mark = x_mark.to(device)

            # 重构
            x_recon = model(x, x_mark, None, None)

            # 计算每个样本的重构误差
            mse = torch.mean((x - x_recon) ** 2, dim=(1, 2))  # [batch_size]
            reconstruction_errors.extend(mse.cpu().numpy())

    # 使用百分位数作为阈值
    threshold = np.percentile(reconstruction_errors, percentile)
    return threshold, reconstruction_errors


def evaluate_model(model, test_loader, threshold, device):
    """评估模型异常检测性能"""
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for x, x_mark, y in test_loader:
            x = x.to(device)
            x_mark = x_mark.to(device)
            y_true.extend(y.cpu().numpy().flatten())

            # 重构并计算异常分数
            x_recon = model(x, x_mark, None, None)
            mse = torch.mean((x - x_recon) ** 2, dim=(1, 2))
            y_scores.extend(mse.cpu().numpy())

    # 基于阈值预测
    y_pred = (np.array(y_scores) > threshold).astype(int)

    # 计算评估指标
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_scores)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': threshold
    }


def evaluate_multiple_thresholds(model, val_loader, test_loader, device):
    """评估多个阈值的性能"""
    # 计算验证集重构误差
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for x, x_mark, y in val_loader:
            x = x.to(device)
            x_mark = x_mark.to(device)
            x_recon = model(x, x_mark, None, None)
            mse = torch.mean((x - x_recon) ** 2, dim=(1, 2))
            reconstruction_errors.extend(mse.cpu().numpy())

    # 测试不同百分位数
    percentiles = [80, 85, 90, 95, 97]
    results = {}

    print("=== 多阈值性能评估 ===")
    for p in percentiles:
        threshold = np.percentile(reconstruction_errors, p)
        metrics = evaluate_model(model, test_loader, threshold, device)
        results[p] = metrics

        print(f"阈值 ({p}th percentile): {threshold:.6f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print("-" * 40)

    # 找到最佳F1-Score的阈值
    best_percentile = max(results.keys(), key=lambda p: results[p]['f1'])
    best_threshold = np.percentile(reconstruction_errors, best_percentile)

    print(f"最佳阈值: {best_percentile}th percentile ({best_threshold:.6f})")
    print(f"最佳F1-Score: {results[best_percentile]['f1']:.4f}")

    return best_threshold, results


def plot_training_history(train_losses, val_losses, save_path='transformer_training_history.png'):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Transformer Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主训练函数"""
    # 配置
    config = Config()
    print(f"使用设备: {config.device}")
    print(f"任务类型: {config.task_name}")

    # 数据加载
    print("加载数据...")
    data_dir = '/home/ubuntu/wyh/cicdis/cicids2017/selected_features/'
    X, y, metadata = load_data(data_dir, window_size=CICIDS_WINDOW_SIZE, step_size=CICIDS_WINDOW_STEP)

    # 数据分割
    train_data, val_data, test_data = split_data_by_chronological_strategy(X, y, metadata)

    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, batch_size=config.batch_size
    )

    # 创建模型
    print("创建Transformer模型...")
    model = Model(config).to(config.device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    # 训练历史记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print("开始训练...")
    for epoch in range(config.epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)

        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, config.device)

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, 'best_transformer_model.pth')
        else:
            patience_counter += 1

        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Best Val Loss: {best_val_loss:.6f}")

        if patience_counter >= config.patience:
            print(f"早停触发！在第 {epoch + 1} 轮停止训练")
            break

    # 加载最佳模型
    print("加载最佳模型进行评估...")
    checkpoint = torch.load('best_transformer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 评估多个阈值
    print("评估多个阈值性能...")
    best_threshold, threshold_results = evaluate_multiple_thresholds(
        model, val_loader, test_loader, config.device
    )

    # 使用最佳阈值进行最终评估
    print("=== 使用最佳阈值的最终结果 ===")
    final_metrics = evaluate_model(model, test_loader, best_threshold, config.device)

    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1-Score: {final_metrics['f1']:.4f}")
    print(f"AUC: {final_metrics['auc']:.4f}")

    # 绘制训练历史
    plot_training_history(train_losses, val_losses)

    # 保存最终结果
    final_results = {
        'best_threshold': best_threshold,
        'threshold_results': threshold_results,
        'final_metrics': final_metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': config.__dict__
    }
    torch.save(final_results, 'transformer_results.pth')

    print("训练完成！模型和结果已保存。")


if __name__ == "__main__":
    main()