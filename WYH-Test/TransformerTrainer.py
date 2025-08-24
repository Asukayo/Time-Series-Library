import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from config import CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP


class NetworkTrafficDataset(Dataset):
    """CICIDS2017 网络流量数据集"""

    def __init__(self, X, y, scaler=None, is_training=True):
        """
        Args:
            X: 形状为 [num_windows, window_size, num_features] 的数据
            y: 形状为 [num_windows, window_size] 的标签
            scaler: 特征缩放器
            is_training: 是否为训练模式
        """
        self.X = X
        self.y = y
        self.is_training = is_training
        self.scaler = scaler

        # 对每个窗口进行标签聚合（多数投票）
        self.window_labels = self._aggregate_window_labels()

        # 标准化特征
        if self.scaler is not None:
            self.X = self._normalize_features()

    def _aggregate_window_labels(self):
        """将窗口内的标签聚合为单个标签（存在恶意流量即标记为恶意）"""
        window_labels = []
        for window_y in self.y:
            # 只要窗口中存在恶意流量就标记为恶意
            has_malicious = np.any(window_y > 0)
            window_labels.append(1 if has_malicious else 0)
        return np.array(window_labels)

    def _normalize_features(self):
        """标准化特征"""
        original_shape = self.X.shape
        # 重塑为 [num_samples, num_features] 进行标准化
        X_reshaped = self.X.reshape(-1, original_shape[-1])

        if self.is_training:
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            X_normalized = self.scaler.transform(X_reshaped)

        # 重塑回原始形状
        return X_normalized.reshape(original_shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])  # [seq_len, features]
        y = torch.LongTensor([self.window_labels[idx]])  # [1]

        # 创建时间标记（用于padding mask）
        x_mark = torch.ones(x.shape[0])  # [seq_len]

        return x, x_mark, y


class TransformerConfig:
    """Transformer模型配置"""

    def __init__(self, seq_len, enc_in, num_class=2):
        self.task_name = 'classification'

        # 数据维度
        self.seq_len = seq_len
        self.enc_in = enc_in  # 输入特征数
        self.c_out = enc_in  # 输出特征数（用于重构）
        self.num_class = num_class  # 分类类别数

        # Transformer架构参数
        self.d_model = 512
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 3
        self.d_ff = 2048
        self.factor = 5
        self.dropout = 0.3
        self.activation = 'relu'

        # 嵌入参数
        self.embed = 'timeF'
        self.freq = 'h'

        # 预测相关（对分类任务无效）
        self.pred_len = 0
        self.dec_in = enc_in


def load_data(data_dir, window_size=100, step_size=20):
    """加载处理后的数据"""
    X_file = os.path.join(data_dir, f'selected_X_w{window_size}_s{step_size}.npy')
    y_file = os.path.join(data_dir, f'selected_y_w{window_size}_s{step_size}.npy')
    metadata_file = os.path.join(data_dir, f'selected_metadata_w{window_size}_s{step_size}.pkl')

    print("加载数据...")
    X = np.load(X_file)
    y = np.load(y_file)

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    print(f"数据形状: X{X.shape}, y{y.shape}")
    print(f"特征数量: {len(metadata['feature_names'])}")

    return X, y, metadata


def split_data_chronologically(X, y, train_ratio=0.6, val_ratio=0.2):
    """按时间顺序分割数据"""
    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    print(f"数据分割:")
    print(f"  训练集: {len(X_train)} ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"  验证集: {len(X_val)} ({len(X_val) / total_samples * 100:.1f}%)")
    print(f"  测试集: {len(X_test)} ({len(X_test) / total_samples * 100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """创建数据加载器"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # 创建标准化器
    scaler = StandardScaler()

    # 创建数据集
    train_dataset = NetworkTrafficDataset(X_train, y_train, scaler=scaler, is_training=True)
    val_dataset = NetworkTrafficDataset(X_val, y_val, scaler=scaler, is_training=False)
    test_dataset = NetworkTrafficDataset(X_test, y_test, scaler=scaler, is_training=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (x_enc, x_mark_enc, targets) in enumerate(tqdm(train_loader)):
        x_enc = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        targets = targets.squeeze().to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(x_enc, x_mark_enc, None, None)
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for x_enc, x_mark_enc, targets in val_loader:
            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            targets = targets.squeeze().to(device)

            outputs = model(x_enc, x_mark_enc, None, None)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 保存预测结果用于计算其他指标
            all_outputs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)

    # 计算其他指标
    all_predicted = (np.array(all_outputs) > 0.5).astype(int)
    precision = precision_score(all_targets, all_predicted, zero_division=0)
    recall = recall_score(all_targets, all_predicted, zero_division=0)
    f1 = f1_score(all_targets, all_predicted, zero_division=0)
    auc = roc_auc_score(all_targets, all_outputs)

    return avg_loss, accuracy, precision, recall, f1, auc


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.00001, device='cuda'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 记录训练历史
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []

    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 10

    print("开始训练...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate_epoch(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)

        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
              f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # 早停检查
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"新的最佳模型! F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停触发，在epoch {epoch + 1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores
    }


def test_model(model, test_loader, device='cuda'):
    """测试模型"""
    print("\n评估测试集...")
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate_epoch(
        model, test_loader, nn.CrossEntropyLoss(), device
    )

    print(f"测试结果:")
    print(f"  准确率: {val_acc:.2f}%")
    print(f"  精确率: {val_prec:.4f}")
    print(f"  召回率: {val_rec:.4f}")
    print(f"  F1分数: {val_f1:.4f}")
    print(f"  AUC: {val_auc:.4f}")

    return {
        'accuracy': val_acc,
        'precision': val_prec,
        'recall': val_rec,
        'f1': val_f1,
        'auc': val_auc
    }


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['train_accuracies'], label='Training Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # F1 Score
    ax3.plot(history['val_f1_scores'], label='Validation F1', color='green')
    ax3.set_title('Validation F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True)

    # Learning Curves Overview
    ax4.plot(history['train_accuracies'], label='Training Accuracy', alpha=0.7)
    ax4.plot(history['val_accuracies'], label='Validation Accuracy', alpha=0.7)
    ax4.plot([f * 100 for f in history['val_f1_scores']], label='Validation F1×100', alpha=0.7)
    ax4.set_title('Learning Curves Overview')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Metric Value')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Transformer网络流量异常检测')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/wyh/cicdis/cicids2017/selected_features/',
                        help='处理后数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save_model', type=str, default='transformer_anomaly_model.pth',
                        help='模型保存路径')

    args = parser.parse_args()

    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = 'cpu'

    print(f"使用设备: {args.device}")

    # 加载数据
    X, y, metadata = load_data(args.data_dir, CICIDS_WINDOW_SIZE, CICIDS_WINDOW_STEP)

    # 分割数据
    train_data, val_data, test_data = split_data_chronologically(X, y)

    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        train_data, val_data, test_data, args.batch_size
    )

    # 创建模型配置
    config = TransformerConfig(
        seq_len=CICIDS_WINDOW_SIZE,
        enc_in=len(metadata['feature_names']),
        num_class=2
    )

    # 导入并创建模型
    from models.Transformer import Model
    model = Model(config)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    trained_model, history = train_model(
        model, train_loader, val_loader,
        args.num_epochs, args.learning_rate, args.device
    )

    # 测试模型
    test_results = test_model(trained_model, test_loader, args.device)

    # 绘制训练历史
    plot_training_history(history, 'training_history.png')

    # 保存模型
    # torch.save({
    #     'model_state_dict': trained_model.state_dict(),
    #     'config': config,
    #     'scaler': scaler,
    #     'metadata': metadata,
    #     'test_results': test_results
    # }, args.save_model)

    print(f"\n模型保存至: {args.save_model}")
    print("训练完成!")


if __name__ == "__main__":
    main()