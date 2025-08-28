import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


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