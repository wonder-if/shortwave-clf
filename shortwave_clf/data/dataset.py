from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import json


class AudioDataset(Dataset):
    def __init__(self, data_dir, tag="train", transform=None):
        """
        初始化数据集
        :param data_dir: 数据集的根目录
        :param tag: 数据集的标签，通常是 "train" 或 "test"
        """
        self.transform = transform

        self.signals_path = os.path.join(data_dir, tag, "signals.npy")
        self.labels_path = os.path.join(data_dir, tag, "labels.npy")

        # 加载数据和标签
        self.signals = np.load(self.signals_path)
        self.labels = np.load(self.labels_path)

        # 将标签转换为整数类型，确保标签的正确性
        self.labels = self.labels.astype(np.int64)

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.signals)

    def __getitem__(self, idx):
        """
        根据索引获取数据和标签
        :param idx: 索引
        :return: 返回一个元组，包含数据和标签
        """
        signal = self.signals[idx]
        label = self.labels[idx]

        # transform
        if self.transform:
            signal = self.transform(signal)

        # 为了适配 PyTorch 的模型，通常需要将 NumPy 数组转换为 PyTorch 张量
        signal = torch.from_numpy(signal).float()
        label = torch.from_numpy(np.array([label])).long()

        return signal, label


if __name__ == "__main__":
    dataset_info_config_path = "shortwave_clf/configs/dataset_info.json"
    with open(dataset_info_config_path, "r") as f:
        dataset_info = json.load(f)

    # 创建 Dataset 和 DataLoader
    data_dir = dataset_info["root"] + dataset_info["dir"]
    batch_size = 32  # 您可以根据需要调整批次大小

    # 训练集
    train_dataset = AudioDataset(data_dir, tag="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集
    test_dataset = AudioDataset(data_dir, tag="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 打印数据集信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 测试 DataLoader
    for signals, labels in train_loader:
        print(f"Batch signals shape: {signals.shape}, labels shape: {labels.shape}")
        break
