import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from shortwave_clf.data import AudioDataset, AudioTransforms
from shortwave_clf.models import ResNet
from shortwave_clf.trainers import train_loop

exp_name = "base"


def main_experiment():
    # 加载数据集配置
    dataset_info_config_path = "shortwave_clf/configs/dataset_info.json"
    with open(dataset_info_config_path, "r") as f:
        dataset_info = json.load(f)

    # 数据集路径
    data_dir = dataset_info["root"] + dataset_info["dir"]
    num_classes = dataset_info["num_classes"]
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # 数据加载器
    batch_size = 128  # 根据需要调整批次大小
    transform = AudioTransforms()
    train_dataset = AudioDataset(data_dir, tag="train", transform=transform)
    test_dataset = AudioDataset(data_dir, tag="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 10, shuffle=False)

    # 初始化模型
    model = ResNet(num_classes=num_classes)

    # 定义损失函数和优化器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练参数
    save_dir = f"results/{exp_name}/saved_models"

    # 训练和验证
    train_loop(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs,
        save_dir,
    )


if __name__ == "__main__":
    main_experiment()
