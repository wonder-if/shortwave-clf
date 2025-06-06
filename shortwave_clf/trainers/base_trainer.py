import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import os


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    interval = 200

    start_time = time.time()  # 记录 epoch 开始时间

    for batch_idx, (signals, labels) in enumerate(train_loader):
        signals, labels = signals.to(device), labels.squeeze().to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * signals.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 打印每个 batch 的训练情况
        if (batch_idx + 1) % interval == 0:  # 每10个batch打印一次
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} - "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100 * (correct / total):.2f}%, "
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    end_time = time.time()  # 记录 epoch 结束时间
    epoch_duration = end_time - start_time  # 计算 epoch 耗时

    print(
        f"Training - "
        f"Loss: {epoch_loss:.4f}, "
        f"Acc: {100 * epoch_acc:.2f}%, "
        f"Time: {epoch_duration:.2f}s"
    )

    return epoch_loss, epoch_acc


def train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_dir,
):
    best_acc = 0.0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        _, epoch_acc = validate_model(model, val_loader, criterion, device)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = {}  # 用于记录每个类的正确数
    class_total = {}  # 用于记录每个类的总数

    with torch.no_grad():
        # 使用 tqdm 创建进度条
        pbar = tqdm(val_loader, desc="Validation")
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.squeeze().to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新每个类的正确数和总数
            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                if label_idx not in class_correct:
                    class_correct[label_idx] = 0
                    class_total[label_idx] = 0
                class_correct[label_idx] += 1 if pred.item() == label_idx else 0
                class_total[label_idx] += 1

            # 更新进度条信息
            pbar.set_postfix(
                {
                    "Lo ss": f"{running_loss / total:.4f}",
                    "Acc": f"{100 * (correct / total):.2f}%",
                }
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # 输出每个类的准确率
    print("\nValidation Results:")
    print(f"Total Loss: {epoch_loss:.4f}, Total Acc: {100 * epoch_acc:.2f}%")
    for class_idx in class_correct:
        class_acc = 100 * class_correct[class_idx] / class_total[class_idx]
        print(f"Class {class_idx} - Acc: {class_acc:.2f}%")

    return epoch_loss, epoch_acc
