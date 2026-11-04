"""
train.py —— 模型训练脚本（第二步运行）

做什么：
用 ImageNet 预训练的 ResNet18 做迁移学习，微调出「正常/肺炎」二分类模型。

迁移学习两步走（代码里对应两处）：
1. 加载预训练权重：torchvision.models.resnet18(weights='IMAGENET1K_V1')
2. 替换最后一层全连接：原本输出 1000 类，改成 2 类

训练策略（对新手最友好的方案）：
- 只训练最后一层 + 稍微微调整个网络（学习率很小，防止把预训练知识冲掉）
- 数据增强：训练集随机翻转+旋转，让模型见到更多变化，防止死记硬背（过拟合）
- 类别不平衡（正常:肺炎≈1:3）：在损失函数里给少数类更大权重

输出：
- models/best_model.pth   验证集 Recall 最高的模型权重
- results/training_curves.png   训练/验证的损失和 Recall 曲线
- results/confusion_matrix.png  混淆矩阵（报告必放）
- results/metrics.json          所有评估指标

运行方式：
    python src/train.py            # CPU 也能跑，但慢（约 1~2 小时）
    python src/train.py --epochs 5 # 快速试跑
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK 编码，强制改成 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # 无显示器环境也能画图
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ============ 配置区 ============
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "chest_xray"
MODEL_PATH = ROOT / "models" / "best_model.pth"
RESULTS_DIR = ROOT / "results"

BATCH_SIZE = 32          # 一次喂给模型多少张图
NUM_EPOCHS = 10          # 全部数据过几遍
LR_HEAD = 1e-3           # 最后一层的学习率（大一些，从头学）
LR_BACKBONE = 1e-4       # 网络主体的学习率（小 10 倍，保护预训练知识）
SEED = 42
# ================================

# ImageNet 预训练模型要求的归一化参数（固定写法，照抄即可）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    """固定随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_dataloaders(batch_size: int):
    """
    构建数据加载器。
    训练集做数据增强（随机翻转/旋转），验证和测试集只做缩放+归一化。
    """
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),      # 随机水平翻转
        transforms.RandomRotation(10),          # 随机旋转 ±10 度
        transforms.Grayscale(num_output_channels=3),  # 胸片是灰度图，复制成3通道以适配预训练模型
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int = 2):
    """迁移学习核心：预训练 ResNet18 + 替换最后一层"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 把最后一层（输出1000类）替换成 2 类
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """跑一遍训练集，返回平均损失和正确率"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()   # 反向传播算梯度
        optimizer.step()  # 更新参数

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()  # 评估阶段不需要算梯度，省显存
def evaluate(model, loader, criterion, device):
    """在验证/测试集上评估，返回损失、正确率、全部预测结果"""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), all_preds, all_labels


def main(epochs: int):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}  （cpu 表示用 CPU 训练，会慢）")

    train_loader, val_loader, test_loader, classes = get_dataloaders(BATCH_SIZE)
    print(f"类别: {classes}  (0=正常, 1=肺炎)")
    print(f"训练集 {len(train_loader.dataset)} 张 | 验证集 {len(val_loader.dataset)} 张 | 测试集 {len(test_loader.dataset)} 张")

    model = build_model(num_classes=len(classes)).to(device)

    # ---- 类别不平衡处理：给少数类（正常）更大的损失权重 ----
    train_labels = [label for _, label in train_loader.dataset.samples]
    counts = np.bincount(train_labels)
    weights = counts.sum() / (len(counts) * counts)  # 逆频率权重
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"类别数量 {counts.tolist()} → 损失权重 {weights.cpu().numpy().round(2).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ---- 差分学习率：主体小步走，最后一层大步走 ----
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": LR_BACKBONE},
        {"params": model.fc.parameters(), "lr": LR_HEAD},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- 训练循环：每轮结束在验证集上评估，保存 Recall 最高的模型 ----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_recall": []}
    best_recall = 0.0
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(val_recall)

        marker = ""
        if val_recall >= best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), MODEL_PATH)
            marker = "  ⭐ 已保存最优模型"

        print(f"Epoch {epoch:2d}/{epochs} | 训练损失 {train_loss:.4f} 训练正确率 {train_acc:.4f} "
              f"| 验证损失 {val_loss:.4f} 验证Recall {val_recall:.4f} | 用时 {time.time()-t0:.0f}s{marker}")

    print(f"\n训练完成！最佳验证 Recall = {best_recall:.4f}")

    # ---- 测试集最终评估 ----
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    metrics = {
        "accuracy": float(np.mean(np.array(test_preds) == np.array(test_labels))),
        "precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "test_loss": float(test_loss),
    }
    print("\n===== 测试集结果 =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["NORMAL", "PNEUMONIA"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {RESULTS_DIR / 'confusion_matrix.png'}")

    # ---- 训练曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train_acc")
    axes[1].plot(history["val_recall"], label="val_recall")
    axes[1].set_title("Accuracy / Recall"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"训练曲线已保存: {RESULTS_DIR / 'training_curves.png'}")
    print("\n下一步运行:  python src/gradcam.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="训练轮数")
    args = parser.parse_args()
    main(args.epochs)
