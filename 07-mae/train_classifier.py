"""加载预训练 MAE，冻结编码器，只训练 CIFAR-10 分类头。"""

import argparse
import csv
import json
from pathlib import Path

from matplotlib.figure import Figure
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mini_mae.classifier import MAEClassifier
from mini_mae.data import build_dataloaders
from mini_mae.model import MiniMAE
from mini_mae.config import MODEL_CONFIG, EXPERIMENT_NAME, BATCH_SIZE


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_images = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_images += count

    if total_images == 0:
        raise ValueError("数据加载器为空")
    return total_loss / total_images, total_correct / total_images


def save_history(history, output_dir):
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    fig = Figure(figsize=(12, 4))
    axes = fig.subplots(1, 2)
    epochs = [row["epoch"] for row in history]
    for ax, metric, title in zip(axes, ["loss", "accuracy"],
                                 ["Cross entropy", "Accuracy"]):
        for split in ["train", "val"]:
            ax.plot(epochs, [row[f"{split}_{metric}"] for row in history],
                    marker="o", label=split)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "curves.png", dpi=150)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=root / "outputs" / EXPERIMENT_NAME / "best_model.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs" / EXPERIMENT_NAME / "classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs 和 batch-size 必须大于零")
    if not args.weights.is_file():
        raise FileNotFoundError(f"找不到 MAE 权重：{args.weights}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(
        root / "data", batch_size=args.batch_size,
        train_size=45000, val_size=1000, seed=42,
    )

    # 架构配置必须与重建预训练保持一致。
    mae = MiniMAE(**MODEL_CONFIG)
    mae.load_state_dict(torch.load(
        args.weights, map_location="cpu", weights_only=True
    ))
    model = MAEClassifier(mae).to(device)
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=1e-3, weight_decay=0.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best_accuracy = [], -1.0
    best_path = args.output_dir / "best_classifier.pt"
    print(f"设备：{device}；编码器冻结，只训练分类头")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, device)
        history.append(dict(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            train_accuracy=train_accuracy, val_accuracy=val_accuracy,
        ))
        save_history(history, args.output_dir)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_accuracy:.2%} | "
              f"val_loss={val_loss:.4f} acc={val_accuracy:.2%}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # 保存编码器和分类头，后续分类不依赖原始 MAE 文件。
            torch.save(model.state_dict(), best_path)

    # 仅在验证集选好模型后，评估一次官方测试集。
    model.load_state_dict(torch.load(
        best_path, map_location=device, weights_only=True
    ))
    test_dataset = datasets.CIFAR10(
        root=str(root / "data"), train=False, download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loss, test_accuracy = run_epoch(model, test_loader, device)
    result = dict(best_val_accuracy=best_accuracy,
                  test_loss=test_loss, test_accuracy=test_accuracy,
                  classes=test_dataset.classes,
                  pretrained_weights=str(args.weights.resolve()))
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"测试 loss={test_loss:.4f}，准确率={test_accuracy:.2%}")
    print(f"分类结果保存至：{args.output_dir}")


if __name__ == "__main__":
    main()
