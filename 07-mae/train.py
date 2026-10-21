from pathlib import Path
import csv
import json

from matplotlib.figure import Figure
import torch
from mini_mae.data import build_dataloaders
from mini_mae.model import MiniMAE
from mini_mae.config import MODEL_CONFIG, EXPERIMENT_NAME, PRETRAIN_EPOCHS, BATCH_SIZE


def save_training_history(history, output_dir):
    """保存每轮指标与损失曲线，不打开窗口。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "val_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs, [row["train_loss"] for row in history],
            marker="o", label="Train")
    ax.plot(epochs, [row["val_loss"] for row in history],
            marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked pixel MSE")
    ax.set_title("MAE reconstruction loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)


def main():
    # 1.设置路径
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_config.json").write_text(
        json.dumps(MODEL_CONFIG, indent=2), encoding="utf-8"
    )

    # 2创建数据加载器
    train_loader, val_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        train_size=45000,
        val_size=1000,
    )
    # 3.选择设备，创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniMAE(**MODEL_CONFIG).to(device)

    # 4.创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05,
    )
    # 5.训练设置
    epochs = PRETRAIN_EPOCHS
    best_val_loss = float("inf")
    mask_ratio=0.50
    history = []
    print("使用设备", device)

    for epoch in range(1, 1 + epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_ratio=mask_ratio,
        )
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            mask_ratio=mask_ratio,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        # 每轮保存，即使后续中断也保留已完成轮次的记录
        save_training_history(history, output_dir)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )
        # 7. 保存验证损失最低的模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"已保存模型：{save_path}")


def train_one_epoch(
        model, loader, optimizer,
        device, mask_ratio
):
    model.train()
    total_loss = 0.0
    total_images = 0
    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        loss, _, _ = model(
            images,
            mask_ratio=mask_ratio,
        )
        loss.backward()
        optimizer.step()

        # 记录损失，用于计算整轮平均值
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size
    return total_loss / total_images


@torch.no_grad()
def evaluate(
        model,
        loader,
        device,
        mask_ratio=0.75,
):
    model.eval()
    total_loss = 0.0
    total_images = 0
    for images, _ in loader:
        images = images.to(device)
        loss, _, _ = model(
            images,
            mask_ratio,
        )
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / total_images


if __name__ == "__main__":
    main()
