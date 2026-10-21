# 读取图片,返回图片批次

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_dataset(data_dir):
    transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    return dataset


def build_dataloaders(
        data_dir,
        batch_size=64,
        train_size=45000,
        val_size=1000,
        seed=42,
):
    dataset = build_dataset(data_dir)
    if train_size <= 0 or val_size <= 0:
        raise ValueError("训练集和验证集大小必须大于零")
    if train_size + val_size > len(dataset):
        raise ValueError("需要的图片数量超过数据集大小")
    # 生成随机排序的图片索引
    generator = torch.Generator().manual_seed(seed)

    # random生成不重复的随机排列
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    ).tolist()

    # 固定末尾的验证图片，扩大训练集时保持验证集不变
    train_indices = indices[:train_size]
    val_indices = indices[-val_size:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # 将单张图片组成批次
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    train_loader, val_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        train_size=45000,
        val_size=1000,
    )

    images, labels = next(iter(train_loader))

    print("训练图片数:", len(train_loader.dataset))
    print("验证图片数:", len(val_loader.dataset))
    print("训练批次数:", len(train_loader))
    print("图片批次形状:", images.shape)
    print("标签批次形状:", labels.shape)
