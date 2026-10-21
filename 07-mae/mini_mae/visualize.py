from pathlib import Path
import matplotlib

# 在导入 pyplot 前选择独立窗口后端，避开 PyCharm 的 InterAgg 后端。
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import torch
from mini_mae.data import build_dataloaders
from mini_mae.model import MiniMAE
from mini_mae.config import MODEL_CONFIG, EXPERIMENT_NAME


def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    _, val_dataloader = build_dataloaders(
        data_dir=project_root / "data",
        batch_size=4,
        train_size=45000,
        val_size=1000,
        seed=42,
    )
    images, _ = next(iter(val_dataloader))
    images = images.to(device)
    # 2. 加载训练好的参数
    model = MiniMAE(**MODEL_CONFIG).to(device)
    state_dict = torch.load(
        project_root / "outputs" / EXPERIMENT_NAME / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    # 固定本次展示的随机遮挡，方便重复比较
    torch.manual_seed(42)

    with torch.no_grad():
        loss, pred, mask = model(images, mask_ratio=0.50)
        target = model.patchify(images)
        pixel_mask = mask.unsqueeze(-1)
        mask_patches = target * (1 - pixel_mask)
        reconstructed_patches = (
                target * (1 - pixel_mask)
                + pred * pixel_mask
        )
        masked_images = model.unpatchify(mask_patches)
        reconstructed_images = model.unpatchify(
            reconstructed_patches
        )

        # 4. 转到 CPU，用于显示
        originals = images.cpu()
        masked_images = masked_images.cpu()
        reconstructed_images = reconstructed_images.cpu()

        count = originals.shape[0]

        fig, axes = plt.subplots(
            count,
            3,
            figsize=(9, count * 3),
            squeeze=False,
        )

        titles = ["Original", "Masked", "Reconstructed"]

        for row in range(count):
            samples = [
                originals[row],
                masked_images[row],
                reconstructed_images[row],
            ]

            for col, image in enumerate(samples):
                # [C, H, W] → [H, W, C]
                display_image = image.permute(1, 2, 0).clamp(0, 1)

                axes[row, col].imshow(
                    display_image,
                    interpolation="nearest",
                )
                axes[row, col].axis("off")

                if row == 0:
                    axes[row, col].set_title(titles[col])

        fig.suptitle(f"Batch masked MSE: {loss.item():.6f}")
        fig.tight_layout()

        output_dir = project_root / "outputs" / EXPERIMENT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "reconstruction.png"

        fig.savefig(save_path, dpi=150)
        print(f"对比图已保存：{save_path}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
