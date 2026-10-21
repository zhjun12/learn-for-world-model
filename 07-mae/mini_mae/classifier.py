"""冻结 MAE 编码器的线性分类模型。"""

import torch
from torch import nn


class MAEClassifier(nn.Module):
    def __init__(self, mae, num_classes=10):
        super().__init__()
        self.patch_embed = mae.patch_embed
        self.pos_encoder = mae.pos_encoder
        self.encoder = mae.encoder

        # 先冻结已有参数，再创建可训练的分类头。
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.head = nn.Linear(self.pos_encoder.shape[-1], num_classes)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.patch_embed.eval()
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            tokens = self.patch_embed(images) + self.pos_encoder
            features = self.encoder(tokens).mean(dim=1)
        return self.head(features)
