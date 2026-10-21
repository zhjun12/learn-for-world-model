import torch.nn as nn
import torch


class PatchEmbed(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size 和 patch_size 必须大于零")
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("输入形状应为 [B, C, H, W]")
        _, C, H, W = images.shape
        if (C, H, W) != (
                self.in_channels,
                self.image_size,
                self.image_size,
        ):
            raise ValueError(
                f"期望输入 [B, {self.in_channels}, "
                f"{self.image_size}, {self.image_size}]，"
                f"实际收到 {tuple(images.shape)}"
            )
        x = self.proj(images)  # [B, D, G, G]
        x = x.flatten(start_dim=2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B,N,D]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, depth=2, mlp_ratio=4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # 每个 token 的特征维度
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),  # 每个 block 内 MLP 的隐藏维度
            dropout=0.0,
            activation='gelu',
            batch_first=True,  # 输入采用 [B, N, D]
            norm_first=True,  # 注意力和mlp前先做layernorm
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        # PyTorch 克隆出的各层初始参数值相同；
        # 这里分别初始化每层的矩阵参数。
        for block in self.blocks.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        return self.blocks(x)


class MiniMAE(nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 embed_dim=64,
                 encoder_depth=2,
                 num_heads=2,
                 decoder_dim=32,
                 decoder_depth=1,
                 ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.pos_encoder = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=encoder_depth,
        )

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        # 2.用于填充被遮挡位置的可学习向量
        self.mask_token = nn.Parameter(
            # randn是标准正态分布,rand是均匀分布
            torch.randn(1, 1, decoder_dim) * 0.02
        )
        # 3.解码器的位置编码
        self.pos_decoder = nn.Parameter(
            torch.randn(1, num_patches, decoder_dim) * 0.02
        )
        # 4.解码器的Transformer
        self.decoder = TransformerEncoder(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            depth=decoder_depth,
        )
        # 5.每个图块需要预测的像素数
        self.patch_dim = patch_size * patch_size * self.patch_embed.in_channels

        # 6.将解码后的特征转化为图块像素
        self.pixel_head = nn.Linear(decoder_dim, self.patch_dim)

    def random_masking(self, tokens, mask_ratio):
        B, N, D = tokens.shape
        num_keep = int(N * (1 - mask_ratio))
        if not 0 < num_keep < N:
            raise ValueError("必须至少保留一个图块，并遮挡一个图块")
        # 随机选择
        noise = torch.rand(B, N, device=tokens.device)  # [8,64]
        order = noise.argsort(dim=1)
        keep_ids = order[:, :num_keep]  # [8,16]
        # [:,None]增加一个维度
        batch_ids = torch.arange(B, device=tokens.device)[:, None]  # [8,1]
        visible = tokens[batch_ids, keep_ids]
        # 原图空间顺序，0可见,1遮挡
        mask = tokens.new_ones(B, N)
        mask[batch_ids, keep_ids] = 0
        return visible, mask, keep_ids

    def forward_encoder(self, images, mask_ratio=0.75):
        tokens = self.patch_embed(images)
        tokens = tokens + self.pos_encoder
        visible, mask, keep_ids = self.random_masking(tokens, mask_ratio)
        latent = self.encoder(visible)
        return latent, mask, keep_ids

    def forward_decoder(self, latent, keep_ids):
        visible_features = self.decoder_embed(latent)
        B, K, D = visible_features.shape
        N = self.patch_embed.num_patches
        # 先用mask_token填满全部位置
        full_tokens = self.mask_token.expand(B, N, D).clone()
        # 把可见图块特征放回原来的空间位置
        batch_ids = torch.arange(B, device=latent.device)[:, None]
        full_tokens[batch_ids, keep_ids] = visible_features
        # 告诉解码器每个图块处于哪个位置
        full_tokens = full_tokens + self.pos_decoder

        # 根据可见图块推断完整序列的特征
        decoder = self.decoder(full_tokens)

        # 将每个位置的特征转换为图块像素
        pred = self.pixel_head(decoder)
        return pred

    def patchify(self, images):
        B, C, H, W = images.shape

        P = self.patch_embed.patch_size  # 切完图片的尺寸
        grid_h = H // P
        grid_w = W // P
        # 将高和宽分别拆成：图块数量 × 图块尺寸
        x = images.reshape(B, C, grid_h, P, grid_w, P)
        # 调整为：批次、图块行、图块列、块内行、块内列、通道
        x = x.permute(0, 2, 4, 3, 5, 1)
        target = x.reshape(B, grid_h * grid_w, P * P * C)
        return target

    def unpatchify(self, patches):
        B, N, patch_dim = patches.shape

        P = self.patch_embed.patch_size
        C = self.patch_embed.in_channels
        G = self.patch_embed.grid_size

        if N != G * G or patch_dim != P * P * C:
            raise ValueError("图块形状与模型配置不匹配")

        x = patches.reshape(B, G, G, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        images = x.reshape(B, C, G * P, G * P)

        return images

    def forward_loss(self, images, pred, mask):
        # 原图的真实图块像素
        target = self.patchify(images)

        # 每个像素的平方误差:[B,N,48]
        squared_error = (pred - target) ** 2
        # 每个图块内部取平均:[B,N]
        loss_per_patch = squared_error.mean(dim=-1)
        # mask:0表示可见,1表示遮挡
        loss = (loss_per_patch * mask).sum() / mask.sum()
        return loss

    def forward(self, images, mask_ratio=0.75):
        latent, mask, keep_ids = self.forward_encoder(
            images, mask_ratio
        )
        pred = self.forward_decoder(latent, keep_ids)
        loss = self.forward_loss(images, pred, mask)

        return loss, pred, mask


if __name__ == "__main__":
    model = MiniMAE()
    model.eval()

    images = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        loss, pred, mask = model(images, mask_ratio=0.75)


    print("图片形状:", images.shape)
    print("预测图块:", pred.shape)
    print("遮挡标记:", mask.shape)
    print("每张图遮挡数量:", mask.sum(dim=1))
    print("重建损失:", loss.item())
