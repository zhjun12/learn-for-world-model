"""训练、可视化和分类共享同一套模型结构。"""

MODEL_CONFIG = dict(
    image_size=32,
    patch_size=4,
    embed_dim=192,
    encoder_depth=6,
    num_heads=6,
    decoder_dim=96,
    decoder_depth=2,
)

EXPERIMENT_NAME = "mae_masked_50"
PRETRAIN_EPOCHS = 20
BATCH_SIZE = 32
