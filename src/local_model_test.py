import torch
from models.vit import Block, VisionTransformer
from timm.layers import Mlp, PatchEmbed
from torch.nn import LayerNorm
from torchinfo import summary

# vit_small_patch16_224 の設定
model_config = {
    "img_size": (224, 224),
    "patch_size": 16,
    "in_chans": 3,
    "num_classes": 10,
    "global_pool": 'token',
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
    "init_values": None,
    "class_token": True,
    "pos_embed": 'learn',
    "no_embed_class": False,
    "reg_tokens": 0,
    "pre_norm": False,
    "fc_norm": None,
    "dynamic_img_size": False,
    "dynamic_img_pad": False,
    "drop_rate": 0.0,
    "pos_drop_rate": 0.0,
    "patch_drop_rate": 0.0,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "weight_init": '',  # 空の文字列に設定
    "fix_init": False,
    "embed_layer": PatchEmbed,
    "norm_layer": LayerNorm,
    "act_layer": None,
    "block_fn": Block,
    "mlp_layer": Mlp,
}

model = VisionTransformer(**model_config)
summary(model)

state_dict = torch.load('./models/vit_small_patch16_224.pt', map_location=torch.device('cpu'), weights_only=True)
msg = model.load_state_dict(state_dict)
print("model.load_state_dict msg : ", msg)