def load_model_config(num_classes, embed_layer, norm_layer, block_fn, mlp_layer):
    """
    JSON ファイルから設定を読み込み、指定したクラスインスタンスを設定に挿入する関数。

    Parameters:
        embed_layer (class): PatchEmbedのクラスインスタンス
        norm_layer (class): LayerNormのクラスインスタンス
        block_fn (class): Blockのクラスインスタンス
        mlp_layer (class): Mlpのクラスインスタンス

    Returns:
        dict: 完全なモデル設定
    """
    model_config = {
    "img_size": (224, 224),
    "patch_size": 16,
    "in_chans": 3,
    "num_classes": num_classes,
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
    "embed_layer": None,
    "norm_layer": None,
    "act_layer": None,
    "block_fn": None,
    "mlp_layer": None,
}

    # クラスインスタンスを設定に代入
    model_config["embed_layer"] = embed_layer
    model_config["norm_layer"] = norm_layer
    model_config["block_fn"] = block_fn
    model_config["mlp_layer"] = mlp_layer

    return model_config

# 使用例
# 必要なクラスのインスタンスを渡して使用
if __name__ == "__main__":
    # ここでクラスインスタンスを定義またはインポート
    from timm.layers import PatchEmbed, Mlp
    from models.vit import Block
    from torch.nn import LayerNorm

    # 関数呼び出し
    config = load_model_config(num_classes, PatchEmbed, LayerNorm, Block, Mlp)
    print(config)
