from timm.layers import Mlp, PatchEmbed
from torch.nn import LayerNorm

from models.vit import Block, VisionTransformer
from models.model_config import load_model_config
from models.sigmoid_attention import Sigmoid_Attention

MODEL_TYPE=["type1"]

"""
type1 : This Model in which only the final layer is replaced with Sigmoid Attention.
type2
"""

def load_Sigmoid_Attention_model(model_type='type1', num_classes=200):
    
    assert model_type in MODEL_TYPE, f"Invalid model_type '{model_type}'. Valid options are: {MODEL_TYPE}"

    model_config = load_model_config(num_classes, PatchEmbed, LayerNorm, Block, Mlp)
    print("model_config : ", model_config)

    if model_type == "type1":
        model = VisionTransformer(**model_config)
        # replace
        model.blocks[-1].attn = Sigmoid_Attention(dim=model_config['embed_dim'], 
                                                  num_heads=model_config['num_heads'], 
                                                  qkv_bias=model_config['qkv_bias']
                                                  )
        
        return model
    
    if model_type == "type2":
        pass

if __name__ == '__main__':
    
    num_classes = 200
    model = load_Sigmoid_Attention_model(model_type='type1', num_classes=num_classes)
    print(model)
