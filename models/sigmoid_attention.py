import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import use_fused_attn
from torch.jit import Final


class Sigmoid_Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, 
                x: torch.Tensor, 
                output_attentions=False,
                attn_info=None,
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        #print("attn_info : ", attn_info)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if output_attentions:
           attention = attn#.detach()

        sigmoid_att = torch.sigmoid(attn)
        # attn_info
        if attn_info is not None:
            print("Replace Sigmoid Attntion : bath=0 data all head")
            for i in range(6):  # dim=1 で 6 回繰り返す
                mask = attn_info != -1  # -1ではない部分のマスク
                sigmoid_att[0, i, 0, 1:][mask] = attn_info[mask]  # 値を置き換え
            print(f"Replace {mask.sum().item()} patch")
                
        inverse_sigmoid_att = torch.log(sigmoid_att / (1 - sigmoid_att))
        
        # STE : backward()のみ，inverse_sigmoid_attを考慮しないようにする!!
        attn_ = attn + (inverse_sigmoid_att - attn).detach()

        attn = attn_.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if output_attentions:
            return x, attention
        else:
            return x
