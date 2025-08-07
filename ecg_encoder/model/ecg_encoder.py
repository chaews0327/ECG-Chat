"""
1D ViT: ECG (Signal) Encoder
REF: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/transformer.py
이외 블로그 구현 등 참고
"""


from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, d_model, num_head, mlp_ratio, ls_init_value, act_layer, norm_layer, is_cross_attention=False):
        super().__init__()
        
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_head)
        self.ls_1 = nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)
            
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = nn.Identity()
        
    
    def attention(self, q, k=None, v=None, attn_mask=None):
        # cross attention과 self attention이 함께 동작할 수 있도록
        k = k if k is not None else q
        v = v if v is not None else q
        attn_mask = attn_mask if attn_mask is not None else None
        
        return self.attn(q, k, v, need_weights=False, attn_mask=attn_mask)[0]


    def forward(self, q, k=None, v=None, attn_mask=None):
        # cross attention의 경우
        if hasattr(self, "ln_1_kv"):
            k = self.ln_1_kv(k)
            v = self.ln_1_kv(v)
        else:
            k = v = None
            
        x = q + self.ls_1(self.attention(self.ln_1(q), k, v, attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, width, layers, heads, mlp_ratio=4.0, ls_init_value=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.width = width
        self.layers = layers
            
        self.resblocks = nn.ModuleList([
            ResidualBlock(width, heads, mlp_ratio, ls_init_value, act_layer, norm_layer) for _ in range(layers)
        ])
        
        
    def forward(self, x, attn_mask=None):
        x = x.transpose(0, 1)
                
        for block in self.resblocks:
            x = block(x, attn_mask)
        x = x.transpose(0, 1)
        
        return x


class ECGEncoder(nn.Module):
    def __init__(self, seq_length, patch_size, lead_num, width, layers, heads, mlp_ratio, ls_init_value,
            patch_dropout=0, attentional_pool=False, attn_pooler_queries=256, attn_pooler_heads=8,
            pos_embed_type='learnable', no_ln_pre=False, final_ln_after_pool=False, pool_type='tok',
            output_tokens=False, output_dim=512, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.output_tokens = output_tokens
        self.seq_length = seq_length  # 전체 길이
        self.lead_num = lead_num  # 채널
        self.patch_size = patch_size
        self.patch_nums = seq_length // patch_size
        self.final_ln_after_pool = final_ln_after_pool
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(
            in_channels=lead_num,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)
        
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn(self.patch_nums+1, width))
        
        self.patch_dropout = nn.Identity()
        self.ln_pre = norm_layer(width)
        
        self.transformer = TransformerEncoder(width, layers, heads, mlp_ratio,
                                              ls_init_value=ls_init_value,
                                              act_layer=act_layer,
                                              norm_layer=norm_layer)
        
        self.attn_pool = None
        pool_dim = width
        self.pool_type = pool_type
        
        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(torch.randn(pool_dim, output_dim))
        
    
    def forward(self, x, output_last_transformer_layer=False):
        x = self.conv1(x)  # (*, width, num_patch)
        x = x.permute(0, 2, 1)  # (*, num_patch, width)
        
        x = torch.cat([self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x], dim=1)  # (*, num_patch+1, width)
        x = x + self.positional_embedding  # (x, num_patch+1, width)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)
        
        if output_last_transformer_layer:
            return x
        
        x = self.ln_post(x)
        pooled, tokens = x[:, 0], x[:, 1:]
        
        if self.proj is not None:
            pooled = pooled @ self.proj
            
        if self.output_tokens:
            return pooled, tokens
            
        return pooled


@dataclass
class CLIPEcgCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 50
    seq_length: int = 5000
    lead_num: int = 12

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None
    
    
def build_ecg_encoder(embed_dim, ecg_cfg):
    if isinstance(ecg_cfg, dict):
        ecg_cfg = CLIPEcgCfg(**ecg_cfg)
    ecg_heads = ecg_cfg.width // ecg_cfg.head_width

    ecg_model = ECGEncoder(
        seq_length=ecg_cfg.seq_length,
        patch_size=ecg_cfg.patch_size,
        lead_num=ecg_cfg.lead_num,
        width=ecg_cfg.width,
        layers=ecg_cfg.layers,
        heads=ecg_heads,
        mlp_ratio=ecg_cfg.mlp_ratio,
        ls_init_value=ecg_cfg.ls_init_value,
        patch_dropout=ecg_cfg.patch_dropout,
        attentional_pool=ecg_cfg.attentional_pool,
        attn_pooler_queries=ecg_cfg.attn_pooler_queries,
        attn_pooler_heads=ecg_cfg.attn_pooler_heads,
        pos_embed_type=ecg_cfg.pos_embed_type,
        no_ln_pre=ecg_cfg.no_ln_pre,
        final_ln_after_pool=ecg_cfg.final_ln_after_pool,
        pool_type=ecg_cfg.pool_type,
        output_tokens=ecg_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        )
    
    return ecg_model