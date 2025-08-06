"""
Transformer Decoder: Multimodal (Signal, Text)
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/transformer.py
REF: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html
"""


from dataclasses import dataclass
from .text_encoder import CLIPTextCfg

import torch
import torch.nn as nn

from .ecg_encoder import TransformerEncoder, ResidualBlock


class MultimodalDecoder(TransformerEncoder):
    def __init__(self, width, layers, heads,
                 context_length=77,
                 mlp_ratio=4.0,
                 ls_init_value=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 output_dim=512):
        super().__init__(width, layers, heads, mlp_ratio, ls_init_value, act_layer,norm_layer)
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualBlock(width, heads, mlp_ratio, ls_init_value, act_layer, norm_layer, is_cross_attention=True)
            for _ in range(layers)
        ])

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        
            
    def forward(self, ecg, txt):
        text_embs = txt.permute(1, 0, 2)  # Q
        image_embs = ecg.permute(1, 0, 2)  # K, V
        seq_len = text_embs.shape[0]
        
        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            text_embs = resblock(text_embs)
            text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)
            
        x = text_embs.permute(1, 0, 2)
        x = self.ln_final(x)
        
        if self.text_projection is not None:
            x = x @ self.text_projection
            
        return x


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    
    
def build_multimodal_decoder(embed_dim, multimodal_cfg):
    if isinstance(multimodal_cfg, dict):
        multimodal_cfg = MultimodalCfg(**multimodal_cfg)
        
    decoder = MultimodalDecoder(context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    )

    return decoder