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
        # 추후 xavier 등으로 변경? 원본 코드에서는 init_parameters 함수를 별도로 정의해주고 있음을 확인
        self.text_projection = nn.Parameter(torch.randn(width, output_dim))
        
            
    def forward(self, ecg, txt):
        # output_tokens=True: 입력으로 들어오는 것은 tokens
        # Resblock과 attn을 직접 가져다쓰기 때문에 해당 함수에서 permute가 필요함
        text_embs = txt.permute(1, 0, 2)  # Q; (B, T, D) -> (T, B, D)
        image_embs = ecg.permute(1, 0, 2)  # K, V; (B, T, D) -> (T, B, D)
        
        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            text_embs = resblock(text_embs)
            text_embs = cross_attn(text_embs, k=image_embs, v=image_embs)
            
        x = text_embs.permute(1, 0, 2)  # (T, B, D) -> (B, T, D)
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