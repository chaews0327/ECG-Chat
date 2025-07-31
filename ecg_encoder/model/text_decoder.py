"""
Transformer Decoder: Multimodal (Signal, Text)
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/transformer.py
REF: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html
"""


import torch
import torch.nn as nn


class MultimodalDecoder(nn.Module):
    def __init__(self, width, num_layer, num_head, dropout=0.0):
        super().__init__()
        
        self.width = width
        self.num_layer = num_layer
            
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln_1_q": nn.LayerNorm(width),
                "ln_1_kv": nn.LayerNorm(width),
                "attention": nn.MultiheadAttention(width, num_head, dropout=dropout),
                "ln_2": nn.LayerNorm(width),
                "ff": nn.Sequential(
                    nn.Linear(width, 4*width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4*width, width),
                    nn.Dropout(dropout),
                )
            })
            for _ in range(num_layer)
        ])
            
            
    def forward(self, img, txt):
        txt = txt.permute(1, 0, 2)  # Q
        img = img.permute(1, 0, 2)  # K, V
        
        for block in self.blocks:
            x = txt
            q = block["ln_1_q"](txt)
            kv = block["ln_1_kv"](img)
            
            attn_out = block["attention"](q, kv, kv)[0]
            x = txt + attn_out
            
            x_ln = block["ln_2"](x)
            x = x + block["ff"](x_ln)
            
        x = x.permute(1, 0, 2)
        
        # FIXME: 원본 코드 내 return 형태 확인
        return x
