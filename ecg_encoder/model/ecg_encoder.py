"""
1D ViT: ECG (Signal) Encoder
REF: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/transformer.py
이외 블로그 구현 등 참고
"""


import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
        TODO: 우선 원본 코드에 따라 TransformerEncoder를 별도 구현을 진행하였긴 하나,
        nn.TransformerEncoderLayer를 사용하지 말아야 할 이유를 찾지 못하였음. (layer normalization 위치 조정 가능)
    """
    def __init__(self, width, num_layer, num_head, dropout=0.0):
        super().__init__()
        
        self.width = width
        self.num_layer = num_layer
                
        self.ln_1 = nn.LayerNorm(width)
        self.attention = nn.MultiHeadAttention(width, num_head)
        self.ln_2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4*width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 4*width),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([])
        for _ in range(num_layer):
            self.blocks.append(
                nn.ModuleDict({
                    "ln_1": self.ln_1,
                    "attention": self.attention,
                    "ln_2": self.ln_2,
                    "ff": self.ff
                })
            )
        
        
    def forward(self, x):
        x = x.transpose(0, 1)
        
        for block in self.blocks:
            residual = x
            x_ln = block["ln_1"](x)
            attn_out = block["attention"](x_ln, x_ln, x_ln)[0]
            x = residual + attn_out

            residual = x
            x_ln = block["ln_2"](x)
            x = residual + block["ff"](x_ln)

        x = x.transpose(0, 1)
        return x


class ECGEncoder(nn.Module):
    def __init__(self, lead_num, seq_length, patch_size, width, num_layer, num_head, kernel_size, stride, padding):
        super().__init__()
        
        self.lead_num = lead_num
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patch = seq_length // patch_size
        self.class_token = nn.Parameter(torch.randn(width))
        self.pos_embedding = nn.Parameter(torch.randn((self.num_patch)))
        
        self.conv1 = nn.Conv1d(in_channels=lead_num, out_channels=width, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.transformer = TransformerEncoder(width, num_layer, num_head, dropout=0.0)
                
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        
        class_token = self.class_token.unsqueeze(0).repeat(self.width, 1, 1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_embedding.unsqueeze(0)
        x = nn.LayerNorm(x)
        x = self.transformer(x)
        
        return x
