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
            
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(width),
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
    def __init__(self, lead_num, seq_length, patch_size, width, num_layer, num_head):
        super().__init__()
        
        self.lead_num = lead_num  # 채널
        self.seq_length = seq_length  # 전체 길이
        self.patch_size = patch_size  # 커널 크기
        self.num_patch = seq_length // patch_size
        self.class_token = nn.Parameter(torch.randn(1, 1, width))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch+1, width))
        self.norm = nn.LayerNorm(width)  # pre-layer norm 사용
        
        self.conv1 = nn.Conv1d(
            in_channels=lead_num,
            out_channels=width,
            kernel_size=patch_size,
            bias=True)
        self.transformer = TransformerEncoder(width, num_layer, num_head, dropout=0.0)
                
    
    def forward(self, x):
        x = self.conv1(x)  # (*, width, num_patch)
        x = x.permute(0, 2, 1)  # (*, num_patch, width)
        
        class_token = self.class_token.repeat(x.shape[0], -1, -1)  # (*, 1, width)
        x = torch.cat([class_token, x], dim=1)  # (*, num_patch+1, width)
        x = x + self.pos_embedding  # (x, num_patch+1, width)
        x = self.norm(x)
        x = self.transformer(x)
        
        # FIXME: 원본 코드 내 return 형태 확인
        return x
