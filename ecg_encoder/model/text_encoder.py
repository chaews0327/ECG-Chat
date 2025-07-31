"""
Transformer: Pretrained
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/hf_model.py
REF: https://huggingface.co/ncbi/MedCPT-Query-Encoder
"""


import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Linear(hidden_size, output_dim)
        
        
    def forward(self, x):
        encoded = self.tokenizer(
            x, 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=64,
        )
        
        x = self.model(**encoded)
        x = self.proj(x)
        
        # FIXME: 원본 코드 내 return 형태 확인
        return x
    