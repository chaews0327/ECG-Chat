"""
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/hf_model.py
REF: https://huggingface.co/ncbi/MedCPT-Query-Encoder
"""


from typing import Optional
from dataclasses import dataclass

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .hf_configs import arch_dict


class MeanPooler(nn.Module):
    def forward(self, x, attn_mask):
        masked_output = x.last_hidden_state * attn_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attn_mask.sum(-1, keepdim=True)


class TextEncoder(nn.Module):
    def __init__(self, model_name, output_dim,
                 pooler_type=None,
                 pretrained=True,
                 output_tokens=False):
        super().__init__()
        self.output_dim = output_dim
        self.output_tokens = output_tokens
        
        uses_transformer_pooler = (pooler_type == "cls_pooler")
        
        self.config = AutoConfig.from_pretrained(model_name)
        create_func, model_args = (AutoModel.from_pretrained, model_name) if pretrained else (
            AutoModel.from_config, self.config)
        if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
            self.transformer = create_func(model_args)
            self.transformer = self.transformer.encoder
        else:
            self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
            
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = MeanPooler()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        hidden_size = (d_model + output_dim) // 2
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim, bias=False),
        )
        
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Linear(hidden_size, output_dim)
        
        
    def forward(self, x):
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        tokens = out.last_hidden_state
        
        if self.output_tokens:
            return projected, tokens
        return projected


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models
    

def build_text_encoder(embed_dim, text_cfg):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
        
    return TextEncoder(
        text_cfg.hf_model_name,
        output_dim=embed_dim,
        proj_type=text_cfg.hf_proj_type,
        pooler_type=text_cfg.hf_pooler_type,
        pretrained=text_cfg.hf_model_pretrained,
        output_tokens=text_cfg.output_tokens,
    )