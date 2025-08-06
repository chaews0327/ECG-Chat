"""
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/coca_model.py
하이퍼 파라미터의 설정은 다음 링크를 따름: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/model_configs/coca_ViT-B-32.json
"""


import json
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from ecg_encoder.model.ecg_encoder import CLIPEcgCfg, build_ecg_encoder
from ecg_encoder.model.text_encoder import CLIPTextCfg, build_text_encoder
from ecg_encoder.model.multimodal_decoder import MultimodalCfg, build_multimodal_decoder


class CoCa(nn.Module):
    def __init__(self, cfg,
                 quick_gelu=False,
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 pad_id=0):
        super().__init__()
        
        with open(cfg, "r") as f:
            config = json.load(f)
        embed_dim = config['embed_dim']
        ecg_cfg = CLIPEcgCfg(**config['ecg_cfg'])
        text_cfg = CLIPTextCfg(**config['text_cfg'])
        multimodal_cfg = MultimodalCfg(**config['multimodal_cfg'])

        self.ecg = build_ecg_encoder(embed_dim, ecg_cfg)
        self.text = build_text_encoder(embed_dim, text_cfg)
        self.text_decoder = build_multimodal_decoder(text_cfg.vocab_size, multimodal_cfg)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.pad_id = pad_id

        self.context_length = multimodal_cfg.context_length
        
    
    def forward(self, ecg, text=None,
                ecg_latent=None,
                ecg_embs=None,
                output_labels=True):
        if (ecg_latent or ecg_embs) is None:
            ecg_latent, token_embs = self.ecg(ecg)
            ecg_latent = F.normalize(ecg_latent, dim=-1)

        if text is None:
            return {"ecg_features": ecg_latent, "ecg_embs": ecg_embs}
        
        text_latent, token_embs = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1)
        
        labels = text[:, 1:] if output_labels else None
        if output_labels:
            # align text_embs and thus logits with labels for teacher-forcing caption loss
            token_embs = token_embs[:, :-1]

        logits = self.text_decoder(ecg_embs, token_embs)
        out_dict = {
            "ecg_features": ecg_latent,
            "text_features": text_latent,
            "logits": logits,
            "logit_scale": self.logit_scale.exp()
        }
        if labels is not None:
            out_dict["labels"] = labels
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict