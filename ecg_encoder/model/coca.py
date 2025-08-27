"""
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/coca_model.py
하이퍼 파라미터의 설정은 다음 링크를 따름: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/model_configs/coca_ViT-B-32.json
함수 내의 Input/Output은 원본 코드의 설정을 그대로 따라감
"""


import json
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from transformers import (
        LogitsProcessorList,
        TopPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList
)

from ecg_encoder.model.ecg_encoder import CLIPEcgCfg, build_ecg_encoder
from ecg_encoder.model.text_encoder import CLIPTextCfg, build_text_encoder
from ecg_encoder.model.multimodal_decoder import MultimodalCfg, build_multimodal_decoder


class CoCa(nn.Module):
    def __init__(self, cfg,
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 pad_id=0):
        super().__init__()
        
        with open(cfg, "r") as f:
            config = json.load(f)  # 설정 불러오기
        
        # Model Configuration 가져오기
        embed_dim = config['embed_dim']
        ecg_cfg = CLIPEcgCfg(**config['ecg_cfg'])
        text_cfg = CLIPTextCfg(**config['text_cfg'])
        multimodal_cfg = MultimodalCfg(**config['multimodal_cfg'])

        # 모델 생성
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
        
        if ecg_latent is None or ecg_embs is None:
            ecg_latent, ecg_embs = self.ecg(ecg)
            ecg_latent = F.normalize(ecg_latent, dim=-1)  # 각 배치 별로 unit vector 생성 (유사도 계산)

        if text is None:
            return {"ecg_features": ecg_latent, "ecg_embs": ecg_embs}
        
        text_latent, token_embs = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1)
        
        labels = text[:, 1:] if output_labels else None
        if output_labels:  # 정답이 있을 때 (Training): Teacher Forcing 사용
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
    
    
    def generation(self, ecg, text=None, seq_len=30, max_seq_len=77,
        temperature=1., top_p=0.1, pad_token_id=None, eos_token_id=None,
        sot_token_id=None, min_seq_len=5, repetition_penalty=1.0,
        fixed_output_length=False):  # Eval/Test 시 아래의 함수로 이어서 진행
        
        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),  # 지나치게 짧은 문장 방지
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),  # 반복 토큰 등장 방지
                ]
            )
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=seq_len)])  # seq_len 도달 시 종료

            device = ecg.device  # 디바이스 통일
            logit_warper = TopPLogitsWarper(top_p)
            
            ecg_latent, ecg_embs = self.ecg(ecg)  # (B, D), (B, T, D)
            if text is None:
                text = torch.ones((ecg.shape[0], 1), device=device, dtype=torch.long) * sot_token_id  # SOT: (B, 1)
            
            was_training = self.training  # 현재 상태 저장
            
            # 텍스트가 1D일 시: 배치 추가
            num_dims = len(text.shape)
            if num_dims == 1: # (T,)
                text = text.unsqueeze(0)  # (1, T)
                
            self.eval()
            out = text
            
            while True:  # seq_len 도달 시 생성 종료
                x = out[:, -max_seq_len:]  # (B, T): max_seq_len만큼의 길이 유지 (현재는 무의미함)
                logits = self(ecg, x, ecg_latent, ecg_embs, False)["logits"][:, -1, :]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)  # 마스크 위치 계산
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id  # 기본은 PAD로 설정

                if mask.all():  # 전부 마스킹됨 (PAD/EOS)
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]  # 마스킹되지 않은 부분의 logit값
                    filtered_logits = logit_processor(x[~mask, :], logits)  # 길이/반복 필터링
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)  # top-p
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (x.shape[1] + 1 == seq_len):
                        # 마지막 토큰일 시 EOS 삽입
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)  # 샘플링

                out = torch.cat((out, sample), dim=-1)
                if stopping_criteria(out, None).all():  # 최대 길이 도달 시 생성 종료
                    break

            if num_dims == 1:
                out = out.squeeze(0)
                
            decoded = self.text.tokenizer.batch_decode(out, skip_special_tokens=False)

            self.train(was_training)  # 기존 상태로 변경
            return decoded