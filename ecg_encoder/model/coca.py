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
    TopKLogitsWarper,
    BeamSearchScorer,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    MaxLengthCriteria,
    StoppingCriteriaList,
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
        self.text_decoder = build_multimodal_decoder(text_cfg.vocab_size, multimodal_cfg)  # vocab size 변경
        
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

        logits = self.text_decoder(ecg_embs, token_embs)  # (64, 100, 768) (64, 76, 768)
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
        temperature=1., top_p=0.1, top_k=1, pad_token_id=None, eos_token_id=None,
        sot_token_id=None, min_seq_len=5, repetition_penalty=1.0,
        fixed_output_length=False, generation_type="beam_search",
        num_beams=6, num_beam_groups=3,):  # Eval/Test 시 아래의 함수로 이어서 진행
        
        device = ecg.device  # 디바이스 통일
        
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
            
            if generation_type == "beam_search":
                output = self.beamsearch_generation(
                    ecg_inputs=ecg,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    pad_len = seq_len - output.shape[1]
                    return torch.cat((
                            output,
                            torch.ones(output.shape[0], pad_len, device=device, dtype=output.dtype) * self.pad_id
                        ),
                        dim=1
                    )
                return output
            
            elif generation_type == "top_p":
                logit_warper = TopPLogitsWarper(top_p)
            elif generation_type == "top_k":
                logit_warper = TopPLogitsWarper(top_k)
            
            ecg_latent, ecg_embs = self.ecg(ecg)  # (B, D), (B, T, D)
            ecg_latent = F.normalize(ecg_latent, dim=-1)
            
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
                    for processor in logit_processor:
                        if hasattr(processor, "eos_token_id"):
                            if not torch.is_tensor(processor.eos_token_id):
                                processor.eos_token_id = torch.tensor([processor.eos_token_id], device=device)
                            else:
                                processor.eos_token_id = processor.eos_token_id.to(device)
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
            
            self.train(was_training)  # 기존 상태로 변경
            return out
        
    
    def beamsearch_generation(self, ecg_inputs, pad_token_id=None, eos_token_id=None, sot_token_id=None,
            num_beams=6, num_beam_groups=3, min_seq_len=5, stopping_criteria=None, logit_processor=None,
            logit_warper=None,):
        
        device = ecg_inputs.device
        batch_size = ecg_inputs.shape[0]
        ecg_inputs = torch.repeat_interleave(ecg_inputs, num_beams, dim=0)
        ecg_latent, ecg_embs = self.ecg(ecg_inputs)

        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            outputs = self(
                ecg_inputs,
                input_ids,
                ecg_latent=ecg_latent,
                ecg_embs=ecg_embs,
                output_labels=False,
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logit_processor(
                    group_input_ids.to("cpu"), next_token_logits.to("cpu"), current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1).to("cpu")
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, None).any().item():
            # if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']
