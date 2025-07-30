"""
1. ECG Encoder
2. Pretrained Text Encoder
3. Multimodal Text Decoder

각 모델 별 구조는 따로(별도의 파일)? 이후 본 파일에서는 전체 구조에 대한 합치기 및 forward 등의 함수만 작성?
"""


import torch
from torch import nn
import numpy as np


class CoCa(nn.Module):
    def __init__(self):
        # TODO
        ecg_encoder = ecg_encoder()
        text_encoder = text_encoder()
        text_decoder = text_decoder()
        
        