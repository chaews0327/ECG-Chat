"""
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/training/main.py
"""


import logging
import re
import random
import sys
import json

import numpy as np
import torch

from ecg_encoder.model.coca import CoCa
from ecg_encoder.model.factory import get_tokenizer, get_model_preprocess_cfg
from ecg_encoder.training.data import get_data
from ecg_encoder.training.parameters import parse_args
from ecg_encoder.training.evaluate import test
from ecg_encoder.training.transform import ecg_transform, PreprocessCfg


LATEST_CHECKPOINT_NAME = "epoch_10.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args):
    args = parse_args(args)

    random_seed(args.seed, 0)
        
    model = CoCa(args.config)
    cfg_dict = get_model_preprocess_cfg(model.ecg)  # 안전하게 dict 얻기
    pp_cfg = PreprocessCfg(**cfg_dict) 
    preprocess_train = ecg_transform(pp_cfg)
    preprocess_val = ecg_transform(pp_cfg)
    
    optimizer = None
    scaler = None

    start_epoch = 0
    if args.resume is not None:  # 체크포인트 존재 시
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
                
        if 'epoch' in checkpoint:
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(new_state_dict, strict=False)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])

    with open(args.config, "r") as f:
        config = json.load(f)
        
    token_model = config["text_cfg"]["hf_tokenizer_name"]
    tokenizer = get_tokenizer(token_model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    
    if args.eval:
        test(model, data, start_epoch)
        return
    
    
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])