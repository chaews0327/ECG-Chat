import logging
import random
import sys
import os
import json
from datetime import datetime

import numpy as np
import torch
from torch import optim
from transformers import get_cosine_schedule_with_warmup

from ecg_encoder.model.coca import CoCa
from ecg_encoder.model.factory import get_tokenizer, get_model_preprocess_cfg
from ecg_encoder.training.data import get_data
from ecg_encoder.training.parameters import parse_args
from ecg_encoder.training.train import train
from ecg_encoder.training.evaluate import test
from ecg_encoder.training.transform import ecg_transform, PreprocessCfg
from ecg_encoder.training.loss import create_loss
from ecg_encoder.training.scheduler import cosine_lr


LATEST_CHECKPOINT_NAME = "epoch_10.pt"


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    args = parse_args(args)
    device = args.cuda_device
    
    # 모델 이름 자동 생성
    if args.name is None:
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = '-'.join([
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"wfep_{args.wfep}",
            f"{date_str}",
        ])
        
    # 모델 체크포인트 설정
    log_base_path = os.path.join(args.logs, args.name)
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if args.train:
        os.makedirs(args.checkpoint_path, exist_ok=True)

    random_seed(args.seed)
    
    model = CoCa(args.config).to(device)
    cfg_dict = get_model_preprocess_cfg(model.ecg)
    pp_cfg = PreprocessCfg(**cfg_dict) 
    preprocess_train = ecg_transform(pp_cfg, is_train=True)
    preprocess_val = ecg_transform(pp_cfg, is_train=False)
    
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)
    
    # Weight Decay를 적용/미적용할 파라미터 정의
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    
    with open(args.config, "r") as f:
        config = json.load(f)
        
    start_epoch = 0
    
    # 토크나이저 생성
    token_model = config["text_cfg"]["hf_tokenizer_name"]
    tokenizer = get_tokenizer(token_model)
    
    # 데이터셋 생성
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        tokenizer=tokenizer,
    )
    
    # Test 시 optimizer 및 scaler 미정의
    if args.eval:
        optimizer = None
    
    # Train 시 optimizer 및 scheduler 정의
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    if args.resume is not None:  # 체크포인트 존재 시
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
    
    if args.train:
        loss = create_loss(args)
        
        for epoch in range(start_epoch, args.epochs):
            l1, l2 = train(args, model, data, loss, epoch, optimizer, scheduler)
            completed_epoch = epoch + 1
            metrics, attn_data = test(args, model, data, completed_epoch)
            
            import matplotlib.pyplot as plt
            def plot_losses(
                loss_data, title,
                save_path: str,
            ):
                plt.figure(figsize=(10, 6))
                
                epochs = range(1, len(loss_data) + 1)
                plt.plot(epochs, loss_data)

                plt.title(title)
                plt.xlabel("Batches")
                plt.ylabel("Loss Value")
                plt.grid(True, linestyle='--')
                plt.savefig(save_path)
                plt.close()

                print(f"Graph saved to: {save_path}")
            
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0):
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                    )
                if args.delete_previous_checkpoint:
                    previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                    if os.path.exists(previous_checkpoint):
                        os.remove(previous_checkpoint)
                        
                # attention 데이터 저장
                save_path = os.path.join(args.checkpoint_path, f"attn_map_epoch_{completed_epoch}.pt")
                torch.save(attn_data, save_path)
                logging.info(f"Attention data saved to: {save_path} for epoch {completed_epoch}.")
                        
            plot_losses(l1, "contrastive loss", os.path.join(args.checkpoint_path, f"contrastive_loss_{completed_epoch}.png"))
            plot_losses(l2, "captioning loss", os.path.join(args.checkpoint_path, f"captioning_loss_{completed_epoch}.png"))
    
    if args.eval:
        metrics, attn_data = test(args, model, data, start_epoch)
        
        # attention 데이터 저장
        os.makedirs(args.checkpoint_path, exist_ok=True)
        save_path = os.path.join(args.checkpoint_path, f"attn_map_epoch_{start_epoch}.pt")
        torch.save(attn_data, save_path)
        logging.info(f"Attention data saved to: {save_path} for epoch {start_epoch}.")
        return
    
    
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])