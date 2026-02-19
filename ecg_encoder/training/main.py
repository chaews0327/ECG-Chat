import logging
import random
import sys
import os
import json
from datetime import datetime
from pathlib import Path

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

from ecg_encoder.model.melp.models.melp_model import MELPModel


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
    
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import wfdb

    def debug_visualize_stages(dataset, preprocess_train, preprocess_val):
        """
        1. 파일에서 직접 Raw 로드
        2. preprocess_val 적용 (Norm + Resize)
        3. preprocess_train 적용 (Aug + Norm + Resize)
        """
        # 1. 원본 데이터 직접 로드 (첫 번째 샘플)
        raw_path = dataset.path[0]
        print(raw_path)
        raw_data, _ = wfdb.rdsamp(raw_path)
        np.save("test.npz",raw_data)
        raw_data[np.isnan(raw_data)] = 0
        raw_data[np.isinf(raw_data)] = 0
        
        # 모델 입력 규격에 맞게 변환 (C, T) -> (1, C, T)
        raw_tensor = torch.Tensor(raw_data.astype(np.float32)).T.unsqueeze(0)
        
        # 2. 단계별 변환 적용 (In-place 방지를 위해 clone 사용)
        # (1) Original (Normalize/Resize 전)
        # (2) Val Transform (Normalize + Resize)
        with torch.no_grad():
            norm_res_data = preprocess_val(raw_tensor.clone()).squeeze(0)
            # (3) Train Transform (Augmentation + Normalize + Resize)
            # 증강 확률이 p=0.5 등이면 여러 번 시도해서 바뀐 걸 찾아야 할 수 있음
            aug_data = preprocess_train(raw_tensor.clone()).squeeze(0)

        # 3. 시각화 (Lead I 기준)
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
        lead_idx = 0 
        
        # Plot 1: Original Raw
        axes[0].plot(raw_data[:, lead_idx], color='black', linewidth=0.7)
        axes[0].set_title(f"Raw Signal (from {os.path.basename(raw_path)})")
        
        # Plot 2: Normalized & Resized
        axes[1].plot(norm_res_data[lead_idx].numpy(), color='blue', linewidth=0.7)
        axes[1].set_title("Normalization & Resize")
        
        # Plot 3: Augmented
        axes[2].plot(aug_data[lead_idx].numpy(), color='red', linewidth=0.7)
        axes[2].set_title("Normalization & Resize & Augmentation")

        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylabel("Amplitude")

        plt.tight_layout()
        save_path = "transformation_steps.png"
        plt.savefig(save_path)
        print(f"=== Debug plot saved to {save_path} ===")
    
    if 'train' in data:
        print("시각화 디버깅을 시작합니다...")
        # data['train'].dataloader.dataset은 ECGTextDataset 객체입니다.
        debug_visualize_stages(
            data['train'].dataloader.dataset, 
            preprocess_train, 
            preprocess_val
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


def get_ecg_encoder(model_name, checkpoint_path, device):
    model = MELPModel(
        ecg_encoder_name="ecgfm", 
        ecg_encoder_weight=checkpoint_path, # 체크포인트 경로 전달
        device=device
    )
    
    # 2. MELP 내부에 이미 가중치 로드 로직이 포함되어 있습니다 (init_ecg_encoder)
    # 만약 위 생성자에서 로드가 안 된다면 아래처럼 수동 로드
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'], strict=False)

    # 3. 인코더 모듈만 추출
    ecg_encoder = model.ecg_encoder 
    
    # 4. 전처리 도구(preprocess)는 MELP 내부 형식을 따름
    # MELP는 별도의 preprocess_val 객체 대신 모델 내부에 로직이 녹아있을 수 있음
    preprocess_val = None 
    
    # 5. 모델 설정 반환 (LLaVA의 Tower가 참조할 수 있게)
    model_config = {
        "ecg_cfg": {
            "width": 768, # ECGFM small 기준
            "seq_length": 5000,
            "patch_size": 50 # 아키텍처에 따라 확인 필요
        }
    }

    ecg_encoder.to(device)
    ecg_encoder.eval() # 추론 모드
    
    return ecg_encoder, preprocess_val, model_config
    
    
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])