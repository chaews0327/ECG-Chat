#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
open_clip_dir="$script_dir/.."
export PYTHONPATH="$script_dir:$open_clip_dir:$PYTHONPATH"

# MODEL_VERSION="vicuna-7b-v1.5"
MODEL_VERSION=llama-2-7b-chat

CUDA_VISIBLE_DEVICES=0 python llava/train/test_mem.py \
    --lora_enable True \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/ecg_instruct_45k.json \
    --ecg_folder /data/ecg/public/mimic-iv-ecg/physionet.org/files/mimic-iv-ecg/1.0 \
    --ecg_tower /home/chaewon/medicalai/my-ecg-chat/logs/model_coca_roberta-ViT-B-32-lr_0.0001-b_96-wfep_False-2025_12_17-17_06_15/checkpoints/epoch_20.pt \
    --open_clip_config coca_ViT-B-32 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_projector_type linear \
    --mm_use_ecg_start_end False \
    --mm_use_ecg_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True