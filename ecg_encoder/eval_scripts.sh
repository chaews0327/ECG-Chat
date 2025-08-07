python3 -m ecg_encoder.training.main \
    --model coca_roberta-ViT-B-32 \
    --config "./ecg_encoder/model/config.json" \
    --ptbxl-path="./data" \
    --resume "./ecg_encoder/checkpoints/epoch_10.pt" \
    --lr 1e-4 \
    --wd 0.1 \
    --eval