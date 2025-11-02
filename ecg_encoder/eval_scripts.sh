python3 -m ecg_encoder.training.main \
    --model coca_roberta-ViT-B-32 \
    --config "./ecg_encoder/model/config.json" \
    --mimic-iv-ecg-path="/Users/chaewonshin/Desktop/user/UNIST/ICBM/Project/ECG/ECG-Chat/data/mimic-iv-ecg" \
    --resume "/Users/chaewonshin/Desktop/user/UNIST/ICBM/Project/ECG/ecg_encoder/checkpoints/epoch_10.pt" \
    --lr 1e-4 \
    --wd 0.1 \
    --batch-size 96 \
    --cuda-device "cpu" \
    --eval