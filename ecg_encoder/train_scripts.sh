# add --wfep if you want use the Waveform Data Enhancement
python3 -m ecg_encoder.training.main \
    --mimic-iv-ecg-path="/Users/chaewonshin/Desktop/user/UNIST/ICBM/Project/ECG/ECG-Chat/data/mimic-iv-ecg" \
    --warmup 10000 \
    --batch-size 96 \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 20 \
    --model coca_roberta-ViT-B-32 \
    --config "./ecg_encoder/model/config.json" \
    --grad-clip-norm 0.5 \
    --cuda-device "cuda:2" \
    --lock-text \
    --cuda-device "cpu" \
    --resume "/Users/chaewonshin/Desktop/user/UNIST/ICBM/Project/ECG/ecg_encoder/checkpoints/epoch_10.pt" \
    --wfep \
    --train \