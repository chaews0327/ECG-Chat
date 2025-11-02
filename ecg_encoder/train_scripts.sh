# add --wfep if you want use the Waveform Data Enhancement
python3 -m ecg_encoder.training.main \
    --mimic-iv-ecg-path="/data/mimic-iv-ecg/physionet.org/files/mimic-iv-ecg/1.0" \
    --warmup 0 \
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
    --wfep \
    --train \