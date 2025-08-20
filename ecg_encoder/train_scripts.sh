# add --wfep if you want use the Waveform Data Enhancement
python3 -m ecg_encoder.training.main \
    --mimic-iv-ecg-path="/data/mimic-iv-ecg/physionet.org/files/mimic-iv-ecg/1.0" \
    --warmup 10000 \
    --batch-size 64 \
    --lr 1e-4 \
    --wd 0.1 \
    --epochs 10 \
    --model coca_roberta-ViT-B-32 \
    --config "./ecg_encoder/model/config.json" \
    --grad-clip-norm 0.5 \
    --delete-previous-checkpoint \
    --wfep \
    --train \