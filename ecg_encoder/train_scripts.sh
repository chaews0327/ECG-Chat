# add --wfep if you want use the Waveform Data Enhancement
python3 -m ecg_encoder.training.main \
    --ptbxl-path="/data/medicalai/samples/hyperkalemia" \
    --warmup 10000 \
    --batch-size 16 \
    --lr 1e-4 \
    --wd 0.1 \
    --epochs 20 \
    --model coca_roberta-ViT-B-32 \
    --config "./ecg_encoder/model/config.json" \
    --grad-clip-norm 0.5 \
    --delete-previous-checkpoint \
    --wfep \
    --train \