python3 -m ecg_encoder.training.main \
    --model coca_roberta-ViT-B-32 \
    --config "./ecg_encoder/model/config.json" \
    --mimic-iv-ecg-path="/data/mimic-iv-ecg/physionet.org/files/mimic-iv-ecg/1.0" \
    --resume "/home/chaewon/medicalai/my-ecg-chat/logs/mimic_wde/model_coca_roberta-ViT-B-32-lr_0.0001-b_64-p_amp/checkpoints/epoch_20.pt" \
    --lr 1e-4 \
    --wd 0.1 \
    --wfep \
    --eval