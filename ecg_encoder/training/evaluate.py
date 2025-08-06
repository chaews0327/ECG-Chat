"""
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/training/train.py
"""

import logging
from torch.cuda.amp import autocast
import numpy as np
import torch
import torch.nn.functional as F


def test(model, data, epoch, args):
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    dataloader = data['test'].dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute very quickly
    all_ecg_features, all_text_features = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            ecgs, texts = batch
            ecgs = ecgs.to(device)
            texts = texts.to(device)
            with autocast(device_type=device, dtype=torch.float16):
                model_out = model(ecgs, texts)
                ecg_features = model_out["ecg_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]
                # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes problematic
                all_ecg_features.append(ecg_features.cpu())
                all_text_features.append(text_features.cpu())
                batch_size = ecgs.shape[0]

            num_samples += batch_size

        test_metrics = get_clip_metrics(
            ecg_features=torch.cat(all_ecg_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
            set_name="hyperkalemia"+"_"
        )
        metrics.update(
            {**test_metrics, "epoch": epoch, "num_samples": num_samples}
        )

    logging.info(
        f"Test Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    return metrics


def get_clip_metrics(ecg_features, text_features, logit_scale):
    metrics = {}
    logits_per_ecg = (logit_scale * ecg_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_ecg.t().detach().cpu()

    logits = {f"ecg_to_text": logits_per_ecg, f"text_to_ecg": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics