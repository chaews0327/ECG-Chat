import logging
import numpy as np
import torch
import torch.nn.functional as F
import random


def test(args, model, data, epoch):
    metrics = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    if args.train:
        dataloader = data['val'].dataloader
    elif args.eval:
        dataloader = data['test'].dataloader
    num_samples = 0

    all_ecg_features, all_text_features = [], []
    all_ecgs, all_texts = [], []
    
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            ecgs, texts, raw_texts = batch
            ecgs = ecgs.to(device)
            texts = texts.to(device)

            model_out = model(ecgs, texts, output_labels=False)
            ecg_features = model_out["ecg_features"]
            text_features = model_out["text_features"]
            logit_scale = model_out["logit_scale"]
            
            all_ecg_features.append(ecg_features.cpu())
            all_text_features.append(text_features.cpu())
            batch_size = ecgs.shape[0]

            num_samples += batch_size
            
            all_ecgs.append(ecgs.cpu())
            all_texts.extend(raw_texts)
                        
        test_metrics = get_clip_metrics(
            ecg_features=torch.cat(all_ecg_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        metrics.update(
            {**test_metrics, "epoch": epoch, "num_samples": num_samples}
        )

    logging.info(
        f"Test Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    
    print_topk_ecg_to_text_matches(
        ecg_features=torch.cat(all_ecg_features),
        text_features=torch.cat(all_text_features),
        all_texts=all_texts,
        logit_scale=logit_scale.cpu(),
        top_k=5,
        n_samples=10,
        seed=args.seed
    )
    
    print_topk_generations(
        model=model,
        all_texts=all_texts,
        ecgs=torch.cat(all_ecgs),
        n_samples=10,
        seed=args.seed
    )
    
    return metrics


def get_clip_metrics(ecg_features, text_features, logit_scale):
    metrics = {}
    logits_per_ecg = (logit_scale * ecg_features @ text_features.t()).detach().cpu()  # [i][j]: ECG i와 Text j의 유사도
    logits_per_text = (logit_scale * text_features @ ecg_features.t()).detach().cpu()  # Text i와 ECG j의 유사도

    logits = {f"ecg_to_text": logits_per_ecg, f"text_to_ecg": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)  # 각 ECG에 대응되는 GT Text

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)  # 유사도가 높은 순서대로 인덱스 정렬
        preds = torch.where(ranking == ground_truth)[1]  # (row index)/(ranking)
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1  # 인덱스는 1부터 시작해야 함
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def print_topk_ecg_to_text_matches(ecg_features, text_features, all_texts, logit_scale, top_k=5, n_samples=10, seed=42):
    random.seed(seed)
    logits = (logit_scale * ecg_features @ text_features.t()).detach().cpu()
    sample_indices = random.sample(range(len(logits)), k=n_samples)

    for i in sample_indices:
        print(f"\nECG Index: {i}")
        print(f"GT: {all_texts[i]}")

        topk_idx = torch.argsort(logits[i], descending=True)[:top_k]
        for rank, idx in enumerate(topk_idx):
            if idx.item() == i:
                print(f"Top{rank+1}: {all_texts[idx]} (O)")
            else:
                print(f"Top{rank+1}: {all_texts[idx]} (X)")
                
                
def print_topk_generations(model, all_texts, ecgs, n_samples=10, seed=42):
    random.seed(seed)
    sample_indices = random.sample(range(len(ecgs)), k=n_samples)
    ecgs_sample = ecgs[sample_indices].to(next(model.parameters()).device)

    generations = model.generation(ecgs_sample, top_p=0.9,
                                   sot_token_id=model.text.tokenizer.cls_token_id,
                                   eos_token_id=model.text.tokenizer.sep_token_id)

    print("\n[Randomly Sampled Generations]")
    for i, idx in enumerate(sample_indices):
        print(f"[{i}] ECG Index: {idx}")
        print(f"GT: {all_texts[idx]}")
        print(f"GEN: {generations[i]}")
