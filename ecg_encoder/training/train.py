import time
import logging
import math

import torch
from torchmetrics.aggregation import MeanMetric


def train(args, model, data, loss, epoch, optimizer, scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    
    dataloader = data['train'].dataloader
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    
    losses_m = {}
    batch_time_m = MeanMetric()
    data_time_m = MeanMetric()
    end = time.time()
    
    for i, batch in enumerate(dataloader):        
        ecg, text = batch
        ecg = ecg.to(device=device)
        text = text.to(device=device)
        
        data_time_m.update(torch.tensor(time.time()-end))
        optimizer.zero_grad()
        
        model_out = model(ecg, text)
        logit_scale = model_out["logit_scale"]
        losses = loss(**model_out, output_dict=True)
        losses = sum(losses.value())  # contrastive + caption loss
            
        losses.backward()
        optimizer.step()
        scheduler.step()
        
        # 시간 측정
        batch_time_m.update(torch.tensor(time.time()-end))
        end = time.time()
        batch_count = i + 1

        if i % 1 == 0:
            batch_size = len(ecg)
            num_samples = batch_count * batch_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / dataloader.num_batches

            # 로그 저장
            for key, val in loss.items():
                if key not in losses_m:
                    losses_m[key] = MeanMetric()
                losses_m[key].update(torch.tensor(val.item()), weight=batch_size)

            # 출력할 로그 정리
            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join([
                f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                for loss_name, loss_m in losses_m.items()
            ])

            samples_per_second = args.batch_size / batch_time_m.val

            # 로그 출력
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )
            