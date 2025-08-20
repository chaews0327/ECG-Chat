import time
import logging
import math

import torch


def train(args, model, data, loss, epoch, optimizer, scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    
    dataloader = data['train'].dataloader
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    
    losses_m = {}
    batch_time_sum = 0.0
    batch_time_count = 0
    data_time_sum = 0.0
    data_time_count = 0
    end = time.time()
    
    for i, batch in enumerate(dataloader):
        ecg, text = batch
        ecg = ecg.to(device=device)
        text = text.to(device=device)
        
        data_time_sum += time.time() - end
        data_time_count += 1
        optimizer.zero_grad()
        
        model_out = model(ecg, text)
        logit_scale = model_out["logit_scale"]
        losses = loss(**model_out, output_dict=True)
        total_loss = sum(losses.values())  # contrastive + caption loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 시간 측정
        batch_time_sum += time.time() - end
        batch_time_count += 1
        end = time.time()
        batch_count = i + 1

        if i % 1 == 0:
            batch_size = len(ecg)
            num_samples = batch_count * batch_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / dataloader.num_batches

            # 로그 저장
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = {"sum": 0.0, "count": 0}
                losses_m[key]["sum"] += val.item() * batch_size
                losses_m[key]["count"] += batch_size

            # 출력할 로그 정리
            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join([
                f"{loss_name.capitalize()}: {losses_m[loss_name]['sum'] / losses_m[loss_name]['count']:.5g}"
                for loss_name, loss_m in losses_m.items()
            ])

            avg_batch_time = batch_time_sum / batch_time_count
            avg_data_time = data_time_sum / data_time_count
            samples_per_second = args.batch_size / avg_batch_time

            # 로그 출력
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {avg_data_time:.3f} "
                f"Batch (t): {avg_batch_time:.3f}, {samples_per_second:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )
            