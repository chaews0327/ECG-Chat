import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_num_logits = 0
        self.labels = None
    
    
    def get_logits(self, ecg_features, text_features, logit_scale):
        logits_per_ecg = logit_scale * ecg_features @ text_features.T
        logits_per_text = logit_scale * text_features @ ecg_features.T
        
        return logits_per_ecg, logits_per_text
        
        
    def get_ground_truth(self, num_logits):
        if self.prev_num_logits != num_logits:
            labels = torch.arange(num_logits, dtype=torch.long)
            self.prev_num_logits = num_logits
        else:
            labels = self.labels
        return labels
    
    
    def forward(self, ecg_features, text_features, logit_scale, output_dict=False):
        logits_per_ecg, logits_per_text = self.get_logits(ecg_features, text_features, logit_scale)
        labels = self.get_ground_truth(logits_per_ecg.shape[0])
        loss = (F.cross_entropy(logits_per_ecg, labels)+F.cross_entropy(logits_per_text, labels)) / 2
        
        if output_dict:
            return {"contrastive_loss": loss}
        
        return loss


class Loss(ContrastiveLoss):  # Contrastive Loss + Caption Loss
    def __init__(self, contrastive_weight, caption_weight):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.caption_weight = caption_weight
        self.caption_loss = nn.CrossEntropyLoss()
        
    
    def forward(self, ecg_features, text_features, logits, labels, logit_scale, output_dict=False):
        contrastive_loss = super().forward(ecg_features, text_features, logit_scale, output_dict)
        contrastive_loss = contrastive_loss * self.contrastive_weight
        
        caption_loss = self.caption_loss(logits.permute(0, 2, 1), labels)
        caption_loss = caption_loss * self.caption_weight

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "caption_loss": caption_loss}

        return contrastive_loss, caption_loss
    
    
def create_loss(args):
    return Loss(
        contrastive_weight=args.contrastive_loss_weight,
        caption_weight=args.caption_loss_weight,
    )