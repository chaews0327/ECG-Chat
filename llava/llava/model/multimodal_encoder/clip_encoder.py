import torch
import torch.nn as nn
import sys

from ecg_encoder.training.main import get_ecg_encoder


class CLIPECGTower(nn.Module):
    def __init__(self, ecg_tower, args, delay_load=False):
        super().__init__()

        self.model_config = None
        self.ecg_processor = None
        self.ecg_tower = None
        self.is_loaded = False

        self.ecg_tower_name = ecg_tower
        self.model_name = getattr(args, 'open_clip_config', None)
        if self.model_name is None:
            raise ValueError('No open_clip config for building ECG encoder!')

        self.load_model(self.model_name)

        ecg_config = self.model_config.get('ecg_cfg', {})

        self.hidden_size = ecg_config.get('width', 768)
        self.seq_length = ecg_config.get('seq_length', 5000)
        self.patch_size = ecg_config.get('patch_size', 50)
        self.device = next(self.ecg_tower.parameters()).device
        self.dtype = next(self.ecg_tower.parameters()).dtype

        self.num_patches_per_side = self.seq_length // self.patch_size
        self.num_patches = self.seq_length // self.patch_size


    def load_model(self, model_name, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.ecg_tower, self.ecg_processor, self.model_config = get_ecg_encoder(model_name, checkpoint_path=self.ecg_tower_name, device='cpu')
        self.ecg_tower.requires_grad_(False)

        self.is_loaded = True
        print("Loaded {} model".format(self.ecg_tower_name))


    @torch.no_grad()
    def forward(self, ecgs):
        ref_param = next(self.ecg_tower.parameters())
        self.device = ref_param.device
        self.dtype = ref_param.dtype

        if isinstance(ecgs, list):
            ecg_features = []
            for ecg in ecgs:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                        r, b, t = self.ecg_tower._encode_ecg(ecg.to(device=self.device, dtype=torch.float32).unsqueeze(0))
                        feat = self._combine_ecg_features(r, b, t)
                ecg_features.append(feat.to(self.dtype))
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                    r, b, t = self.ecg_tower._encode_ecg(ecgs.to(device=self.device, dtype=torch.float32))
                    feat = self._combine_ecg_features(r, b, t)
            ecg_features = feat.to(self.dtype)

        return ecg_features


    def _combine_ecg_features(self, rhythm, beat, token):
        if rhythm.dim() == 2:
            rhythm = rhythm.unsqueeze(1)
        
        feats = [rhythm, beat, token]
        max_dim = max(f.shape[-1] for f in feats)
        
        padded = []
        for f in feats:
            if f.shape[-1] < max_dim:
                pad_size = max_dim - f.shape[-1]
                p = torch.zeros((*f.shape[:-1], pad_size), device=f.device, dtype=f.dtype)
                f = torch.cat([f, p], dim=-1)
            padded.append(f)
        
        return torch.cat(padded, dim=1) # [Batch, Total_Tokens, 768]


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

