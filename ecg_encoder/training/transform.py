import torch.nn as nn
import torch
from typing import Tuple
from dataclasses import dataclass


# https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/open_clip/constants.py
# 해당 파일에서 MIMIC_IV_MEAN/STD 값을 가져와서 사용 (적절한 값이 없는 상태이므로)
ECG_MEAN = [0] * 12
ECG_STD = [1] * 12


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        for i in range(len(self.mean)):
            x[:, i, :] = (x[:, i, :] - self.mean[i]) / self.std[i]
        return x


class Resize(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, length = x.shape
        if length < self.seq_length:
            new_x = torch.zeros((b, c, self.seq_length))
            new_x[:, :, 0:length] = x
        elif length > self.seq_length:
            new_x = x[:, :, 0:self.seq_length]
        else:
            new_x = x
        return new_x


class Compose:
    """
    Data augmentation module that transforms any
    given data example with a chain of augmentations.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


@dataclass
class PreprocessCfg:
    seq_length: int = 5000
    duration: int = 10
    sampling_rate: int = 500
    dataset: str = None
    mean: Tuple[float, ...] = None
    std: Tuple[float, ...] = None
    resize_mode: str = 'shortest'

    @property
    def num_channels(self):
        return 12

    @property
    def input_size(self):
        return self.num_channels, self.seq_length
    

def ecg_transform(cfg: PreprocessCfg):
    ecg_size = ecg_size=(cfg.num_channels, cfg.seq_length)
    mean=cfg.mean
    std=cfg.std
    resize_mode=cfg.resize_mode
        
    if mean is not None:
        normalize = Normalize(mean=mean, std=std)
    else:
        normalize = Normalize(mean=ECG_MEAN, std=ECG_STD)
    resize = Resize(seq_length=ecg_size[1])

    
    transforms = []
    transforms.extend([
        normalize,
        resize
    ])
    return Compose(transforms)