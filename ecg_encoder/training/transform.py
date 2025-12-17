import torch.nn as nn
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from .augmentations import BaselineWander, RandomMasking, CutMix


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


class RandomApply(torch.nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, ecg):
        if self.p < torch.rand(1):
            return ecg
        for t in self.transforms:
            ecg = t(ecg)
        return ecg

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


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


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    dur: Optional[Tuple[float, float]] = 10
    sr: Optional[int] = 500
    

def ecg_transform(cfg, is_train):
    ecg_size = (cfg.num_channels, cfg.seq_length)  # (D, T)
    mean=cfg.mean  # 현재의 cfg에서는 정해져 있지 않음
    std=cfg.std  # 현재의 cfg에서는 정해져 있지 않음
        
    if mean is not None:
        normalize = Normalize(mean=mean, std=std)
    else:  # (0, 1)로 처리됨
        normalize = Normalize(mean=ECG_MEAN, std=ECG_STD)
    resize = Resize(seq_length=ecg_size[1])

    aug_cfg = AugmentationCfg()

    dur = aug_cfg.dur
    sr = aug_cfg.sr

    if is_train:
        train_transform = [
            RandomApply([BaselineWander(fs=sr), ], p=0.5),
            RandomApply([CutMix(fs=sr)], p=0.5),
            RandomApply([RandomMasking(fs=sr)], p=0.3)
        ]
        train_transform.extend([
            normalize,
            resize
        ])
        train_transform = Compose(train_transform)
        return train_transform
    else:
        transforms = []
        transforms.extend([
            normalize,
            resize
        ])
        return Compose(transforms)