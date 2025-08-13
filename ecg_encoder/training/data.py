"""
REF: https://github.com/YubaoZhao/ECG-Chat/blob/master/open_clip/training/data.py
구조를 동일하게 맞추기 위해 PTB-XL 함수 기준 수정 진행
"""


import os
import pickle
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class ECGTextDataset(Dataset):
    def __init__(self, path, mode, texts, transforms=None, tokenizer=None, is_train=True):
        super(ECGTextDataset, self).__init__()
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.path = path
        self.mode = mode
        self.y = texts
        self.is_train = is_train

    def tokenize(self, text):
        text = text.lower()
        encoded = self.tokenizer(text)
        return encoded[0]

    def load_data(self, idx):
        with open(self.path[idx], 'rb') as f:
            data = pickle.load(f)[self.mode[idx]]['ecg'][0]
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0

        data = torch.Tensor(data.astype(np.float32))
        data = torch.unsqueeze(data, 0)

        if self.transforms is not None:
            data = self.transforms(data)
        data = torch.squeeze(data, 0)
        return data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.load_data(idx)
        y = self.y[idx]
        return x, self.tokenize(y)


class ECGValDataset(ECGTextDataset):
    def __init__(self, dir, path, diagnostics, transforms=None, tokenizer=None):
        abs_path = [os.path.join(dir, p) for p in path]
        super(ECGValDataset, self).__init__(abs_path, None, transforms, tokenizer)
        self.diagnostics = diagnostics

    def __len__(self):
        return self.diagnostics.shape[0]

    def __getitem__(self, idx):
        x = self.load_data(idx)
        diagnostic = self.diagnostics[idx, :]
        return x, diagnostic


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_wave_info(data):
    keys = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval',
            'QTc_Interval', 'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak']
    text_describe = ""
    text_describe += f" RR: {data['RR_Interval']}"
    text_describe += f" PR: {data['PR_Interval']}"
    text_describe += f" QRS: {data['QRS_Complex']}"
    text_describe += f" QT/QTc: {data['QT_Interval']}/{data['QTc_Interval']}"
    text_describe += f" P/R/T Wave: {data['P_Wave_Peak']}/{data['R_Wave_Peak']}/{data['T_Wave_Peak']}"
    return text_describe


def load_ptbxl(path, is_train, test_fold=0):
    Y = pd.read_csv('preprocess/diagnostics_feature.csv')

    if is_train:
        Y = Y[Y.strat_fold != test_fold]
    else:
        Y = Y[Y.strat_fold == test_fold]

    X_rel = Y.ECG_ID.values
    modes = Y.Mode.values
    y = Y.Label.values
    X = [os.path.join(path, str(x)+'.pkl') for x in X_rel]

    texts = []
    for i in range(len(Y)):
        text = str(y[i])
        texts.append(text + get_wave_info(Y.iloc[i]))

    return X, modes, texts


def make_dataloader(args, dataset, is_train, dist_sampler=True, drop_last=None):
    num_samples = len(dataset)
    # sampler = DistributedSampler(dataset) if args.distributed and dist_sampler else None  # 분산학습 진행 X
    sampler = None
    shuffle = is_train and sampler is None
    drop_last = is_train if drop_last is None else drop_last
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=drop_last,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_all_ecg_text_dataset(args, preprocess_train, preprocess_test, epoch=0, tokenizer=None):
    datasets = {}
    # X_train, text_train, X_val, text_val, m_X_test, m_text_test\
    #    = load_mimic_iv_ecg(args.mimic_iv_ecg_path, wfep=args.wfep)
    X_train, mode_train, text_train = load_ptbxl(args.ptbxl_path, is_train=True)  # Train/Test split하도록 변경
    train_dataset = ECGTextDataset(X_train, mode_train, text_train, transforms=preprocess_train, tokenizer=tokenizer,
                                   is_train=True)
    datasets['train'] = make_dataloader(args, train_dataset, is_train=True)

    X_test, mode_test, text_test = load_ptbxl(args.ptbxl_path, is_train=False)
    ptbxl_test_dataset = ECGTextDataset(X_test, mode_test, text_test, transforms=preprocess_test,
                                        tokenizer=tokenizer, is_train=False)
    datasets['test'] = make_dataloader(args, ptbxl_test_dataset, is_train=False, dist_sampler=False)
    return datasets


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns

    data = get_all_ecg_text_dataset(args, preprocess_train, preprocess_val, epoch=epoch, tokenizer=tokenizer)

    return data