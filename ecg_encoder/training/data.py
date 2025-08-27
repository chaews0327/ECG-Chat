import os
import wfdb
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class ECGTextDataset(Dataset):
    def __init__(self, path, texts, transforms=None, tokenizer=None, is_train=True):
        super(ECGTextDataset, self).__init__()
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.path = path
        self.y = texts
        self.is_train = is_train

    def tokenize(self, text):
        text = text.lower()
        encoded = self.tokenizer(text)
        return encoded[0]

    def load_data(self, idx):
        data = wfdb.rdsamp(self.path[idx])[0]
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0

        data = torch.Tensor(data.astype(np.float32)).T
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
        return x, self.tokenize(y), y


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


@dataclass
class DataInfo:
    dataloader: DataLoader


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


def load_mimic_iv_ecg(path, wfep):
    db = pd.read_csv(os.path.join(path, 'machine_measurements.csv')).set_index('study_id')
    record_list = pd.read_csv('preprocess/filtered_record_list.csv').set_index('study_id')
    all_idx = record_list.index.values
    
    # train/test split: (8:1:1)
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        
    def data(index_list):
        reports = []
        X = []
        n_reports = 18  # 하나의 기록에 리포트 18개
        bad_reports = ["--- Warning: Data quality may affect interpretation ---",
                       "--- Recording unsuitable for analysis - please repeat ---",
                       "Analysis error",
                       "conduction defect",
                       "*** report made without knowing patient's sex ***",
                       "--- Suspect arm lead reversal",
                       "--- Possible measurement error ---",
                       "--- Pediatric criteria used ---",
                       "--- Suspect limb lead reversal",
                       "-------------------- Pediatric ECG interpretation --------------------",
                       "Lead(s) unsuitable for analysis:",
                       "LEAD(S) UNSUITABLE FOR ANALYSIS:",
                       "PACER DETECTION SUSPENDED DUE TO EXTERNAL NOISE-REVIEW ADVISED",
                       "Pacer detection suspended due to external noise-REVIEW ADVISED"]

        for i in index_list:
            row = record_list.loc[i]  # 기록/파일 경로/환자 데이터/WDE DB
            m_row = db.loc[i]  # 리포트가 들어 있는 DB
            report_txt = ""  # 초기화
            for j in range(n_reports):
                report = m_row[f"report_{j}"]  # 각 기록에 대한 리포트 가져오기
                if type(report) == str:
                    is_bad = False
                    for bad_report in bad_reports:
                        if report.find(bad_report) > -1:  # bad report의 기록이 있을 시
                            is_bad = True
                            break
                    report_txt += (report + " ") if not is_bad else ""  # 정상 리포트만 이어붙이기
            if report_txt == "":
                continue
            report_txt = report_txt[:-1].lower()  # 마지막 공백 제거 및 소문자 변환
            # 불필요한 정보 제거
            report_txt = (report_txt.replace("---", "")
                          .replace("***", "")
                          .replace(" - age undetermined", ""))

            # 약어 -> 문장 치환
            report_txt = (report_txt.replace('rbbb', 'right bundle branch block')
                          .replace('lbbb', 'light bundle branch block')
                          .replace('lvh', 'left ventricle hypertrophy')
                          .replace("mi", "myocardial infarction")
                          .replace("lafb", "left anterior fascicular block")
                          .replace("pvc(s)", "ventricular premature complex")
                          .replace("pvcs", "ventricular premature complex")
                          .replace("pac(s)", "atrial premature complex")
                          .replace("pacs", "atrial premature complex"))
            if wfep:  # WDE 정보 포함일 시
                report_txt = report_txt + get_wave_info(row)
            reports.append(report_txt)
            X.append(os.path.join(path, row["path"]))
        return X, reports
    
    train_x, train_y = data(train_idx)
    val_x, val_y = data(val_idx)
    test_x, test_y = data(test_idx)
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def collate_fn(batch):
        ecgs, tokens, raw_texts = zip(*batch)
        ecgs = torch.stack(ecgs, dim=0)
        tokens = torch.stack(tokens, dim=0)
        return ecgs, tokens, list(raw_texts)


def make_dataloader(args, dataset, is_train, drop_last=None):
    num_samples = len(dataset)
    shuffle = is_train
    drop_last = is_train if drop_last is None else drop_last
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader)


def get_all_ecg_text_dataset(args, preprocess_train, preprocess_test, tokenizer=None):
    datasets = {}
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mimic_iv_ecg(args.mimic_iv_ecg_path, wfep=args.wfep)
    train_dataset = ECGTextDataset(X_train, Y_train, transforms=preprocess_train, tokenizer=tokenizer, is_train=True)
    datasets['train'] = make_dataloader(args, train_dataset, is_train=True)
    
    val_dataset = ECGTextDataset(X_val, Y_val, transforms=preprocess_test, tokenizer=tokenizer, is_train=False)
    datasets['val'] = make_dataloader(args, val_dataset, is_train=False)

    test_dataset = ECGTextDataset(X_test, Y_test, transforms=preprocess_test, tokenizer=tokenizer, is_train=False)
    datasets['test'] = make_dataloader(args, test_dataset, is_train=False)
    
    return datasets


def get_data(args, preprocess_fns, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns

    data = get_all_ecg_text_dataset(args, preprocess_train, preprocess_val, tokenizer=tokenizer)

    return data