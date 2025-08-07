"""
Hyperkalemia 데이터셋으로부터 Feature 가져오기 + DF 형태로 재구성
기존의 5개의 데이터셋으로부터의 signal preprocessing 단계에 대응됨 (기존에는 Wave Feature + 진단 결과)

input: pkl
output: DF (Wave Feature + Hyperkalemia Label)
"""


import numpy as np
import pandas as pd
import pickle

import argparse
import os

from tqdm import tqdm


def get_analysis_from_feature(path, sampling_rate):
    """
    pkl 파일 내의 추출되어 있는 feature를 평균 처리 후 return
    """
    with open(path, 'rb') as f:
        ecg_dict = pickle.load(f)

    label = ecg_dict['original']['prob'][0]
    rr = ecg_dict['original']['feature']['II.r.intvs']  # lead.r.intvs
    pr = ecg_dict['original']['feature']['II.pr.intvs']  # lead.pr.intvs / summary.pr_intv
    qrs = ecg_dict['original']['feature']['II.qrs.durs']  # lead.qrs.durs / summary.qrs_dur
    qt = ecg_dict['original']['feature']['summary.qt_intv']  # only summary -> (t.offsets - r.onsets)
    qtc = ecg_dict['original']['feature']['summary.qt_corrected']  # only summary -> qt/np.sqrt(rr)
    pp = ecg_dict['original']['feature']['II.p.peaks']  # lead.p.peaks
    rp = ecg_dict['original']['feature']['II.r.peaks']  # lead.r.peaks
    tp = ecg_dict['original']['feature']['II.t.peaks']  # lead.t.peaks
        
    return label, int(dropna(rr)), int(dropna(pr)), int(dropna(qrs)), int(dropna(qt)), \
            int(dropna(qtc)), int(dropna(pp)), int(dropna(rp)), int(dropna(tp))


# FIXME: 특정 함수 내에서만 호출되는 함수 -> 내부 함수로 넣는 게 적합한가?
def dropna(x):
    """
    NaN 제외 후 mean 계산 / only NaN일 시 0으로 계산 진행 후 return
    """
    if not isinstance(x, (list, tuple, np.ndarray)):
        x = [x]

    x_cleaned = [np.nan if (v == "None" or v is None) else v for v in x]
    x_array = np.asarray(x_cleaned, dtype=float)
    valid = x_array[~np.isnan(x_array)]
    
    if len(valid) > 0:
        return np.mean(valid)
    else:
        return 0


def calculate_waveforms(data_dir, path_list, sampling_rate=500):
    """
    모든 입력에 대한 feature를 가져와 DF 변환에 적합하도록 딕셔너리 형태로 변환
    """
    data_dict = {
        'ECG_ID': [],  # ECG 해시 값
        'Label': [],  # Hyperkalemia 라벨 (확률)
        'RR_Interval': [],
        'PR_Interval': [],
        'QRS_Complex': [],
        'QT_Interval': [],
        'QTc_Interval': [],
        'P_Wave_Peak': [],
        'R_Wave_Peak': [],
        'T_Wave_Peak': [],
    }

    for i in tqdm(range(len(path_list))):
        path = path_list[i]
        label, rr, pr, qrs, qt, qtc, pp, rp, tp = get_analysis_from_feature(os.path.join(data_dir, path), sampling_rate)
        
        data_dict['ECG_ID'].append(os.path.splitext(path)[0])
        data_dict['Label'].append(label)
        data_dict['RR_Interval'].append(rr)
        data_dict['PR_Interval'].append(pr)
        data_dict['QRS_Complex'].append(qrs)
        data_dict['QT_Interval'].append(qt)
        data_dict['QTc_Interval'].append(qtc)
        data_dict['P_Wave_Peak'].append(pp)
        data_dict['R_Wave_Peak'].append(rp)
        data_dict['T_Wave_Peak'].append(tp)
        
    return data_dict


def preprocess(args):
    """
    모든 ECG 입력에 대한 텍스트를 DF 형태로 저장
    """
    DATA_DIR = args.data_dir
    SAVE_DIR = args.save_dir
    
    path_list = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]

    data_dict = calculate_waveforms(data_dir=DATA_DIR, path_list=path_list, sampling_rate=250)
    df = pd.DataFrame(data_dict)
    df["strat_fold"] = np.arange(len(df)) % 5  # 10-fold
    df.to_csv(os.path.join(SAVE_DIR, "diagnostics_feature.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/data/medicalai/samples/hyperkalemia")
    parser.add_argument("--save-dir", type=str, default=".")
    args = parser.parse_args()
    
    preprocess(args)