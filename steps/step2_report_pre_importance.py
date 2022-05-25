import itertools
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import product
from os import nice
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from numpy.lib.stride_tricks import as_strided
from omegaconf import MISSING  # Do not confuse with dataclass.MISSING
from omegaconf import DictConfig, OmegaConf
from pandas.core.indexes.datetimes import date_range
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           binary_opening)
from sklearn.feature_selection import mutual_info_classif, r_regression
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, fbeta_score,
                             make_scorer, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

from dataset import load_dataset

cfg = OmegaConf.load('conf/config.yaml')

stations = [
    '부산항신항',
    '인천항',
    '평택당진항',
    '군산항',
    '대산항',
    '목포항',
    '여수광양항',
    '해운대',        
]

rows = []
for station in stations:        
    X, y, groups =  load_dataset(
        root = Path.cwd(),
        station = station
        )
    latest = X.filter(like='lag0')
    latest.columns
    latest.columns = latest.columns.astype(str).str.replace('_lag0','')
    latest = latest.assign(ws = lambda x: np.sqrt(x.eval('u**2 + v**2')))

    value_vars = ['air_temp', 'sea_air_pre', 'u', 'v', 'humidity', 'sea_temp', 'ASTD',
        'dew_T', 'T_DT', 'sst-Td', 'vis', 'ws']


    renamer= {
        'air_temp':'temp',
        'sea_air_pre':'qff',
        'humidity': 'rh',            
        'sea_temp':'sst',
        'dew_T':'Td',
        'T_DT':'temp-Td',
    }

    features = latest[value_vars].rename(columns=renamer)


    pred_hours = [1,2,3]
    for pred_hour, col, importance_type in product(pred_hours, features.columns, ['mic','r_reg']):
        pass
        label = y[f"hj_label_{pred_hour}"]
        xy = pd.DataFrame(data=dict(X=features[col], y=label)).dropna()
        mi = mutual_info_classif
        value = {
            'mic': mutual_info_classif,
            'r_reg': r_regression,
        }[importance_type](xy[['X']], xy.y)[0]

        row = {
            'station_name':station,
            'pred_hour': pred_hour,
            'importance_type':importance_type,
            'variable': col,
            'value':value
        }

        rows.append(row)

table = pd.DataFrame(rows)
id_vars = ['station_name', 'pred_hour', 'importance_type', 'variable']
table = table.set_index(id_vars).unstack().droplevel(0,axis=1).reset_index()

dest = Path(cfg.importance_dest)
dest.parent.mkdir(parents=True, exist_ok=True)
table.to_excel(dest, index=False)










    





