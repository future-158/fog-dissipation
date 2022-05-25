import itertools
import logging
import math
import os
import sys
import time
from datetime import datetime
from functools import partial
from os import nice
from pathlib import Path
from omegaconf import OmegaConf

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from pandas.core.indexes.datetimes import date_range
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           binary_opening)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, make_scorer, precision_score,
                             recall_score, roc_auc_score, cohen_kappa_score)
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

stat_dfs = []


for station in stations:
    X, y, groups =  load_dataset(root = Path.cwd(), station = station)
    latest = X.filter(like='lag0')
    latest.columns = latest.columns.astype(str).str.replace('_lag0','')
    latest = latest.assign(ws = lambda x: np.sqrt(x.eval('u**2 + v**2')))

    value_vars = ['air_temp', 'humidity', 'sea_air_pre', 'sea_temp', 'ws', 'vis']
    latest = latest[value_vars]
    stat = latest.describe().loc[['min','mean','max']]
    stat.index.name = 'agg_type'

    stat = stat.reset_index()
    stat['station_name'] = station

    id_vars = ['station_name', 'agg_type']
    melted = pd.melt(stat, id_vars=id_vars, value_vars=value_vars)

    stat_dfs.append(melted)

stat_table = pd.concat(stat_dfs)
dest = Path(cfg.stat_dest)
dest.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(dest, mode='w') as writer:
    stat_table.to_excel(writer, sheet_name='stat_table', index=False)







