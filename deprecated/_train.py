#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import warnings
from collections import namedtuple
from datetime import date, datetime
from functools import partial
from itertools import product
from pathlib import Path

import catboost
import joblib
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tqdm
from numpy.core.numeric import cross
from numpy.lib.stride_tricks import as_strided
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, make_scorer, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GroupShuffleSplit, KFold,
                                     RandomizedSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import Bunch, resample, shuffle, validation

from utils import calc_metrics, get_data
from dataset import load_dataset



def get_train_test(
    station='인천항',
    pred_hour:int=3):

    data = get_data(station)
    data['label'] = data['ttd'].le(pred_hour * 3600)

    dropcols =  ['cls_vis',           'g',
          'datetime',      'gstart',        'gend',         'ttd',
                 '항']

    groups = data['g']
    data = data.drop(columns=dropcols)
    label = data.pop('label').astype(np.int64)

    
    gss = StratifiedGroupKFold(n_splits=5)
    cvp = cross_val_predict(
        RandomizedSearchCV(*get_model_grid(model_name='cb'), cv=gss, n_iter=1, scoring='f1'),
        # get_model_grid('cb')[0],
        data, 
        label, 
        cv=gss,
        groups=groups,
    )




    parts = []
    for train_idx, test_idx in gss.split(data, groups=data['g']):        
        model = RandomizedSearchCV(*get_model_grid(model_name='cb'), cv=gss, n_iter=1, scoring='f1', refit=True)
        model.fit(data.iloc[train_idx], label.iloc[train_idx])
        pred = model.predict(data.iloc[test_idx])
        
        obs_pred_part = label.iloc[test_idx].to_frame(name='obs').assign(pred=pred)
        parts.append(obs_pred_part)
    obs_pred = pd.concat(parts)


def get_model_grid(model_name:str = ''):
    if model_name == 'cb':
        model =  catboost.CatBoostClassifier(
                            n_estimators=1000,
                            eval_metric='F1',
                            task_type='GPU',
                            devices='7',
                            early_stopping_rounds=50,
                            learning_rate=1e-1,
                            verbose=0)
        param_grid = {'n_estimators': np.arange(100,200)}
        return model, param_grid


                
def gss(x, y):
    full_index = pd.date_range(y.index.min(), y.index.max(), freq='10T')
    g = mask = pd.Series(index=full_index, dtype=float)
    mask = mask.fillna(0)
    mask[y.index] = 1
    mask = mask.fillna(0)
    g = (mask==0).cumsum()
    g = g[mask==1]
    gss = GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)
    for train_idx, val_idx in gss.split(x, y, g):
        yield x.iloc[train_idx], y.iloc[train_idx], x.iloc[val_idx], y.iloc[val_idx]

    
def calc_metrics(obs, pred):
    confusion_matrix(np.array(y_test, dtype=int), pred)
    usecols = ['ACC', 'PAG', 'POD', 'CSI', 'F1', 'POFD', 'tn', 'fp', 'fn', 'tp']
    tn, fp, fn, tp = confusion_matrix(np.array(obs, dtype=int), pred).flatten()
    POFD= fp / (tn + fp) * 100
    performance = dict(
                    ACC=accuracy_score(obs, pred) * 100,
                    CSI= tp / (tp + fn + fp) * 100,
                    PAG= tp / (tp + fp) * 100,
                    POD= recall_score(obs, pred) * 100,
                    FAR=fp / (tp + fp) * 100,
                    F1=fbeta_score(obs, pred, beta=1) * 100,
                    POFD=POFD,
                    POCD=100-POFD,
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    tp=tp,
                    )
    return dict(zip(usecols, map(performance.get, usecols)))


        # scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 10)
        # params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.5, 2)
        # params['depth'] = trial.suggest_int('depth', 5, 8) # default 6
        # params['l2_leaf_reg'] = trial.suggest_int('l2_leaf_reg', 1, 10)  # default 3

# ports= ['인천항', '대산항', '군산항', '평택당진항']

if __name__ == '__main__':
    stations = ['평택당진항',
    '목포항',
    '대산항',
    # '울산항',
    '해운대',
    # '여수광양항', # 3시간 이내 100%
    '부산항',
    '포항항', # 3시간 이내 100%
    '인천항',
    '군산항',
    # '부산항신항'
    ]
    pred_hours = [1,2,3]

    rows = []
    for station, pred_hour in tqdm.tqdm(product(stations, pred_hours)):
        data = get_data(station)
        data['label'] = data['ttd'].le(pred_hour * 3600)

        dropcols =  ['cls_vis',           'g',
            'datetime',      'gstart',        'gend',         'ttd',
                    '항']

        groups = data['g']
        data = data.drop(columns=dropcols)
        label = data.pop('label').astype(np.int64)
        print(station)
        print(label.value_counts())



    
        x_train, x_test, y_train, y_test = get_train_test(station=station,pred_hour=pred_hour)
        assert len(y_train.index.intersection(y_test.index) ) == 0

        model = get_model()
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, shuffle=False)

    # groupkfold split 
    # latest 20% train/test
    # train / test
    # train -> train/tesz

        model.fit(
            x_train, y_train,
            eval_set=[(x_val,y_val)],
            verbose=2
        )

        pred = model.predict(x_test)
        metric = calc_metrics(np.array(y_test, dtype=float), pred)
        metric['station'] = station
        metric['pred_hour'] = pred_hour
        print(metric)
        rows.append(metric)
        del model

    # 라벨링 바꾼 경우 
    table = pd.DataFrame(rows).set_index(['station',	'pred_hour']).sort_index()
