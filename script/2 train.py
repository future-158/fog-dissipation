#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import warnings
from collections import namedtuple
from datetime import date, datetime
from functools import partial
from itertools import product
from pathlib import Path
import joblib
# import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import os
from numpy.lib.stride_tricks import as_strided
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, make_scorer, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split, GroupShuffleSplit)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import Bunch, resample, shuffle, validation
from utils import calc_metrics
import catboost




def get_data(pred_hour: int = 0):
    pred_hour = 3
    features = joblib.load('scratch/extracted_features_{}'.format(pred_hour))
    y = joblib.load('scratch/data')['y'][pred_hour]
    test_mask = y.index >= pd.to_datetime('2020-05-01')

    return (
        features[~test_mask], y[~test_mask],
        features[test_mask], y[test_mask],
    )



def get_model():
    return catboost.CatBoostClassifier(
                    n_estimators=1000,
                    eval_metric='F1',
                    # task_type='GPU',
                    early_stopping_rounds=50,
                    learning_rate=1e-1,
                    verbose=0)


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


def get_objective(config):
    def objective(trial):
        input_path = config['input_path']
        pred_hour = config['pred_hour']
        x_train, y_train, *_ = get_data(pred_hour)
        # scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 10)
        scale_pos_weight = trial.suggest_float('scale_pos_weight', 0.5, 2)
        params = {}
        params['depth'] = trial.suggest_int('depth', 5, 8) # default 6
        params['l2_leaf_reg'] = trial.suggest_int('l2_leaf_reg', 1, 10)  # default 3

        cross_val_scores = []
        iterations = []
        for x_train, y_train, x_val, y_val in gss(train_x, train_y):
            sample_weight = np.where(y_train==1, scale_pos_weight, 1)
            model = get_model()
            model.set_params(**params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val,y_val)],
                sample_weight=sample_weight)
            best_iteration_ = model.best_iteration_
            # add using howe
            pred = model.predict(x_val).round()
            score = accuracy_score(y_val,pred)
            cross_val_scores.append(score)
            iterations.append(best_iteration_)

        cross_val_score = np.mean(cross_val_scores)
        best_iteration_ = int(np.mean(iterations))

            # epoch = model.model.get_booster().best_iteration
            # last_write_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # trial.set_user_attr(key="onnx", value=onx_path)
        trial.set_user_attr(key="cross_val_score", value=cross_val_score)
        trial.set_user_attr(key='params', value=model.get_params())
        trial.set_user_attr(key='scale_pos_weight', value=scale_pos_weight)
        trial.set_user_attr(key='best_iteration_', value=best_iteration_)
        return cross_val_score
    return objective

def run_config(**config):
    # out_dir = Path('../output') / config['jobid']
    # out_dir.mkdir(parents=True, exist_ok=True)
    # dest = out_dir /  str(config['pred_hour'])
    study = optuna.create_study(direction='maximize')
    #optuna.logging.get_verbosity
    #optuna.logging.set_verbosity
    study.optimize(
        get_objective(config), 
        # n_trials=config['n_trials'], 
        timeout=config['time_budget'], 
        gc_after_trial=True,
        # callbacks=[
        #     ],
        )

    # best_params = study.best_params
    # best_value = study.best_value
    user_attrs = study.best_trial.user_attrs
    input_path = config['input_path']
    pred_hour = config['pred_hour']

    x_train, y_train, x_test, y_test = get_data(input_path, pred_hour)
    scale_pos_weight = user_attrs['scale_pos_weight']
    best_iteration_ = user_attrs['best_iteration_']
    params = user_attrs['params']
    
    model = get_model()
    model.set_params(**params)
    model.set_params(n_estimators=best_iteration_)
    sample_weight = np.where(y_train==1, scale_pos_weight, 1)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    pred = model.predict(x_test) # cls return 0 | 1 supposely
    metric = calc_metrics(y_test, pred)

    payload = {
        'metric':metric,
    }

    template['jobid'] = '20210524-i-dissipation-001'
    root = Path('../output')

    out_dir = root / template['jobid']    
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f'{pred_hour}'
    joblib.dump(payload,dest)
    return metric

# study_file point to path or blob
# metrics tp, tn, ... or 
# appendonly
# index, obs, pred (spec num_val x 2 dataframe -> json)



if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    template = {}
    template['time_budget'] = 60*10
    template['input_path'] = '../data/i.csv'


    for pred_hour in [1,2,3]:
        config = {
            **template,
            'pred_hour': pred_hour}
        print('run config: ', config)
        _ = run_config( **config)