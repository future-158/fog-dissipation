import itertools
import logging
import math
import os
import sys
import time
from functools import partial
from os import nice
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import date_range
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, make_scorer, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
# logging.Logger(name='eda', level=logging.DEBUG)
import sys;print(sys.executable)

def get_data(input_path = '', pred_hour=1):
    valid_cols = ['DB', 'YR', 'MN', 'DY', 'HH', 'MM', 'air_temp', 'sea_air_pre', 'win_vel', 'win_dir', 'humidity', 'sea_temp', 'vis']
    renamer = {
        '시간': 'datetime',
        '기온(℃)': 'air_temp',
        '수온(℃)': 'sea_temp',
        '기압(hPa)': 'sea_air_pre',
        '습도(%)': 'humidity',
        '시정(20km)': 'vis',
        '풍향(수치)': 'win_dir',
        '풍속(m/s)': 'win_vel'}

    numeric_cols = ['air_temp', 'sea_air_pre', 'win_vel', 'win_dir', 'humidity', 'sea_temp', 'vis']
    # input_path = input_path
    df = pd.read_csv(input_path, na_values=[-99999, '-'], parse_dates=['시간'])
    df = df.rename(columns=renamer)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.vis = df.vis.clip(0,3000)

    wd = df['win_dir']
    wd_rad = np.deg2rad(wd)
    ws = df['win_vel']
    df['u'] = ws * np.sin(wd_rad)
    df['v'] = ws * np.cos(wd_rad)

    usecols = numeric_cols = ['air_temp', 'sea_air_pre',
        'u', 'v', 'humidity', 'sea_temp', 'vis'] # win_dir, win_vel -> u, v
    df[numeric_cols] = df[numeric_cols].interpolate(method='nearest').fillna(method='ffill').fillna(method='bfill')    
    df = df.set_index('datetime')
    df = df[usecols]

    aggregator = {
        'air_temp': 'mean',
        'sea_air_pre': 'mean',
        'u': 'mean',
        'v': 'mean',
        'humidity': 'mean',
        'sea_temp': 'mean',
        'vis': 'min'}

    df = df.resample('10T').agg(aggregator)

    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['T'] = df.air_temp + 273.15
    df['ASTD'] = df.air_temp - df.sea_temp
    df['dew_T']= df.air_temp-((100-df.humidity)/5)*np.sqrt(df['T']/300)-0.00135*(df.humidity-84)**2+0.35
    df['T_DT'] = df.air_temp - df.dew_T
    df['sst-Td'] =  df.sea_temp - df['dew_T']

    #lag operation

    assert df.index.is_monotonic
    assert df.index.to_series().diff().value_counts().nunique() == 1
    assert df.index.to_series().diff().value_counts().idxmax() == pd.Timedelta(minutes=10)

    usecols = ['humidity', 'T_DT', 'vis', 'ASTD', 'sst-Td', 'u', 'v']
    lagged_cols = []
    lag_num = 12

    # df.loc[seafog_mask, 'vis'] = df.loc[seafog_mask, 'vis'].clip(0, 1100) -> rolling min.
    for i in range(lag_num+1):
        lagged = df.loc[:,usecols].shift(i).add_prefix(f'{i:02}_')
        lagged_cols.append(lagged)

    x = pd.concat(lagged_cols,
                    axis=1)
    x = x.assign(
        hour=df.hour,
        month=df.month,
        press=df.sea_air_pre)

    vis = df.vis
    dissipated = vis.shift(-60).rolling(60, closed='right').min().gt(1100)    
    dissipated = dissipated.shift(-60*pred_hour)        
    na_mask = np.logical_or(x.isna().any(axis=1), dissipated.isna())

    current_fog_mask = vis.le(1000)
    valid_mask = np.logical_and(current_fog_mask, ~na_mask)
    x = x[valid_mask]
    y = dissipated[valid_mask].astype(int)

    test_mask = y.index >= pd.to_datetime('2020-05-01')

    x_test = x[test_mask]
    y_test = y[test_mask]
    x_train = x[~test_mask]
    y_train = y[~test_mask]
    return x_train, y_train, x_test, y_test

    # train, val, test split

def calc_metrics(obs, pred):
    usecols = ['ACC', 'CSI', 'PAG', 'POD', 'FAR', 'f1_score', 'POFD', 'POCD', 'tn', 'fp', 'fn', 'tp']
    obs = obs.round()
    pred = pred.round()
    tn, fp, fn, tp = confusion_matrix(obs, pred).flatten()
    POFD=fp / (tn + fp) * 100
    performance = dict(
                    ACC=accuracy_score(obs, pred) * 100,
                    CSI=tp / (tp + fn + fp) * 100,
                    PAG=100 - fp / (tp + fp) * 100,
                    POD=recall_score(obs, pred) * 100,
                    FAR=fp / (tp + fp) * 100,
                    f1_score=fbeta_score(obs, pred, beta=1) * 100,
                    POFD=POFD,
                    POCD=100-POFD,
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    tp=tp,
                    )
    return dict(zip(usecols, map(performance.get, usecols)))


if __name__=='__main__':
    input_path = '../data/i.csv'
    for pred_hour in [1,2,3]:
        x_train, y_train, x_test, y_test=  get_data(input_path, pred_hour)




    

