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


def load_dataset(root: Path, station:str = ''):
    root = root / 'data' / 'clean'
    vaisala_stations = ['인천항','평택당진항','부산항','부산항신항']

    if station in vaisala_stations:
        start_dt = '2017-01-01'
        target_name = '시정(20km)'
    else:
        start_dt = '2018-01-01'
        target_name = '시정(3km)'
    
    input_path = Path(root) / f'{station}.csv'
    dtypes = {
        # '시간': datetime,
        '항': str,
        # '항': pd.StringDtype, # failed
        '풍향(수치)': float,
        '풍속(m/s)': float,
        '기온(℃)': float,
        '수온(℃)': float,
        '기압(hPa)': float,
        '습도(%)': float,
        '시정(3km)': float,
        '시정(20km)': float}
    

    renamer = {
        '시간': 'datetime',
        '기온(℃)': 'air_temp',
        '수온(℃)': 'sea_temp',
        '기압(hPa)': 'sea_air_pre',
        '습도(%)': 'humidity',
        target_name: 'vis',
        '풍향(수치)': 'win_dir',
        '풍속(m/s)': 'win_vel',
        '항':'station',
        }

    usecols = ['win_dir', 'win_vel', 'air_temp', 'sea_temp', 'sea_air_pre',
       'humidity', 'vis']
    
    numeric_cols = ['air_temp', 'sea_air_pre', 'win_vel', 'win_dir', 'humidity', 'sea_temp', 'vis']

    df = pd.read_csv(
        input_path, 
        parse_dates=['시간'], infer_datetime_format=True, index_col=['시간'],
        dtype=dtypes)

    df = df.rename(columns=renamer)
    df = df.loc[start_dt:, usecols]


    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # df.vis = df.vis.clip(0,3000)
    
    wd = df['win_dir']
    wd_rad = np.deg2rad(wd)
    ws = df['win_vel']
    df['u'] = ws * np.sin(wd_rad)
    df['v'] = ws * np.cos(wd_rad)

    usecols = numeric_cols = ['air_temp', 'sea_air_pre',
        'u', 'v', 'humidity', 'sea_temp', 'vis'] # win_dir, win_vel -> u, v
    # df[numeric_cols] = df[numeric_cols].interpolate(method='nearest').fillna(method='ffill').fillna(method='bfill')    
    df = df[usecols]
    aggregator = {
        'air_temp': 'mean',
        'sea_air_pre': 'mean',
        'u': 'mean',
        'v': 'mean',
        'humidity': 'mean',
        'sea_temp': 'mean',
        'vis': 'mean'}

    df = df.resample('10T', label='right', closed='right').aggregate(aggregator)

    df.vis = df.vis.interpolate('linear', limit=6)
    df.vis = df.vis.fillna(df.vis.mean())
    covariates = ['air_temp', 'sea_air_pre','u', 'v', 'humidity', 'sea_temp']
    df[covariates] = df[covariates].interpolate(how='linear', limit=18)


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

    lagcols = [
        'air_temp', 'sea_air_pre', 'u', 'v', 'humidity', 'sea_temp', 
        'ASTD', 'dew_T', 'T_DT', 'sst-Td','vis']
            
    nolagcols=['hour', 'month']
    # curr_fog = df.vis.rolling(6).median().le(1000)
    curr_fog = np.where(df.vis<=1000, 1, 0)
    curr_fog = binary_opening(curr_fog, np.ones((7,))) # 10% reduction
    curr_fog = pd.Series(curr_fog, index=df.vis.index).astype(int)
    curr_fog[:] = curr_fog.rolling(6, min_periods=1).sum().ge(1)

    label_1 = df.vis.shift(-6*1).rolling(6*1).mean().gt(1000).astype(np.int64).to_frame(name='label_1')
    label_2 = df.vis.shift(-6*2).rolling(6*2).mean().gt(1000).astype(np.int64).to_frame(name='label_2')
    label_3 = df.vis.shift(-6*3).rolling(6*3).mean().gt(1000).astype(np.int64).to_frame(name='label_3')

    new_label_1 = df.vis.shift(-6*1).rolling(6*1).quantile(0.1).gt(1000).astype(np.int64).to_frame(name='new_label_1')
    new_label_2 = df.vis.shift(-6*2).rolling(6*2).quantile(0.1).gt(1000).astype(np.int64).to_frame(name='new_label_2')
    new_label_3 = df.vis.shift(-6*3).rolling(6*3).quantile(0.1).gt(1000).astype(np.int64).to_frame(name='new_label_3')


    hj_label_1 = df.vis.le(1000).shift(-6*2).rolling(6).sum().eq(0).astype(np.int64).to_frame(name='hj_label_1')
    hj_label_2 = df.vis.le(1000).shift(-6*3).rolling(6).sum().eq(0).astype(np.int64).to_frame(name='hj_label_2')
    hj_label_3 = df.vis.le(1000).shift(-6*4).rolling(6).sum().eq(0).astype(np.int64).to_frame(name='hj_label_3')


    lagged = pd.concat(
        [df.shift(lagT)[lagcols].add_suffix(f'_lag{lagT}') for lagT in range(0,120,10)],
        axis=1
    )

    X = lagged.join(df[nolagcols])
    X = X[curr_fog]

    y = pd.concat([
        label_1, label_2, label_3,
        new_label_1,new_label_2,new_label_3,
        hj_label_1, hj_label_2,hj_label_3
    ], axis=1)[curr_fog]

    fog = curr_fog[curr_fog==1]
    pd.testing.assert_index_equal(y.index, fog.index)
    g = (fog.index.to_series().diff() != pd.Timedelta('10T')).cumsum().to_frame(name='g').reset_index()
    first = g.groupby('g')['시간'].transform('first')
    last = g.groupby('g')['시간'].transform('last')    
    ttd = last - g.시간
    y['ttd_1H'] = (ttd < pd.Timedelta(hours=1)).values
    y['ttd_2H'] = (ttd < pd.Timedelta(hours=2)).values
    y['ttd_3H'] = (ttd < pd.Timedelta(hours=3)).values
    groups = g.g.values
    return X,y, groups

if __name__=='__main__':
    cfg = OmegaConf.load('config.yaml')
    stations = [
        # '부산항',
        '부산항신항',
        '인천항',
        '평택당진항',
        '군산항',
        '대산항',
        '목포항',
        '여수광양항',
        '해운대',        
        # '울산항',
        # '포항항',
    ]

    rows = []

    
    for station in stations:
        X, y, groups =  load_dataset(station = station)
        row = y.sum()
        row['count'] = y.shape[0]
        row['station'] = station
        rows.append(row)

    stats = pd.DataFrame(rows)
    id_vars = ['station']
    value_vars = ['count', 'hj_label_1', 'hj_label_2', 'hj_label_3']
    names = ['COUNT', '소산 1H', '소산 2H', '소산 3H']

    stats = stats.set_index(id_vars)[value_vars]
    stats.columns = names
    dest = cfg.stat_dest

    stats = stats.reset_index()
    table =stats
    tag = 'stat'
    version = '4'
    sheet_name = f'{tag}_{version}'
    with pd.ExcelWriter(dest, mode='a') as writer:
        table.to_excel(writer, sheet_name=sheet_name, index=False)
    





