from imp import source_from_cache
from sre_constants import NOT_LITERAL_IGNORE
import sys
from this import d;print(sys.executable)
from catboost.core import train;print(sys.executable)
from numpy.random import sample
from pathlib import Path
import os
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from typing import List
import numpy as np
from sklearn.utils import resample
import joblib
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import delayed, Parallel
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import catboost
import argparse
import scipy as sp
from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm, Pool, EFstrType

# from BorutaShap import BorutaShap, load_data

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

pred_hour = 3
def get_data(pred_hour: int = 0):
    features = joblib.load('scratch/extracted_features_{}'.format(pred_hour))
    y = joblib.load('scratch/data')['y'][pred_hour]
    y = y.astype(float)
    test_mask = y.index >= pd.to_datetime('2020-05-01')
    features = features.loc[:,features.isna().sum().le(100)]

    return (
        features[~test_mask], y[~test_mask],
        features[test_mask], y[test_mask],
    )

def get_train_val(x, y, n_splits=2):
    full_index = pd.date_range(y.index.min(), y.index.max(), freq='10T')
    g = mask = pd.Series(index=full_index, dtype=float)
    mask = mask.fillna(0)
    mask[y.index] = 1
    mask = mask.fillna(0)
    g = (mask==0).cumsum()
    g = g[mask==1]
    gss = GroupShuffleSplit(n_splits=n_splits, train_size=.8, random_state=42)
    for train_idx, val_idx in gss.split(x, y, g):
        yield x.iloc[train_idx], y.iloc[train_idx], x.iloc[val_idx], y.iloc[val_idx]

n_iterations = 20
n_splits = 5

def _do_tests(self, dec_reg, hit_reg, _iter):
    active_features = np.where(dec_reg >= 0)[0]
    hits = hit_reg[active_features]
    # get uncorrected p values based on hit_reg
    to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
    to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

def get_model():
    return CatBoostClassifier(
        iterations=1000,
        max_depth=7,
        learning_rate=0.1,
        random_seed=0, 
        # eval_metric='F1',
        eval_metric='Accuracy',
        task_type='GPU',
        # devices='1',
        early_stopping_rounds=30
        )

x, y, *_ = get_data(pred_hour)

table = pd.DataFrame(
    index=[*x.columns, *['shadow_' + x for x in x.columns]],
    columns = np.arange(n_iterations * n_splits),
    dtype=int)

for num_iteration in range(n_iterations):
    x, y, *_ = get_data(pred_hour)
    shadow = x.add_prefix('shadow_')
    shadow.loc[:,:] = shadow.sample(frac=1, replace=False).values
    x_cat = pd.concat([x,shadow], axis=1)
    cols = x_cat.columns
    shadow_start = len(cols)//2    

    for i, (x_train, y_train, x_val, y_val) in enumerate(get_train_val(x_cat,y,n_splits)):
        model = get_model()
        col = n_splits * num_iteration  + i
        model.fit(x_train, y_train, eval_set=[(x_val,y_val)])
        fi = model.feature_importances_
        table[col] = fi

joblib.dump(table, 'scratch/table')
# heats += np.where(fi[:shadow_start]>shadow_max, 1, 0)
# np.where(fi[:shadow_start]>shadow_max, 1, 0)

# np.where(fi[])






