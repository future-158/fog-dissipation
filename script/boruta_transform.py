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

source = 'scratch/table'
table = joblib.load(source)

def _do_tests(self, dec_reg, hit_reg, _iter):
    active_features = np.where(dec_reg >= 0)[0]
    hits = hit_reg[active_features]
    # get uncorrected p values based on hit_reg
    to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
    to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

table = table.T
shadow_max = table.filter(like='shadow').max(axis=1).values[:, None]
shadow_mask = table.columns.str.contains('shadow')
heats = (table.loc[:, ~shadow_mask] > shadow_max).sum()

usecols = heats[heats.gt(55)].index
x, y = joblib.load('scratch/data').values()
# recalc? to low.

















