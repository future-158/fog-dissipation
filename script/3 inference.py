from sys import call_tracing
import pandas as pd
from pathlib import Path
from collections import defaultdict
from numpy.lib.stride_tricks import as_strided
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, make_scorer, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import Bunch, resample, shuffle, validation
import joblib
from itertools import zip_longest
import json
import argparse
from pathlib import Path


def interpret_cmat(tn, fp, fn, tp):
    l2cols = ['ACC', 'CSI', 'PAG', 'POD', 'FAR', 'POFD', 'POCD']
    metrics = dict(
                    ACC= (tp + tn) / (tn+fp+fn+tp),
                    CSI= tp / (tp + fn + fp),
                    PAG= tp / (tp + fp),
                    POD= tp / (tp + fn),
                    FAR= fp / (tp + fp),
                    POFD= fp / (tn + fp), 
                    POCD = tn / (tn + fp), 
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    tp=tp,
                    )

    return {k:v *100 if k in l2cols else v for k, v in metrics.items()}



jobid = '20210524-i-dissipation-001'
source_dir = Path('../output') / jobid

rows = []
for file in source_dir.glob('[0-9]*'):
    pred_hour = file.name
    pred_hour = int(pred_hour)
    metric = joblib.load(file)['metric']
    row = {**metric, 'pred_hour':pred_hour}
    rows.append(row)

table = pd.DataFrame(rows)

table.to_excel('../product/2021-05-25 dissipation.xlsx')


