from imp import source_from_cache
import sys;print(sys.executable)
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
from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm, Pool

# from BorutaShap import BorutaShap, load_data


def get_data(pred_hour: int = 0):
    pred_hour = 3
    features = joblib.load('scratch/extracted_features_{}'.format(pred_hour))
    y = joblib.load('scratch/data')['y'][pred_hour]
    test_mask = y.index >= pd.to_datetime('2020-05-01')

    return (
        features[~test_mask], y[~test_mask],
        features[test_mask], y[test_mask],
    )

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


def get_model():
    return CatBoostClassifier(
        iterations=100,
        max_depth=5,
        learning_rate=0.1,
        random_seed=0, 
        eval_metric='F1',
        task_type='GPU',
        devices='0',
        )

def select_features(x, y) -> List[str]:    
    x_train, x_val, y_train, y_val = train_test_split(x,y)
    x_train = pd.DataFrame(x_train, index=y_train.index)
    feature_names = ['features_{}'.format(x) for x in x_train.columns]
    x_train.columns = feature_names

    cat_features = [column for column, dtype in x_train.dtypes.items() if dtype==object]
    train_pool = Pool(x_train, y_train, feature_names=feature_names, cat_features=cat_features)
    test_pool = Pool(x_val, y_val, feature_names=feature_names, cat_features=cat_features)
    model = get_model()    

    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues
    algorithm = EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange
    steps=1



    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=list(range(train_pool.num_col())),
        num_features_to_select=50,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
    )

    selected_features = summary['selected_features']
    return selected_features

if __name__ == '__main__':

    # todo 전체데이터중 random index + random feature index
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int)
    source_dir  = Path('/home/jhj/Proj/jax/code/extract/scratch')

    files = [x for x in source_dir.glob('epoch*')]
    shapelets = [joblib.load(file)['shapelet'] for file in files]
    distances = [joblib.load(file)['distance'] for file in files]
    shapelets = np.concatenate(shapelets)
    distances = np.concatenate(distances)
    batch_size = 1000
    epochs = shapelets.shape[0] // batch_size


    y = joblib.load('../pipe/transformed')['y_train']
    for i in range(epochs):
        print(f'epoch {i} / epochs processing')
        shapelet = shapelets[i*batch_size:(i+1)*batch_size]
        distance = distances[i*batch_size:(i+1)*batch_size]
        x = distance.T
        selected_features = select_features(x, y)
        out_dir = Path('/home/jhj/Proj/jax/code/select/pipe')
        dest = out_dir / str(i)
        selected_shapelet = shapelet[selected_features]
        joblib.dump(selected_shapelet, dest)



