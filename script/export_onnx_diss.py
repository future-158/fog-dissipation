"""
CUDA_VISIBLE_DEVICES='2' taskset --cpu-list 80-100 python export_onnx_diss.py -m station=SF_0002,SF_0003,SF_0004,SF_0005,SF_0006,SF_0007,SF_0008,SF_0009 pred_hour=1,2,3
(
    docker run \
    --shm-size 1g --gpus=device=5 \
    --rm -it \
    -v $PWD:/workspace -w /workspace \
    autogluon/autogluon:0.3.1-rapids0.19-cuda10.2-jupyter-ubuntu18.04-py3.7 \
    python export_onnx_diss.py
    )
"""
import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import autogluon.core as ag
import hydra
import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from autogluon.core.utils import try_import_lightgbm
from autogluon.tabular import TabularDataset, TabularPredictor
# from networkx.algorithms.shortest_paths.weighted import johnson
from omegaconf import DictConfig, OmegaConf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     train_test_split)

from dataset import load_dataset
from utils import calc_metrics

os.environ['CUDA_VISIBLE_DEVICES'] ='1'
@hydra.main(config_path=".", config_name="export")
def main(cfg : DictConfig) -> None:
    # cfg = OmegaConf.load('config.yaml')
    time_limit= cfg.time_limit
    station = cfg.station
    station_name = cfg.station_name[station]
    pred_hour = cfg.pred_hour
    target_name = cfg.target_name
    tokp = cfg.topk
    # out_dir = Path('result/{}T_{}H'.format(sample_distance, pred_hour))
    # out_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups = load_dataset(station_name)
    
    X = X.astype(np.float32)
    y = y[target_name]
    data = X.assign(label=y)
    data = data.dropna()

    label = 'label'  # specifies which column do we want to predict
    save_path = 'ag_hpo_models/'  # where to save trained models

    groups = data.index.year*366 + data.index.dayofyear
    sgk = StratifiedGroupKFold(n_splits=5)
    # sgk = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    train_idx, test_idx = list(sgk.split(data,data.label, groups=groups))[0]
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    train_idx, val_idx = list(sgk.split(train_data, train_data[label], groups=groups[train_idx]))[0]
    val_data = train_data.iloc[val_idx]
    train_data = train_data.iloc[train_idx]

    print('train_ratio:\n',train_data['label'].value_counts())
    print('val_ratio:\n',val_data['label'].value_counts())
    print('test_ratio:\n',test_data.label.value_counts())

    import time
    time.sleep(5)

    

    predictor = TabularPredictor(
        label=label, 
        path=save_path,
        eval_metric='accuracy',
        # sample_weight='balance_weight',    
        problem_type='binary'
        # ignored_columns=None,
        )
        # group shuffle split or cross validation not implemented


    hyperparameters = {
    'CAT': {},
    # 'GBM': {},
    # 'XGB': {},
    # 'RF': {}
    }

    # hyperparameters = {
    # 'GBM' : {'n_estimators':np.arange(100,500,50)},
    # 'CAT': {'n_estimators':np.arange(100,500,50)},
    # 'XGB': {'n_estimators':np.arange(100,500,50)},
    # 'RF': {'n_estimators': np.arange(100,500,50)},
    # }

    results = predictor.fit(
        train_data,
        val_data,
        ag_args_fit={'num_gpus': 1, 'num_cpus':40},
        # hyperparameter_tune=True,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs='auto',
        time_limit=time_limit,
    )

    results = predictor.fit_summary()  # display detailed summary of fit() process
    # predictor.class_labels_internal_map
    # predictor.delete_models(models_to_keep='best') # ensemble model contains best models
    topk_models = predictor.get_model_names()[:tokp]
    [ensemble_model] = predictor.fit_weighted_ensemble(topk_models)
    predictor.delete_models(models_to_keep=[ensemble_model])
    
    distilled_model_names = predictor.distill(time_limit=time_limit, hyperparameters={'RF': {}}, teacher_preds='soft', augment_method='spunge', augment_args={'size_factor': 1}, verbosity=3, models_name_suffix='spunge')
    # distilled_model_names = predictor.distill(
    #     train_data = train_data.dropna(),
    #     tuning_data = val_data.dropna(),
    #     # augmentation_data=test_data.drop(columns=['label']).dropna(),
    #     # teacher_preds='hard',
    #     time_limit=cfg.distill_time_limit,
    #     # augment_method=None if cfg.distill_time_limit < 600 else 'spunge',
    #     # augment_method = None,
    #     hyperparameters={'RF': {
    #         'max_depth':10,
    #         'max_features': 17,
    #         'n_estimators': 50
    #         }}
    #         )

    # distilled_model_names = predictor.distill(
    #     train_data = pd.concat([train_data, val_data]),
    #     tuning_data = test_data,
    #     time_limit=cfg.distill_time_limit,
    #     # augment_method=None if cfg.distill_time_limit < 600 else 'spunge',
    #     augment_method = None,
    #     # hyperparameters={'RF': {
    #     #     'max_depth':10,
    #     #     'max_features': 17,
    #     #     'n_estimators': 50
    #     #     }}
    #     #     )
    #     hyperparameters={'RF':{}})
    

    model_to_deploy = distilled_model_names[0]
    predictor.delete_models(models_to_keep=model_to_deploy, dry_run=False)
    
    results['test_leaderboard'] = predictor.leaderboard(test_data)
    results['post_distill_perf'] = {
        'test': calc_metrics(
            test_data.label,
            predictor.predict(test_data,model=model_to_deploy),
            binary=False),
        }

    distill_model_path = Path(predictor.path) / 'models' / model_to_deploy /'model.pkl'
    distill_model = joblib.load(distill_model_path)

    clr = distill_model.model
    raw_model_dest = Path(cfg.model_path).with_suffix('.pkl')
    joblib.dump(clr, raw_model_dest)
    return

    assert test_data.columns[-1] == 'label'
    num_features = test_data.shape[1] - 1 # colums size - label
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    onx = convert_sklearn(clr, initial_types=initial_type)
    with open(cfg.model_path, "wb") as f:
        f.write(onx.SerializeToString())

    chk_pred = predictor.predict(test_data).values
    chk_proba = predictor.predict_proba(test_data).values
    # predictor.delete_models(models_to_keep = [], dry_run=False)
    # del predictor        
    
    sess = rt.InferenceSession(cfg.model_path)
    # yhat = predictor.predict(test_data)
    # proba = predictor.predict_proba(test_data)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    # pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
    proba = sess.run([label_name], {input_name: test_data.iloc[:,:-1].values.astype(np.float32)})[0]
    yhat = proba.round().flatten()

    _ = calc_metrics(test_data.label, yhat, binary=False)

    info = {}
    info['usecols'] = train_data.columns.tolist()

    with open(cfg.info_path, 'w') as f_out:
        json.dump(info, f_out)

    # obs_pred = test_result['obs_pred'] = test_data.label.to_frame(name='obs').assign(yhat=yhat).join(proba)

if __name__ == '__main__':
    main()    





