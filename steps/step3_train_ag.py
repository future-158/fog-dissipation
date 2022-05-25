import argparse
import os
from itertools import product
from pathlib import Path

import autogluon.core as ag
import hydra
import joblib
import numpy as np
import pandas as pd
from autogluon.core.utils import try_import_lightgbm
from autogluon.tabular import TabularDataset, TabularPredictor
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
# from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from dataset import load_dataset
from utils import calc_metrics

@hydra.main(config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    time_limit= cfg.time_limit
    station = cfg.station
    station_name = cfg.station_name[station]
    pred_hour = cfg.pred_hour
    target_name = cfg.target_name
    tokp = cfg.topk
    # out_dir = Path('result/{}T_{}H'.format(sample_distance, pred_hour))
    # out_dir.mkdir(parents=True, exist_ok=True)

    

    X, y, groups = load_dataset(
        Path(get_original_cwd()),
        station_name
        )
    y = y[target_name]
    data = X.assign(label=y)
    label = 'label'  # specifies which column do we want to predict
    save_path = 'data/ag_hpo_models/'  # where to save trained models
    Path(save_path).mkdir(parents=True, exist_ok=True)

    groups = data.index.year*366 + data.index.dayofyear
    sgk = StratifiedGroupKFold(n_splits=5)

    train_idx, test_idx = list(sgk.split(data,y, groups=groups))[0]
    print('train_ratio:\n',data['label'].iloc[train_idx].value_counts(normalize=True))
    print('val_ratio:\n',data['label'].iloc[test_idx].value_counts(normalize=True))

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    train_idx, val_idx = list(sgk.split(train_data, train_data[label], groups=groups[train_idx]))[0]
    val_data = train_data.iloc[val_idx]
    train_data = train_data.iloc[train_idx]

    predictor = TabularPredictor(
        label=label, 
        path=save_path,
        eval_metric='accuracy',
        sample_weight='balance_weight',    
        # ignored_columns=None,
        )
        # group shuffle split or cross validation not implemented

    hyperparameters = {
    'CAT': {},
    'GBM': {},
    'XGB': {},
    'RF': {}
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
        ag_args_fit={
            # 'num_gpus': 1,
            'num_cpus':10
            },
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

    pre_ensemble_perf = {
        'test': calc_metrics(
            test_data.label,
            predictor.predict(test_data,model=ensemble_model),
            binary=False
            ),
        'val': calc_metrics(
            val_data.label,
            predictor.predict(val_data,model=ensemble_model),
            binary=False
            )
        }

    # predictor.delete_models(models_to_keep=[ensemble_model])
    kv = predictor.refit_full()
    post_ensemble_perf = {
        'test': calc_metrics(
            test_data.label,
            predictor.predict(test_data,model=kv[ensemble_model]),
            binary=False
            ),
        'val': calc_metrics(
            val_data.label,
            predictor.predict(val_data,model=kv[ensemble_model]),
            binary=False
            )
        }

    results['pre_ensemble_perf'] = pre_ensemble_perf
    results['post_ensemble_perf'] = post_ensemble_perf

    # predictor.model = kv[ensemble_model]    
    # joblib.dump(predictor, Path(out_dir) / f'predictor_{cv_idx}.pkl')
    dest = Path(get_original_cwd()) / 'data' / 'result' / f'{station}_{pred_hour}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, dest)
   
if __name__ == '__main__':
    main()    
    # CUDA_VISIBLE_DEVICES='0' taskset --cpu-list 50-90 python train_ag.py -m station=SF_0002,SF_0003,SF_0004,SF_0005,SF_0006,SF_0007,SF_0008,SF_0009 pred_hour=1,2,3 
