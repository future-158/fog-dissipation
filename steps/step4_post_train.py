import argparse
import os
from itertools import product
from pathlib import Path

import autogluon.core as ag
import hydra
import joblib
from hydra.utils import get_original_cwd, to_absolute_path
import numpy as np
import pandas as pd
from autogluon.core.utils import try_import_lightgbm
from autogluon.tabular import TabularDataset, TabularPredictor
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     train_test_split)

from dataset import load_dataset
from utils import calc_metrics


cfg = OmegaConf.load('conf/config.yaml')
files = [x for x in Path('data/result').glob('SF_*')]
rows = []
for file in files:
    station_code, pred_hour = file.stem.rsplit('_', maxsplit=1)
    pred_hour = int(pred_hour)
    row = joblib.load(file)['post_ensemble_perf']['test']
    row['pred_hour'] = pred_hour
    row['station_code'] = station_code
    rows.append(row)
id_var = ['station_code', 'pred_hour']
table = pd.DataFrame(rows).set_index(id_var).sort_index()

table = table.reset_index()

table.station_code = table.station_code.map(cfg.station_name)
table = table.rename(columns={'station_code':'station_name'})

dest = Path(cfg.report_dest )
dest.parent.mkdir(parents=True, exist_ok=True)
# with pd.ExcelFile(dest) as reader:
#     sheet_name = str(len(reader.sheet_names) + 1)
# with pd.ExcelWriter(dest, mode='a') as writer:
#     table.to_excel(writer, sheet_name=sheet_name, index=False)
with pd.ExcelWriter(dest, mode='w') as writer:
    table.to_excel(writer, sheet_name='metrics', index=False)
