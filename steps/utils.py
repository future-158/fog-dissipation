from os import nice
import numpy as np
import pandas as pd
from sklearn.metrics import *
from typing import *

def calc_metrics(obs: pd.Series, pred:np.ndarray, binary=True) -> Dict:
    if binary:
        tn, fp, fn, tp = confusion_matrix(obs, pred).flatten()
        return dict(
            ACC=accuracy_score(obs,pred),
            CSI=jaccard_score(obs, pred),
            PAG=precision_score(obs, pred),
            POD=recall_score(obs, pred),
            F1=f1_score(obs,pred),
            TN=tn,
            FP=fp,
            FN=fn,
            TP=tp,
        )

    else:
        metrics = {}
        metrics['ACC'] = accuracy_score(obs, pred )
        metrics['macro_CSI']  = jaccard_score(obs,pred, average='macro')
        metrics['macro_PAG'] = precision_score(obs,pred, average='macro')
        metrics['macro_POD'] = recall_score(obs, pred, average='macro')
        metrics['macro_F1'] = f1_score(obs, pred, average='macro')

        metrics['micro_CSI']  = jaccard_score(obs,pred, average='micro')
        metrics['micro_PAG'] = precision_score(obs,pred, average='micro')
        metrics['micro_POD'] = recall_score(obs, pred, average='micro')
        metrics['micro_F1'] = f1_score(obs, pred, average='micro')
        return metrics

    
    


