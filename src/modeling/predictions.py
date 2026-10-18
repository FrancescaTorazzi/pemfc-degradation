import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import utils

def TimeSeries_predict_and_metrics(model, dataset, batch_size=64, device='cpu'):
    loader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()

    all_preds = []
    all_targets = []
    all_indices = []

    # loop predictions
    with torch.no_grad():
        for X_batch, y_batch, _, index_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()

            all_preds.append(preds)
            all_targets.append(y_batch)
            all_indices.append(index_batch)

    y_pred = torch.cat(all_preds).numpy()  # shape: (N, T_y)
    y_true = torch.cat(all_targets).numpy()
    indices_y = torch.cat(all_indices).numpy()  # shape: (N,)

    # ricostruzione della serie temporale completa 
    full_pred = []
    full_true = []
    full_index = []

    for i in range(len(indices_y)):
        for t in range(y_pred.shape[1]):  # T_y
            full_index.append(indices_y[i] + t)
            full_pred.append(y_pred[i, t])
            full_true.append(y_true[i, t])

    df_pred = pd.DataFrame({
        "index": full_index,
        "y_true": full_true,
        "y_pred": full_pred
    }).sort_values("index").reset_index(drop=True)

    df_metrics = metrics(
        df_pred["y_true"].to_numpy(),
        df_pred["y_pred"].to_numpy()
    )

    return df_pred, df_metrics

def metrics(y_true, y_pred) -> pd.DataFrame:
    """
    Compute common regression metrics between two aligned series.
    Returns a one-row DataFrame with columns [rmse, mae, r2].
    """
    # Conversione esplicita a float, riga per riga
    y_true_clean = []
    y_pred_clean = []

    for yt, yp in zip(y_true, y_pred):
        try:
            yt_val = float(yt.item())
            yp_val = float(yp.item())
            y_true_clean.append(yt_val)
            y_pred_clean.append(yp_val)
        except:
            continue  # Skippa valori non numerici o NaN-like

    assert len(y_true_clean) == len(y_pred_clean), "y_true and y_pred must have the same length"

    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae  = mean_absolute_error(y_true_clean, y_pred_clean)
    r2   = r2_score(y_true_clean, y_pred_clean)

    return pd.DataFrame({
        'rmse': [rmse],
        'mae':  [mae],
        'r2':   [r2],
    })

import pandas as pd 


import pandas as pd 
def get_final_metrics(history: dict, test_metrics: dict = None) -> pd.DataFrame:
    
    import pandas as pd

    def safe_get(d, k):
        return d[k][-1] if k in d and len(d[k]) > 0 else None

    train_metrics = {
        'mse':   safe_get(history, "train_mse"),
        'mae':   safe_get(history, "train_mae"),
        'rmse':  safe_get(history, "train_rmse"),
        'r2':    safe_get(history, "train_r2"),
        'da':    safe_get(history, "train_da"),
        'total': safe_get(history, "train_total"),
    }

    val_metrics = {
        'mse':   safe_get(history, "val_mse"),
        'mae':   safe_get(history, "val_mae"),
        'rmse':  safe_get(history, "val_rmse"),
        'r2':    safe_get(history, "val_r2"),
        'da':    safe_get(history, "val_da"),
        'total': safe_get(history, "val_total"),
    }

    df = pd.DataFrame([
        {"set": "train", **train_metrics},
        {"set": "val",   **val_metrics},
    ])

    if test_metrics:
        df.loc[len(df)] = {"set": "test", **test_metrics}

    # ordinamento opzionale per leggibilità
    cols = ['set', 'mse', 'rmse', 'mae', 'r2', 'da', 'total']
    df = df[[c for c in cols if c in df.columns]]

    return df

def rmse_after_impurity_from_aligned(
        df_aligned: pd.DataFrame,
        ranges_test: None,        
):
    idx = np.asarray(df_aligned["index"].values)
    y_true = np.asarray(df_aligned["y_true"].values, dtype=float)
    y_pred = np.asarray(df_aligned["y_pred"].values, dtype=float)
    order = np.argsort(idx)
    idx, y_true = y_pred = idx[order], y_true[order], y_pred[order]
    resid_all = []
    by_range = []
    for k, (start_idx, end_idx, start_impurity_idx) in enumerate(ranges_test):
        m = (idx >= int(start_impurity_idx)) & (idx < int(end_idx))
        resid = y_true[m] - y_pred[m]
        resid_all.append(resid)

        rmse_r = float(np.sqrt(np.mean(resid**2)))
        
    resid_all = np.concatenate(resid_all)
    rmse_overall = float(np.sqrt(np.mean(resid_all**2)))
    return rmse_overall
        
    

def rnn_csv(df_aligned: pd.DataFrame, ranges_test: None, test_files: list) -> pd.DataFrame:
    df_tmp = df_aligned.copy()
    if df_tmp.index.name != 'idx':
        df_tmp = df_tmp.reset_index().rename(columns= {'index': 'idx'})
    cols = ['idx', 'y_pred', 'y_true']
    out = df_tmp[['idx', 'y_pred', 'y_true']].copy()
    out['in_impurity']= 0 
    seg_col = np.full(len(out), 'UNASSIGNED', dtype=object)
    for name,(start_idx, end_idx, start_impurity_idx) in zip(test_files, ranges_test):
        # segment_id per tutto il range del segmento
        mask_seg = (out['idx'] >= start_idx) & (out['idx'] <= end_idx)
        seg_col[mask_seg.to_numpy()] = name

        # in_impurity solo dalla soglia di impurity fino a end_idx
        mask_imp = (out['idx'] >= start_impurity_idx) & (out['idx'] <= end_idx)
        out.loc[mask_imp, 'in_impurity'] = 1
        
        seg_col[ mask_imp.to_numpy()] = name
    out['segment_id'] = seg_col
    return out 
  
# def rmse with impurity
def rmse_impurity(df: pd.DataFrame) -> dict:
    """
    ritorna un dizionario con {segment_id: rmse_impurity}.
    L'RMSE è calcolato solo sulle righe con in_impurity == 1
    """
    result = {}
    for seg, group in df.groupby("segment_id"):
        g_imp = group[group["in_impurity"]==1]
        err = g_imp["y_pred"] - g_imp["y_true"]
        result[seg] = float(np.sqrt(np.mean(err**2)))
    return result 
















