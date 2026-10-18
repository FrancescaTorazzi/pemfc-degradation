from pathlib import Path 
import os 
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf 
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import pandas as pd, json
from datetime import datetime
import pickle
import numpy as np
import joblib
from src.utils import set_environment, y_feature
from src.dataset import DynamicDataset
from src.import_data import ImportData_LSTM
from src.modeling.models import PEMFC_GRU
from src.modeling.trainer import Trainer
from src.plots import plot_history, plot_true_vs_pred_series, plot_scatter_ytrue_vs_ypred_from_df, plot_histories, plot_da_total, plot_history_log, plot_source_test
from src.modeling.predictions import get_final_metrics, rnn_csv, rmse_impurity
from src.savings import save_experiment



import torch
def _ensure_abs(path_like: str | Path, base: Path | None = None) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    # base = root del progetto: cartella genitore di 'src'
    if base is None:
        base = Path(__file__).resolve().parents[1]
    return base / p

@hydra.main(config_path="conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:

#   tipe of divece. GPU, mps (mac), or CPU
    requested_device = str(cfg.device).lower()

    if requested_device =='auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')    

    else: 
        device = torch.device(requested_device)
    print(f"selected device: {device}")
    
    #seed
    set_environment(seed=cfg.seed, device=device)
    
    # -------- Dataset ---------------------------------------------
    ds = ImportData_LSTM(
    Source_File=cfg.data.Source_File,
    Target_File=cfg.data.Target_File,
    verbose_cols=cfg.data.verbose_cols,
    include_CO=cfg.data.include_CO
    )

    # information from Datasets 
    ds_source = ds.source_dataset
    ds_source_da = ds.source_dataset_da
    ds_target = ds.target_dataset
    ds_test = ds.test_dataset

    X_features = ds.X_features # actually the columns are the same for every dataset, the only difference is target that doesn't have CellV (names of columns X)
    y_feature = ds.y_feature # name of column y
    y_scaler = ds.source_y_scaler
    
    #ranges
    ranges_source = ds.ranges_source
    ranges_target = ds.ranges_target
    ranges_test = ds.ranges_test
    # nomi file
    source_file= list(ds.d_source["source_file"].keys())
    target_file = list(ds.d_target["target_file"].keys())

    # -------- Dataloader ---------------------------------------------
    # Source
    dl_source = DynamicDataset(
    dataset=ds_source,
    X_features=X_features,
    y_features=y_feature,
    T_x=cfg.data.T_x,
    T_y=cfg.data.T_y,
    s=cfg.data.s,
    dataframes_lenght=ranges_source,     
    train_val_split=(cfg.data.train_split, cfg.data.val_split),
    is_training = True,
       )
    train_dl_source = dl_source
    val_dl_source  = dl_source.get_val_dataset()

    # source per la DA
    dl_source_da = DynamicDataset(
    dataset=ds_source,
    X_features=X_features,
    y_features=y_feature,
    T_x=cfg.data.T_x,
    T_y=cfg.data.T_y,
    s=cfg.data.s,
    dataframes_lenght=ranges_source,     # <-- tutta la base dati
    train_val_split=(0.8, 0.2),
    is_training = True,
    )
    train_dl_source_da = dl_source_da
    val_dl_source_da  = dl_source_da.get_val_dataset()

    # Target i dati di target arrivano fino a 1000 punti dopo l'introduzione della CO
    dl_target = DynamicDataset(
    dataset=ds_target,
    X_features=X_features,
    y_features=None,
    T_x=cfg.data.T_x,
    T_y=cfg.data.T_y,
    s=cfg.data.s,
    dataframes_lenght=ranges_target,     # <-- tutta la base dati
    train_val_split=(0.8, 0.2),
    is_training = True,
    )
    train_dl_target = dl_target
    val_dl_target  = dl_target.get_val_dataset()

    # Test
    dl_test = DynamicDataset(
    dataset=ds_test,
    X_features=X_features,
    y_features=y_feature,
    T_x=cfg.data.T_x,
    T_y=cfg.data.T_y,
    s=1,
    dataframes_lenght=ranges_test,     # <-- tutta la base dati
    train_val_split=(1.0, 0.0),
    is_training = False,
    )
    dl_test = dl_test

    # ------- MODEL ------------
    # Model
    model = PEMFC_GRU(
    input_size=len(X_features),
    hidden_size=cfg.model.hidden_size,
    output_size=len(y_feature),
    num_layers=cfg.model.num_layers,
    dropout=cfg.model.dropout
    ).to(device)

    # Train
    train = Trainer(
    model= model,
    device=device,
    da_type = cfg.train.da_type,
    da_lambda= cfg.train.da_lambda,
    learning_rate= cfg.train.learning_rate,
    epochs= cfg.train.epochs,
    batch_size = cfg.train.batch_size,
    weight_decay= cfg.train.weight_decay,
    )

    history, state_dict, best_val_loss =train.fit(
    train_dl_source= train_dl_source,
    train_dl_source_da= train_dl_source_da,
    train_dl_target= train_dl_target,
    val_dl_source= val_dl_source,
    val_dl_source_da= val_dl_source_da,
    val_dl_target= val_dl_target,
    patience=cfg.train.patience,
    restore_best_weights= True,
    save_last_weights_path=None
    )
#   testing results on source 
    dl_source_for_plot = DynamicDataset(
    dataset=ds_source,
    X_features=X_features,
    y_features=y_feature,
    T_x=cfg.data.T_x,
    T_y=cfg.data.T_y,
    s=1,
    dataframes_lenght=ranges_source,     # <-- tutta la base dati
    train_val_split=(1.0, 0.0),
    is_training = False,
   
)
    train_dl_source = dl_source

    df_aligned_dl_source, test_metrics_source = train.test_LSTM(
        test_dl=dl_source_for_plot, 
        y_scaler=y_scaler,
        y_feature=y_feature
    )


    # Test
    df_aligned_dl_test, test_metrics = train.test_LSTM(
    test_dl=dl_test, 
    y_scaler=y_scaler,
    y_feature=y_feature
    )

    # ------- Plot -------------- 
    fig_history = plot_history(history, metrics=['loss', 'mse'])
    fig_series_rnn_source = plot_true_vs_pred_series(df_aligned_dl_source, title="RNN - Test set: real vs predicted", ranges_test=ranges_source)
    fig_series_rnn_test = plot_true_vs_pred_series(df_aligned_dl_test, title="RNN - Test set: real vs predicted", ranges_test=ranges_test)
    fig_scatter_rnn = plot_scatter_ytrue_vs_ypred_from_df(df_pred=df_aligned_dl_test, title="RNN - Test: y_true vs y_pred")
    fig_da_total = plot_da_total(history, title=f"Training history ({train.da_type.upper()})")
    history_log = plot_history_log(history, metrics=['loss', 'mse'], title='Training History')
    fig_all_histories = plot_histories(history)
    fig_source_test_pred = plot_source_test(df_source=df_aligned_dl_source, df_test= df_aligned_dl_test, ranges_source=ranges_source, ranges_test=ranges_test)
    # ------ Metrics ------ 
    final_metrics_df = get_final_metrics(history=history, test_metrics=test_metrics)
    rnn_metrics_dict = final_metrics_df.to_dict(orient="list") 
  
    # ------ Savings ----------------
    ckpt_payload = {
        "state_dict": state_dict,
        "history": history,
        "best_val_loss": float(best_val_loss),    }
    
    rnn_csv_ = rnn_csv(df_aligned=df_aligned_dl_test, ranges_test=ranges_test, test_files=target_file)
    dict_ = rmse_impurity(df=rnn_csv_)
    rnn_metrics_dict["rmse_impurity"] = dict_

    y_col = list(y_feature.values())[0] if isinstance(y_feature, dict) else y_feature[0]
    y_full = ds_source[y_col].tolist()

    sliding_windows_payload= {
        "config": {
            "T_x": cfg.data.T_x,
            "T_y": cfg.data.T_y,
            "s": cfg.data.s
        },
        "ranges": {
            "source_ranges": ranges_source,
            "source_files": source_file,
            "target_ranges": ranges_target,
            "target_files": target_file,            
        },
        "train": {
            "starts_train_source": list(getattr(train_dl_source, "starts_train", None)),
            "indices_y_train_soruce":list(getattr(train_dl_source, "indices_y_train", None)),
            "starts_val_source": list(getattr(train_dl_source, "starts_val", None)),
            "indices_y_val_soruce": list(getattr(train_dl_source, "indices_y_val", None)),
        },
        "series": {
            "y_col": y_col,
            "y": y_full
        }
    }

    # salvataggio esperimenti (CSV/JSON/PKL + path pasi)
    extra_config = {
        "seed": cfg.seed,
        "device": str(device),
        "data": {
            "Source_File": cfg.data.Source_File,
            "Target_File": cfg.data.Target_File,
            "T_x": cfg.data.T_x, "T_y": cfg.data.T_y, "s": cfg.data.s,
            "train_split": cfg.data.train_split, "val_split": cfg.data.val_split,
            "X_features": X_features,
            "y_features": y_feature
        },
        "model": {
            
            "hidden_size": cfg.model.hidden_size,
            "num_layers": cfg.model.num_layers,
            "dropout": cfg.model.dropout
        },
        "training": {
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.learning_rate,
            "epochs": cfg.train.epochs,
            "weight_decay": cfg.train.weight_decay,
            "patience": cfg.train.patience,
            "da_type": cfg.train.da_type,
            "da_lambda": cfg.train.da_lambda
        },
    }
    # ------------ SALVATAGGIO ----------------------------------
    save_experiment(
        cfg,
        rnn={
            "model": model,                         # salva model.pth in RNN/
            "metrics": rnn_metrics_dict,            # json
            "predictions_df":  rnn_csv_,          # csv
            "figs": {
                "history.png":            fig_history,
                "test_true_vs_pred_source.png":  fig_series_rnn_source,
                "test_true_vs_pred_test.png":  fig_series_rnn_test,
                "all_histories": fig_all_histories,
                "series source & pred": fig_source_test_pred,
                
            },
            "model_state_dict": state_dict,
            "model_checkpoint": ckpt_payload,            
        },
        
        base_dir=_ensure_abs(cfg.output.dir),       # "runs"
        overrides=None,
        overwrite=False,
        
    )


if __name__ == "__main__":
    run()