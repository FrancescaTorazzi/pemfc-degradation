import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from sklearn.preprocessing import StandardScaler
import os
import random
import numpy as np
import torch
from sklearn.preprocessing import FunctionTransformer
from src.dataframe import select_df, load_data_index

from src.features import feature_to_remove, y_feature # from src.features import feature_to_remove, y_feature


def load_data_idx(df_dict: dict):
    """
    Prende in input un dizionario creato da select_df + load_data_index e restituisce:
        - data_f: DataFrame concatenato 
        - ranges: lista di tuple (experiment_name, start_idx, end_idx, impurity_start_idx)
    """
    # identifica automaticamente la chiave tra 'tr_files__dfs' o ' te_files__dfs'
    dfs_key = [k for k in df_dict.keys() if k.endswith('__dfs')][0]
    idxs_key = dfs_key.replace('__dfs', '__idxs')

    # estrai Dataframe completo
    data_f = df_dict[dfs_key]

    # Estrai i range
    ranges = []
    for folder_name, (start_idx, end_idx, impurity_start_idx) in df_dict[idxs_key].items():
        ranges.append((start_idx, end_idx, impurity_start_idx ))

    return data_f, ranges

 
    
def scale_data(data_f, scaler_X = None, scaler_y = None):

    if all((scaler_X,scaler_y)):
        X_tr_df = data_f.drop(columns = list(y_feature.values()))
        y_tr_df = data_f[list(y_feature.values())]

        #scaler_X.fit(X_tr_df) -> sbagliato   
        #scaler_y.fit(y_tr_df)

        X_df_sc = scaler_X.transform(X_tr_df)
        y_df_sc = scaler_y.transform(y_tr_df)

        X_df_sc = pd.DataFrame(X_df_sc, columns = X_tr_df.columns)
        y_df_sc = pd.DataFrame(y_df_sc, columns = y_tr_df.columns)

        scaled_data = pd.concat([X_df_sc, y_df_sc], axis=1)

        return scaled_data
    else:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_tr_df = data_f.drop(columns = list(y_feature.values()))
        y_tr_df = data_f[list(y_feature.values())]

        scaler_X.fit(X_tr_df)
        scaler_y.fit(y_tr_df)

        X_df_sc = scaler_X.transform(X_tr_df)
        y_df_sc = scaler_y.transform(y_tr_df)

        X_df_sc = pd.DataFrame(X_df_sc, columns = X_tr_df.columns)
        y_df_sc = pd.DataFrame(y_df_sc, columns = y_tr_df.columns)

        scaled_data = pd.concat([X_df_sc, y_df_sc], axis=1)

        return scaled_data, scaler_X, scaler_y

def clean_df_train(
    df: pd.DataFrame,
    thresh: float = 0.95,
    verbose: bool = False,
):
    """
    * Estrae y_label e rimuove le sue colonne.
    * Elimina cols_to_remove (se presenti).
    * Elimina colonne costanti.
    * Elimina colonne con correlazione > thresh.
    * Ricongiunge y_label alla fine.
    * Se verbose: stampa colonne rimosse e riepilogo.
    """

    def verb(cols: List[str], per_line: int = 6, sep: str = " | ") -> None:
        if not cols:
            print("(none)")
            return
        w = max(map(len, cols)) + len(sep)
        for i in range(0, len(cols), per_line):
            line = sep.join(s.ljust(w) for s in cols[i : i + per_line])
            print(line)
            if i + per_line < len(cols):
                print("-" * len(line))

    start_cols = df.shape[1]
    df = df.copy()

    # target
    y_columns = list(y_feature.values())
    y = df[y_columns]  # Select the target columns
    df.drop(columns=y_columns, inplace=True)  # Remove the target columns from the DataFrame

    # keep these columns 
    columns_to_keep = ["Recirculated_Rate", "Relative_Humidity", "Purge_timing"]
    df_protected = df[columns_to_keep].copy()
    df = df.drop(columns=[col for col in columns_to_keep if col in df.columns] )

    # remove unwanted
    cols_to_remove = feature_to_remove 
    removed = [c for c in cols_to_remove if c in df.columns]
    df.drop(columns=removed, inplace=True)

    # constant columns
    const = df.columns[df.nunique(dropna=False) == 1].tolist()
    df.drop(columns=const, inplace=True)

    # high-corr columns
    num_df = df.select_dtypes(include=[np.number]) # solo numeriche
    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > thresh)]
    df.drop(columns=high_corr, inplace=True)

    # add protected columns 
    for col in columns_to_keep:
        if col not in df.columns and col in df_protected.columns:
            df[col] = df_protected[col]

    # ricongiungi target
    df_cleaned_train = pd.concat([df, y], axis=1)
    X_features = {i: c_name for i,c_name in enumerate(df.columns) if c_name not in list(y_feature.values())}
    all_columns = df_cleaned_train.columns

    if verbose:
        print(f"Removed by list ({len(removed)}):");   verb(removed);   print()
        print(f"Removed constants ({len(const)}):");   verb(const);   print()
        print(f"Removed corr> {thresh} ({len(high_corr)}):"); verb(high_corr); print()
        print(f"Cols: start={start_cols}, final={df.shape[1]}, removed={start_cols - df.shape[1]}")
        print("Final columns:"); verb(df_cleaned_train.columns.tolist())

    return df_cleaned_train, X_features, all_columns


def filter_co_accomulated(df: pd.DataFrame, include_CO: bool = True) -> pd.DataFrame:
    """rimuove la colonna 'CO_accumulated' se include_CO è False"""
    if not include_CO and 'CO_accumulated' in df.columns:
        df = df.drop(columns=['CO_accumulated'])
    return df

# il seed


def set_environment(seed: int = 42, device: str = 'cpu') -> torch.Generator:
    """set_environment.

    Setup the execution environment by tuning pytorch parameters and
    by setting a random seed for reproducibility.

    :param seed: random seed to set
    :type  seed: int
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def clip_target_df(df, ranges, plus=1000):
    """
    Gives back a df.iloc(: beta_idx + plus)
    """
    first = ranges[0]
    beta_idx = int(first[2])
    stop = beta_idx + plus
    df_clip = df.iloc[:stop].reset_index(drop=True)
    start_idx = ranges[0][0]
    end_idx = len(df_clip)
    impurity_start_idx = ranges[0][2]
    new_ranges = [(start_idx, end_idx, impurity_start_idx)]
    return df_clip, new_ranges

   
def joint_X_scaler(df_source, df_target):
    """Fit a single X scaler on the joint data (source + target)."""   
    X_source = df_source.drop(columns=list(y_feature.values()))
    X_target = df_target.drop(columns=list(y_feature.values()))
    X_joint = pd.concat([X_source, X_target], axis=0, ignore_index=True)
    scaler_X = StandardScaler()
    scaler_X.fit(X_joint)
    return scaler_X


