import pandas as pd 
import os 
import numpy as np 
import pandas as pd 
from src.config import UPDATED_DATA
from collections import OrderedDict
def select_df(
        source_key: list[str],
        target_key: list[str]
):
    # gat all file paths and stems 
    l = sorted([f for f in os.listdir(UPDATED_DATA) if (UPDATED_DATA / f).is_dir()])
    all_files = l

    # create the full paths uing path 
    all_paths = {f:UPDATED_DATA / f for f in all_files}

    def get_names(filters: list[str]) -> list[str]:
        if filters == ['all']:
            return l 
        
        if isinstance(filters, list) and len(filters)>1:
            return [f for f in l if any (k in f for k in filters)]
        else: 
            if isinstance(filters, list):
                return[f for f in l if all(k in f for k in filters)]
            else:
                return[f for f in l if filters in f]

    source_file = get_names(source_key)
    target_file = get_names(target_key)
    test_file = get_names(target_key)

    overlap = set(source_file).intersection(set(target_file))
    
    if overlap:
        raise Exception (f"Source and Target/Test file cannot overlap: {overlap}")
    # se uno dei due insiemi è vuoto, c'è un errore nei filtri 
    elif bool(source_file) != bool(target_file):
        raise Exception('Misspelled or impossible keys combinations')
    else:
        pass 
    
    source_file = OrderedDict((folder_name, all_paths[folder_name]) for folder_name in source_file)
    target_file = OrderedDict((folder_name, all_paths[folder_name]) for folder_name in target_file)
    test_file = OrderedDict((folder_name, all_paths[folder_name]) for folder_name in test_file)

    for name in source_file:
        print(f" source_file: {name}")

    for name in target_file:
        print(f"target_file: {name}")

    for name in test_file:
        print(f"test_file: {name}")

    unique_id = 'SOURCE__'+''.join(part for k in source_key for part in k.split()) + 'TARGET__'+''.join(part for k in target_key for part in k.split())

    d_source = {
        'id_keys': unique_id,
        'source_file':source_file,
    }
    d_target = {
        'id_keys': unique_id,
        'target_file': target_file,
    }
    return d_source, d_target

def load_data_index(dict_to_copy: dict, key_) -> dict[pd.DataFrame, dict]:
    """
    Load reference and impurity data from multiple folders, compute the comulative indices. 
    Args: 
        file list_ dict where keys are folder names, and values are Path objects.

    Retuns: 
        dfs: concatenated DataFrame of all reference and impurity data. 
        info_diz: dictionary with full and reference index ranges mapped to the concatenated dfs. 
    """
    d = OrderedDict(dict_to_copy.items())# copia il dizionario originale (per non modificarlo direttamente)
    dfs = []
    info_diz = {}
    s_full = 0 # start index for full data

   


    for folder, folder_path in d[key_].items(): # prende ogni cartella (chiave folder) e il relativo path (folder_path) del dizionario d[key_]
        df_reference = pd.read_csv(folder_path / 'reference.csv')
        df_impurity = pd.read_csv(folder_path / 'impurity.csv')

        df_all = pd.concat([df_reference, df_impurity], ignore_index = True)
        
        #rimuovi le prime 15 righe per togliere lo spike iniziale 
        df_all = df_all.iloc[10:].reset_index(drop = True)

        dfs.append(df_all)

        len_ref = len(df_reference)
        len_full = len(df_all)

        start_full = s_full
        end_full = s_full + len_full - 1 

        impurity_start_idx = start_full +  len_ref - 1  

        # salva una tuple con gli indici di inizio/fine per: tutti i dati, solo i dati di reference. 
        info_diz[folder] = (start_full, end_full, impurity_start_idx)

        # inidice prossima cartella 
        s_full += len_full
    
    dfs = pd.concat(dfs, ignore_index = True)


    d[key_ + '__idxs'] = info_diz
    d[key_ + '__dfs'] = dfs

    return d 