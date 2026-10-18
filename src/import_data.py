import numpy as np
from typing import List, Tuple
import random
from pathlib import Path

from src import utils # import src.utils as utils
from src.dataframe import select_df, load_data_index
from src.features import y_feature

"""
qua ci sarà da modificare per 3 dataset source / target / test; inoltre in target c'è da togliere la colonna delle Y
"""

class ImportData_LSTM():
    
    def __init__(
            self,
            Source_File: List[str] = [],
            Target_File: List[str] = None,
            Test_File: List[str] = None,
            verbose_cols: bool = False,
            include_CO: bool = True,
            #train_val_split: Tuple[float, float] = (0.8, 0.2),
            
):     
        Source_File = Source_File  # <--- perchè adesso non passiamo più una semplice lista python ma una lista Config di hydra
        Target_File = Target_File    # <--- perchè adesso non passiamo più una semplice lista python ma una lista Config di hydra 
        Test_File = Test_File

        self.verbose_cols = verbose_cols
        self.y_feature = y_feature
        self.d_source, self.d_target = select_df(source_key=Source_File, target_key=Target_File)
        self.d_test = self.d_target
        self.include_CO = include_CO

        # 1. generate dictionaries 
        #d_tr, d_te = select_df(Train_Files, Test_Files)

        # 2. read the CSV files
        self.df_source_da_dict = load_data_index( self.d_source, key_ = 'source_file')
        self.df_source_dict = load_data_index( self.d_source, key_ = 'source_file')
        self.df_target_dict = load_data_index( self.d_target, key_ = 'target_file')
        self.df_test_dict = load_data_index(self.d_test, key_ = 'target_file')
        
        #load train validation and test csv into dataframes
        self.df_source_da_raw, self.ranges_source_da = utils.load_data_idx(self.df_source_da_dict)
        self.df_source_raw, self.ranges_source  = utils.load_data_idx(self.df_source_dict)
        self.df_target_raw, self.ranges_target  = utils.load_data_idx(self.df_target_dict)
        self.df_test_raw, self.ranges_test = utils.load_data_idx( self.df_test_dict)

        #include CO_accomulated 
        self.df_source_da_raw = utils.filter_co_accomulated(self.df_source_da_raw, include_CO=self.include_CO)
        self.df_source_raw = utils.filter_co_accomulated(self.df_source_raw, include_CO=self.include_CO)
        self.df_target_raw = utils.filter_co_accomulated(self.df_target_raw, include_CO=self.include_CO)
        self.df_test_raw = utils.filter_co_accomulated(self.df_test_raw, include_CO=self.include_CO)


        # in source and target dataset slicing in it up to 1000 lines after the start of CO accumulated
        self.df_source_da_raw, self.ranges_source_da = utils.clip_target_df(df = self.df_source_da_raw,ranges=self.ranges_source_da, plus=1000)
        self.df_target_raw, self.ranges_target = utils.clip_target_df(df = self.df_target_raw,ranges=self.ranges_target, plus=1000)

        #clean the train dataset and apply the same final columns also to validation and test datasets
        self.df_source, self.X_features, self.col_to_use = utils.clean_df_train(self.df_source_raw, verbose=self.verbose_cols)
        self.df_source_da = self.df_source_da_raw[self.col_to_use]
        self.df_target = self.df_target_raw[self.col_to_use]
        self.df_test = self.df_test_raw[self.col_to_use]

        #lastly scale the dataframes (?)
        self.joint_X_scaler = utils.joint_X_scaler(self.df_source, self.df_target)
        
        self.source_dataset, self.source_X_scaler, self.source_y_scaler = utils.scale_data(self.df_source)
        self.source_dataset_da, self.source_X_scaler_da, self.source_y_scaler_da = utils.scale_data(self.df_source_da)
        self.target_dataset, self.target_X_scaler, _ = utils.scale_data(self.df_target)
        self.test_dataset = utils.scale_data(self.df_test, self.joint_X_scaler, self.source_y_scaler) 



        # remove column y form target
        self.target_dataset = self.target_dataset.drop(columns= "CellV_avg")




