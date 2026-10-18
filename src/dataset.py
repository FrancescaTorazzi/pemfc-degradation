import numpy as np
import pandas as pd
import torch
from torch import utils
from typing import Union, Tuple
import random

# togliamo tutti i seed c'è la fun seed in utils

class PEMFCDataset(utils.data.Dataset):
    def __init__(self,
                dataset: pd.DataFrame,
                X_features: dict[int, str],
                y_features: dict[int, str] | None = None,
                ):
        
        self.dataset = dataset
        self.features_int = list(X_features.keys())
        self.X = dataset[list(X_features.values())].values
        

        #logic for y None
        if y_features is None:
            self.targets_int = []
            self.y = None
        else:
            self.targets_int = list(y_features.keys())
            self.y = dataset[list(y_features.values())].values
            




class DynamicDataset(PEMFCDataset):
    def __init__(self, 
                 dataset: pd.DataFrame,
                 X_features: dict[int, str],
                 y_features: dict[int, str] | str = None,
                 T_x: int = 1,
                 T_y: int = 1,
                 s: int = 1,
                 dataframes_lenght = pd.Series,
                 is_training = True,
                 train_val_split: tuple = (0.8, 0.2)
                 ):
        
        """
        Examples:
            e.g. 1):
            df.index = [0,1,2,3,4,5,6,7,8,9]
            s = 3,
            T_x = T_y = 3
            then:
            X:  [[0, 1, 2], [3, 4, 5]]
            y:  [[3, 4, 5], [6, 7, 8]

            e.g. 2):
            df.index = [0,1,2,3,4,5,6,7,8,9]
            s = 3,
            T_x = 4,
            T_y = 3
            then:
            X:  [[0, 1, 2, 3], [3, 4, 5, 6]]  
            y:  [[3, 4, 5], [6, 7, 8]]
        """
        
        super().__init__(dataset, X_features, y_features)

        self.T_x = T_x
        self.T_y = T_y
        self.s = s 
        self.is_training = is_training
        
        

        idx_reference = np.array([i[1] - i[0] for i in dataframes_lenght]) - (max(self.T_x, self.T_y) + self.s - 1) # ranges_train = {start_idx, end_idx, impurity_start_idx}
        if any(i < 0 for i in idx_reference):
            raise Exception('Sequence length too long for one or more data blocks.')

        # Ottieni tupla (start_idx, ordine temporale)
        all_starts = [
            (idx, i)
            for (start, end, _), n in zip(dataframes_lenght, idx_reference)
            for i, idx in enumerate(np.arange(start, start + n + 1, s))
        ]

        
        if self.is_training == True:   
            # Shuffle degli start
            all_starts = list(all_starts)
            random.shuffle(all_starts)
        else: 
            all_starts = sorted(all_starts, key=lambda x: x[0])
            self.beta = np.ones(len(dataset))
            
        all_starts = [x[0] for x in all_starts]  # tieni solo gli indici
        

        # Calcola gli indici target (per y)
        # modificato! perche contenteneva un errore di logica, l'indice y deve essere calcolato come st + T_x non come come st + s
        all_y_indices = [st + self.T_x for st in all_starts]
            
        # Split train/val
        split_idx = int(len(all_starts) * train_val_split[0])
        self.starts_train = all_starts[:split_idx]
        self.starts_val = all_starts[split_idx:]

        self.indices_y_train = all_y_indices[:split_idx]
        self.indices_y_val = all_y_indices[split_idx:]

        # dimensione dell'ouput y quando y manca ( dummy a 1 col se non noto)
        self._y_out_dim = 1 if (self.y is None) else self.y.shape[1]

    def __len__(self):
        return len(self.starts_train)

    def __getitem__(self, idx):
        st = self.starts_train[idx]
        X_seq = self.X[st : st + self.T_x]
        index_y = self.indices_y_train[idx]

        if self.y is None:
            y_seq = np.zeros((self.T_y, self._y_out_dim), dtype = np.float32)
        else:
            #anche questo modificato per il motivo a riga 92
            y_seq = self.y[index_y : index_y + self.T_y]

        return(
            torch.tensor(X_seq, dtype = torch.float),
            torch.tensor(y_seq, dtype = torch.float),
            index_y
        ) 
        


    def get_val_dataset(self):
        class ValDataset(torch.utils.data.Dataset):
            def __init__(self, parent):
                self.X = parent.X
                self.y = parent.y
                self.starts = parent.starts_val
                self.T_x = parent.T_x
                self.T_y = parent.T_y
                self.s = parent.s 
                self.indices_y = parent.indices_y_val
                self._y_out_dim = parent._y_out_dim

            def __len__(self):
                return len(self.starts)

            def __getitem__(self, idx):
                st = self.starts[idx]
                X_seq = self.X[st : st + self.T_x]
                index_y = self.indices_y[idx]

                if self.y is None:
                    y_seq = np.zeros((self.T_y, self._y_out_dim), dtype = np.float32)
                else:
                    #anche questo modificato per il motivo a riga 92, l'indice y deve essere calcolato come st + T_x non come come st + s
                    y_seq = self.y[index_y : index_y + self.T_y]
                

                return(
                    torch.tensor(X_seq, dtype = torch.float),
                    torch.tensor(y_seq, dtype = torch.float),
                    index_y
                )
                

 
        return ValDataset(self)

    


