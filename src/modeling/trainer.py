import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
import os
from itertools import cycle
from src.plots import plot_full_series
from src.utils import set_environment
from src.modeling.losses import TransferLoss
from pathlib import Path
import copy

"""
qua bisognda modifcare il codice sia per la loss, inoltre adesso non ci sarà un solo batch
"""
 
 
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        da_type: str = "coral",   # coral | mmd | dare
        da_lambda: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        seed: int = 42
    ):

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Using device: {self.device}")
    
        # model
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay= weight_decay )
        self.seed = seed
        self.loss = F.mse_loss
        
        # Domain Adaptation
        self.da_type = da_type
        self.da_lambda = da_lambda
        self.transfer_criterion = TransferLoss(loss_type = self.da_type)
    
    #infinite loaders logic
    @staticmethod
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch  

    def _get_loader(self, dataset, shuffle=False, drop_last=False):
        return DataLoader(dataset, batch_size = self.batch_size , shuffle=shuffle, drop_last= drop_last) 

    def fit(self, 
            train_dl_source,
            train_dl_source_da, 
            train_dl_target,
            val_dl_source_da,
            val_dl_source,
            val_dl_target, 
            patience=5, 
            restore_best_weights=True, 
            save_last_weights_path=None):
        
        train_dl_source = self._get_loader(train_dl_source, shuffle=True, drop_last=True)
        train_dl_source_da = self._get_loader(train_dl_source_da, shuffle=True, drop_last=True)
        train_dl_target = self._get_loader(train_dl_target, shuffle=True, drop_last=True)
        
        val_dl_source = self._get_loader(val_dl_source, shuffle=False, drop_last=False)
        val_dl_source_da = self._get_loader(val_dl_source_da, shuffle=False, drop_last=False)
        val_dl_target = self._get_loader(val_dl_target, shuffle=False, drop_last=False)

        #infinite loaders
        it_s = self.infinite_loader(train_dl_source)
        it_s_da = self.infinite_loader(train_dl_source_da)

        it_t = self.infinite_loader(train_dl_target)

        n_steps = max(len(train_dl_source),  len(train_dl_source_da), len(train_dl_target))

        history = {
            'train_loss': [], 'train_mse': [], 'train_mae': [], 'train_r2': [], 'train_rmse': [],
            'val_loss':   [], 'val_mse':   [], 'val_mae':   [], 'val_r2':   [], 'val_rmse': [],
            'train_da': [], 'train_total': [], 'val_da': [], 'val_total': []
        }

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            # === TRAINING ===
            self.model.train()
            sq_losses, abs_losses = [], []
            ys_true, ys_pred = [], []
            da_losses, total_losses = [], []

            for _ in tqdm(range(n_steps), desc=f"Epoch {epoch}/{self.epochs} [training]", leave=False):

                Xs, ys,  _ = next(it_s)
                Xs_da, _, _ = next(it_s_da)
                Xt, _,  _ = next(it_t)
                
                Xs = Xs.to(self.device); ys = ys.to(self.device)
                Xs_da = Xs_da.to(self.device)
                Xt = Xt.to(self.device)
                           
                self.optimizer.zero_grad()

                # predictions
                yhat_s, _ = self.model(Xs)
                _, feat_s = self.model(Xs_da)
                _, feat_t = self.model(Xt)

                # supervised source (weighted)
                task_loss = self.loss(yhat_s, ys)
                da_loss = self.transfer_criterion(feat_s, feat_t)
                w_da_loss = self.da_lambda*da_loss
                # total loss
                loss = task_loss + self.da_lambda * da_loss
                loss.backward()
                self.optimizer.step()

                # logging
                sq_losses.append(task_loss.item())
                abs_losses.append(F.l1_loss(yhat_s, ys).item())
                da_losses.append(w_da_loss.item())
                total_losses.append(loss.item())
                ys_true.append(ys.detach().cpu().numpy())
                ys_pred.append(yhat_s.detach().cpu().numpy())

            # metrics
            train_loss = float(np.mean(sq_losses))
            train_mse = train_loss
            train_mae = float(np.mean(abs_losses))
            y_true_tr = np.concatenate(ys_true)
            y_pred_tr = np.concatenate(ys_pred)
            ss_res_tr = np.sum((y_true_tr - y_pred_tr)**2)
            ss_tot_tr = np.sum((y_true_tr - y_true_tr.mean())**2)
            train_r2 = 1 - ss_res_tr / ss_tot_tr
            train_rmse = float(np.sqrt(train_mse)) 
            train_da   = float(np.mean(da_losses)) 
            train_total= float(np.mean(total_losses)) 

            # === VALIDATION ===
            self.model.eval()
            val_sq, val_abs = [], []
            ys_true, ys_pred = [], []

            # (1) MSE su source 
            with torch.no_grad():
                for Xs, ys, _ in tqdm(val_dl_source, desc=f"Epoch {epoch}/{self.epochs} [Validation:source]", leave=False):
                    Xs = Xs.to(self.device); ys = ys.to(self.device)
                    yhat_s, _ = self.model(Xs)
                    loss_s = self.loss(yhat_s, ys)
                    val_sq.append(loss_s.item())
                    val_abs.append(F.l1_loss(yhat_s, ys).item())
                    ys_true.append(ys.detach().cpu().numpy())
                    ys_pred.append(yhat_s.detach().cpu().numpy())


            val_loss = float(np.mean(val_sq)) 
            val_mse  = val_loss
            val_mae  = float(np.mean(val_abs)) 
            y_true_va = np.concatenate(ys_true) 
            y_pred_va = np.concatenate(ys_pred) 
            ss_res_va = np.sum((y_true_va - y_pred_va)**2)
            ss_tot_va = np.sum((y_true_va - y_true_va.mean())**2) 
            val_r2 = 1 - ss_res_va / ss_tot_va
            val_rmse = float(np.sqrt(val_mse)) 

            # (2) misura la distanza tra i due domini
            val_da_losses = []
            with torch.no_grad():
                it_vs = self.infinite_loader(val_dl_source)
                it_vs_da = self.infinite_loader(val_dl_source_da)
                it_vt = self.infinite_loader(val_dl_target)

                n_val_steps = max(len(val_dl_source), len(val_dl_target))

                for _ in tqdm(range(n_val_steps), desc=f"Epoch {epoch}/{self.epochs} [Validation:target-DA]", leave=False):
                    Xs, ys, _ = next(it_vs)
                    Xs_da, _, _ = next(it_vs_da)
                    Xt, _, _ = next(it_vt)

                    Xs = Xs.to(self.device)
                    Xs_da = Xs_da.to(self.device)
                    Xt = Xt.to(self.device)

                    _, feat_s = self.model(Xs_da)
                    _, feat_t = self.model(Xt)

                    da_val = self.transfer_criterion(feat_s, feat_t)
                    val_da_losses.append(da_val.item())

            val_da = float(np.mean(val_da_losses)) 
            val_total = val_loss + self.da_lambda * val_da
            w_val_da = self.da_lambda * val_da


            history['train_loss'].append(train_loss)
            history['train_mse'].append(train_mse)
            history['train_mae'].append(train_mae)
            history['train_r2'].append(train_r2)
            history['train_rmse'].append(train_rmse)
            history['val_loss'].append(val_loss)
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)
            history['val_rmse'].append(val_rmse)
            history['train_da'].append(train_da)
            history['train_total'].append(train_total)
            history['val_da'].append(w_val_da)
            history['val_total'].append(val_total)


            print(
                f"Epoch {epoch}/{self.epochs}  "
                f"train MSE: {train_mse:.4f}  MAE: {train_mae:.4f}  R²: {train_r2:.4f}  RMSE: {train_rmse:.4f}  "
                f"DA: {train_da:.4f}  TOTAL: {train_total:.4f} | "
                f"val   MSE: {val_mse:.4f}  MAE: {val_mae:.4f}  R²: {val_r2:.4f}  RMSE: {val_rmse:.4f}  "
                f"DA: {val_da:.4f}  TOTAL: {val_total:.4f}"
            )

            # === EARLY STOPPING LOGIC ===
            monitor = val_total
            if monitor < best_val_loss:
                best_val_loss = monitor
                epochs_no_improve = 0
                if restore_best_weights:
                    best_weights = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    if restore_best_weights and best_weights is not None:
                        self.model.load_state_dict(best_weights)
                    break
        
            # best weights 
            if restore_best_weights and best_weights is not None:
                self.model.load_state_dict(best_weights)

        self.history = history
        

        return history, self.model.state_dict(), best_val_loss


    
    def test_LSTM(self, 
            test_dl,
            y_scaler,
            y_feature):

            """Evaluate on test set and print MSE/RMSE/MAE/R²."""

            test_loader = self._get_loader(test_dl, shuffle=False)

            self.model.eval()
            sq_losses, abs_losses = [], []

            ys_true, ys_pred, all_indices = [], [], []
        
            

            with torch.no_grad():
                for X, y,  index_y in tqdm (test_loader, desc="Test Set"):
                    X, y,  = X.to(self.device), y.to(self.device)

                    out = self.model(X)
                    preds = out[0]

                    loss = self.loss(preds, y)
                    sq_losses.append(loss.item())
                    abs_losses.append(F.l1_loss(preds, y).item())

                    ys_true.append(y.cpu().numpy())
                    ys_pred.append(preds.cpu().numpy())
                    all_indices.append(index_y.numpy())  # <- raccogli gli indici
            
            y_true = np.concatenate(ys_true)
            y_pred = np.concatenate(ys_pred)
            all_indices = np.concatenate(all_indices)


            mse = np.mean(sq_losses)
            mae  = np.mean(abs_losses)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot


            print(f"Test MSE: {mse:.4f}  RMSE: {rmse:.4f} MAE: {mae:.4f}  R²: {r2:.4f}")

            # Align predictions with the original dataframe
            y_test_real = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
            y_test_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

            # Use indices to align with test_dataframe
            aligned_dataframe = pd.DataFrame({
                'index': all_indices,
                'y_true': y_test_real,
                'y_pred': y_test_pred 
            })
            #aligned_dataframe = np.arange(y_test_real)#test_dataframe.loc[indices]

            # Plot serie completa
            #fig = plot_full_series(
               # df_pred=aligned_dataframe,
               # title='Test: true vs. predicted'
            #)

            return  aligned_dataframe, {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


    




