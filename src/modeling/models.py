import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf

"""
Anche in questo caso ci saranno delle modifiche nel modello per essere in grado di fare la domain adaptation 
"""

class PEMFC_GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.5):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
       
        self.fc1 = nn.Linear(hidden_size, hidden_size) # feature projector
        self.fc2 = nn.Linear(hidden_size, output_size) # task head ( regressore )

    def forward(self, x: torch.Tensor):
        h, _ = self.gru(x)    # [B, T, H]
        z = self.fc1(h)
        z = self.ln(z)
        z = torch.tanh(z)
        z = self.dropout_layer(z)
        feat = z[:, -1, :]         # [B,H]   
        y = self.fc2(feat)
        y = y.unsqueeze(-1) 
        return y, feat






class PEMFC_LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.5):
        
        super(PEMFC_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout = dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Predict next step

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        out = self.fc1(out)
        out = self.ln(out)
        out = torch.tanh(out)
        out = self.dropout_layer(out)
        return self.fc2(out[:, -1:, :])   # Get prediction from last time step

















