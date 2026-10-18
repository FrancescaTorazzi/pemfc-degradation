import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def plot_history(history: dict,
                 metrics: list = ['rmse', 'mae', 'r2'],
                 title: str = 'Training History') -> None:
    """
    Improved train vs val metrics history with grid and distinct line styles.
    """
    df_hist = pd.DataFrame(history)
    df_hist.index.name = 'epoch'
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        ax.plot(df_hist.index, df_hist[f'train_{m}'], label='train', linestyle='-', linewidth=2)
        ax.plot(df_hist.index, df_hist[f'val_{m}'], label='val', linestyle='--', linewidth=2)
        ax.set_title(m.upper(), fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(m, fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    #plt.show()
    return fig

def plot_da_total(history: dict,
                  title: str = 'Domain Adaptation & Total Loss') -> None:
    """
    Plot Domain Adaptation (DA) and Total Loss history, same style as plot_history.
    Left panel: train_da vs val_da (blue)
    Right panel: train_total vs val_total (blue/orange)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df_hist = pd.DataFrame(history)
    df_hist.index.name = 'epoch'

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # --- Left: DA loss ---
    ax = axes[0]
    ax.plot(df_hist.index, df_hist['train_da'], label='train_da',  linestyle='-', linewidth=2)
    ax.plot(df_hist.index, df_hist['val_da'], label='val_da', linestyle='--', linewidth=2)
    ax.set_title('DA LOSS', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- Right: Total loss ---
    ax = axes[1]
    ax.plot(df_hist.index, df_hist['train_total'], label='train_total', color='tab:blue', linestyle='-', linewidth=2)
    ax.plot(df_hist.index, df_hist['val_total'], label='val_total', color='tab:orange', linestyle='--', linewidth=2)
    ax.set_title('TOTAL LOSS', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig


def plot_metrics_summary(metrics_df: pd.DataFrame,
                         title: str = 'Metrics Summary') -> None:
    """
    Improved bar chart of metrics (rmse, mae, r2) with annotations and grid.
    """
    metrics = metrics_df.columns.tolist()
    values = metrics_df.iloc[0].values
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    for ax, m, v in zip(np.atleast_1d(axes), metrics, values):
        ax.barh([0], [v], height=0.4)
        ax.set_title(m.upper(), fontsize=12)
        ax.set_xlabel(m, fontsize=10)
        ax.set_yticks([])
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.text(v, 0, f'{v:.3f}', va='center', ha='left', fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    #plt.show()
    return fig

def plot_full_series(df_pred, title="True vs. Predicted", figsize=(12, 6)):
    """
    Plotta la serie temporale completa con y_true e y_pred.
    """
    plt.figure(figsize=figsize)
    plt.plot(df_pred['index'], df_pred['y_true'], label='y_true', linewidth=2)
    plt.plot(df_pred['index'], df_pred['y_pred'], label='y_pred', linewidth=2, linestyle='--')
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    return fig

def plot_residuals(df: pd.DataFrame,
                   title: str = 'Residuals Analysis',
                   figsize: tuple[float, float] = (12, 5)) -> None:
    """
    New plot: scatter of residuals vs index and histogram of residuals.
    """
    residuals = df['y_Real'] - df['y_Predicted']
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter residuals vs index
    axes[0].scatter(df.index, residuals, s=10, alpha=0.7)
    axes[0].axhline(0, linestyle='--', linewidth=1)
    axes[0].set_title('Residuals vs Index', fontsize=12)
    axes[0].set_xlabel('Index', fontsize=10)
    axes[0].set_ylabel('Residuals', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Histogram of residuals
    axes[1].hist(residuals.dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_title('Residuals Distribution', fontsize=12)
    axes[1].set_xlabel('Residual', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    #plt.show()
    return fig
def plot_radar(metrics_dict, title="Radar Plot"):
    # Estrai le label dalle colonne del primo DataFrame
    first_df = next(iter(metrics_dict.values()))
    labels = first_df.columns.tolist()
    num_vars = len(labels)

    # Angoli per i raggi del radar
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # chiudi il cerchio

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name, df in metrics_dict.items():
        values = df.iloc[0].tolist()  # una sola riga
        values += values[:1]  # chiudi il cerchio

        ax.plot(angles, values, label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, size=15)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_sliding_windows_split(train_ds, ranges_train, df_train, train_files):
    """
    Plotta, in un'unica figura, le sliding windows usate in train e validation per ciascun esperimento.
    Ogni subplot rappresenta un esperimento.
    """
    starts_train = train_ds.starts_train
    starts_val = train_ds.starts_val
    T_x = train_ds.T_x

    # Mappa degli start -> tipo (train/val)
    start_dict = {idx: 'train' for idx in starts_train}
    start_dict.update({idx: 'val' for idx in starts_val})

    n_exp = len(ranges_train)
    fig, axs = plt.subplots(n_exp, 1, figsize=(13, 4 * n_exp), sharex=False)

    if n_exp == 1:
        axs = [axs]  # per uniformità anche se c'è un solo esperimento

    for ax, name, (start_idx, end_idx, _) in zip(axs, train_files, ranges_train):
        df_exp = df_train.iloc[start_idx:end_idx + 1]
        x_vals = list(range(start_idx, end_idx + 1))

        ax.plot(x_vals, df_exp['CellV_avg'].values, color='lightgray', lw=1, label='Signal')

        for j in range(start_idx, end_idx - T_x + 1):
            if j in start_dict:
                w = df_train['CellV_avg'].iloc[j: j + T_x].values
                c = 'green' if start_dict[j] == 'train' else 'red'
                ax.plot(range(j, j + T_x), w, color=c, alpha=0.6, lw=1)

        train_patch = mpatches.Patch(color='green', label='Train')
        val_patch = mpatches.Patch(color='red', label='Validation')
        ax.set_title(f"{name}  (start={start_idx}, end={end_idx})", fontsize=14, weight='bold')
        ax.set_xlabel("Index", fontsize=11)
        ax.set_ylabel('CellV_avg', fontsize=11)
        ax.legend(handles=[train_patch, val_patch], loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Sliding Windows: Train vs Validation", fontsize=16)
    return fig


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_scatter_ytrue_vs_ypred_from_df(df_pred: pd.DataFrame, 
                                        title: str = "True vs Predicted (Test Set)", 
                                        save_path: str = None):
    """
    Crea uno scatter plot y_true vs y_pred con diagonale 45°, usando un DataFrame allineato.
    
    Parametri:
    - df_pred: DataFrame con colonne ['y_true', 'y_pred']
    - title: titolo del grafico
    - save_path: se specificato, salva il grafico in quel percorso
    """
    y_true = df_pred['y_true'].values
    y_pred = df_pred['y_pred'].values

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    #plt.show()
    return fig

def plot_true_vs_pred_series(
    df_pred: pd.DataFrame,
    title: str = "True vs Predicted on Test Set",
    ranges_test=None,
):
    """
    Plotta la serie temporale dei valori reali vs predetti (layout uniforme RNN).
    df_pred deve avere colonne: ['index', 'y_true', 'y_pred'].
    ranges_test: tupla (start_idx, end_idx, start_impurity_idx) oppure lista di tuple.
    """
    fig = plt.figure(figsize=(13, 4))
    plt.plot(df_pred["index"], df_pred["y_true"], label="True", linewidth=2)
    plt.plot(df_pred["index"], df_pred["y_pred"], label="Predicted", linewidth=2)

    # linee verticali inizio CO
    if ranges_test is not None:
        starts = [ranges_test[-1]] if isinstance(ranges_test, tuple) else [t[-1] for t in ranges_test]
        first = True
        for x in starts:
            plt.axvline(x=x, linestyle="--", color="gray", alpha=0.9, linewidth=1.5,
                        label="CO start" if first else None)
            first = False

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Index", fontsize=11)
    plt.ylabel("Target", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_history_log(history: dict,
                 metrics: list = ['loss', 'mse', 'rmse', 'mae', 'r2'],
                 title: str = 'Training History') -> None:
    """
    Improved train vs val metrics history with grid and distinct line styles.
    Uses logarithmic scale only for 'loss'.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df_hist = pd.DataFrame(history)
    df_hist.index.name = 'epoch'
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)

    for ax, m in zip(np.atleast_1d(axes), metrics):
        ax.plot(df_hist.index, df_hist[f'train_{m}'], label='train', linestyle='-', linewidth=2)
        ax.plot(df_hist.index, df_hist[f'val_{m}'], label='val', linestyle='--', linewidth=2)
        ax.set_title(m.upper(), fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(m, fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        # 🔹 Scala logaritmica solo per la LOSS
        if m.lower() == 'loss':
            ax.set_yscale('log')
            ax.set_ylabel(f'{m} (log scale)', fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

def plot_histories(history: dict, title: str = "Training Summary"):
    """
    2x2:
    (1) Regression Loss   (train_loss vs val_loss)
    (2) MSE               (train_mse  vs val_mse)
    (3) DA Loss           (train_da   vs val_da)
    (4) Total Loss        (train_total vs val_total)
    """
    df = pd.DataFrame(history)
    df.index.name = "epoch"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # (1) Regression Loss
    ax = axes[0, 0]
    ax.plot(df.index, df["train_loss"], label="train", linewidth=2)
    ax.plot(df.index, df["val_loss"],   label="val",   linestyle="--", linewidth=2)
    ax.set_title("Regression Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5); ax.legend()

    # (2) MSE
    ax = axes[0, 1]
    ax.plot(df.index, df["train_mse"], label="train", linewidth=2)
    ax.plot(df.index, df["val_mse"],   label="val",   linestyle="--", linewidth=2)
    ax.set_title("MSE"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.grid(True, linestyle="--", alpha=0.5); ax.legend()

    # (3) DA Loss
    ax = axes[1, 0]
    ax.plot(df.index, df["train_da"], label="train", linewidth=2)
    ax.plot(df.index, df["val_da"],   label="val",   linestyle="--", linewidth=2)
    ax.set_title("DA Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5); ax.legend()

    # (4) Total Loss
    ax = axes[1, 1]
    ax.plot(df.index, df["train_total"], label="train", linewidth=2)
    ax.plot(df.index, df["val_total"],   label="val",   linestyle="--", linewidth=2)
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5); ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

import matplotlib.pyplot as plt
import pandas as pd

def plot_source_test(df_source: pd.DataFrame,
                     df_test: pd.DataFrame,
                     ranges_source=None,
                     ranges_test=None,
                     title: str = "Predictions: Source vs Test"):
    """
    Grid 1x2: sinistra = source, destra = test.
    Ogni df deve avere colonne ['index', 'y_true', 'y_pred'].
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    # --- Source ---
    ax = axes[0]
    ax.plot(df_source["index"], df_source["y_true"], label="True", linewidth=2)
    ax.plot(df_source["index"], df_source["y_pred"], label="Predicted", linewidth=2)
    if ranges_source is not None:
        starts = [ranges_source[-1]] if isinstance(ranges_source, tuple) else [t[-1] for t in ranges_source]
        for x in starts:
            ax.axvline(x=x, linestyle="--", color="gray", alpha=0.9, linewidth=1.5)
    ax.set_title("Source", fontsize=12)
    ax.set_xlabel("Index"); ax.set_ylabel("Target")
    ax.legend(); ax.grid(alpha=0.3)

    # --- Test ---
    ax = axes[1]
    ax.plot(df_test["index"], df_test["y_true"], label="True", linewidth=2)
    ax.plot(df_test["index"], df_test["y_pred"], label="Predicted", linewidth=2)
    if ranges_test is not None:
        starts = [ranges_test[-1]] if isinstance(ranges_test, tuple) else [t[-1] for t in ranges_test]
        for x in starts:
            ax.axvline(x=x, linestyle="--", color="gray", alpha=0.9, linewidth=1.5, label="CO start")
    ax.set_title("Test", fontsize=12)
    ax.set_xlabel("Index"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
