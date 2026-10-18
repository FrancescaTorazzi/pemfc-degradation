# salvataggio.py
from __future__ import annotations
import json

import pandas as pd
from omegaconf import OmegaConf
import torch

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
from datetime import datetime
import json, pickle, shutil
import pandas as pd
import numpy as np

ArrayLike = Union[pd.Series, pd.Index, np.ndarray, list, tuple]

#def _ensure_dir(p: Union[str, Path]) -> Path:
    #p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _to_serializable(x: Any) -> Any:
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, (int, float, str, bool)) or x is None: return x
    if isinstance(x, (list, tuple)): return [_to_serializable(i) for i in x]
    if isinstance(x, dict): return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, pd.DataFrame): return x.to_dict(orient="list")
    if isinstance(x, (pd.Series, pd.Index, np.ndarray)): return x.tolist()
    if isinstance(x, (pd.Series, pd.Index, np.ndarray)): return _to_serializable(x.tolist()) 
    return str(x)

def _unique_dir(base: Path, name: str) -> Path:
    d = base / name
    if not d.exists(): return d
    k = 2
    while True:
        cand = base / f"{name}-{k}"
        if not cand.exists(): return cand
        k += 1

def _maybe_copy(src: Optional[Union[str, Path]], dst: Path) -> Optional[str]:
    if not src: return None
    src = Path(src)
    if not src.exists(): return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)

def _write_preds_csv(dst_dir: Path, y_true: Optional[ArrayLike], y_pred: Optional[ArrayLike], fallback_src: Optional[Union[str, Path]]):
    dst = dst_dir / "predictions.csv"
    if fallback_src:
        return _maybe_copy(fallback_src, dst)
    if y_true is not None and y_pred is not None:
        df = pd.DataFrame({"y_true": pd.Series(y_true).reset_index(drop=True),
                           "y_pred": pd.Series(y_pred).reset_index(drop=True)})
        df.to_csv(dst, index=False); return str(dst)
    return None

def _write_metrics_json(dst_dir: Path, metrics: Optional[Mapping[str, Any]], fallback_src: Optional[Union[str, Path]]):
    dst = dst_dir / "metrics.json"
    if fallback_src:
        return _maybe_copy(fallback_src, dst)
    if metrics is not None:
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(dict(metrics)), f, indent=2, ensure_ascii=False)
        return str(dst)
    return None

def save_full_run(
    output_root: Union[str, Path],
    exp_name: str,
    *,
    # --- RNN ---
    rnn_y_true: Optional[ArrayLike] = None,
    rnn_y_pred: Optional[ArrayLike] = None,
    rnn_metrics: Optional[Mapping[str, Any]] = None,
    rnn_predictions_csv: Optional[Union[str, Path]] = None,  # se hai già il CSV
    rnn_metrics_json: Optional[Union[str, Path]] = None,     # se hai già il JSON
    rnn_plots: Optional[Mapping[str, Union[str, Path]]] = None,  # name->path dei PNG già salvati
    rnn_final_metrics_csv: Optional[Union[str, Path]] = None,
    rnn_model_path: Optional[Union[str, Path]] = None,       # salvo il path in un .txt
    rnn_extra: Optional[Mapping[str, Any]] = None,

    # --- POLY ---
    poly_y_true: Optional[ArrayLike] = None,
    poly_y_pred: Optional[ArrayLike] = None,
    poly_metrics: Optional[Mapping[str, Any]] = None,
    poly_predictions_csv: Optional[Union[str, Path]] = None,
    poly_metrics_json: Optional[Union[str, Path]] = None,
    poly_plots: Optional[Mapping[str, Union[str, Path]]] = None,
    poly_extra: Optional[Mapping[str, Any]] = None,

    overwrite_exp_dir: bool = False,   # se False e la cartella esiste → crea suffisso -2, -3...
) -> Path:
    """
    Organizza e salva quanto già prodotto in:
      <output_root>/<exp_name>/RNN/ ...  e  <output_root>/<exp_name>/POLY/ ...
    NON rigenera plot: copia i file esistenti e/o scrive CSV/JSON da dati passati.
    Ritorna il Path della cartella finale dell'esperimento.
    """
    root = _ensure_dir(output_root)
    exp_dir = (root / exp_name)
    if exp_dir.exists() and not overwrite_exp_dir:
        exp_dir = _unique_dir(root, exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---------- RNN ----------
    if any(v is not None for v in [rnn_y_true, rnn_y_pred, rnn_metrics, rnn_predictions_csv, rnn_metrics_json, rnn_plots, rnn_model_path]):
        rdir = _ensure_dir(exp_dir / "RNN")
        artifacts_rnn: Dict[str, str] = {}

        # predictions.csv (copio se già esiste, altrimenti scrivo da y_true/y_pred)
        p_csv = _write_preds_csv(rdir, rnn_y_true, rnn_y_pred, rnn_predictions_csv)
        if p_csv: artifacts_rnn["predictions_csv"] = p_csv

        # metrics.json (copio se già esiste, altrimenti scrivo da dict)
        m_json = _write_metrics_json(rdir, rnn_metrics, rnn_metrics_json)
        if m_json: artifacts_rnn["metrics_json"] = m_json

        # altri plot già salvati (li copio dentro RNN/)
        if rnn_plots:
            for name, src in rnn_plots.items():
                copied = _maybe_copy(src, rdir / f"{Path(src).name}")
                if copied: artifacts_rnn[f"plot_{name}"] = copied

        # final_metrics.csv se lo hai (lo copiò così com'è)
        if rnn_final_metrics_csv:
            copied = _maybe_copy(rnn_final_metrics_csv, rdir / "final_metrics.csv")
            if copied: artifacts_rnn["final_metrics_csv"] = copied

        # path pesi (scrivo solo il path, non copio)
        if rnn_model_path:
            (rdir / "model_path.txt").write_text(str(rnn_model_path))
            artifacts_rnn["model_path_txt"] = str(rdir / "model_path.txt")

        # manifest.pkl
        manifest_rnn = {
            "timestamp": ts, "exp_name": f"{exp_dir.name}/RNN",
            "artifacts": artifacts_rnn,
            "data": {
                "y_true_len": int(len(rnn_y_true)) if rnn_y_true is not None else None,
                "y_pred_len": int(len(rnn_y_pred)) if rnn_y_pred is not None else None,
                "metrics": _to_serializable(rnn_metrics) if rnn_metrics is not None else None,
            },
            "extra": _to_serializable(dict(rnn_extra) if rnn_extra else {}),
        }
        with open(rdir / "manifest.pkl", "wb") as f:
            pickle.dump(manifest_rnn, f)

    # ---------- POLY ----------
    if any(v is not None for v in [poly_y_true, poly_y_pred, poly_metrics, poly_predictions_csv, poly_metrics_json, poly_plots]):
        pdir = _ensure_dir(exp_dir / "POLY")
        artifacts_poly: Dict[str, str] = {}

        p_csv = _write_preds_csv(pdir, poly_y_true, poly_y_pred, poly_predictions_csv)
        if p_csv: artifacts_poly["predictions_csv"] = p_csv

        m_json = _write_metrics_json(pdir, poly_metrics, poly_metrics_json)
        if m_json: artifacts_poly["metrics_json"] = m_json

        if poly_plots:
            for name, src in poly_plots.items():
                copied = _maybe_copy(src, pdir / f"{Path(src).name}")
                if copied: artifacts_poly[f"plot_{name}"] = copied

        manifest_poly = {
            "timestamp": ts, "exp_name": f"{exp_dir.name}/POLY",
            "artifacts": artifacts_poly,
            "data": {
                "y_true_len": int(len(poly_y_true)) if poly_y_true is not None else None,
                "y_pred_len": int(len(poly_y_pred)) if poly_y_pred is not None else None,
                "metrics": _to_serializable(poly_metrics) if poly_metrics is not None else None,
            },
            "extra": _to_serializable(dict(poly_extra) if poly_extra else {}),
        }
        with open(pdir / "manifest.pkl", "wb") as f:
            pickle.dump(manifest_poly, f)

    return exp_dir







def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Set overwrite=True to replace.")
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, ensure_ascii=False, indent=2)


def _write_df_csv(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Set overwrite=True to replace.")
    df.to_csv(path, index=False)


def _save_figs(figs: Optional[Dict[str, Any]], out_dir: Path, overwrite: bool) -> None:
    """
    figs: dict { 'nome_file.png': matplotlib.figure.Figure }
    """
    if not figs:
        return
    for name, fig in figs.items():
        out_path = out_dir / name
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {out_path}. Set overwrite=True to replace.")
        # salva senza dipendenze extra
        fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _save_torch_state(state: Any, path: Path, overwrite: bool) -> None:
    """
    Salva i pesi solo se torch è disponibile e 'state' è valido.
    Accetta sia un model.state_dict() sia un torch.nn.Module (da cui estraiamo state_dict()).
    """
    if state is None:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Set overwrite=True to replace.")
    try:
        import torch  # import lazy per non richiedere torch se non necessario
    except Exception as e:
        raise RuntimeError("PyTorch non disponibile ma è stato richiesto il salvataggio dei pesi.") from e

    # Se ci passi direttamente il modulo, prendi .state_dict()
    if hasattr(state, "state_dict"):
        torch.save(state.state_dict(), path)
    else:
        torch.save(state, path)


def save_experiment(
    cfg: Any,
    *,
    rnn: Optional[Dict[str, Any]] = None,
    poly: Optional[Dict[str, Any]] = None,
    base_dir: str | Path = "runs",
    overrides: Optional[list[str]] = None,
    overwrite: bool = False,
    csv_sep: str=","
) -> Dict[str, Path]:
    """
    Salva i risultati ordinati in:
        runs/<exp_name>/{RNN, POLY}/ e confhydra/.

    Parametri attesi (tutti opzionali) nei dict `rnn` e `poly`:
      - 'metrics': dict               -> salvato come metrics.json
      - 'predictions_df': DataFrame   -> salvato come predictions.csv
        (oppure passare 'y_true' e 'y_pred' come array/list e verrà creato un DF)
      - 'figs': dict {filename.png: matplotlib.figure.Figure}
      - 'state_dict' OR 'model' (solo in rnn): pesi PyTorch da salvare come model.pth

    Ritorna un dict con i percorsi principali creati.
    """
    exp_name = getattr(cfg.output, "exp_name", None) if hasattr(cfg, "output") else None
    if not exp_name:
        # fallback: prova a leggere da dict stile OmegaConf
        try:
            exp_name = cfg["output"]["exp_name"]
        except Exception:
            raise ValueError("Non trovo cfg.output.exp_name. Assicurati che sia definito nella config.")

    base_dir = Path(base_dir)
    exp_dir = base_dir / exp_name
    rnn_dir = exp_dir / "RNN"
    conf_dir = exp_dir / "confhydra"

    # Crea cartelle
    _ensure_dir(rnn_dir)
    _ensure_dir(conf_dir)

    # --- confhydra ---
    # Config risolta
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg_path = conf_dir / "config_resolved.yaml"
    if cfg_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {cfg_path}. Set overwrite=True to replace.")
    cfg_path.write_text(config_yaml, encoding="utf-8")


    # Overrides usati a runtime (se li passi dal chiamante)
    if overrides:
        (conf_dir / "overrides.txt").write_text("\n".join(overrides), encoding="utf-8")

    # --- RNN ---
    if rnn is not None:
        # pesi
        if "state_dict" in rnn:
            _save_torch_state(rnn["state_dict"], rnn_dir / "model.pth", overwrite)
        elif "state_dict" in rnn:
            _save_torch_state(rnn["state_dict"], rnn_dir / "model.pth", overwrite)
        elif "model" in rnn:
            _save_torch_state(rnn["model"], rnn_dir / "model.pth", overwrite)
        
        # checkpoint pesi + history -> model_ckpt.pth (+ history.json)
        if "model_checkpoint" in rnn and rnn["model_checkpoint"] is not None:
            _save_torch_state(rnn["model_checkpoint"], rnn_dir / "model_ckpt.pth", overwrite)
            hist = rnn["model_checkpoint"].get("history")
            if hist is not None:
                _write_json(rnn_dir / "history.json", hist, overwrite)

        # metriche
        if "metrics" in rnn and rnn["metrics"] is not None:
            _write_json(rnn_dir / "metrics.json", rnn["metrics"], overwrite)

        # predictions
        if "predictions_df" in rnn and rnn["predictions_df"] is not None:
            rnn["predictions_df"].to_csv(rnn_dir / "predictions.csv",
                                         index=False, sep=(csv_sep))
        else:
            y_true = rnn.get("y_true")
            y_pred = rnn.get("y_pred")
            if y_true is not None and y_pred is not None:
                df_pred = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                df_pred.to_csv(rnn_dir /"predictions.csv",
                              index=False, sep=csv_sep)

        # eventuali figure
        _save_figs(rnn.get("figs"), rnn_dir, overwrite)

        if  rnn is not None and rnn.get("sliding_windows") is not None:
            sw_path = rnn_dir / "sliding_windows.pkl"
            if sw_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {sw_path}. Set overwrite=True to replace")
            with open(sw_path, "wb") as f:
                pickle.dump(rnn["sliding_windows"], f)

    return {
        "exp_dir": exp_dir,
        "rnn_dir": rnn_dir,
        "conf_dir": conf_dir,
        "config_yaml": cfg_path,
        "windows_pkl": sw_path if (rnn and rnn.get("sliding_windows") is not None) else None,
        "poly_params_pkl": poly_dir / "model_params.pkl" if (poly and poly.get("params") is not None) else None
    }