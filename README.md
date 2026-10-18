# PEMFC Degradation Modelling with Domain Adaptation

Research code developed for an MSc thesis on data-driven degradation modelling of proton exchange membrane fuel cells (PEMFCs).

The project uses recurrent neural networks to model degradation from multivariate time-series data. It includes GRU and LSTM architectures and supports unsupervised domain adaptation between different operating domains using CORAL, MMD, and DARE-based objectives.

## Repository structure

```text
src/
├── conf/config.yaml       # Experiment configuration
├── main.py                # Training and evaluation entry point
├── import_data.py         # Data loading and preprocessing
├── dataset.py             # Time-series window generation
├── modeling/
│   ├── models.py          # GRU and LSTM models
│   ├── trainer.py         # Training and validation
│   ├── losses.py          # Domain-adaptation losses
│   └── predictions.py     # Predictions and metrics
├── plots.py               # Visualisation utilities
├── savings.py             # Experiment output management
└── utils.py               # Preprocessing and reproducibility utilities
```

## Data availability

The experimental datasets are not included because they contain confidential research data. Data files, exploratory notebooks, trained models, and experiment outputs are excluded from version control.

The code therefore requires a compatible local dataset and appropriate configuration before it can be run.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Set the local data references and model parameters in `src/conf/config.yaml`, then run:

```bash
python -m src.main
```

## Methods

- GRU and LSTM models for multivariate time-series regression
- Sliding-window sequence construction
- Supervised training on a source domain
- Unsupervised adaptation to a target domain
- CORAL, maximum mean discrepancy (MMD), and DARE-based losses

## Disclaimer

This repository contains research code developed for an academic thesis. The experimental data and detailed test conditions are not publicly available.
