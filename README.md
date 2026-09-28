# Weather Intelligence API

**Student:** Nandeeswari Kumaravelu
**Student ID:** 25980436
**Course:** 36120 — Advanced Machine Learning Application (AT2)

## Setup and reproduction

**Requirements:** Python 3.12, git, internet access (Open-Meteo API and TestPyPI).

### 1. Create the environment

```bash
git clone <this-repo-url>
cd 36120-26SP-AT2-25980436-experiments
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn xgboost requests joblib ipykernel jupyter
```

### 2. Install the custom package from TestPyPI

The notebooks import `nba_draft_features` (version 0.3.0, extended from AT1 with
`aggregation.py` and `targets.py`). `--no-deps` is used because the dependencies were
installed from the real PyPI in step 1.

```bash
python -m pip install --no-deps -i https://test.pypi.org/simple/ nba-draft-features==0.3.0
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=at2-venv312 --display-name="AT2 (Python 3.12)"
```

In VS Code or JupyterLab, select the kernel **AT2 (Python 3.12)** when opening a notebook.

### 4. Run order

1. `python fetch_weather.py` pulls hourly Sydney data (2015-2025) from the Open-Meteo
   Historical Weather API, aggregates it to daily values and computes CCI and WHC.
   Outputs go to `data/raw/` and `data/processed/`. The processed files are already
   committed, so this step is optional.
2. Run all cells in `notebooks/comfort_climate/36120-26SP-AT2-25980436-comfort_climate.ipynb`.
3. Run all cells in `notebooks/weather_hazard/36120-26SP-AT2-25980436-weather_hazard.ipynb`.
   Each notebook saves its trained model to `models/<target>/` and its metrics to a
   `*_metrics.json` file (final cell).
4. `python export_for_api.py` writes the feature lookup tables used by the API to
   `models/comfort_climate/` and `models/weather_hazard/`.

### 5. Deployment

The trained models are served by a separate FastAPI repository:
https://github.com/nandeeswari19-web/36120-26SP-AT2-25980436-api

Deployed API (Render): `https://three6120-26sp-at2-25980436-api.onrender.com`

### Data and leakage notes

Only data before 2026 is used for training, validation and testing. The train,
validation and test sets are split chronologically (70/15/15) to avoid leaking future
information through the lag and rolling features.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Assignment 2

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         weather_intelligence_mod and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── weather_intelligence_mod   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes weather_intelligence_mod a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

