import pandas as pd
from autogluon.tabular import TabularPredictor
import torch
import os

# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = r"C:\Users\Hzaab\Desktop\MLSD project\data\raw\train.csv"
TARGET = "fake"

# Where AutoGluon will save everything
AUTOML_PATH = r"C:\Users\Hzaab\Desktop\MLSD project\scratch\autogluon_f1"

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
train_df = train_df.drop(columns=[train_df.columns[0]])

os.environ["RAY_DISABLE_DASHBOARD"] = "1"

predictor = TabularPredictor(
    label=TARGET,
    path=AUTOML_PATH,
    eval_metric="f1",
)

hyperparameters = {
    "REALTABPFN-V2": {},
    "TABICL": {},
    "TABM": {},
    "CAT": {},
    "GBM_PREP": [
        {"device": "cpu"},
    ],
    "GBM": [
        {"device": "cpu"},
    ],
    # Disable the families that are crashing:
    "TABDPT": [],
    "MITRA": [],
}

predictor.fit(
    train_data=train_df,
    presets="extreme_quality",
    hyperparameters=hyperparameters,
    fit_strategy="sequential",
    num_gpus=1,
    num_cpus=8,
    memory_limit=10,
    ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
)