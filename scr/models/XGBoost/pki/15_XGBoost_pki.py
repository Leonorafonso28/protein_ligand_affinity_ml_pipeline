import optuna
import xgboost as xgb
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
import numpy
import random
from sklearn.metrics import classification_report
import pandas as pd

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Load and binarize data
X_train = np.load("data/models/XGBoost_pKi/X_train_pki.npy")
y_train_reg = np.load("data/models/XGBoost_pKi/y_train_pki.npy")
X_val = np.load("data/models/XGBoost_pKi/X_val_pki.npy")
y_val_reg = np.load("data/models/XGBoost_pKi/y_val_pki.npy")

# Convert pKi to binary: active (1) if >6.5, else inactive (0)
y_train = (y_train_reg >= 6.5).astype(int)
y_val = (y_val_reg >= 6.5).astype(int)

# Optuna objective function
def objective(trial):
    params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "lambda": trial.suggest_float("lambda", 1e-5, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_seed": 42
    }

    model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


    y_pred = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    return f1

# Run Optuna
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)) 
study.optimize(objective, n_trials=400)

# Retrieve best parameters
best_params = study.best_trial.params
best_f1 = study.best_trial.value
print("Best params:")
print(best_params)
print(f"Best Validation F1: {best_f1:.4f}")

# Save best hyperparameters to JSON
import json
with open("data/models/XGBoost_pKi/best_xgboost_hyperparams_pki_classification1.json", "w") as f:
    json.dump(best_params, f, indent=4)