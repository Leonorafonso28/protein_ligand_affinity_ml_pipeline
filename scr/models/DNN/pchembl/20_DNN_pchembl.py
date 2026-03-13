import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import optuna
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef)

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Load the data
X_train = np.load("data/models/DNN_pchembl/X_train_pchembl.npy")
y_train = np.load("data/models/DNN_pchembl/y_train_pchembl.npy")
X_val = np.load("data/models/DNN_pchembl/X_val_pchembl.npy")
y_val = np.load("data/models/DNN_pchembl/y_val_pchembl.npy")
X_test = np.load("data/models/DNN_pchembl/X_test_pchembl.npy")
y_test = np.load("data/models/DNN_pchembl/y_test_pchembl.npy")
X_blind = np.load("data/models/DNN_pchembl/X_blind_pchembl.npy")
y_blind = np.load("data/models/DNN_pchembl/y_blind_pchembl.npy")
X_blind_protein = np.load("data/models/DNN_pchembl/X_blind_protein_pchembl.npy")
y_blind_protein = np.load("data/models/DNN_pchembl/y_blind_protein_pchembl.npy")
X_blind_ligand = np.load("data/models/DNN_pchembl/X_blind_ligand_pchembl.npy")
y_blind_ligand = np.load("data/models/DNN_pchembl/y_blind_ligand_pchembl.npy")

# Convert pchembl to binary: active (1) if >6.5, else inactive (0)
y_train = (y_train >= 6.5).astype(int)
y_val = (y_val >= 6.5).astype(int)
y_test = (y_test >= 6.5).astype(int)
y_blind = (y_blind >= 6.5).astype(int)
y_blind_protein = (y_blind_protein >= 6.5).astype(int)
y_blind_ligand = (y_blind_ligand >= 6.5).astype(int)

# Define the model builder
def create_model(trial):
    model = keras.Sequential()
    n_layers = trial.suggest_int("n_layers", 1, 5)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "elu"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    for i in range(n_layers):
        units = trial.suggest_int(f"n_units_l{i}", 16, 128)
        dropout = trial.suggest_float(f"dropout_l{i}", 0.0, 0.5)
        model.add(layers.Dense(units, activation=activation,
                               kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1))

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy" 
        )

    return model

# Define the objective function
def objective(trial):
    model = create_model(trial)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 50, 150)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stop]
    )

    preds = (model.predict(X_val).flatten() > 0.5).astype(int)
    f1 = f1_score(y_val, preds)
    return f1  

# Run optimization
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
with open("data/models/DNN_pchembl/best_DNN_hyperparams_pchembl_classification.json", "w") as f:
    json.dump(best_params, f, indent=4)