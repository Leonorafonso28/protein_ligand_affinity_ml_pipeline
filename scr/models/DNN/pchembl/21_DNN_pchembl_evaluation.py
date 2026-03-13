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

#Load best hyperparams
import json
with open("data/models/DNN_pchembl/best_DNN_hyperparams_pchembl_classification.json", "r") as f:
    best_params = json.load(f)

# Build final model using best hyperparameters
def build_model(params):
    model = keras.Sequential()
    for i in range(params["n_layers"]):
        units = params[f"n_units_l{i}"]
        dropout = params[f"dropout_l{i}"]
        model.add(layers.Dense(units, activation=params["activation"],
                               kernel_regularizer=keras.regularizers.l2(params["weight_decay"])))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy"
    )
    return model

final_model = build_model(best_params)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=best_params["batch_size"],
    epochs=best_params["epochs"],
    verbose=1,
    callbacks=[early_stop]
)

# Evaluation function
def evaluate_metrics(y_true, y_probs, name=""):
    y_pred = (y_probs >= 0.5).astype(int)
    print(f"\n--- {name} ---")
    print(f"AUC:      {roc_auc_score(y_true, y_probs):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_true, y_pred):.4f}")
    print(f"Precision:{precision_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"MCC:      {matthews_corrcoef(y_true, y_pred):.4f}")

# Evaluate
evaluate_metrics(y_test, final_model.predict(X_test).flatten(), "Test Set")
evaluate_metrics(y_blind, final_model.predict(X_blind).flatten(), "Blind Set")
evaluate_metrics(y_blind_protein, final_model.predict(X_blind_protein).flatten(), "Blind Protein Set")
evaluate_metrics(y_blind_ligand, final_model.predict(X_blind_ligand).flatten(), "Blind Ligand Set")

# Save well-predicted examples from blind set
preds_blind_proba = final_model.predict(X_blind).flatten()
preds_blind_labels = (preds_blind_proba >= 0.5).astype(int)
correct_indices = np.where(preds_blind_labels == y_blind)[0]
well_predicted_df = pd.DataFrame(X_blind[correct_indices])
well_predicted_df["true_label"] = y_blind[correct_indices]
well_predicted_df["predicted_proba"] = preds_blind_proba[correct_indices]
well_predicted_df["predicted_label"] = preds_blind_labels[correct_indices]
well_predicted_df.to_csv("data/models/DNN_pchembl/well_predicted_blind_features.csv", index=False)
print(f"\nSaved {len(well_predicted_df)} well-predicted examples to 'well_predicted_blind_features.csv'.")