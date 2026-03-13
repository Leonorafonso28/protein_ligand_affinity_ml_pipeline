import optuna
import xgboost as xgb
import numpy as np
import joblib
import json
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
y_train =( np.load("data/models/XGBoost_pKi/y_train_pki.npy") >= 6.5).astype(int)
X_val = np.load("data/models/XGBoost_pKi/X_val_pki.npy")
y_val= (np.load("data/models/XGBoost_pKi/y_val_pki.npy") >= 6.5).astype(int)

#Load best hyperparams 
with open("data/models/XGBoost_pKi/best_xgboost_hyperparams_pki_classification1.json", "r") as f:
    best_params = json.load(f)
    
#Build final model using best hyperparameters
final_model = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=SEED)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=1)

# Save final model
joblib.dump(final_model, "data/models/XGBoost_pKi/xgboost_pki_classifier1.pkl")

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
    print(classification_report(y_true, y_pred, digits=4))

# Evaluate on test set
X_test = np.load("data/models/XGBoost_pKi/X_test_pki.npy")
y_test = (np.load("data/models/XGBoost_pKi/y_test_pki.npy") > 6.5).astype(int)
preds_test_proba = final_model.predict_proba(X_test)[:, 1]
evaluate_metrics(y_test, preds_test_proba, name="Test Set")

# Evaluate on blind set
X_blind = np.load("data/models/XGBoost_pKi/X_blind_pki.npy")
y_blind = (np.load("data/models/XGBoost_pKi/y_blind_pki.npy") > 6.5).astype(int)
preds_blind_proba = final_model.predict_proba(X_blind)[:, 1]
evaluate_metrics(y_blind, preds_blind_proba, name="Blind Set (no overlap protein or smiles")

# Identify correct predictions
preds_blind_labels = (preds_blind_proba >= 0.5).astype(int)
correct_indices = np.where(preds_blind_labels == y_blind)[0]

# Extract feature rows for correctly predicted examples
well_predicted_features = X_blind[correct_indices]
true_labels = y_blind[correct_indices]
predicted_probs = preds_blind_proba[correct_indices]

# Create DataFrame
well_predicted_df = pd.DataFrame(well_predicted_features)
well_predicted_df["true_label"] = true_labels
well_predicted_df["predicted_proba"] = predicted_probs
well_predicted_df["predicted_label"] = preds_blind_labels[correct_indices]

# Save to CSV
well_predicted_df.to_csv("data/models/XGBoost_pKi/well_predicted_blind_features1.csv", index=False)
print(f"Saved {len(well_predicted_df)} well-predicted examples from blind set to 'well_predicted_blind_features.csv'.")

# Evaluate on blind protein set
X_blind_protein = np.load("data/models/XGBoost_pKi/X_blind_protein_pki.npy")
y_blind_protein = (np.load("data/models/XGBoost_pKi/y_blind_protein_pki.npy") > 6.5).astype(int)
preds_blind_protein_proba = final_model.predict_proba(X_blind_protein)[:, 1]
evaluate_metrics(y_blind_protein, preds_blind_protein_proba, name="Blind Protein Set")

# Evaluate on blind ligand set
X_blind_ligand = np.load("data/models/XGBoost_pKi/X_blind_ligand_pki.npy")
y_blind_ligand = (np.load("data/models/XGBoost_pKi/y_blind_ligand_pki.npy") > 6.5).astype(int)
preds_blind_ligand_proba = final_model.predict_proba(X_blind_ligand)[:, 1]
evaluate_metrics(y_blind_ligand, preds_blind_ligand_proba, name="Blind Ligand Set")