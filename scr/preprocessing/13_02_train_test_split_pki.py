import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import random

random.seed(42)
np.random.seed(42)

# Load and clean the dataset
df = pd.read_csv("features_processed_pKi.csv")
df["CANONICAL_SMILES"] = df["CANONICAL_SMILES"].astype(str).str.strip()
df["SEQUENCE_1L"] = df["SEQUENCE_1L"].astype(str).str.strip()
df = df[df["CANONICAL_SMILES"].notna() & df["SEQUENCE_1L"].notna()].copy()

# Sample 5% unique SMILES and SEQUENCES
unique_smiles = df["CANONICAL_SMILES"].unique()
unique_sequences = df["SEQUENCE_1L"].unique()
sampled_smiles = set(np.random.choice(unique_smiles, int(0.15 * len(unique_smiles)), replace=False))
sampled_sequences = set(np.random.choice(unique_sequences, int(0.15 * len(unique_sequences)), replace=False))

#  Create blind_protein and blind_ligand
blind_protein = df[df["SEQUENCE_1L"].isin(sampled_sequences)].copy()
blind_ligand = df[df["CANONICAL_SMILES"].isin(sampled_smiles)].copy()

# Blind: Only keep rows where both SMILES and SEQUENCE are in the sampled sets
df_blind = df[df["CANONICAL_SMILES"].isin(sampled_smiles) & df["SEQUENCE_1L"].isin(sampled_sequences)].copy()

blind_protein = blind_protein[
    ~((blind_protein["SEQUENCE_1L"].isin(df_blind["SEQUENCE_1L"])) & 
      (blind_protein["CANONICAL_SMILES"].isin(df_blind["CANONICAL_SMILES"])))
]

blind_ligand = blind_ligand[
    ~((blind_ligand["SEQUENCE_1L"].isin(df_blind["SEQUENCE_1L"])) & 
      (blind_ligand["CANONICAL_SMILES"].isin(df_blind["CANONICAL_SMILES"])))
]
# Remove all rows from df that contain the sampled SMILES or SEQUENCES
df_rest = df[~df["CANONICAL_SMILES"].isin(sampled_smiles) & ~df["SEQUENCE_1L"].isin(sampled_sequences)].copy()

# Add pair_key to all sets
for d in [df_rest, df_blind, blind_ligand, blind_protein]:
    d["pair_key"] = d["PDB ID"].astype(str) + "|" + d["Chain ID"].astype(str) + "|" + d["SEQUENCE_1L"] + "|" + d["LIGAND_ID"].astype(str)

# Split df_rest into train/val/test
train_val_df, test_df = train_test_split(df_rest, test_size=0.10, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)

# Targets
y_train = train_df["pKi"].values
y_val = val_df["pKi"].values
y_test = test_df["pKi"].values
y_blind = df_blind["pKi"].values
y_blind_protein = blind_protein["pKi"].values
y_blind_ligand = blind_ligand["pKi"].values

# Drop columns
drop_cols = ["pKi", "pair_key", "PDB ID", "Chain ID", "UniProt_ID", "PDB_Start", "PDB_End", "UniProt_Start", "UniProt_End",
             "LIGAND_ID", "LIGAND_TYPE", "POSITIONS", "NAME", "SEQUENCE_3L_STD", "SEQUENCE_1L", "CANONICAL_SMILES",
             "INCHIKEY", "PDB_ID", "Experimental_Method", "Resolution", "Organism", "molecule_chembl_id", "target_chembl_id",
             "assay_type", "pchembl_value", "relation", "standard_relation", "standard_units", "standard_value",
             "target_organism", "type", "units", "value", "target_type", "PDB_coverage", "z", "z_reason",
             "UniProt_Start.1", "UniProt_End.1", "standard_value.1", "value.1", "pKi.1", "pchembl_value.1"]

def clean(dfset):
    return dfset.drop(columns=[col for col in drop_cols if col in dfset.columns])

X_train = clean(train_df)
X_val = clean(val_df)
X_test = clean(test_df)
X_blind = clean(df_blind)
X_blind_protein = clean(blind_protein)
X_blind_ligand = clean(blind_ligand)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_blind = scaler.transform(X_blind)
X_blind_protein = scaler.transform(X_blind_protein)
X_blind_ligand = scaler.transform(X_blind_ligand)

# Save outputs
np.save("X_train_pki.npy", X_train)
np.save("y_train_pki.npy", y_train)
np.save("X_val_pki.npy", X_val)
np.save("y_val_pki.npy", y_val)
np.save("X_test_pki.npy", X_test)
np.save("y_test_pki.npy", y_test)
np.save("X_blind_pki.npy", X_blind)
np.save("y_blind_pki.npy", y_blind)
np.save("X_blind_protein_pki.npy", X_blind_protein)
np.save("y_blind_protein_pki.npy", y_blind_protein)
np.save("X_blind_ligand_pki.npy", X_blind_ligand)
np.save("y_blind_ligand_pki.npy", y_blind_ligand)

# Final sanity check
results = {
    "Train": len(train_df),
    "Val": len(val_df),
    "Test": len(test_df),
    "Blind": len(df_blind),
    "Blind_protein": len(blind_protein),
    "Blind_lig": len(blind_ligand),
    "Overlap Pair Keys Blind": len(set(df_rest["pair_key"]) & set(df_blind["pair_key"])),
    "Overlap SMILES Blind": len(set(df_rest["CANONICAL_SMILES"]) & set(df_blind["CANONICAL_SMILES"])),
    "Overlap Sequences Blind": len(set(df_rest["SEQUENCE_1L"]) & set(df_blind["SEQUENCE_1L"])),
    "Overlap Pair Keys Blind Protein": len(set(df_rest["pair_key"]) & set(blind_protein["pair_key"])),
    "Overlap SMILES Blind Protein": len(set(df_rest["CANONICAL_SMILES"]) & set(blind_protein["CANONICAL_SMILES"])),
    "Overlap Sequences Blind Protein": len(set(df_rest["SEQUENCE_1L"]) & set(blind_protein["SEQUENCE_1L"])),
    "Overlap Pair Keys Blind Lig": len(set(df_rest["pair_key"]) & set(blind_ligand["pair_key"])),
    "Overlap SMILES Blind Lig": len(set(df_rest["CANONICAL_SMILES"]) & set(blind_ligand["CANONICAL_SMILES"])),
    "Overlap Sequences Blind Lig": len(set(df_rest["SEQUENCE_1L"]) & set(blind_ligand["SEQUENCE_1L"]))
}

print(results)