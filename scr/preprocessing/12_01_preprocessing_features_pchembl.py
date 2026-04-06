import pandas as pd
import numpy as np

features_total = pd.read_csv("data/features/features_combined_ligand_protein_pchembl.csv")
print(f"Initial shape before equal organisms: {features_total.shape}")

#Kepp only rows where Organism (PDB) == to target_organism (ChEMBL)
features_total = features_total[features_total["Organism"] == features_total["target_organism"]]
print(f"Initial shape after equal organisms: {features_total.shape}")

# Count Homo sapiens vs others
homo_sapiens_count = features_total[features_total["Organism"] == "Homo sapiens"].shape[0]
other_organisms_count = features_total.shape[0] - homo_sapiens_count

print(f"Homo sapiens entries: {homo_sapiens_count}")
print(f"Non-Homo sapiens entries: {other_organisms_count}")
print(f"Total: {features_total.shape[0]}")

#Kepp only rows where Organism (PDB) and target_organism (ChEMBL) == "Homo sapiens"
features_total = features_total[features_total["Organism"] == "Homo sapiens"]
print(f"Initial shape only Homo sapiens: {features_total.shape}")

columns_to_keep = [
    "PDB ID", "Chain ID", "UniProt_ID", "PDB_Start", "PDB_End", "UniProt_Start", "UniProt_End",
    "LIGAND_ID", "LIGAND_TYPE", "POSITIONS", "NAME", "SEQUENCE_3L_STD", "SEQUENCE_1L", 
    "CANONICAL_SMILES", "INCHIKEY", "PDB_ID", "Experimental_Method", "Resolution", "Organism",
    "molecule_chembl_id", "target_chembl_id", "assay_type", "pchembl_value", "relation",
    "standard_relation", "standard_units", "standard_value", "target_organism", "type", 
    "units", "value", "target_type", "pKi", "PDB_coverage", "z", "z_reason"
]

columns_to_keep = [col for col in columns_to_keep if col in features_total.columns]
features_meta = features_total[columns_to_keep]

# Select only numeric columns
features_total_numerics = features_total.select_dtypes(include=np.number)

# Remove columns with NaN values
features_total_numerics = features_total_numerics.dropna(axis=1)

# Remove columns with zero variance
features_total_nv = features_total_numerics.loc[:, features_total_numerics.var() != 0]

# Save dataset
final_dataset = pd.concat([features_meta, features_total_nv], axis=1)
final_dataset.to_csv("data/processed/features_processed_pchembl.csv", index=False)
print(f"Final shape: {features_total_nv.shape}") 