import pandas as pd
from scipy.stats import zscore

# Step 1: Load and filter dataset
df = pd.read_csv("data/features/ChEMBL_activities.csv")
df = df.drop_duplicates()

# Step 2: Keep only single protein binding assays with valid pchembl_value
chembl_df = df[
    (df["target_type"] == "SINGLE PROTEIN") &
    (df["assay_type"] == "B") &
    (df["pchembl_value"].notna())
].copy()

# Step 3: Compute sequence coverage and ensure Resolution is numeric
chembl_df["PDB_coverage"] = (
    chembl_df["PDB_End"] - chembl_df["PDB_Start"] + 1
) / (chembl_df["UniProt_End"] - chembl_df["UniProt_Start"] + 1)
chembl_df["Resolution"] = pd.to_numeric(chembl_df["Resolution"], errors="coerce")

# Step 4: Sort by best coverage and resolution
# Step 4: Remove duplicated PDB chains with same sequence (keep best by coverage and resolution)
chembl_df = chembl_df.sort_values(
    by=["SEQUENCE_1L", "PDB ID", "Chain ID", "PDB_coverage", "Resolution"],
    ascending=[True, True, True, False, True]
)
chembl_df = chembl_df.drop_duplicates(
    subset=["SEQUENCE_1L", "PDB ID", "Chain ID"],
    keep="first"
)

# Step 5: Keep best structure per (UniProt_ID, molecule_chembl_id, pchembl_value)
representative_df =chembl_df.drop_duplicates(
    subset=["molecule_chembl_id", "pchembl_value", "SEQUENCE_1L"]
)

# Step 6: Group by interaction (UniProt_ID + molecule_chembl_id)
interaction_group = representative_df.groupby(["molecule_chembl_id", "SEQUENCE_1L"])

# Step 7: Separate and process duplicates using z-score logic
deduplicated_entries = []
retained_entries = []

for (mol, seq), group in interaction_group:
    unique_pchembl = group["pchembl_value"].nunique()

    if unique_pchembl == 1:
        # All values identical → keep one and tag reason
        entry = group.iloc[[0]].copy()
        entry["z"] = float("nan")
        entry["z_reason"] = "identical_or_single_value"
        retained_entries.append(entry)
    else:
        # Apply z-score to pchembl_value
        group = group.copy()
        try:
            group["z"] = zscore(group["pchembl_value"])
        except Exception:
            group["z"] = float("nan")

        # Check for all NaN z-scores
        if group["z"].isna().all():
            # Tag the NaN reason
            group["z_reason"] = "identical_or_single_value"
            selected = group.iloc[[0]]
        else:
            # Filter non-outliers
            filtered = group[group["z"].abs() <= 1]
            if not filtered.empty:
                selected = filtered.loc[[filtered["z"].abs().idxmin()]]
                selected["z_reason"] = "within_range"
            else:
                # All outliers → take closest to mean
                selected = group.loc[[group["pchembl_value"].sub(group["pchembl_value"].mean()).abs().idxmin()]]
                selected["z_reason"] = "outlier_kept"

        deduplicated_entries.append(selected)

# Step 8: Combine final dataset
final_df = pd.concat(retained_entries + deduplicated_entries, ignore_index=True)

# Step 9: Save and log
final_df.to_csv("data/features/deduplicated_filtered_pchembl_dataset.csv", index=False)
print(f"Initial filtered rows: {len(chembl_df)}")
print(f"Final deduplicated rows: {len(final_df)}")

duplicates_remaining = final_df.groupby(["UniProt_ID", "molecule_chembl_id", "SEQUENCE_1L"]).size()
multi_entry_pairs = duplicates_remaining[duplicates_remaining > 1]
print(f"Number of UniProt_ID - molecule_chembl_id - sequence variants with multiple entries: {len(multi_entry_pairs)}")

duplicates= final_df.groupby(["molecule_chembl_id", "SEQUENCE_1L"]).size()
multi_entry_pairs = duplicates[duplicates > 1]
print(f"Number of molecule_chembl_id - sequence with multiple entries: {len(multi_entry_pairs)}")

seq_variation = final_df.groupby(["UniProt_ID", "molecule_chembl_id"])["SEQUENCE_1L"].nunique()
multi_seq_pairs = seq_variation[seq_variation > 1]
print(f"Number of UniProt_ID - molecule_chembl_id pairs with multiple sequences: {len(multi_seq_pairs)}")