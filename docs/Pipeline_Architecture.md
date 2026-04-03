# **Overview**

This repository implements an end-to-end data pipeline for constructing a curated protein–ligand binding affinity database and training machine learning models for drug discovery applications. The pipeline is organised into four sequential stages, each composed of standalone, modular scripts with well-defined inputs and outputs.

**Technology stack:** Python 3.9+, BioPython, Gemmi, RDKit, DeepChem, ESM-2 (Meta), ChEMBL Web Resource Client, Optuna, XGBoost, TensorFlow, pandas, NumPy, scikit-learn.

# **Stage 1 — Data Acquisition & Structural Cleaning (Scripts 00–04)**

**Goal:** Retrieve, validate, and standardise protein structure files from the RCSB PDB.

## **Script 00 — PDB Integrity Validation**

Pre-flight sanity check before any data is downloaded. Queries the three RCSB holdings endpoints (current, unreleased, removed) and asserts mutual exclusivity. Any overlap halts the pipeline. Ensures no retracted or unreleased structures enter the training corpus.

## **Script 01 — Data Acquisition**

Fetches the full list of current PDB entries and filters for protein structures with at least one polymer entity. Retrieves experimental method, resolution, and organism metadata. Downloads structure files using a three-tier fallback strategy: BioPython (.ent.gz) → wwPDB FTP → RCSB HTTPS (.cif/.pdb). As of February 2025, 231,564 total PDB structures were registered; after filtering, 226,867 protein entries were retained.

## **Scripts 02, 03, 04 — CIF/ENT Cleaning & Conversion**

CIF files are converted to ENT format via Gemmi (files with multi-character chain names are skipped). All ENT and PDB files are cleaned to retain only ATOM and HETATM records, removing water molecules (HOH). UniProt sequence mappings are extracted from DBREF records throughout, linking structural data to canonical protein sequences. Two UniProt mapping tables are produced (one per file type) and later merged in Stage 3.
Stage 1 outputs: cleaned .ent files per PDB entry, protein_data.csv (metadata), uniprot_mappings.csv, uniprot_cif_mappings.csv.

# **Stage 2 — Sequence Processing & Dataset Assembly (Scripts 05–08)**

**Goal:** Split structures by chain, filter ligands, and assemble the master protein–ligand dataset.

## **Script 05 — Chain Splitting & Sequence Extraction**

Splits each multi-chain ENT file into one file per protein chain. Performs residue standardisation using a lookup table of 180+ non-standard → standard amino acid mappings (PDBFixer, AMBER, selenocysteine, pyrrolysine). Ambiguous residues (ASX, GLX, UNK) are removed; chains containing any ambiguous residue are discarded entirely (ambiguity_threshold = 0). Per-chain FASTA files are generated. All HETATM records from the original file are written to every chain file for downstream ligand association. After filtering, 223,483 PDB structures and 744,432 chains were retained, with 41,431 unique ligands identified.

## **Script 06 — Ligand Filtering & SMILES Retrieval**

Applies a six-stage chemical filter to all unique ligand codes. Predefined exclusion lists remove ions, metals, solvents, and buffers without API calls. RCSB metadata filters remove biopolymer fragments (DNA, RNA, peptide-linking, terminus types). Saccharide subtypes (D-saccharide, L-saccharide) are explicitly excluded. For retained ligands, canonical SMILES are retrieved from the RCSB chemical component dictionary and standardised with RDKit. A relaxed Lipinski filter excludes ligands failing two or more of: MW > 750 Da, rotatable bonds > 15, H-donors > 10, H-acceptors > 15. After filtering: 37,472 drug-like ligands retained; 3,959 excluded.

## **Script 07 — Dataset Assembly**

Merges chain-level sequence data, validated ligands, and experimental metadata into the master flat table. Applies a minimum chain length filter of ≥ 30 residues. Explodes multi-ligand rows into one row per protein–chain–ligand triplet. Result: filtered_df.csv with 302,487 unique chains and 37,399 unique ligands.

## **Script 08 — File Synchronisation**

Copies only FASTA and ENT files corresponding to entries in filtered_df.csv to filtered output directories. Rewrites ENT files to retain only HETATM lines for ligands that passed the chemical filter. Verification step reports any missing files.
Stage 2 outputs: filtered_df.csv, per-chain .ent and .fasta files (filtered), filtered_ligands.csv, excluded_ligands.csv.

# **Stage 3 — Bioactivity Integration & Feature Engineering (Scripts 09–11)**

**Goal:** Link the structural dataset to ChEMBL bioactivity data and extract ML features.

## **Script 09 — ChEMBL Bioactivity Retrieval**

Bridges the PDB-derived structural dataset with ChEMBL by a two-way cross-reference: ligands are mapped to ChEMBL molecule IDs via InChIKey (with canonical SMILES as fallback); proteins are mapped to ChEMBL target IDs via UniProt accession. Bioactivity records (pChEMBL value, Ki, standard units, assay type, target organism) are retrieved for each protein–ligand pair from the ChEMBL API. This anchors every bioactivity label to a known 3D co-crystal structure — a key differentiator from purely chemoinformatic datasets.

## ** Scripts 10.1 / 10.2 — Label Curation (pKi and pChEMBL branches)**

Two parallel curation workflows produce two labelled datasets:

- **pKi branch (10.1):** Filters for single-protein binding assays (assay_type = B), exact Ki measurements in nM (standard_relation = =), positive values only. Converts Ki to pKi = −log₁₀(Ki × 10⁻⁹). Structural deduplication selects the highest-coverage, best-resolution PDB structure per interaction. Z-score deduplication (threshold |z| ≤ 1) resolves conflicting measurements for the same molecule–sequence pair.

- **pChEMBL branch (10.2):** Same structural deduplication and z-score logic, applied to the pChEMBL value column directly. No unit conversion required; covers Ki, Kd, IC50, EC50 on a unified −log₁₀ molar scale. Broader coverage than the pKi branch at the cost of label heterogeneity.

## **Scripts 11.1 / 11.2 — Feature Extraction**

Two parallel feature extraction scripts (one per label branch) compute:

**Ligand features:** 210 RDKit molecular descriptors via DeepChem's RDKitDescriptors featuriser (molecular weight, logP, TPSA, ring counts, hydrogen bond donors/acceptors, topological and electronic indices). Computed over unique SMILES only.
Protein features: 1,280-dimensional mean-pooled embeddings from ESM-2 650M (esm2_t33_650M_UR50D, layer 33), computed over unique sequences only. Sequences containing unknown residues (X) are skipped.

Both feature sets are merged into a single feature matrix per branch (~1,490 features per row).

**Stage 3 outputs:** ChEMBL_activities.csv, deduplicated_filtered_Ki_dataset.csv, deduplicated_filtered_pchembl_dataset.csv, features_combined_ligand_protein_pKi.csv, features_combined_ligand_protein_pchembl.csv.

# **Stage 4 — Preprocessing, Splitting & Modelling (Scripts 12–21)**

**Goal:** Clean feature matrices, split data rigorously, and train and evaluate ML models.

## **Scripts 12.1 / 12.2 — Feature Preprocessing**

Two parallel preprocessing scripts (one per branch) apply: (1) organism consistency filter — retains only rows where the PDB source organism matches the ChEMBL target organism; (2) Homo sapiens filter — restricts to human targets; (3) NaN column removal — drops any numeric column with at least one missing value; (4) zero-variance removal — drops constant features. Final feature matrices are the concatenation of metadata columns and the cleaned numeric block.

## **Scripts 13.1 / 13.2 — Dataset Splitting**

Six-partition splitting strategy applied independently to each branch (see Dataset Card for full detail). StandardScaler fitted on the training partition only; transform applied to all other partitions. All arrays saved as .npy files.
Scripts 15, 16 — XGBoost Hyperparameter Optimisation
Optuna TPE sampler, 400 trials, validation F1 as objective. Search space: L1/L2 regularisation, colsample_bytree, subsample, learning rate, max depth, min child weight, number of estimators. Best parameters saved to JSON.

## **Scripts 14, 17 — XGBoost Evaluation**
Final XGBoost binary classifier trained with best hyperparameters. Both datasets use pKi ≥ 6.5 / pChEMBL ≥ 6.5 as the activity threshold (corresponding to approximately 316 nM). Evaluated across all six partitions using AUC-ROC, accuracy, recall, precision, F1, and MCC.

## **Scripts 18, 20 — DNN Hyperparameter Optimisation**

Same Optuna framework as XGBoost. Search space: number of hidden layers (1–5), units per layer (16–128), dropout (0–0.5), activation (ReLU/tanh/ELU), L2 weight decay, learning rate, batch size, number of epochs. Early stopping with patience = 10, best weights restored.

## **Scripts 19, 21 — DNN Evaluation**

Final DNN trained with best hyperparameters. Evaluated on the same six partitions using the same metric set.
Stage 4 outputs: features_processed_pchembl.csv, features_processed_pKi.csv, 12 .npy arrays per branch (X and y for train/val/test/blind/blind-protein/blind-ligand), trained model files (.pkl for XGBoost, saved weights for DNN), well_predicted_blind_features.csv per model.