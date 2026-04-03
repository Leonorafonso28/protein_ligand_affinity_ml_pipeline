## 1. Dataset Overview

This dataset was constructed from scratch by integrating three public data sources — the RCSB Protein Data Bank (PDB), UniProt, and ChEMBL — to produce a curated, structure-informed protein–ligand binding affinity corpus for machine learning model development in drug discovery.

Each entry in the final dataset represents a unique protein chain–ligand co-complex observed in a PDB crystal structure, annotated with a binding affinity label retrieved from ChEMBL and featurised with RDKit molecular descriptors and ESM-2 protein language model embeddings.

Two parallel label branches are provided:

| Branch | Label | Assay scope |
|--------|-------|-------------|
| **pKi** | −log₁₀(Kᵢ × 10⁻⁹) | Equilibrium Ki measurements only (single-protein, binding assay, nM, exact relation) |
| **pChEMBL** | ChEMBL-standardised −log₁₀ molar | All binding assay types with a valid pChEMBL value (Ki, Kd, IC50, EC50) |

The pKi branch is smaller but more homogeneous in label type; the pChEMBL branch offers broader coverage at the cost of higher label heterogeneity. Both branches use a binary activity threshold of **≥ 6.5** (≈ 316 nM), a widely used convention in ChEMBL-based activity classification benchmarks.

---

## 2. Data Sources & Provenance

| Source | Version / Snapshot | Access method |
|--------|--------------------|---------------|
| RCSB PDB | February 19, 2025 (231,564 total entries) | REST API + wwPDB FTP + RCSB HTTPS |
| UniProt | Extracted from DBREF records in PDB files | Embedded in structure files |
| ChEMBL | Current at retrieval (2025) | `chembl-webresource-client` Python library |
| ESM-2 | `esm2_t33_650M_UR50D` (650M parameters, 33 layers) | `facebookresearch/esm` (GitHub) |
| RDKit descriptors | Stable `Descriptors._descList` | DeepChem `RDKitDescriptors` featuriser |

**Key provenance guarantees:**

- All PDB entries were validated against the RCSB holdings API before download. Unreleased and removed/retracted entries were explicitly excluded (Script 00).
- SMILES representations were standardised to canonical form using RDKit prior to any feature computation or ChEMBL cross-referencing, ensuring consistent molecular identity across all sources.
- InChIKey was used as the primary identifier for ChEMBL ligand mapping; canonical SMILES served as fallback.
- An organism consistency filter was applied: only rows where the PDB source organism matched the ChEMBL target organism were retained.
- The final datasets are restricted to **Homo sapiens** targets only.

---

## 3. Dataset Construction

### 3.1 Structural Data Acquisition

Starting from 231,564 PDB entries registered as of February 19, 2025, entries without at least one protein polymer entity were removed, leaving **226,867 protein structures**. For each retained entry, the PDB ID, experimental method, resolution, and source organism were extracted. Structure files were downloaded in `.ent`, `.cif`, or `.pdb` format using a three-tier fallback strategy (BioPython → wwPDB FTP → RCSB HTTPS). CIF files were converted to ENT format using Gemmi; all ENT files were cleaned to retain only `ATOM` and `HETATM` records, with water molecules (HOH) removed.

### 3.2 Chain Splitting & Sequence Processing

Each multi-chain ENT file was split into one file per protein chain. Residue standardisation was applied using a lookup table of 180+ non-standard → standard amino acid mappings derived from PDBFixer, AMBER, and selenocysteine/pyrrolysine conventions. Histidine protonation variants (HIP, HID, HIE) were mapped to HIS. Ambiguous residues (ASX/B, GLX/Z, UNK/X) were removed rather than converted; chains containing any ambiguous residue were discarded entirely (`ambiguity_threshold = 0`). Chains shorter than 30 residues were removed to exclude peptide fragments with insufficient structural context. All amino acid sequences were converted from 3-letter to 1-letter codes and written to per-chain FASTA files.

After chain processing: **223,483 PDB structures retained → 744,432 protein chains with 41,431 unique ligands identified.**

### 3.3 Ligand Filtering

A six-stage chemical filter was applied to all unique ligand codes:

| Stage | Filter | Method |
|-------|--------|--------|
| 1 | Predefined exclusion list (ions, metals, solvents, buffers: FE, ZN, CA, HOH, PO4, SO4, etc.) | Lookup, no API call |
| 2 | Non-3-letter codes (invalid PDB component codes) | String length check |
| 3 | Forbidden words in RCSB metadata (ion, metal, solvent, water, buffer, salt) | RCSB API |
| 4 | Biopolymer types (DNA, RNA, peptide-linking, terminus, saccharide subtypes) | RCSB type field |
| 5 | Invalid or unparseable SMILES | RDKit |
| 6 | Relaxed Lipinski filter: exclude if failing ≥ 2 of MW > 750 Da, rotatable bonds > 15, H-donors > 10, H-acceptors > 15 | RDKit |

Retained ligand types: non-polymer, peptide-like, D-saccharide, L-saccharide, saccharide.

**Result: 37,472 drug-like ligands retained; 3,959 excluded.**

The master structural dataset (`filtered_df.csv`) — produced by merging chain data, filtered ligands, and experimental metadata — contains **302,487 unique protein chain–ligand pairs** with 37,399 unique ligands.

### 3.4 Bioactivity Integration

PDB ligands were mapped to ChEMBL molecule IDs via InChIKey (SMILES fallback); PDB protein chains were mapped to ChEMBL target IDs via UniProt accession. Bioactivity records were retrieved from the ChEMBL API for each protein–ligand pair, collecting: pChEMBL value, standard value, standard units, standard relation, target organism, assay type, and target type.

**Label curation — shared steps (both branches):**

1. Filter for `target_type = SINGLE PROTEIN` and `assay_type = B` (binding assays only).
2. Select the highest-coverage, best-resolution PDB structure per (molecule, sequence) interaction.
3. Z-score deduplication: for conflicting measurements of the same interaction, retain the value with |z| closest to 0 among those with |z| ≤ 1. If all values are outliers, retain the one nearest to the group mean.

**pKi-specific steps:** additional filters for `type = Ki`, `standard_relation = =`, `standard_units = nM`, `standard_value > 0`. pKi computed as −log₁₀(standard_value × 10⁻⁹).

**pChEMBL-specific steps:** filter for non-null `pchembl_value`; no unit conversion required (already on −log₁₀ molar scale).

### 3.5 Feature Extraction

| Feature group | Method | Dimensionality |
|---------------|--------|----------------|
| Ligand descriptors | RDKit via DeepChem `RDKitDescriptors` (computed over unique SMILES) | ~210 |
| Protein embeddings | ESM-2 650M, mean-pool of layer-33 token representations over positions 1…len(seq) | 1,280 |
| **Combined** | Concatenation of ligand descriptors + protein embeddings | **~1,490** |

Sequences containing unknown residues (X) were skipped during ESM-2 inference. All feature computation was performed over unique inputs only, then merged back into the full dataset.

### 3.6 Preprocessing

Applied identically to both branches:

1. **Organism consistency filter** — retain only rows where PDB `Organism` matches ChEMBL `target_organism`.
2. **Homo sapiens filter** — retain only human targets.
3. **NaN column removal** — drop any numeric column containing at least one missing value.
4. **Zero-variance removal** — drop any constant numeric column.

---

## 4. Dataset Statistics

### pChEMBL Dataset

| Property | Value |
|----------|-------|
| Total protein–ligand pairs | 14,215 |
| Unique proteins | 9,417 |
| Unique ligands | 7,270 |
| Feature columns (post-filtering) | 1,476 |
| Active pairs (pChEMBL ≥ 6.5) | 9,406 (66.2%) |
| Inactive pairs (pChEMBL < 6.5) | 4,630 (33.8%) |
| Label value range | 5.14 – 8.74 |
| Dominant ligand type | Non-polymer (14,161 entries) |
| Dominant experimental method | X-ray crystallography (13,698) |
| Structure resolution range (majority) | 1.59 – 2.90 Å |

### pKi Dataset

| Property | Value |
|----------|-------|
| Total protein–ligand pairs | 3,744 |
| Unique proteins | 2,537 |
| Unique ligands | 1,767 |
| Feature columns (post-filtering) | 1,472 |
| Active pairs (pKi ≥ 6.5) | 2,748 (73.9%) |
| Inactive pairs (pKi < 6.5) | 971 (26.1%) |
| Label value range | 5.60 – 9.22 |
| Dominant ligand type | Non-polymer (3,714 entries) |
| Dominant experimental method | X-ray crystallography (3,570) |
| Structure resolution range (majority) | 1.55 – 2.90 Å |

> **Note on class imbalance:** Both datasets are imbalanced, with active compounds representing the majority class. F1-score was used as the primary optimisation metric during hyperparameter search; MCC and AUC-ROC are the primary evaluation metrics to account for this imbalance.

---

## 5. Data Splitting Strategy

### 5.1 Rationale

A naive random split is not appropriate for this dataset. As documented in the leakage analysis notebook, both datasets exhibit high molecular redundancy at the ligand, scaffold, protein, and pair level. Placing chemically or structurally similar entities in both train and test would result in inflated metrics that do not reflect genuine generalisation to novel compounds or targets — the core challenge in drug discovery.

A **six-partition blind splitting strategy** is used instead, evaluating generalisation along three orthogonal axes:

- **Interpolation** — within the known chemical and protein space (Test set)
- **Target extrapolation** — to novel protein sequences (Blind Protein set)
- **Scaffold extrapolation** — to novel chemical matter (Blind Ligand set)

All splits are performed at the level of **unique molecular identities** (SMILES and protein sequences), never at the row level. Seed is fixed at 42 for full reproducibility.

### 5.2 Partition Definitions

| Partition | Construction | Leakage constraint |
|-----------|-------------|-------------------|
| **Train** | 80% of data after blind exclusion | No SMILES or sequence overlap with any blind set |
| **Validation** | 10% of data after blind exclusion | No SMILES or sequence overlap with any blind set |
| **Test** | 10% of data after blind exclusion | No SMILES or sequence overlap with any blind set |
| **Blind** | Rows where both SMILES and sequence are in their respective blind pools | Both protein and ligand entirely unseen during training |
| **Blind Protein** | Rows where only the sequence is in the blind pool (Blind rows excluded) | Protein unseen; ligand may appear in training |
| **Blind Ligand** | Rows where only the SMILES is in the blind pool (Blind rows excluded) | Ligand unseen; protein may appear in training |

**Blind pool construction:** 15% of unique SMILES strings and 15% of unique protein sequences are randomly sampled (seed = 42) independently to form the two blind pools. All rows involving blind-pool SMILES or sequences are removed from the remaining data before the train/val/test split is performed.

**Feature standardisation:** `StandardScaler` is fitted exclusively on the training partition and applied to all other partitions. The scaler must be persisted to disk for inference-time consistency.

### 5.3 Partition Sizes

| Partition | pChEMBL | pKi |
|-----------|---------|-----|
| Train | 8,257 | 2,194 |
| Validation | 1,033 | 275 |
| Test | 1,033 | 275 |
| Blind | 312 | 82 |
| Blind Protein | 1,763 | 435 |
| Blind Ligand | 1,817 | 483 |
| **Total** | **14,215** | **3,744** |

### 5.4 Verified Overlaps

| Overlap check | Blind | Blind Protein | Blind Ligand |
|---------------|-------|---------------|--------------|
| Protein–ligand pair overlap with Train | ✅ 0 | ✅ 0 | ✅ 0 |
| Sequence overlap with Train | ✅ 0 | ✅ 0 | ⚠️ Expected (by design) |
| SMILES overlap with Train | ✅ 0 | ⚠️ Expected (by design) | ✅ 0 |

**Expected overlaps (by design):**

- Blind Protein set contains ligands seen during training: **252 ligands (pKi)** · **932 ligands (pChEMBL)**
- Blind Ligand set contains proteins seen during training: **120 proteins (pKi)** · **526 proteins (pChEMBL)**

These overlaps are intentional and reflect realistic prospective scenarios where a model must generalise to a new target (Blind Protein) or a new compound class (Blind Ligand), while having seen the complementary modality during training.

---

## 6. Known Limitations

- **No sequence identity clustering:** Protein chains were not clustered by sequence identity (e.g., MMseqs2 at 30%) prior to splitting. Highly similar proteins may appear in both training and the Blind Protein set, which may partially explain the strong Blind Protein performance observed. Future work should apply sequence-identity-based clustering to provide a stricter generalisation guarantee.
- **No scaffold-based ligand splitting:** The Blind Ligand partition uses random SMILES sampling rather than Bemis-Murcko scaffold splitting. Scaffold-based splitting would provide a more stringent test of scaffold hopping ability and better simulate prospective virtual screening scenarios.
- **Homo sapiens only:** All non-human targets are excluded. Cross-species transfer learning or applications to non-human targets would require retraining on a broader organism scope.
- **ESM-2 sequence truncation:** Sequences exceeding 1,022 tokens are silently truncated by the ESM-2 tokeniser. Affected sequences are not flagged in the current pipeline.
- **Label heterogeneity (pChEMBL):** The pChEMBL branch aggregates Ki, Kd, IC50, and EC50 measurements on a common scale. This broader label diversity introduces greater noise relative to the pKi branch and may explain the slightly lower classification performance observed for pChEMBL models.
- **Class imbalance:** Both datasets are imbalanced (active majority). No oversampling or undersampling was applied; imbalance was addressed solely through metric selection (F1, MCC) and the optimisation objective.
