# Feature Engineering

## Overview

This document describes the feature extraction strategy used to represent protein–ligand pairs as 
numerical vectors for machine learning. Each entry in the dataset is a unique protein chain–ligand 
co-complex drawn from experimentally validated PDB structures. The feature vector concatenates 
two independent feature sets — one for the ligand and one for the protein — producing a final 
input dimensionality of **1,490 features per sample**.

---

## 1. Ligand Representation — RDKit Molecular Descriptors

Ligands are represented using **210 physicochemical and topological descriptors** computed from 
their canonical SMILES strings via [DeepChem's `RDKitDescriptors` featuriser](https://deepchem.io/), 
which wraps RDKit's full descriptor list.

### Descriptor Categories

| Category | Examples |
|---|---|
| Molecular size & weight | `MolWt`, `HeavyAtomMolWt`, `ExactMolWt` |
| Lipophilicity | `MolLogP`, `MolMR` |
| Polar surface area | `TPSA` |
| H-bond capacity | `NumHDonors`, `NumHAcceptors` |
| Flexibility | `NumRotatableBonds` |
| Ring & aromaticity | `RingCount`, `NumAromaticRings` |
| Topological indices | `BertzCT`, `Chi0`, `Kappa1` |
| Electronic & charge | various partial charge descriptors |

### Processing

- SMILES are **standardised to canonical form** with RDKit before featurisation, ensuring that 
  the same molecule always maps to the same descriptor vector regardless of input notation.
- Descriptors are computed over **unique SMILES only** to avoid redundant computation; results 
  are then merged back to the full dataset.
- Rows for which RDKit cannot parse the SMILES receive `NaN` descriptor values and are 
  removed during preprocessing (Script 12).

---

## 2. Protein Representation — ESM-2 Sequence Embeddings

Proteins are represented using **1,280-dimensional embeddings** produced by the 
[ESM-2 650M parameter model](https://github.com/facebookresearch/esm) 
(`esm2_t33_650M_UR50D`, 33 transformer layers).

### Extraction Strategy

- Embeddings are extracted from **layer 33** (the final transformer layer), which encodes the 
  richest sequence-level representations.
- A **mean-pool over all token positions** (excluding BOS/EOS special tokens) produces a 
  fixed-size 1,280-dimensional vector per sequence, regardless of protein length.
- Computation is performed over **unique sequences only**; the resulting embeddings are merged 
  back to all rows sharing the same sequence.
- Sequences are processed in `torch.no_grad()` with `model.eval()` to disable gradient 
  tracking and dropout.

### Why ESM-2?

ESM-2 is a protein language model trained on ~250M protein sequences. Its embeddings have 
been shown to encode structural, functional, and evolutionary information without requiring 3D 
coordinates, enabling generalisation to targets without available crystal structures. This makes 
it particularly suitable for the blind protein generalisation scenario evaluated in this pipeline.

### Practical Notes

- Sequences exceeding **1,022 tokens** are truncated by the ESM tokeniser. Long sequences 
  should be logged and handled explicitly in production.
- GPU inference is strongly recommended for datasets with more than a few thousand unique 
  sequences.
- The ESM-2 package must be installed from the 
  [facebookresearch/esm GitHub repository](https://github.com/facebookresearch/esm), 
  not from PyPI.

---

## 3. Combined Feature Vector

| Block | Source | Dimensionality |
|---|---|---|
| RDKit molecular descriptors | Ligand canonical SMILES | ~200 (after NaN/zero-variance filtering) |
| ESM-2 embeddings | Protein amino acid sequence | 1,280 |
| **Total** | | **~1,490** |

The two blocks are concatenated along `axis=1` to form the final feature matrix fed to the 
machine learning models.

---

## 4. Feature Preprocessing

Applied in Scripts 12.1 and 12.2, after feature extraction and before dataset splitting:

1. **Organism consistency filter** — retains only rows where the PDB source organism 
   (from EBI PDBe) matches the ChEMBL target organism, removing expression-host artefacts.
2. **Homo sapiens filter** — restricts the dataset to human targets, the primary scope for 
   clinical drug discovery.
3. **NaN column removal** — any feature column containing at least one missing value is 
   dropped (`dropna(axis=1)`). This removes RDKit descriptors that failed for some molecules 
   and ESM-2 columns not generated for skipped sequences.
4. **Zero-variance removal** — constant features (variance == 0) are dropped as they carry 
   no discriminative information.
5. **StandardScaler** — all remaining numeric features are standardised (zero mean, unit 
   variance) using a `StandardScaler` fitted **exclusively on the training partition** and 
   applied to all other splits.

### Dataset sizes after preprocessing

| Dataset | Initial pairs | Retained pairs | Retained columns |
|---|---|---|---|
| pChEMBL | 17,222 | 14,215 | 1,476 |
| pKi | 5,410 | 3,744 | 1,472 |

---

## 5. Design Rationale

The combination of **interpretable physicochemical descriptors** (RDKit) with **learned protein 
representations** (ESM-2) was chosen as a strong, reproducible baseline for the following reasons:

- RDKit descriptors are **chemically interpretable** and stable across library versions, 
  facilitating feature importance analysis and regulatory explainability.
- ESM-2 embeddings capture **evolutionary and structural context** from sequence alone, 
  avoiding dependency on solved 3D structures for the protein modality.
- Mean-pooling produces a **fixed-size representation** compatible with standard tabular ML 
  models (gradient boosting, random forests) as well as neural architectures.
- Both feature sets are computed over **unique inputs** (unique SMILES / unique sequences) 
  before merging, avoiding redundant computation at scale.

---

## 6. Known Limitations & Future Work

- **No sequence identity filtering** was applied before splitting. High sequence similarity 
  between training and the blind protein set may inflate blind protein performance. 
  A sequence clustering step (e.g., CD-HIT at 30% identity) is recommended for production.
- **NaN column removal is aggressive** — a single missing value drops the entire column. 
  Median imputation for RDKit descriptors prior to this step could improve feature retention.
- **ESM-2 truncation** at 1,022 tokens affects long protein sequences silently. An explicit 
  truncation log and sliding-window strategy should be added.
- **Graph-based representations** (molecular graphs for ligands, protein contact graphs) 
  and **3D structure-based features** (binding pocket descriptors, docking scores) are 
  natural extensions for future model iterations.
- **Scaffold-based ligand splitting** (RDKit MurckoScaffold) would provide a more 
  stringent blind ligand evaluation than random SMILES sampling.