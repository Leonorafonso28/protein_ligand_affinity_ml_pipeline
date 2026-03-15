import pandas as pd
import torch
import esm
import deepchem as dc
from rdkit.Chem import Descriptors

#Directories
input_file = "data/features/deduplicated_filtered_Ki_dataset.csv"
output_file = "data/features/features_combined_ligand_protein_pKi.csv"

#Read and verified columns
df = pd.read_csv(input_file)

required_cols = ["CANONICAL_SMILES", "SEQUENCE_1L"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

#Ligand feature extraction
featurizer = dc.feat.RDKitDescriptors()
descriptor_names = [desc[0] for desc in Descriptors._descList]
smiles_set = df["CANONICAL_SMILES"].dropna().unique()

ligand_features = {}
for smiles in smiles_set:
    try:
        features = featurizer.featurize([smiles])[0]
        if features is not None:
            ligand_features[smiles] = features
    except Exception as e:
        print(f"Error processing {smiles}: {e}")


ligand_df = pd.DataFrame.from_dict(ligand_features, orient="index", columns=descriptor_names)
ligand_df.reset_index(inplace=True)
ligand_df.rename(columns={"index": "CANONICAL_SMILES"}, inplace=True)

#Merging information in the original dataset
df = df.merge(ligand_df, on="CANONICAL_SMILES", how="left")

#Protein feature extraction
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

#Proteins embeddings 
sequence_set = df["SEQUENCE_1L"].dropna().unique()
protein_features = {}

for sequence in sequence_set:
    if "X" in sequence:
        print(f"Skipping sequence: {sequence[:10]}...")
        continue

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    mean_embedding = token_representations[0, 1:len(sequence)+1].mean(dim=0)
    protein_features[sequence] = mean_embedding.tolist()

#Adding embeddings to dataframe
embedding_dim = len(next(iter(protein_features.values())))
protein_df = pd.DataFrame.from_dict(protein_features, orient="index")
protein_df.columns = [f"ESM2_{i+1}" for i in range(embedding_dim)]
protein_df.reset_index(inplace=True)
protein_df.rename(columns={"index": "SEQUENCE_1L"}, inplace=True)

#join to original dataframe
df = df.merge(protein_df, on="SEQUENCE_1L", how="left")

#Save final file
df = df.copy()
df.to_csv(output_file, index=False)
print(f"Final dataset saved in: {output_file}")