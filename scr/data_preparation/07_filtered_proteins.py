import pandas as pd 

#Directory
filtered_ligands_path = "data/interim/filtered_ligands.csv"
ligands_per_chain_path = "data/interim/ligands_per_chain.csv"
protein_data_path = "data/interim/protein_data.csv"
output_path = "data/interim/expanded_filtered_dataset.csv"

#Remove NO_LIGAND
df=pd.read_csv(ligands_per_chain_path)
df=df[df["LIGAND_ID"]!="NO_LIGAND"]

#Expand multiple ligands per chain
df["LIGAND_ID"]=df["LIGAND_ID"].str.split(",")
df = df.explode("LIGAND_ID")

#Remove chain with less than 30 aminoacids
df = df[df["SEQUENCE_1L"].str.len() >= 30]

#Keeping only ligands that are in filtered_ligands
filtered_ligands = pd.read_csv(filtered_ligands_path)[["LIGAND_ID","LIGAND_TYPE", "NAME" ,"CANONICAL_SMILES", "INCHIKEY"]]
df = df[df["LIGAND_ID"].isin(filtered_ligands["LIGAND_ID"])]

#Joining the datasets filtered_ligands and ligands_per_chain
merged_df = df.merge(filtered_ligands, on="LIGAND_ID", how="left")
columns_to_keep = ["PDB ID", "Chain ID", "LIGAND_ID", "LIGAND_TYPE", "POSITIONS", "NAME", "SEQUENCE_3L_STD", "SEQUENCE_1L", "CANONICAL_SMILES", "INCHIKEY"]
merged_df = merged_df[columns_to_keep]

#Remove duplicates
merged_df = merged_df.drop_duplicates(subset=["PDB ID", "Chain ID", "LIGAND_ID"])

#Join protein_data with PDB ID
protein_data = pd.read_csv(protein_data_path)
protein_data_filtered = protein_data[protein_data["PDB_ID"].isin(merged_df["PDB ID"])]

final_df = merged_df.merge(protein_data_filtered[["PDB_ID", "Experimental_Method", "Resolution", "Organism"]], 
                           left_on="PDB ID", right_on="PDB_ID", how="left")

final_df.to_csv("data/interim/filtered_df.csv", index=False)