import os
import shutil
import pandas as pd

#Directories
merged_data_path = "data/interim/filtered_df.csv"
fasta_dir = "data/interim/fasta_sequences"
pdb_dir = "data/interim/split_pdb_files"
filtered_fasta_dir = "data/interim/fasta_sequences_filtered"
filtered_pdb_dir = "data/interim/split_pdb_files_filtered"

os.makedirs(filtered_fasta_dir, exist_ok=True)
os.makedirs(filtered_pdb_dir, exist_ok=True)

# Loading filtered dataset
merged_df = pd.read_csv(merged_data_path, dtype=str)

# Extracting valid ligands from the filtered dataset
valid_ligands = set(merged_df["LIGAND_ID"].unique())

# Extracting unique PDB-Chain pairs, keeping case distinction
unique_pdb_chains = set(merged_df["PDB ID"] + "_" + merged_df["Chain ID"])

# Obtaining the actual list of available files without modifying capitalization
existing_fasta_files = {f.replace(".fasta", ""): f for f in os.listdir(fasta_dir) if f.endswith(".fasta")}
existing_pdb_files = {f.replace(".ent", ""): f for f in os.listdir(pdb_dir) if f.endswith(".ent")}

#Filtering FASTA files based on unique PDB-Chain pairs
for pdb_id_chain in unique_pdb_chains:
    if pdb_id_chain in existing_fasta_files:
        fasta_file = os.path.join(fasta_dir, existing_fasta_files[pdb_id_chain])
        dest_file = os.path.join(filtered_fasta_dir, existing_fasta_files[pdb_id_chain])
        shutil.copy(fasta_file, dest_file)
    else:
        print(f"Warning: FASTA file not found for {pdb_id_chain}")

# Filtering PDB .ent files
for pdb_id_chain in unique_pdb_chains:
    if pdb_id_chain in existing_pdb_files:
        ent_file_path = os.path.join(pdb_dir, existing_pdb_files[pdb_id_chain])
        filtered_ent_path = os.path.join(filtered_pdb_dir, existing_pdb_files[pdb_id_chain])
        
        with open(ent_file_path, 'r') as file:
            lines = file.readlines()

        #Filtering only the desired HETATM lines
        filtered_lines = []
        for line in lines:
            if line.startswith("HETATM"):
                ligand_id = line[17:20].strip()
                if ligand_id in valid_ligands:
                    filtered_lines.append(line)  
            else:
                filtered_lines.append(line)  

        #Saving the filtered .ent file
        with open(filtered_ent_path, 'w') as file:
            file.writelines(filtered_lines)
    else:
        print(f"Warning: PDB file not found {pdb_id_chain}")

print("\nFASTA and PDB files filtered successfully!")

# Verification
csv_file = merged_data_path
pdb_directory = filtered_pdb_dir
fasta_directory = filtered_fasta_dir

df = pd.read_csv(csv_file, dtype=str)
expected_pdb_chains = set(df["PDB ID"] + "_" + df["Chain ID"])
existing_pdb_files = {f.replace(".ent", "") for f in os.listdir(pdb_directory) if f.endswith(".ent")}
existing_fasta_files = {f.replace(".fasta", "") for f in os.listdir(fasta_directory) if f.endswith(".fasta")}

missing_pdb_files = expected_pdb_chains - existing_pdb_files
missing_fasta_files = expected_pdb_chains - existing_fasta_files

print("\nVerification Results:")
print(f"Total expected PDB chains from CSV: {len(expected_pdb_chains)}")
print(f"Total existing .ent files in {pdb_directory}: {len(existing_pdb_files)}")
print(f"Total existing FASTA files in {fasta_directory}: {len(existing_fasta_files)}")

if not missing_pdb_files and not missing_fasta_files:
    print("\nAll PDB chains from the CSV are present in both directories!")
else:
    print("\nSome files are missing:")
    if missing_pdb_files:
        print(f"\nMissing PDB files ({len(missing_pdb_files)}):")
        print("\n".join(missing_pdb_files))
    if missing_fasta_files:
        print(f"\nMissing FASTA files ({len(missing_fasta_files)}):")
        print("\n".join(missing_fasta_files))
