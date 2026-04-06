import os
import pandas as pd
from Bio import PDB

# Directories
input_dir = "data/raw/pdb_files"
output_dir = "data/interim/cleaned_pdb_files"
os.makedirs(output_dir, exist_ok=True)

# Data storage for UniProt mappings
uniprot_data = []
no_uniprot_pdbs = []  # Stores PDB IDs with missing uniprot id

def extract_uniprot_info(pdb_file):
    """Extracts UniProt ID, corresponding chains, sequence mapping, and conflicts from a PDB file."""
    pdb_id = os.path.basename(pdb_file).split(".")[0]  # Extract PDB ID from filename
    has_uniprot_info = False

    with open(pdb_file, "r") as f:
        for line in f:
            parts = line.split()
            
            # Extract UniProt Mapping from DBREF
            if line.startswith("DBREF"):
                try:
                    if len(parts) < 7:  # Skip invalid DBREF lines
                        continue

                    chain_id = parts[2]
                    
                    # Handle missing or non-integer values safely
                    pdb_start = int(parts[3]) if parts[3].isdigit() else None
                    pdb_end = int(parts[4]) if parts[4].isdigit() else None

                    if "UNP" in parts:
                        unp_index = parts.index("UNP") + 1
                        uniprot_id = parts[unp_index] if unp_index < len(parts) else None

                        # Ensure UniProt start and end exist
                        uniprot_start = int(parts[unp_index + 2]) if unp_index + 2 < len(parts) and parts[unp_index + 2].isdigit() else None
                        uniprot_end = int(parts[unp_index + 3]) if unp_index + 3 < len(parts) and parts[unp_index + 3].isdigit() else None

                        uniprot_data.append([pdb_id, chain_id, uniprot_id, pdb_start, pdb_end, uniprot_start, uniprot_end])
                        has_uniprot_info = True  # Mark as having UniProt info
                except (IndexError, ValueError) as e:
                    print(f" Error parsing DBREF in {pdb_file}: {line.strip()} → {e}")

    # If no UniProt info was found, mark the PDB as with no uniprot id
    if not has_uniprot_info:
        no_uniprot_pdbs.append(pdb_id)

# Function to clean PDB files
def clean_pdb_file(input_pdb, output_pdb):
    """Cleans PDB by keeping only ATOM, HETATM, and removing waters."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    # Write filtered PDB
    with open(output_pdb, "w") as out:
        with open(input_pdb, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")) and " HOH " not in line:
                    out.write(line)

# Process all PDB files in the folder
for pdb_file in os.listdir(input_dir):
    if pdb_file.endswith(".pdb") or pdb_file.endswith(".ent"):
        input_pdb_path = os.path.join(input_dir, pdb_file)
        output_pdb_path = os.path.join(output_dir, pdb_file)

        print(f"Processing PDB: {pdb_file}...")

        # Extract UniProt and conflict data
        extract_uniprot_info(input_pdb_path)

        # Clean PDB file
        clean_pdb_file(input_pdb_path, output_pdb_path)
        print(f" Cleaned file saved: {output_pdb_path}")

# Convert extracted UniProt data into a DataFrame and save it
df_uniprot = pd.DataFrame(uniprot_data, columns=["PDB_ID", "Chain", "UniProt_ID", "PDB_Start", "PDB_End", "UniProt_Start", "UniProt_End"])
df_uniprot.to_csv("data/interim/uniprot_mappings.csv", index=False)

# Print Summary of PDBs with no Uniprot IDs
if no_uniprot_pdbs:
    print("\n The following PDB files had missing UniProt:")
    for pdb_id in no_uniprot_pdbs:
        print(f"   - {pdb_id}")

print("\n Cleaning complete! All cleaned PDB files are in:", output_dir)
print(" UniProt mappings saved to: uniprot_mappings.csv")