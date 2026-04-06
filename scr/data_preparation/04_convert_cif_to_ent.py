import os
import pandas as pd
import gemmi  # CIF and PDB handling library
from Bio import PDB  # BioPython for PDB cleaning

# Directories
input_dir = "data/raw/pdb_files"  # Folder containing CIF files
converted_dir = "data/interim/converted_cif_to_ent_files"  # Folder for converted ENT files
cleaned_dir = "data/interim/cleaned_pdb_files"  # Folder for cleaned ENT files
os.makedirs(converted_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

# Data storage
uniprot_data = []
no_uniprot_pdbs = []
problematic_cifs = []

# Function to convert CIF to ENT, skipping files with long chain names
def convert_cif_to_ent(input_cif, output_ent):
    """Converts a CIF file to ENT format, skipping ones with long chain names."""
    try:
        structure = gemmi.read_structure(input_cif)  # Read CIF

        # Check for long chain names
        for model in structure:
            for chain in model:
                if len(chain.name) > 1:  # PDB/ENT format supports only 1-character chain names
                    print(f" Skipping {input_cif} - Chain name too long: {chain.name}")
                    problematic_cifs.append(os.path.basename(input_cif))
                    return  # Skip conversion

        structure.write_pdb(output_ent)  # Save as ENT (same format as PDB, just different extension)
        print(f" Converted {input_cif} → {output_ent}")

    except Exception as e:
        print(f" Error converting {input_cif}: {e}")
        problematic_cifs.append(os.path.basename(input_cif))

# Function to extract UniProt info from ENT
def extract_uniprot_info(pdb_file):
    """Extracts UniProt ID, corresponding chains, and sequence mapping from a PDB file."""
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

    # If no UniProt info was found, mark the PDB as problematic
    if not has_uniprot_info:
        no_uniprot_pdbs.append(pdb_id)

# Function to clean ENT files (remove water and non-ATOM/HETATM lines)
def clean_pdb_file(input_pdb, output_pdb):
    """Cleans PDB by keeping only ATOM, HETATM, and removing water."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    # Write filtered PDB (ENT format)
    with open(output_pdb, "w") as out:
        with open(input_pdb, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")) and " HOH " not in line:
                    out.write(line)

# Convert CIF files to ENT
for file in os.listdir(input_dir):
    if file.endswith(".cif"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(converted_dir, file.replace(".cif", ".ent"))

        print(f"Processing CIF: {file}...")
        convert_cif_to_ent(input_path, output_path)

# Process converted ENT files
for pdb_file in os.listdir(converted_dir):
    if pdb_file.endswith(".ent"):
        input_pdb_path = os.path.join(converted_dir, pdb_file)
        output_pdb_path = os.path.join(cleaned_dir, pdb_file)

        print(f"Processing ENT: {pdb_file}...")

        # Extract UniProt information
        extract_uniprot_info(input_pdb_path)

        # Clean ENT file
        clean_pdb_file(input_pdb_path, output_pdb_path)
        print(f" Cleaned file saved: {output_pdb_path}")

# Save UniProt data to CSV
df_uniprot = pd.DataFrame(uniprot_data, columns=["PDB_ID", "Chain", "UniProt_ID", "PDB_Start", "PDB_End", "UniProt_Start", "UniProt_End"])
df_uniprot.to_csv("data/interim/uniprot_cif_mappings.csv", index=False)

# Report problematic CIF files
if problematic_cifs:
    print("\n The following CIF files had long chain names and were skipped:")
    for cif in problematic_cifs:
        print(f"   - {cif}")

# Report ENT files with missing UniProt info
if no_uniprot_pdbs:
    print("\n The following ENT files had missing UniProt mappings:")
    for pdb_id in no_uniprot_pdbs:
        print(f"   - {pdb_id}")

print("\nProcessing complete!")
print(f"Cleaned ENT files saved in: {cleaned_dir}")
print(f"UniProt mappings saved in: uniprot_mappings.csv")