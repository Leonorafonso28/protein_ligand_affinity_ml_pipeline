import os
import pandas as pd
import gemmi  # CIF and PDB handling library
from Bio import PDB  # BioPython for PDB cleaning
import re

# Directories
input_dir = "data/raw/pdb_files"  # Folder containing CIF/PDB files
converted_dir = "data/interim/converted_cif_to_ent_files"  # Folder for converted ENT files
cleaned_dir = "data/interim/cleaned_cif_pdb_files"  # Folder for cleaned ENT files
os.makedirs(converted_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

# Data storage
uniprot_data = []
no_uniprot_pdbs = []
problematic_cifs = []

# Function to extract UniProt info from PDB/ENT 
def extract_uniprot_info(pdb_file):
    """Extracts UniProt ID, corresponding chains, and sequence mapping from a PDB or ENT file.
    Ensures that PDB files are not marked as missing if the corresponding ENT file has UniProt mappings.
    """
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

    # Check if the corresponding ENT file exists and has UniProt mappings
    ent_file_path = os.path.join(cleaned_dir, f"{pdb_id}.ent")
    if os.path.exists(ent_file_path):
        with open(ent_file_path, "r") as ent_file:
            for line in ent_file:
                if line.startswith("DBREF"):
                    has_uniprot_info = True  # If ENT has UniProt, mark it

    # If UniProt data was found in PDB or ENT, ensure it's not in the missing list
    if has_uniprot_info:
        if pdb_id in no_uniprot_pdbs:
            no_uniprot_pdbs.remove(pdb_id)  # Remove from missing list if already added
    else:
        no_uniprot_pdbs.append(pdb_id)  # Only add to missing list if no UniProt ID found


# Function to clean PDB/ENT files and keep only ATOM, HETATM (no water)
def clean_pdb_file(input_pdb, output_pdb):
    """Cleans PDB by keeping only ATOM, HETATM, and removing water. 
    If any residue sequence is invalid, the entire file is skipped.
    """
    parser = PDB.PDBParser(QUIET=True)
    invalid_residue_found = False  # Flag to check if the file should be skipped

    # Read input PDB lines
    with open(input_pdb, "r") as f:
        lines = f.readlines()

    # Check for invalid residue sequence before writing to output file
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                resseq = int(line[22:26].strip())  # Try converting residue ID to int
            except ValueError:
                print(f"Skipping PDB {input_pdb} due to invalid residue sequence in line: {line.strip()}")
                invalid_residue_found = True  # Mark the file for skipping
                break  # Stop processing this PDB file

    # If the file contains invalid residues, do NOT write an output file
    if invalid_residue_found:
        if os.path.exists(output_pdb):
            try:
                os.remove(output_pdb)  # Ensure file is not locked before deleting
            except PermissionError:
                print(f"Warning: Unable to delete {output_pdb}, file might be in use.")
        return  # Skip further processing

    # Write cleaned PDB if no invalid residues were found
    with open(output_pdb, "w") as out:
        for line in lines:
            if line.startswith(("ATOM", "HETATM")) and " HOH " not in line:  # Remove water molecules
                out.write(line)

    print(f"Cleaned PDB saved as {output_pdb}")

# Function to convert CIF to ENT, skipping files with long chain names
def convert_cif_to_ent(input_cif, output_ent):
    """Converts a CIF file to ENT format, skipping ones with long chain names, keeping chain names unchanged."""
    try:
        structure = gemmi.read_structure(input_cif)  # Read CIF

        # Check for long chain names and store original chain names
        long_chain_found = False
        for model in structure:
            for chain in model:
                if len(chain.name) > 1:  # PDB/ENT format supports only 1-character chain names
                    print(f"Skipping {input_cif} - Chain name too long: {chain.name}")
                    problematic_cifs.append(os.path.basename(input_cif))
                    long_chain_found = True
                    break  # Stop checking further chains in this file
            if long_chain_found:
                return  # Skip conversion

        # Write to PDB format (ENT)
        structure.write_pdb(output_ent)  
        print(f"Converted {input_cif} → {output_ent}")

    except Exception as e:
        print(f"Error converting {input_cif}: {e}")
        problematic_cifs.append(os.path.basename(input_cif))

# Process PDB and CIF files
for file in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file)

    # Process CIF files
    if file.endswith(".cif"):
        output_path = os.path.join(converted_dir, file.replace(".cif", ".ent"))
        print(f"Processing CIF: {file}...")
        convert_cif_to_ent(input_path, output_path)

    # Process PDB files
    elif file.endswith(".pdb"):
        output_path = os.path.join(converted_dir, file.replace(".pdb", ".ent"))
        print(f"Processing PDB: {file}...")

        # Extract UniProt information from .pdb file
        extract_uniprot_info(input_path)

        # Clean PDB file and save as ENT
        clean_pdb_file(input_path, output_path)

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

# Report PDB/ENT files with missing UniProt info
if no_uniprot_pdbs:
    print("\nThe following PDB/ENT files had missing UniProt mappings:")
    for pdb_id in no_uniprot_pdbs:
        print(f"   - {pdb_id}")

print("\nProcessing complete!")
print(f"Cleaned ENT files saved in: {cleaned_dir}")
print("UniProt mappings saved in: uniprot_cif_mappings.csv")