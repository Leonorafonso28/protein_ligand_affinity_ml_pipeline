import pandas as pd
import requests
import re
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load dataset
df = pd.read_csv('data/interim/ligands_per_chain.csv')

# Remove rows where LIGAND_ID is "NO_LIGAND"
df = df[df['LIGAND_ID'] != "NO_LIGAND"]

# List of known metal ions and solvents to exclude immediately, will be added to the excluded_ligands.txt from PLIC
excluded_ligands = {"FE2", "ZN2", "CU", "CA", "MG", "HOH", "DOD", "PO4", "SO4", "FE", "CA", "ZN", "K", "SO4", "CL", "NI", "CO", "NA", "NH2", "NH3"} 

# Debugging: Check if exclusion works correctly
print("Total Excluded Ligands (after filtering):", len(excluded_ligands))
print("Sample excluded ligands:", list(excluded_ligands)[:10])

# Function to standardize SMILES
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return "Invalid SMILES"

# Function to fetch ligand information from RCSB PDB
def get_ligand_info(ligand_id):
    if len(ligand_id) != 3:
        print(f" Excluding {ligand_id} - Not a 3-letter code")
        return None, None, None, None
    
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        ligand_type = data.get("chem_comp", {}).get("type", "Type not found").lower()
        ligand_name = data.get("chem_comp", {}).get("name", "Name not found").upper()

        # Exclude only standalone forbidden words, but allow nucleotides/phosphates
        for forbidden_word in ["ion", "metal", "solvent", "water", "buffer", "salt"]:
            if re.search(rf"\b{forbidden_word}\b", ligand_name.lower()) or re.search(rf"\b{forbidden_word}\b", ligand_type.lower()):
                print(f" Excluding {ligand_id} - Name contains forbidden word: {ligand_name}")
                return ligand_type, ligand_name, None, None

        # Exclude RNA/DNA and their linking types
        if re.search(r"\b(dna|rna)\b", ligand_type.lower()):
            print(f" Excluding {ligand_id} - Ligand type is RNA/DNA: {ligand_type}")
            return ligand_type, ligand_name, None, None
        
        # Exclude ligands with "peptide linking" or "terminus" in their type
        if re.search(r"peptide.*linking|terminus| linking", ligand_type.lower()):
            print(f" Excluding {ligand_id} - Ligand type contains 'peptide linking' or 'terminus': {ligand_type}")
            return ligand_type, ligand_name, None, None

# Extract all available SMILES_CANONICAL, SMILES, and InChIKey
        smiles_canonical_list = []  # Store multiple canonical SMILES
        smiles_list = []  # Store general SMILES
        inchi_key = "InChIKey not found"

        if "pdbx_chem_comp_descriptor" in data:
            for descriptor in data["pdbx_chem_comp_descriptor"]:
                if descriptor["type"] == "SMILES_CANONICAL":
                    smiles_canonical_list.append(descriptor["descriptor"])
                elif descriptor["type"] == "SMILES":
                    smiles_list.append(descriptor["descriptor"])
                elif descriptor["type"] == "InChIKey":
                    inchi_key = descriptor["descriptor"]

        # Try all SMILES_CANONICAL first, keeping the first valid standardized version
        canonical_smiles = "Invalid SMILES"
        for smiles in smiles_canonical_list:
            standardized = standardize_smiles(smiles)
            if standardized != "Invalid SMILES":
                canonical_smiles = standardized
                break  # Stop after finding the first valid SMILES

        # If no valid SMILES_CANONICAL was found, try general SMILES
        if canonical_smiles == "Invalid SMILES":
            for smiles in smiles_list:
                standardized = standardize_smiles(smiles)
                if standardized != "Invalid SMILES":
                    canonical_smiles = standardized
                    break  # Stop after finding the first valid SMILES

        # If still no valid SMILES found, exclude the ligand
        if canonical_smiles == "Invalid SMILES":
            print(f" Excluding {ligand_id} - No valid SMILES found.")
            return ligand_type, ligand_name, None, None

        return ligand_type, ligand_name, canonical_smiles, inchi_key
    else:
        print(f" Failed to fetch {ligand_id} (Status Code: {response.status_code})")
        return None, None, None, None

# Function to compute molecular properties
def get_molecular_properties(smiles):
    if smiles == "SMILES not found" or smiles == "Invalid SMILES":
        return "N/A", "N/A", "N/A", "N/A"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "N/A", "N/A", "N/A", "N/A"

    mw = Descriptors.MolWt(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    return mw, rot_bonds, h_donors, h_acceptors

# Extract all unique ligands from the dataset
unique_ligands = set(lig.strip() for ligand_entry in df['LIGAND_ID'].dropna() for lig in ligand_entry.split(',') if lig.strip())

ligand_data = []
excluded_due_to_criteria = []

for ligand_id in unique_ligands:
    if ligand_id in excluded_ligands:
        excluded_due_to_criteria.append((ligand_id, "Excluded due to predefined list", "N/A", "N/A", "N/A", "N/A", "N/A"))
        continue  # Skip known ions, metals, water, and buffers

    ligand_type, ligand_name, canonical_smiles, inchi_key = get_ligand_info(ligand_id)

    if canonical_smiles is None and inchi_key is None:  # Length, no SMILES or InChIKey, forbidden words
        excluded_due_to_criteria.append((ligand_id, ligand_name, "Excluded by criteria", "N/A", "N/A", "N/A", "N/A"))
        continue

    mw, rot_bonds, h_donors, h_acceptors = get_molecular_properties(canonical_smiles)

    # Count failed criteria
    try:
        failed_criteria = sum([
            mw > 750 if isinstance(mw, (int, float)) else False, 
            rot_bonds > 15 if isinstance(rot_bonds, (int, float)) else False, 
            h_donors > 10 if isinstance(h_donors, (int, float)) else False, 
            h_acceptors > 15 if isinstance(h_acceptors, (int, float)) else False
        ])
    except ValueError:
        print(f"Skipping ligand due to invalid values: MW={mw}, Rotatable Bonds={rot_bonds}, H-Donors={h_donors}, H-Acceptors={h_acceptors}")
        failed_criteria = 4  # Force exclusion of invalid ligands

    # If a ligand fails 1, exclude it
    if failed_criteria >= 2:
        excluded_due_to_criteria.append((ligand_id, ligand_name, canonical_smiles, mw, rot_bonds, h_donors, h_acceptors))
        continue

    # Add the ligand information to the ligand data
    ligand_data.append([ligand_id, ligand_type, ligand_name, canonical_smiles, inchi_key])

# Save filtered ligands
pd.DataFrame(ligand_data, columns=["LIGAND_ID", "LIGAND_TYPE", "NAME", "CANONICAL_SMILES", "INCHIKEY"]).to_csv("data/intermin/filtered_ligands.csv", index=False)
pd.DataFrame(excluded_due_to_criteria, columns=["LIGAND_ID", "NAME", "CANONICAL_SMILES", "MW", "ROTATABLE_BONDS", "H_DONORS", "H_ACCEPTORS"]).to_csv("data/intermin/excluded_ligands.csv", index=False)

# Print summary
print("\n Filtered ligands saved to 'data/interim/filtered_ligands.csv'")
print(" Excluded ligands saved to 'data/interim/excluded_ligands.csv'")

# Show unique ligand types in filtered ligands
ligand_data_df = pd.DataFrame(ligand_data, columns=["LIGAND_ID", "LIGAND_TYPE", "NAME", "CANONICAL_SMILES", "INCHIKEY"])
unique_ligand_types = ligand_data_df["LIGAND_TYPE"].unique()

print("\nUnique Ligand Types in filtered ligands (kept):")
print(unique_ligand_types)