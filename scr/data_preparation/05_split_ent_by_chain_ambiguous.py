import os
import pandas as pd

# Set of ambiguous or non-standard residues to remove (no conversion possible)
ambiguous_residues = {"ASX" , "GLX", "UNK", "X", "B", "Z"}
#ASX is B and GLX is Z aminoacids, UNK is X

# Mapping non-standard to standard 3-letter codes (for `.ent` files and CSV)
valid_aminoacids = {
    "ALA": "ALA", "ARG": "ARG", "ASN": "ASN", "ASP": "ASP", "CYS": "CYS",
    "GLN": "GLN", "GLU": "GLU", "GLY": "GLY", "HIS": "HIS", "ILE": "ILE",
    "LEU": "LEU", "LYS": "LYS", "MET": "MET", "PHE": "PHE", "PRO": "PRO",
    "SER": "SER", "THR": "THR", "TRP": "TRP", "TYR": "TYR", "VAL": "VAL",
    # Histidine protonation variants
    "HIP": "HIS", "HID": "HIS", "HIE": "HIS", #https://ambermd.org/Questions/HIS.html
    # Lysine variant
    "LYN": "LYS",
    # Aspartic & Glutamic acid variants
    "ASH": "ASP", "GLH": "GLU",    
    #Convertion from PDBFIXER https://github.com/openmm/pdbfixer/blob/master/pdbfixer/pdbfixer.py
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', '5OW':'LYS', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MK8':'LEU', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR',
    # O and U aminoacids to standard 
    "SEC": "CYS", "PYL": "LYS"
}

# 3-letter to 1-letter amino acid mapping (for FASTA & CSV conversion)
aa_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def process_pdb(pdb_file, output_dir, fasta_dir, ambiguity_threshold=0):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fasta_dir, exist_ok=True)
    
    data = []
    structures = {}
    sequences_1letter = {}
    sequences_3letter_pdb = {}
    sequences_3letter_std = {}
    positions = {}
    ligands = {}
    chain_ambiguous_count = {}
    chain_total_count = {}
    all_hetatm = []  # Store all HETATM records in the file
    unique_residues = {}  # Ensure only one residue per position
    unique_ligands = set()
    
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):  
                residue_name = line[17:20].strip()
                chain_id = line[21]
                res_number = line[22:26].strip()
                
                if residue_name not in valid_aminoacids:
                    chain_ambiguous_count[chain_id] = chain_ambiguous_count.get(chain_id, 0) + 1
                    continue  # Skip non-amino acids
                
                if (chain_id, res_number) in unique_residues:
                    continue  # Skip duplicate residues at the same position
                unique_residues[(chain_id, res_number)] = residue_name
                
                if chain_id not in sequences_3letter_pdb:
                    sequences_3letter_pdb[chain_id] = []
                    sequences_3letter_std[chain_id] = []
                    positions[chain_id] = []
                    sequences_1letter[chain_id] = []
                
                sequences_3letter_pdb[chain_id].append(residue_name)
                positions[chain_id].append(res_number)
                standard_residue = valid_aminoacids.get(residue_name, None)
                one_letter = aa_3to1.get(standard_residue, None)
                
                if standard_residue:
                    sequences_3letter_std[chain_id].append(standard_residue)
                if one_letter:
                    sequences_1letter[chain_id].append(one_letter)
                
                chain_total_count[chain_id] = chain_total_count.get(chain_id, 0) + 1
            elif line.startswith("HETATM"):
                all_hetatm.append(line)  # Store all HETATM lines
                chain_id = line[21]
                residue_name = line[17:20].strip()
            
                # Ensure every chain gets all ligands in the file
                if chain_id not in ligands:
                    ligands[chain_id] = set()  # Use a set to avoid duplicate ligands
                ligands[chain_id].add(residue_name)  # Store only unique ligand IDs
            
                # Also track all unique ligands in the file, regardless of chain
                unique_ligands.add(residue_name)


    # Remove chains exceeding ambiguity threshold
    chains_to_remove = {chain_id for chain_id, amb_count in chain_ambiguous_count.items()
                        if amb_count / chain_total_count.get(chain_id, 1) > ambiguity_threshold}

    for chain_id in sequences_3letter_pdb:
        if chain_id in chains_to_remove:
            continue
        
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        ligand_list = list(set(ligands.get(chain_id, ["NO_LIGAND"])))
        key = f"{pdb_id}_{chain_id}"
        structures[key] = []
    
        with open(pdb_file, 'r') as file:
            for line in file:
                chain_id_line = line[21]
                if line.startswith("ATOM") and chain_id_line == chain_id:
                    residue_name = line[17:20].strip()
                    if residue_name in valid_aminoacids:  # Convert non-standard to standard
                        standard_residue = valid_aminoacids[residue_name]
                        line = line[:17] + f"{standard_residue:>3}" + line[20:]
                    structures[key].append(line)
    
                elif line.startswith("HETATM"):  
                    structures[key].append(line)  # Keep ligands unchanged
    
        output_file = os.path.join(output_dir, f"{key}.ent")
        with open(output_file, 'w') as out_file:
            out_file.writelines(structures[key])
        print(f"Saved: {output_file}")

    for chain_id, seq in sequences_1letter.items():
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        fasta_file = os.path.join(fasta_dir, f"{pdb_id}_{chain_id}.fasta")
        with open(fasta_file, 'w') as fasta_out:
            fasta_out.write(f">{pdb_id}_{chain_id}\n")
            fasta_out.write("".join(seq) + "\n")

        data.append([
            pdb_id, chain_id,
            "-".join(sequences_3letter_pdb[chain_id]),
            "-".join(sequences_3letter_std[chain_id]),
            "".join(seq),
            "-".join(positions[chain_id]),
            ",".join(unique_ligands) if unique_ligands else "NO_LIGAND"
        ])

    return data

def process_all_pdb_files(directory, output_dir, fasta_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fasta_dir, exist_ok=True)
    all_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.ent'):
            pdb_file = os.path.join(directory, filename)
            data = process_pdb(pdb_file, output_dir, fasta_dir)
            all_data.extend(data)
    
    df = pd.DataFrame(all_data, columns=['PDB ID', 'Chain ID', 'SEQUENCE_3L_PDB', 'SEQUENCE_3L_STD', 'SEQUENCE_1L', 'POSITIONS', 'LIGAND_ID'])
    df.to_csv('data/intermin/ligands_per_chain.csv', index=False)
    print("Processing complete. Data saved to ligands_per_chain.csv and FASTA files generated.")
    return df

pdb_directory = 'data/intermin/cleaned_pdb_files'
output_directory = 'data/intermin/split_pdb_files'
fasta_directory = 'data/intermin/fasta_sequences'
df = process_all_pdb_files(pdb_directory, output_directory, fasta_directory)