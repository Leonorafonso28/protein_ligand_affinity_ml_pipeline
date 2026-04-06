import os
import requests
import gzip
import shutil
import pandas as pd
from Bio.PDB import PDBList

# Directory to save PDB files
output_dir = "data/raw/pdb_files"
os.makedirs(output_dir, exist_ok=True)

#  Fetch ALL current PDB entries
rcsb_url = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
response = requests.get(rcsb_url)

if response.status_code == 200:
    all_pdb_ids = response.json()
    print(f"\nTotal PDB entries retrieved: {len(all_pdb_ids)}")

    # Convert each PDB ID to lowercase and take the first 1000
    all_pdb_ids = [pdb.lower() for pdb in all_pdb_ids][:1000]
else:
    print("Failed to fetch PDB entry list. Exiting.")
    exit()
    
# Ensure `protein_pdb_ids` contains valid protein structures
protein_pdb_ids = []
protein_data = []

print("\nFetching PDB metadata from RCSB API...")
for pdb_id in all_pdb_ids:
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    entry_response = requests.get(entry_url)

    if entry_response.status_code == 404:
        print(f"Skipping obsolete PDB entry: {pdb_id}")
        continue  # Skip obsolete PDBs

    if entry_response.status_code == 200:
        entry_data = entry_response.json()

        # Debugging: Print API response structure for verification
        print(f"\n JSON Response for {pdb_id}: {list(entry_data.keys())}")

        # Check for experimental method
        experimental_method = entry_data.get("rcsb_entry_info", {}).get("experimental_method", "N/A")
        if experimental_method == "N/A":
            continue  # Skip if no valid method

        # Check for Protein Entity
        if entry_data["rcsb_entry_info"].get("polymer_entity_count_protein", 0) > 0:
            resolution = entry_data["rcsb_entry_info"].get("resolution_combined", ["N/A"])[0]

            # Get Organism Name from EBI PDBe API
            organism = "Unknown"
            ebi_url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/entities/{pdb_id}"
            ebi_response = requests.get(ebi_url)

            if ebi_response.status_code == 200:
                ebi_data = ebi_response.json()
                if pdb_id in ebi_data:
                    for entity in ebi_data[pdb_id]:
                        for source in entity.get("source", []):
                            if "organism_scientific_name" in source:
                                organism = source["organism_scientific_name"]
                                break  # Stop once we find the first valid organism

            protein_pdb_ids.append(pdb_id.lower())
            protein_data.append([pdb_id, experimental_method, organism, resolution])

print(f"\nFiltered experimental protein PDB entries: {len(protein_pdb_ids)}")

#  Initialize BioPython PDB Downloader
pdbl = PDBList()

#  Function to check if a file is gzipped
def is_gzipped(file_path):
    """Check if a file is a valid gzipped file."""
    try:
        with open(file_path, "rb") as f:
            return f.read(2) == b'\x1f\x8b'  # GZIP magic number
    except:
        return False

#  Function to download PDB file (Stops if a valid file is found)
def download_pdb_file(pdb_id):
    downloaded = False  # Track if we successfully get a file

    # First Try BioPython Download
    try:
        file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="pdb")

        if os.path.exists(file_path):
            if is_gzipped(file_path):
                #  Extract `.ent.gz` to `.ent`
                ent_file_path = os.path.join(output_dir, f"{pdb_id}.ent")
                with gzip.open(file_path, "rb") as f_in, open(ent_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(file_path)  # Remove `.gz` file after extraction
                print(f" BioPython extracted: {ent_file_path}")
            else:
                #  BioPython returned a regular `.ent` file, keep as is
                ent_file_path = os.path.join(output_dir, f"{pdb_id}.ent")
                shutil.move(file_path, ent_file_path)
                print(f" BioPython downloaded uncompressed file: {ent_file_path}")

            downloaded = True
    except Exception as e:
        print(f" BioPython download failed for {pdb_id}: {e}")

    # If BioPython fails, try FTP `.ent.gz`
    if not downloaded:
        ftp_urls = [
            f"https://files.wwpdb.org/pub/pdb/data/structures/all/pdb/pdb{pdb_id.lower()}.ent.gz"
        ]
        for ftp_url in ftp_urls:
            ftp_output_path = os.path.join(output_dir, f"{pdb_id}.ent.gz")

            try:
                response = requests.get(ftp_url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(ftp_output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)

                    #  Extract `.ent.gz`
                    if is_gzipped(ftp_output_path):
                        ent_file_path = os.path.join(output_dir, f"{pdb_id}.ent")
                        with gzip.open(ftp_output_path, "rb") as f_in, open(ent_file_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        os.remove(ftp_output_path)  #  Remove `.gz` file after extraction

                        print(f"FTP Download & Extracted: {ent_file_path}")
                        downloaded = True
                        break  # Stop trying FTP if one works
                    else:
                        print(f" FTP file is not a valid gzip file: {ftp_url}")
                        os.remove(ftp_output_path)  # Delete invalid file
                else:
                    print(f" FTP download failed: {ftp_url} (Status Code: {response.status_code})")
            except Exception as e:
                print(f"FTP download failed for {pdb_id}: {e}")

    # If `.ent` is still unavailable, download `.cif` or `.pdb`
    if not downloaded:
        formats = ["cif", "pdb"]
        for fmt in formats:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.{fmt}"
            output_path = os.path.join(output_dir, f"{pdb_id}.{fmt}")

            try:
                response = requests.get(pdb_url, stream=True)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    print(f" Downloaded: {pdb_id}.{fmt}")
                    downloaded = True
                    break
                else:
                    print(f" Failed: {pdb_id}.{fmt} (Status Code: {response.status_code})")
            except Exception as e:
                print(f" Error downloading {pdb_id}.{fmt}: {e}")

    # Stop further downloads if one file is available
    if downloaded:
        print(f" {pdb_id} successfully retrieved. No need for additional downloads.")

# Process Each PDB ID (Stop Downloading Once One File Exists)
for pdb_id in protein_pdb_ids:
    download_pdb_file(pdb_id)

print("\n Download and extraction complete!")

# Track failed downloads
failed_downloads = [
    pdb_id for pdb_id in protein_pdb_ids
    if not any(os.path.exists(os.path.join(output_dir, f"{pdb_id}{ext}")) for ext in [".ent", ".cif", ".pdb"])
]

# Print failed downloads
if failed_downloads:
    print("\n The following PDBs could not be downloaded:")
    print("\n".join(failed_downloads))
else:
    print("\n All selected PDBs were successfully downloaded!")

#  Save data to CSV
df = pd.DataFrame(protein_data, columns=["PDB_ID", "Experimental_Method", "Organism", "Resolution"])
df.to_csv("data/interim/protein_data.csv", index=False)

print("\n First few rows of the dataset:")
print(df.head())

print("\n Download and extraction complete!")

#Save in txt
txt_filename = os.path.join(output_dir, "data/raw/pdb_current_19_02_2025.txt")
    
with open(txt_filename, "w") as txt_file:
    txt_file.write("\n".join(all_pdb_ids))
        
print(f"List of PDBs saved in:\n - {txt_filename}")


