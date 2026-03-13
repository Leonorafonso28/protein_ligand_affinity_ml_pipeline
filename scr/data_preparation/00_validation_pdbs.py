import requests

# URLs for different categories of PDB entries
current_url = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
unreleased_url = "https://data.rcsb.org/rest/v1/holdings/unreleased/entry_ids"
removed_url = "https://data.rcsb.org/rest/v1/holdings/removed/entry_ids"

# Fetch data from each endpoint
def fetch_pdb_ids(url):
    response = requests.get(url)
    if response.status_code == 200:
        return set(map(str.lower, response.json()))  # Convert to lowercase for consistency
    else:
        print(f" Failed to fetch data from {url}")
        return set()

# Load all PDB ID lists
current_pdbs = fetch_pdb_ids(current_url)
unreleased_pdbs = fetch_pdb_ids(unreleased_url)
removed_pdbs = fetch_pdb_ids(removed_url)

# Ensure NO 'current' PDBs exist in 'unreleased'
invalid_unreleased = current_pdbs.intersection(unreleased_pdbs)
if invalid_unreleased:
    print("\n ERROR: The following PDBs are marked as 'current' but are actually UNRELEASED:")
    print("\n".join(invalid_unreleased))
else:
    print("\n CONFIRMED: No 'current' PDBs are in the 'unreleased' list.")

# Ensure NO 'current' PDBs exist in 'removed'
invalid_removed = current_pdbs.intersection(removed_pdbs)
if invalid_removed:
    print("\nERROR: The following PDBs are marked as 'current' but are actually REMOVED (obsolete/theoretical):")
    print("\n".join(invalid_removed))
else:
    print("\n CONFIRMED: No 'current' PDBs are in the 'removed' list.")

# Assert final checks
assert len(invalid_unreleased) == 0, "Inconsistency found: Some 'current' PDBs are actually UNRELEASED!"
assert len(invalid_removed) == 0, "Inconsistency found: Some 'current' PDBs are actually REMOVED!"

# Print total counts
print(f"\n Total Current (Released) PDBs: {len(current_pdbs)}")
print(f" Total Unreleased PDBs: {len(unreleased_pdbs)}")
print(f" Total Removed (Obsolete/Theoretical) PDBs: {len(removed_pdbs)}")

print("\n FINAL CHECK PASSED: ALL 'current' PDBs are truly released and not obsolete.")