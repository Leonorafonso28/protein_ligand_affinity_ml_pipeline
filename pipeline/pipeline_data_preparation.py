import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PREP_STEPS = [
    "scr/data_preparation/00_validation_pdbs.py",
    "scr/data_preparation/01_download_pdb_info.py",
    "scr/data_preparation/02_clean_pdb_and_cif_files.py",
    "scr/data_preparation/03_clean_ent_files.py",
    "scr/data_preparation/04_convert_cif_to_ent.py",
    "scr/data_preparation/05_split_ent_by_chain_ambiguous.py",
    "scr/data_preparation/06_filter_ligands_get_smiles.py",
    "scr/data_preparation/07_filtered_proteins.py",
    "scr/data_preparation/08_filtering_FASTA_ent_files.py"
]

def run_step(step_path: str):
    script = PROJECT_ROOT / step_path
    if not script.exists():
        logger.warning(f"Script not found: {script}")
        return
    logger.info(f"Running {script}")
    subprocess.run(
        ["python", str(script)],
        check=True
    )

def run_pipeline():
    logger.info("Starting data preparation pipeline")
    for step in DATA_PREP_STEPS:
        run_step(step)
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()