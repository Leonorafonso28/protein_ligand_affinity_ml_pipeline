import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_STEPS = [
    "scr/features/09_ChEMBL_bioactivities.py",
    "scr/features/10_01_filter_chembl_ki_data.py.",
    "scr/features/10_02_filter_chembl_pchembl_data.py",
    "scr/features/11_01_feature_ligand_protein_extraction_pki.py",
    "scr/features/11_02_feature_ligand_protein_extraction_pchembl.py"
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
    logger.info("Starting feature extraction pipeline")
    for step in FEATURE_STEPS:
        run_step(step)
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()