import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPROCESSING_STEPS = [
    "src/preprocessing/12_01_preprocessing_features_pchembl.py",
    "src/preprocessing/12_02_preprocessing_features_pki.py",
    "src/preprocessing/13_01_train_test_split_pchembl.py",
    "src/preprocessing/13_02_train_test_split_pki.py",
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
    logger.info("Starting preprocessing pipeline")
    for step in PREPROCESSING_STEPS:
        run_step(step)
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()