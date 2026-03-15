import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_STEPS = [
    "src/models/XGBoost/pki/14_XGBoost_pki.py",
    "src/models/XGBoost/pki/15_XGBoost_pki_evaluation.py",
    "src/models/XGBoost/pchembl/16_XGBoost_pchembl.py",
    "src/models/XGBoost/pchembl/17_XGBoost_pchembl_evaluation.py",
    "src/models/DNN/pki/18_DNN_pki.py",
    "src/models/DNN/pki/19_DNN_pki_evaluation.py",
    "src/models/DNN/pchembl/20_DNN_pchembl.py",
    "src/models/DNN/pchembl/21_DNN_pchembl_evaluation.py",
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
    logger.info("Starting models pipeline")
    for step in MODEL_STEPS:
        run_step(step)
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()