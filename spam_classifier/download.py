import logging
import subprocess
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def download_sms_data(data_dir: Path) -> Path:
    raw_path = data_dir / "raw" / "sms.tsv"
    dvc_file = raw_path.with_suffix(".tsv.dvc")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info("Data already exists locally")
        return raw_path

    logger.info("Trying to pull data with DVC...")
    try:
        result = subprocess.run(
            ["dvc", "pull", str(dvc_file)], check=False, capture_output=True, text=True
        )
        if result.returncode == 0 and raw_path.exists():
            logger.info("Data successfully pulled via DVC")
            return raw_path
    except FileNotFoundError:
        logger.info("DVC not installed or not in PATH")

    logger.info("Downloading SMS Spam Collection from direct source...")
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    try:
        urllib.request.urlretrieve(url, raw_path)
        logger.info(f"Downloaded data to {raw_path}")
    except Exception as e:
        logger.info(f"Download failed: {e}")
        raise

    try:
        subprocess.check_call(["dvc", "add", str(raw_path)])
        logger.info("Data added to DVC tracking")
        subprocess.check_call(["dvc", "push", str(dvc_file)])
        logger.info("Data pushed to DVC remote")
    except Exception as e:
        logger.info(f"Could not add to DVC (optional on first run): {e}")

    return raw_path


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    download_sms_data(data_dir)
