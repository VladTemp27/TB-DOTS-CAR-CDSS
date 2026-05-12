import logging
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from analysis_pipeline import TBAnalysisPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    pipeline = TBAnalysisPipeline()
    df = pipeline.load_raw_data()
    outputs = pipeline.export_all_figures(df)

    logger.info("Exported %d images to %s", len(outputs), pipeline.output_dir)


if __name__ == "__main__":
    main()
