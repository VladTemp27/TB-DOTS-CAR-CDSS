import logging
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from non_temporal_analysis import TBAnalysisPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    dataset_path = BASE_DIR / "dataset" / "non-temporal" / "2015-2025-consolidated-clean.csv"

    pipeline = TBAnalysisPipeline()
    df = pipeline.load_data(dataset_path)
    outputs = pipeline.export_all_figures(df)

    logger.info("Exported %d SVG images to %s", len(outputs), pipeline.output_dir)


if __name__ == "__main__":
    main()
