"""Generate descriptive figures, tables, and statistics for the 7,389-record
non-temporal TB dataset, writing everything into paper/apa/pharma_figures/.

This is a duplicate of static-model/run_analysis.py, redirected so all outputs
land in this directory instead of paper/apa/figures and paper/apa/tables.
"""

import logging
from pathlib import Path
import sys

# Run the sibling analysis_pipeline.py in this folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    logger.info("Loaded %d records × %d columns", df.shape[0], df.shape[1])

    outputs = pipeline.export_all_figures(df)

    logger.info("Exported %d artifacts to %s", len(outputs), pipeline.output_dir)


if __name__ == "__main__":
    main()
