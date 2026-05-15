#!/usr/bin/env python3
"""CLI entrypoint for the SLM-SHAP producer pipeline."""
import argparse
from pathlib import Path

from slm_shap_pipeline.config import PipelineConfig
from slm_shap_pipeline.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="SLM-SHAP producer pipeline")
    parser.add_argument("--condition", choices=["blind", "sighted"], required=True)
    parser.add_argument("--patient-limit", type=int, default=None)
    parser.add_argument("--dry-run-gemini", action="store_true")
    parser.add_argument("--provider", choices=["cli", "api", "medgemma"], default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--medgemma-model", type=Path, default=None, help="Path to MedGemma GGUF file")
    parser.add_argument("--output-base", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    args = parser.parse_args()

    config_kwargs = {}
    if args.output_base:
        config_kwargs["output_base"] = args.output_base
    if args.model_path:
        config_kwargs["model_path"] = args.model_path
    if args.csv_path:
        config_kwargs["csv_path"] = args.csv_path
    if args.provider:
        config_kwargs["provider"] = args.provider
    if args.model:
        config_kwargs["gemini_model"] = args.model
    if args.medgemma_model:
        config_kwargs["medgemma_model_path"] = args.medgemma_model

    config = PipelineConfig(**config_kwargs)
    run(
        condition=args.condition,
        config=config,
        patient_limit=args.patient_limit,
        dry_run_gemini=args.dry_run_gemini,
    )


if __name__ == "__main__":
    main()
