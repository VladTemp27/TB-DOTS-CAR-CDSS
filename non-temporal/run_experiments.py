import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

import pandas as pd
from pathlib import Path
from experiment_pipeline import TBExperimentPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


def main():
    pipeline = TBExperimentPipeline()
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "dataset" / "non-temporal" / "2015-2025-ml-ready.csv"
    results_path = Path("results") / "all_experiments.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df = pipeline.load_data(dataset_path)

    versions = [
        ('Baseline',              'baseline',  [None]),
        ('Resampling_Comparison', 'improved',  ['SMOTE', 'SMOTE - ENN', 'SMOTE - Tomek']),
        ('Extended_Features',     'extended',  ['SMOTE', 'SMOTE - ENN', 'SMOTE - Tomek']),
    ]

    all_results = []
    val_sets = {}

    for v_name, feat_ver, samplers in versions:
        logger.info("=== Starting experiment: %s ===", v_name)
        features = pipeline.get_features(feat_ver)
        X = df[features]
        y = df['Target']

        # 70/20/10 stratified split
        X_temp, X_val, y_temp, y_val = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)
        val_sets[v_name] = (X_val, y_val)
        logger.info("Split — train=%d  test=%d  val=%d", len(X_train), len(X_test), len(X_val))

        preprocessor = pipeline.get_preprocessor(features)

        for sampler in samplers:
            logger.info("Sampling strategy: %s", sampler or 'None')
            models = pipeline.get_models(y_train)
            res = pipeline.run_experiment(
                f"{v_name}_{sampler}", sampler, models,
                X_train, y_train, X_test, y_test, preprocessor)
            res['Version'] = v_name
            all_results.append(res)

            # Export individual experiment table
            pipeline.export_to_latex(
                res[['Model', 'ROC-AUC', 'CV-AUC Mean', 'CV-AUC Std',
                     'Recall (Fail)', 'F1 (Fail)', 'Accuracy']],
                f"results_{feat_ver}_{str(sampler).lower().replace(' ', '_')}",
                f"Performance Results — {feat_ver.capitalize()} / {sampler or 'No Sampling'}",
                f"tab:{feat_ver}_{str(sampler).lower().replace(' ', '_')}"
            )

        # Persist results after each version so partial runs are recoverable
        partial_df = pd.concat(all_results).drop(columns=['_pipeline', 'Version'], errors='ignore')
        partial_df.to_csv(results_path, index=False)
        logger.info("Results saved → %s", results_path)

    master_df = pd.concat(all_results)

    # Composite selection metric: 60% AUC + 40% Recall(Fail)
    master_df['_score'] = 0.6 * master_df['ROC-AUC'] + 0.4 * master_df['Recall (Fail)']
    best_model_row = master_df.sort_values('_score', ascending=False).iloc[0]
    best_pipeline = best_model_row['_pipeline']
    best_version = best_model_row['Version']

    logger.info(
        "Best model: %s / %s (score=%.4f  AUC=%.4f  Recall(Fail)=%.4f)",
        best_model_row['Model'], best_model_row['Sampler'],
        best_model_row['_score'], best_model_row['ROC-AUC'],
        best_model_row['Recall (Fail)']
    )

    # Evaluate top 5 configs on holdout validation set
    top5 = master_df.sort_values('_score', ascending=False).head(5)
    X_val_final, y_val_final = val_sets[best_version]
    logger.info("=== Holdout validation (top 5 configs) ===")
    for _, row in top5.iterrows():
        pipe = row['_pipeline']
        v = row['Version']
        xv, yv = val_sets[v]
        try:
            yv_pred = pipe.predict(xv)
            yv_proba = pipe.predict_proba(xv)[:, 1]
            report = classification_report(yv, yv_pred, output_dict=True, zero_division=0)
            logger.info(
                "  %-22s / %-12s  Val-AUC=%.4f  Val-Acc=%.4f  Val-Recall(Fail)=%.4f",
                row['Model'], row['Sampler'],
                roc_auc_score(yv, yv_proba),
                accuracy_score(yv, yv_pred),
                report['0']['recall'],
            )
        except Exception as e:
            logger.warning("  Holdout eval failed for %s / %s: %s", row['Model'], row['Sampler'], e)

    # Generate evaluation figures — re-split using the best version's feature set
    _version_to_feat = {
        'Baseline': 'baseline',
        'Resampling_Comparison': 'improved',
        'Extended_Features': 'extended',
    }
    best_feat_ver = _version_to_feat.get(best_version, 'improved')
    X_fig = df[pipeline.get_features(best_feat_ver)]
    y_fig = df['Target']
    X_temp_fig, _, y_temp_fig, _ = train_test_split(
        X_fig, y_fig, test_size=0.10, random_state=42, stratify=y_fig)
    _, X_test_fig, _, y_test_fig = train_test_split(
        X_temp_fig, y_temp_fig, test_size=2/9, random_state=42, stratify=y_temp_fig)

    pipeline.generate_figures(master_df, best_pipeline, X_test_fig, y_test_fig)

    # ONNX → ORT export
    try:
        from skl2onnx import to_onnx
        from sklearn.pipeline import Pipeline as SKPipeline
        import numpy as np

        steps = [(name, step) for name, step in best_pipeline.steps if name != 'sampler']
        clean_pipe = SKPipeline(steps)

        best_features = pipeline.get_features(best_feat_ver)
        X_dummy = pd.DataFrame(np.zeros((1, len(best_features))), columns=best_features)

        onnx_model = to_onnx(clean_pipe, X_dummy[:1], target_opset=12,
                              options={id(clean_pipe.steps[-1][1]): {'zipmap': False}})

        model_dir = base_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / "tb_outcome_prediction.onnx"

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info("Serialized ONNX → %s", onnx_path)

        try:
            from onnxruntime.tools import convert_onnx_models_to_ort
            convert_onnx_models_to_ort.convert_onnx_models_to_ort(str(model_dir))
            onnx_path.unlink(missing_ok=True)
            logger.info("Converted to ORT → %s", model_dir / "tb_outcome_prediction.ort")
        except Exception as e:
            logger.warning("ORT conversion failed (keeping .onnx): %s", e)

    except Exception as e:
        logger.warning("Model export failed: %s", e)

    # Export summary tables
    master_export = master_df.drop(columns=['_pipeline', 'Version', '_score'], errors='ignore')

    best_summary = (master_export
                    .sort_values('ROC-AUC', ascending=False)
                    .groupby(['Sampler', 'Model']).head(1))
    pipeline.export_to_latex(
        best_summary[['Sampler', 'Model', 'ROC-AUC', 'Recall (Fail)', 'F1 (Fail)']].head(10),
        "top_models_comparison",
        "Top 10 Performing Model Configurations Across All Experiments",
        "tab:top_models"
    )

    # CV performance table
    cv_table = (master_df
                .groupby(['Model', 'Version'])[['CV-AUC Mean', 'CV-AUC Std']]
                .mean()
                .round(4)
                .reset_index())
    pipeline.export_to_latex(
        cv_table,
        "cv_performance",
        "5-Fold Stratified Cross-Validation ROC-AUC by Model and Feature Version",
        "tab:cv_performance"
    )

    # Extended feature set results
    extended_mask = master_df['Version'] == 'Extended_Features'
    if extended_mask.any():
        ext_df = master_export[extended_mask].sort_values('ROC-AUC', ascending=False)
        pipeline.export_to_latex(
            ext_df[['Model', 'Sampler', 'ROC-AUC', 'CV-AUC Mean',
                    'Recall (Fail)', 'F1 (Fail)', 'Accuracy']].head(10),
            "extended_results",
            "Top Results for Extended Feature Set Experiments",
            "tab:extended_results"
        )

    logger.info("All experiments completed. Tables → paper/apa/tables/  Figures → paper/apa/figures/")


if __name__ == "__main__":
    main()
