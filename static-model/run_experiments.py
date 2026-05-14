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
        ('Main', 'improved', [None, 'SMOTE - ENN']),
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

    # Generate evaluation figures
    best_feat_ver = 'improved'
    X_fig = df[pipeline.get_features(best_feat_ver)]
    y_fig = df['Target']
    X_temp_fig, _, y_temp_fig, _ = train_test_split(
        X_fig, y_fig, test_size=0.10, random_state=42, stratify=y_fig)
    _, X_test_fig, _, y_test_fig = train_test_split(
        X_temp_fig, y_temp_fig, test_size=2/9, random_state=42, stratify=y_temp_fig)

    pipeline.generate_figures(master_df, best_pipeline, X_test_fig, y_test_fig)

    # ONNX export — full sklearn pipeline (preprocessor + classifier) embedded
    # Uses onnxmltools to register the XGBoost converter with skl2onnx (XGBoost 3.x compatible)
    try:
        import numpy as np
        from sklearn.pipeline import Pipeline as SKPipeline
        from skl2onnx import to_onnx, update_registered_converter
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from xgboost.sklearn import XGBClassifier

        update_registered_converter(
            XGBClassifier, 'XGBoostXGBClassifier',
            calculate_linear_classifier_output_shapes, convert_xgboost,
            options={'nocl': [True, False], 'zipmap': [True, False]},
        )

        steps = [(name, step) for name, step in best_pipeline.steps if name != 'sampler']
        clean_pipe = SKPipeline(steps)

        best_features = pipeline.get_features(best_feat_ver)
        X_dummy = pd.DataFrame(np.zeros((1, len(best_features))), columns=best_features)

        onnx_model = to_onnx(
            clean_pipe, X_dummy[:1].astype(np.float32),
            target_opset={'': 12, 'ai.onnx.ml': 3},
            options={id(clean_pipe.steps[-1][1]): {'zipmap': False, 'nocl': True}},
        )

        model_dir = base_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / "tb_outcome_prediction.onnx"

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info("Exported ONNX → %s", onnx_path)

    except Exception as e:
        logger.warning("Model export failed: %s", e)

    # Export model comparison table (APA-formatted, resizeable)
    _export_model_comparison(master_df, best_model_row, pipeline.table_dir)

    logger.info("All experiments completed. Tables → paper/apa/tables/  Figures → paper/apa/figures/")


def _export_model_comparison(master_df, best_model_row, table_dir):
    cols = ['Model', 'Sampler', 'ROC-AUC', 'CV-AUC Mean', 'CV-AUC Std', 'Accuracy',
            'Precision (Fail)', 'Recall (Fail)', 'F1 (Fail)',
            'Precision (Succ)', 'Recall (Succ)', 'F1 (Succ)']
    df = (master_df[cols + ['_score']]
          .sort_values('ROC-AUC', ascending=False)
          .reset_index(drop=True))

    def fmt(v):
        try:
            return f"{float(v):.4f}"
        except (ValueError, TypeError):
            return str(v)

    rows_tex = []
    for _, row in df.iterrows():
        is_best = (row['Model'] == best_model_row['Model'] and
                   row['Sampler'] == best_model_row['Sampler'])
        score = 0.6 * float(row['ROC-AUC']) + 0.4 * float(row['Recall (Fail)'])
        cells = [row['Model'], row['Sampler']] + [fmt(row[c]) for c in cols[2:]] + [f"{score:.4f}"]
        line = ' & '.join(cells) + r' \\'
        rows_tex.append(r'\textbf{' + line.replace(r' \\', r'} \\') if is_best else line)

    header = (r'Model & Sampler & AUC & CV-AUC & $\pm$Std & Acc. & '
              r'Prec.\ (F) & Recall (F) & F1 (F) & Prec.\ (S) & Recall (S) & F1 (S) & Score \\')

    tex = (
        r'\begin{table}[htbp]' + '\n'
        r'\caption{Comprehensive Performance Metrics for All Static Model Configurations'
        r' (Improved Feature Set)}' + '\n'
        r'\label{tab:model_comparison}' + '\n'
        r'\resizebox{\textwidth}{!}{%' + '\n'
        r'\begin{tabular}{llccccccccccc}' + '\n'
        r'\toprule' + '\n'
        + header + '\n'
        r'\midrule' + '\n'
        + '\n'.join(rows_tex) + '\n'
        r'\bottomrule' + '\n'
        r'\end{tabular}%' + '\n'
        r'}' + '\n'
        r'\smallskip' + '\n\n'
        r'{\footnotesize \textit{Note.} F~=~Failure class; S~=~Success class; '
        r'AUC~=~test-set ROC-AUC; CV-AUC~=~5-fold CV mean; $\pm$Std~=~CV standard deviation; '
        r'Acc.~=~Accuracy; Prec.~=~Precision; '
        r'Score~=~composite $0.6 \times \text{AUC} + 0.4 \times \text{Recall\,(F)}$. '
        r'Bold row indicates recommended model.}' + '\n'
        r'\end{table}'
    )

    out = table_dir / 'model_comparison.tex'
    out.write_text(tex)
    import logging
    logging.getLogger(__name__).info("Exported APA table → %s", out)


if __name__ == "__main__":
    main()
