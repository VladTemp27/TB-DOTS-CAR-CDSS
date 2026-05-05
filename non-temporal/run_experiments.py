import logging
logging.getLogger().setLevel(logging.CRITICAL)

import pandas as pd
from experiment_pipeline import TBExperimentPipeline
from sklearn.model_selection import train_test_split

def main():
    pipeline = TBExperimentPipeline()
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "dataset" / "non-temporal" / "2015-2025-ml-ready.csv"
    
    print("Loading data...")
    df = pipeline.load_data(dataset_path)
    
    # We will run two main versions: Baseline (Original Features) and Improved (Extended Features)
    versions = [
        ('Baseline', 'baseline', [None]),
        ('Resampling_Comparison', 'improved', ['SMOTE', 'SMOTE - ENN', 'SMOTE - Tomek'])
    ]
    
    all_results = []
    
    for v_name, feat_ver, samplers in versions:
        print(f"\n--- Running {v_name} Experiments ---")
        features = pipeline.get_features(feat_ver)
        X = df[features]
        y = df['Target']
        
        # 70/20/10 split as per the notebooks
        X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)
        
        preprocessor = pipeline.get_preprocessor(features)
        
        for sampler in samplers:
            print(f"Sampling Strategy: {sampler or 'None'}")
            models = pipeline.get_models(y_train)
            res = pipeline.run_experiment(f"{v_name}_{sampler}", sampler, models, X_train, y_train, X_test, y_test, preprocessor)
            all_results.append(res)
            
            # Export individual experiment table
            pipeline.export_to_latex(
                res[['Model', 'ROC-AUC', 'Recall (Fail)', 'F1 (Fail)', 'Accuracy']], 
                f"results_{feat_ver}_{str(sampler).lower()}",
                f"Performance Results for {feat_ver.capitalize()} with {sampler or 'Baseline'}",
                f"tab:{feat_ver}_{str(sampler).lower()}"
            )

    # Create a Master Comparison Table (Best model from each)
    master_df = pd.concat(all_results)
    
    best_model_row = master_df.sort_values('ROC-AUC', ascending=False).iloc[0]
    best_pipeline = best_model_row['_pipeline']
    
    # Export to ONNX for native web browser inference (onnxruntime-web)
    try:
        from skl2onnx import to_onnx
        from sklearn.pipeline import Pipeline
        import numpy as np
        
        # Strip the imblearn sampler if present to make it compatible with skl2onnx
        steps = [(name, step) for name, step in best_pipeline.steps if name != 'sampler']
        clean_pipe = Pipeline(steps)
        
        # Build a dummy sample to infer input schema
        features = best_pipeline.feature_names_in_
        X_dummy = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        
        onnx_model = to_onnx(clean_pipe, X_dummy[:1], target_opset=12,
                             options={id(clean_pipe.steps[-1][1]): {'zipmap': False}})
        
        model_dir = base_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / "tb_outcome_prediction.onnx"
        
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Exported best model to ONNX format at {onnx_path}")
    except Exception as e:
        print(f"Warning: Failed to export to ONNX format ({e})")
        
    # Remove _pipeline for clean LaTeX export
    master_df = master_df.drop(columns=['_pipeline'])
    
    best_summary = master_df.sort_values('ROC-AUC', ascending=False).groupby(['Sampler', 'Model']).head(1)
    
    pipeline.export_to_latex(
        best_summary[['Sampler', 'Model', 'ROC-AUC', 'Recall (Fail)', 'F1 (Fail)']].head(10),
        "top_models_comparison",
        "Top 10 Performing Model Configurations Across All Experiments",
        "tab:top_models"
    )
    
    print("\nAll experiments completed and LaTeX tables exported.")

if __name__ == "__main__":
    main()
