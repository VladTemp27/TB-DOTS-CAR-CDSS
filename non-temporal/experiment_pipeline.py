import logging
logging.getLogger().setLevel(logging.CRITICAL)

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings

# Suppress LightGBM feature names warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*valid feature names.*")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

class TBExperimentPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.output_dir = Path("results")
        base_dir = Path(__file__).resolve().parent.parent
        self.table_dir = base_dir / "paper" / "apa" / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path):
        df_raw = pd.read_csv(file_path)
        
        # Decode Outcome/Status
        OUTCOME_LABELS_ALPHA = [
            'CURED', 'DIAGNOSED', 'DIED', 'EXCLUDED', 'FAILED', 'LOST TO FF-UP',
            'NOT ENROLLED', 'NOT EVALUATED', 'ON TREATMENT', 'Screened', 'TREATMENT COMPLETED'
        ]
        outcome_encoded_sorted = sorted(df_raw['Outcome/Status'].dropna().unique())
        OUTCOME_DECODE = dict(zip(outcome_encoded_sorted, OUTCOME_LABELS_ALPHA))
        df_raw['Outcome_Label'] = df_raw['Outcome/Status'].map(OUTCOME_DECODE)

        SUCCESS_OUTCOMES = {'TREATMENT COMPLETED', 'CURED'}
        FAILURE_OUTCOMES = {'DIED', 'LOST TO FF-UP', 'FAILED'}

        df = df_raw[df_raw['Outcome_Label'].isin(SUCCESS_OUTCOMES | FAILURE_OUTCOMES)].copy()
        df['Target'] = df['Outcome_Label'].apply(lambda x: 1 if x in SUCCESS_OUTCOMES else 0)
        
        # Derived features (if possible)
        if 'Date Started Tx' in df.columns and 'Date of Diagnosis' in df.columns:
            df['_Date_Started_Tx'] = pd.to_datetime(df['Date Started Tx'], errors='coerce')
            df['_Date_of_Diagnosis'] = pd.to_datetime(df['Date of Diagnosis'], errors='coerce')
            df['Diagnosis_to_Treatment_days'] = (df['_Date_Started_Tx'] - df['_Date_of_Diagnosis']).dt.days.clip(lower=0)
        
        return df

    def get_features(self, version='baseline'):
        baseline = [
            'Age', 'Days_To_Treatment', 'Year', 'Sex', 'Anatomical Site',
            'Registration Group', 'Bacteriologic Status', 'Microscopy Result',
            'Source of Patient', 'Type', 'Province'
        ]
        if version == 'improved':
            return baseline + ['City/Municipality', 'Treatment Health Facility', 
                               'Screening/Diagnosing Health Facility']
        return baseline

    def get_preprocessor(self, features):
        # We'll just use a simple imputer + scaler for numeric, and simple imputer for categorical
        # Actually, the dataset seems to have numeric encoded already for most parts
        # Let's identify numeric columns
        numeric_cols = ['Age', 'Days_To_Treatment', 'Year']
        categorical_cols = [f for f in features if f not in numeric_cols]
        
        return ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), [f for f in numeric_cols if f in features]),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ]), [f for f in categorical_cols if f in features]),
        ])

    def get_models(self, y_train, use_weights=True):
        weight_ratio = (y_train == 0).sum() / y_train.sum()
        class_weight = 'balanced' if use_weights else None
        
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state, class_weight=class_weight),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=self.random_state, class_weight=class_weight, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=self.random_state, scale_pos_weight=weight_ratio if use_weights else 1, n_jobs=-1, verbosity=0),
            'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=self.random_state, class_weight=class_weight, n_jobs=-1, verbose=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=self.random_state, subsample=0.8),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=self.random_state, class_weight=class_weight, C=1.0, gamma='scale'),
        }

    def run_experiment(self, name, sampler_name, model_dict, X_train, y_train, X_test, y_test, preprocessor):
        results = []
        sampler = None
        if sampler_name == 'SMOTE': sampler = SMOTE(random_state=self.random_state)
        elif sampler_name == 'SMOTE - ENN': sampler = SMOTEENN(random_state=self.random_state)
        elif sampler_name == 'SMOTE - Tomek': sampler = SMOTETomek(random_state=self.random_state)

        for m_name, model in model_dict.items():
            print(f"[{name}] Training {m_name}...")
            t0 = time.time()
            if sampler:
                pipe = ImbPipeline(steps=[('preprocessor', preprocessor), ('sampler', sampler), ('classifier', model)])
            else:
                pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0
            
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            results.append({
                'Model': m_name,
                'Sampler': sampler_name or 'None',
                'Accuracy': accuracy_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'Precision (Fail)': report['0']['precision'],
                'Recall (Fail)': report['0']['recall'],
                'F1 (Fail)': report['0']['f1-score'],
                'Precision (Succ)': report['1']['precision'],
                'Recall (Succ)': report['1']['recall'],
                'F1 (Succ)': report['1']['f1-score'],
                'Train Time (s)': round(train_time, 2)
            })
            
        return pd.DataFrame(results)

    def export_to_latex(self, df, filename, caption, label):
        # Format percentages/decimals for LaTeX
        tex_df = df.copy()
        numeric_cols = tex_df.select_dtypes(include=[np.number]).columns
        tex_df[numeric_cols] = tex_df[numeric_cols].map(lambda x: f"{x:.4f}")
        
        latex_str = tex_df.to_latex(
            index=False,
            caption=caption,
            label=label,
            position='htbp',
            column_format='l' + 'c' * (len(df.columns) - 1),
            escape=False,
            bold_rows=True
        )
        
        # Post-processing to make it look even better in APA
        latex_str = latex_str.replace('\\toprule', '\\toprule\n')
        latex_str = latex_str.replace('\\midrule', '\\midrule\n')
        latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\n')
        
        output_path = self.table_dir / f"{filename}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"Exported LaTeX table to {output_path}")

if __name__ == "__main__":
    # Example usage / Test
    pipeline = TBExperimentPipeline()
    try:
        import os
        base_dir = Path(__file__).resolve().parent.parent
        df = pipeline.load_data(base_dir / "dataset" / "non-temporal" / "2015-2025-ml-ready.csv")
        features = pipeline.get_features('baseline')
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        preprocessor = pipeline.get_preprocessor(features)
        models = pipeline.get_models(y_train)
        
        results = pipeline.run_experiment("Baseline", None, models, X_train, y_train, X_test, y_test, preprocessor)
        pipeline.export_to_latex(results, "baseline_results", "Baseline Model Performance", "tab:baseline_results")
    except Exception as e:
        print(f"Error during test run: {e}")
