import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*valid feature names.*")

OUTCOME_LABELS_ALPHA = [
    'CURED', 'DIAGNOSED', 'DIED', 'EXCLUDED', 'FAILED', 'LOST TO FF-UP',
    'NOT ENROLLED', 'NOT EVALUATED', 'ON TREATMENT', 'Screened', 'TREATMENT COMPLETED'
]
# Alphabetical sort: CURED=0, DIAGNOSED=1, DIED=2, EXCLUDED=3, FAILED=4,
# LOST TO FF-UP=5, NOT ENROLLED=6, NOT EVALUATED=7, ON TREATMENT=8,
# Screened=9, TREATMENT COMPLETED=10
OUTCOME_DECODE = {i: label for i, label in enumerate(sorted(OUTCOME_LABELS_ALPHA))}


class TBExperimentPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.output_dir = Path("results")
        base_dir = Path(__file__).resolve().parent.parent
        self.table_dir = base_dir / "paper" / "apa" / "tables"
        self.fig_dir = base_dir / "paper" / "apa" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path):
        logger.info("Loading dataset: %s", file_path)
        df_raw = pd.read_csv(file_path)
        logger.info("Raw dataset: %d rows, %d columns", len(df_raw), len(df_raw.columns))

        # Schema validation
        required_cols = set(self.get_features('extended')) | {'Outcome/Status'}
        missing = required_cols - set(df_raw.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Decode Outcome/Status — now raw LabelEncoder integers (alphabetical order)
        n_unique = df_raw['Outcome/Status'].dropna().nunique()
        if n_unique != len(OUTCOME_LABELS_ALPHA):
            raise ValueError(
                f"Outcome/Status has {n_unique} unique values; expected {len(OUTCOME_LABELS_ALPHA)}. "
                "Update OUTCOME_LABELS_ALPHA if the dataset changed."
            )
        df_raw['Outcome_Label'] = df_raw['Outcome/Status'].map(OUTCOME_DECODE)

        SUCCESS_OUTCOMES = {'TREATMENT COMPLETED', 'CURED'}
        FAILURE_OUTCOMES = {'DIED', 'LOST TO FF-UP', 'FAILED'}

        df = df_raw[df_raw['Outcome_Label'].isin(SUCCESS_OUTCOMES | FAILURE_OUTCOMES)].copy()
        df['Target'] = df['Outcome_Label'].apply(lambda x: 1 if x in SUCCESS_OUTCOMES else 0)

        n_success = (df['Target'] == 1).sum()
        n_failure = (df['Target'] == 0).sum()
        n_excluded = len(df_raw) - len(df)
        logger.info(
            "After outcome filtering: %d samples retained, %d excluded "
            "(success=%d, failure=%d, imbalance ratio=%.2f:1)",
            len(df), n_excluded, n_success, n_failure, n_success / max(n_failure, 1)
        )
        if n_excluded > 0:
            excluded_labels = df_raw.loc[
                ~df_raw['Outcome_Label'].isin(SUCCESS_OUTCOMES | FAILURE_OUTCOMES), 'Outcome_Label'
            ].value_counts().to_dict()
            logger.debug("Excluded outcome categories: %s", excluded_labels)

        return df

    def get_features(self, version='baseline'):
        baseline = [
            'Age_Final', 'Days_To_Treatment', 'Year', 'Sex', 'Anatomical Site',
            'Registration Group', 'Bacteriologic Status', 'Microscopy Result',
            'Source of Patient', 'Type', 'Province'
        ]
        improved = baseline + [
            'City/Municipality', 'Treatment Health Facility',
            'Screening/Diagnosing Health Facility'
        ]
        extended = improved + [
            'Diagnosis_Month', 'Diagnosis_Season', 'Diagnosis_Quarter',
            'Days_Screening_To_Diagnosis', 'Days_To_Microscopy_Result',
            'Days_To_RDT_Result', 'RDT Result',
        ]
        if version == 'extended':
            return extended
        if version == 'improved':
            return improved
        return baseline

    def get_preprocessor(self, features):
        numeric_cols = [c for c in [
            'Age_Final', 'Days_To_Treatment', 'Year',
            'Days_Screening_To_Diagnosis', 'Days_To_Microscopy_Result',
            'Days_To_RDT_Result', 'Diagnosis_Month', 'Diagnosis_Quarter',
            'Diagnosis_DayOfYear'
        ] if c in features]
        categorical_cols = [f for f in features if f not in numeric_cols]

        return ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
            ]), categorical_cols),
        ], remainder='drop')

    def get_models(self, y_train, use_weights=True):
        weight_ratio = (y_train == 0).sum() / y_train.sum()
        class_weight = 'balanced' if use_weights else None

        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state, class_weight=class_weight),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=self.random_state,
                class_weight=class_weight, n_jobs=-1),
            'XGBoost': XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=self.random_state,
                scale_pos_weight=weight_ratio if use_weights else 1,
                n_jobs=-1, verbosity=0),
            'LightGBM': LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=self.random_state, class_weight=class_weight,
                n_jobs=-1, verbose=-1),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=self.random_state, subsample=0.8),
            'SVM (RBF)': SVC(
                kernel='rbf', probability=True, random_state=self.random_state,
                class_weight=class_weight, C=1.0, gamma='scale'),
        }

    def run_experiment(self, name, sampler_name, model_dict, X_train, y_train, X_test, y_test, preprocessor):
        results = []
        sampler = None
        if sampler_name == 'SMOTE':
            sampler = SMOTE(random_state=self.random_state)
        elif sampler_name == 'SMOTE - ENN':
            sampler = SMOTEENN(random_state=self.random_state)
        elif sampler_name == 'SMOTE - Tomek':
            sampler = SMOTETomek(random_state=self.random_state)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for m_name, model in model_dict.items():
            logger.info("[%s] Training %-22s sampler=%-12s  train=%d  test=%d",
                        name, m_name, sampler_name or 'None', len(X_train), len(X_test))
            t0 = time.time()
            if sampler:
                pipe = ImbPipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('sampler', sampler),
                    ('classifier', model)
                ])
            else:
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

            pipe.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)
            f1_fail = report['0']['f1-score']
            recall_fail = report['0']['recall']

            # 5-fold stratified cross-validation on training data
            cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv,
                                       scoring='roc_auc', n_jobs=-1)

            logger.info("[%s] %-22s  AUC=%.4f  CV-AUC=%.4f±%.4f  Acc=%.4f  F1(fail)=%.4f  %.2fs",
                        name, m_name, auc, cv_aucs.mean(), cv_aucs.std(),
                        acc, f1_fail, train_time)

            results.append({
                'Model': m_name,
                'Sampler': sampler_name or 'None',
                'Accuracy': acc,
                'ROC-AUC': auc,
                'CV-AUC Mean': cv_aucs.mean(),
                'CV-AUC Std': cv_aucs.std(),
                'Precision (Fail)': report['0']['precision'],
                'Recall (Fail)': recall_fail,
                'F1 (Fail)': f1_fail,
                'Precision (Succ)': report['1']['precision'],
                'Recall (Succ)': report['1']['recall'],
                'F1 (Succ)': report['1']['f1-score'],
                'Train Time (s)': round(train_time, 2),
                '_pipeline': pipe
            })

        return pd.DataFrame(results)

    def generate_figures(self, master_df, best_pipeline, X_test, y_test):
        logger.info("Generating evaluation figures → %s", self.fig_dir)
        plt.style.use('seaborn-v0_8-whitegrid')
        FIGSIZE = (8, 5)

        # --- per-model best rows (highest ROC-AUC for each model) ---
        best_per_model = master_df.sort_values('ROC-AUC', ascending=False).groupby('Model').first().reset_index()

        # ── Figure 1: ROC Curves ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random chance')
        for _, row in best_per_model.iterrows():
            pipe = row['_pipeline']
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, linewidth=1.5, label=f"{row['Model']} (AUC={auc:.4f})")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — Best Configuration per Model')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(self.fig_dir / 'fig_roc_curves.pdf', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info("Saved fig_roc_curves.pdf")

        # ── Figure 2: Precision-Recall Curves ────────────────────────────
        prevalence = y_test.mean()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.axhline(prevalence, color='k', linestyle='--', linewidth=0.8,
                   label=f'No-skill baseline ({prevalence:.2f})')
        for _, row in best_per_model.iterrows():
            pipe = row['_pipeline']
            y_proba = pipe.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            ax.plot(rec, prec, linewidth=1.5, label=f"{row['Model']} (AP={ap:.4f})")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves — Best Configuration per Model')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        fig.savefig(self.fig_dir / 'fig_pr_curves.pdf', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info("Saved fig_pr_curves.pdf")

        # ── Figure 3: Confusion Matrix (best model on test set) ──────────
        y_pred_best = best_pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        labels = [f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_norm.flatten())]
        labels = np.array(labels).reshape(2, 2)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                    xticklabels=['Pred Failure', 'Pred Success'],
                    yticklabels=['True Failure', 'True Success'])
        ax.set_title('Confusion Matrix — Best Model (Test Set)')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        fig.tight_layout()
        fig.savefig(self.fig_dir / 'fig_confusion_matrix.pdf', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info("Saved fig_confusion_matrix.pdf")

        # ── Figure 4: CV AUC Boxplot ──────────────────────────────────────
        cv_rows = []
        for _, row in master_df.iterrows():
            cv_rows.append({'Model': row['Model'], 'CV-AUC Mean': row['CV-AUC Mean']})
        cv_df = pd.DataFrame(cv_rows)
        model_order = (cv_df.groupby('Model')['CV-AUC Mean']
                       .median().sort_values(ascending=False).index.tolist())

        fig, ax = plt.subplots(figsize=FIGSIZE)
        cv_df_grouped = [cv_df[cv_df['Model'] == m]['CV-AUC Mean'].values for m in model_order]
        ax.boxplot(cv_df_grouped, labels=model_order, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6))
        ax.set_xlabel('Model')
        ax.set_ylabel('5-Fold CV ROC-AUC')
        ax.set_title('Cross-Validation ROC-AUC Distribution per Model')
        plt.xticks(rotation=20, ha='right')
        fig.tight_layout()
        fig.savefig(self.fig_dir / 'fig_cv_boxplot.pdf', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info("Saved fig_cv_boxplot.pdf")

        # ── Figure 5: Feature Importance ──────────────────────────────────
        best_clf = best_pipeline.steps[-1][1]
        feature_names = None

        try:
            feature_names = list(best_pipeline[:-1].get_feature_names_out())
        except Exception:
            pass

        importances = None
        if hasattr(best_clf, 'feature_importances_'):
            importances = best_clf.feature_importances_
        elif hasattr(best_clf, 'coef_'):
            importances = np.abs(best_clf.coef_[0])

        if importances is not None and feature_names is not None:
            n = min(15, len(importances))
            importances = importances / importances.sum()
            idx = np.argsort(importances)[-n:]
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.barh(range(n), importances[idx], color='steelblue', alpha=0.8)
            ax.set_yticks(range(n))
            ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
            ax.set_xlabel('Normalized Importance')
            ax.set_title(f'Top {n} Feature Importances — Best Model')
            fig.tight_layout()
            fig.savefig(self.fig_dir / 'fig_feature_importance.pdf', bbox_inches='tight', dpi=150)
            plt.close(fig)
            logger.info("Saved fig_feature_importance.pdf")
        else:
            logger.warning("Could not extract feature importances from best model — skipping fig_feature_importance.pdf")

        logger.info("All figures saved to %s", self.fig_dir)

    def export_to_latex(self, df, filename, caption, label):
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
        )

        latex_str = latex_str.replace('\\toprule', '\\toprule\n')
        latex_str = latex_str.replace('\\midrule', '\\midrule\n')
        latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\n')

        output_path = self.table_dir / f"{filename}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_str)
        logger.info("Exported LaTeX table → %s", output_path)


if __name__ == "__main__":
    pipeline = TBExperimentPipeline()
    try:
        base_dir = Path(__file__).resolve().parent.parent
        df = pipeline.load_data(base_dir / "dataset" / "non-temporal" / "2015-2025-ml-ready.csv")
        features = pipeline.get_features('baseline')
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        preprocessor = pipeline.get_preprocessor(features)
        models = pipeline.get_models(y_train)
        results = pipeline.run_experiment(
            "Baseline", None, models, X_train, y_train, X_test, y_test, preprocessor)
        pipeline.export_to_latex(results, "baseline_results",
                                  "Baseline Model Performance", "tab:baseline_results")
    except Exception as e:
        print(f"Error during test run: {e}")
        raise
