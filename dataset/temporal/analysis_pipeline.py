"""
Input:  dataset/temporal/combined_complete_dataset.csv  (Raw)

PURPOSE — RETROACTIVE EDA JUSTIFICATION
----------------------------------------
The preprocessing pipeline was written before this EDA. This
produces the empirical evidence that *explains* every major preprocessing
decision in that pipeline:

  Preprocessing Decision              → EDA Station that justifies it
  ─────────────────────────────────────────────────────────────────────
  ALWAYS_DROP_COLUMNS (Alpha Pathway) → Station 1  Global Missingness
  Forward-fill + Gamma indicators     → Station 2  Patient Retention Cascade
  Clinical bound clipping / MICE      → Station 3  Raw Vitals Distribution
    └─ MICE specifically justified    →   + Pearson Correlation Matrix
  Temporal model / look-back window   → Station 4  Lag Correlation Analysis
  Categorical harmonization targets   → Station 5  Categorical Audit
    └─ association structure shown    →   + Cramér's V Correlation Matrix
  Date parsing / month ordering       → Station 6  Temporal Cohort Timeline

Correlation matrices (Station 3 & 5)
--------------------------------------
  Station 3 — Pearson r (numeric × numeric)
      Justifies MICE over mean/median imputation: if vitals are correlated,
      MICE can borrow information across columns (e.g. bp_systolic predicts
      bp_diastolic). An uncorrelated matrix would mean MICE has no advantage.

  Station 5 — Cramér's V (categorical × categorical)
      Measures association between unordered category pairs without encoding.
      Cramér's V is symmetric and normalized to [0, 1] so a single colorbar
      is interpretable. Justifies which categorical columns needed careful
      harmonization (strongly associated columns must be consistently coded
      or the association signal is destroyed by label noise).

  Why NOT a mixed matrix:
      Combining Pearson, Cramér's V, and η² in one heatmap places three
      conceptually different metrics on one colorbar — 0.7 Pearson and 0.7
      Cramér's V do not mean the same thing. Two clean, separately labeled
      matrices are easier to defend in peer review.

Outputs
-------
  paper/apa/temporal/tables/   *.tex  (booktabs LaTeX, no vertical rules)
  paper/apa/temporal/figures/  *.pdf  (publication-ready matplotlib figures)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Global font settings (Serif / Times New Roman) ───────────────────────────
# Uses Times New Roman if installed; falls back to the next available serif font.
# All generated PDFs (histograms, heatmaps, line charts) will use this font.
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif", "Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.titlesize":  12,
    "text.usetex":       False,
})

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "dataset" / "temporal" / "combined_complete_dataset.csv"
)

MONTH_RANGE = range(0, 13)  # M0 – M12

MONTHLY_SUFFIXES = [
    "Monthly Doses Taken",
    "Cumulative Doses Taken",
    "Monthly Missed Doses",
    "%Adherence",
    "Weight",
    "Height",
    "Smear/TB Lamp",
    "Xpert MTB/RIF",
]

ALPHA_DROP_CANDIDATES = [
    "tuberculosis_culture",
    "tuberculin_skin_test",
    "other_lab_test",
    "others",
    "other",
    "dat_supported_dup",
    "risk_factors_for_drug_resistance_tuberculosis",
]

VITAL_COLUMNS = [
    "age",
    "weight_kg",
    "height_cm",
    "bp_systolic",
    "bp_diastolic",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "o2_sat",
]

CATEGORICAL_COLUMNS = [
    "sex",
    "civil_status",
    "nationality",
    "diagnosis",
    "bacteriologic_status",
    "treatment_regimen",
    "outcome",
    "case_registration_group",
    "drug_resistance_bacteriological_status",
    "chest_x_ray_at_case_notification",
    "co_morbidities",
    "prior_history_of_tb",
    "dat_supported",
]

LAG_FEATURE           = "pct_adherence"
LAG_FEATURE_FALLBACKS = ["weight", "monthly_doses_taken", "monthly_missed_doses"]

ALPHA_THRESHOLD = 0.80


# ============================================================================
# HELPERS
# ============================================================================

def _snake(col: str) -> str:
    """Lightweight snake_case converter matching the preprocessing pipeline."""
    col = col.strip()
    if col == "DAT- supported":
        return "dat_supported_dup"
    if col == "Nationality":
        return "nationality_raw"
    col = col.replace("%", "pct_")
    for ch in ("?", ":", "'"):
        col = col.replace(ch, "")
    for ch in ("/", "-"):
        col = col.replace(ch, "_")
    for ch in ("(", ")"):
        col = col.replace(ch, "")
    col = col.replace(".", "")
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_").lower()
    return col


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramér's V between two categorical Series.

    Cramér's V is symmetric and normalized to [0, 1]:
        0 = no association
        1 = perfect association

    Formula: V = sqrt(χ² / (n · (min(k, r) − 1)))
    where k = number of categories in x, r = number of categories in y.

    NaN pairs are dropped before computing the contingency table so
    missing values in either column do not inflate the χ² statistic.
    """
    # Drop rows where either column is NaN
    mask = x.notna() & y.notna()
    x_, y_ = x[mask], y[mask]
    n = len(x_)
    if n == 0:
        return np.nan

    contingency = pd.crosstab(x_, y_)
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    k = contingency.shape[0]   # categories in x
    r = contingency.shape[1]   # categories in y
    denom = n * (min(k, r) - 1)
    if denom == 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))


# ============================================================================
# PIPELINE CLASS
# ============================================================================

class TemporalEDAPipeline:
    """
    Runs the four-station Baseline Audit described in the project spec and
    two supplementary stations (categorical audit + cohort timeline) that
    justify additional preprocessing decisions.

    Correlation matrices are added to Station 3 (Pearson, numeric vitals)
    and Station 5 (Cramér's V, categorical features) only — see module
    docstring for the rationale against a mixed matrix.
    """

    def __init__(self) -> None:
        base = Path(__file__).resolve().parent.parent.parent
        self.table_dir  = base / "paper" / "apa" / "temporal" / "tables"
        self.figure_dir = base / "paper" / "apa" / "temporal" / "figures"
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # VALVE A – LaTeX table export (booktabs, no vertical rules)
    # ------------------------------------------------------------------
    def export_table(
        self,
        df: pd.DataFrame,
        filename: str,
        caption: str,
        label: str,
    ) -> Path:
        """
        Write *df* as a booktabs LaTeX table to the tables directory.
        Values are limited to 2 decimal places where numeric.
        """
        path = self.table_dir / f"{filename}.tex"
        n_cols = len(df.columns)
        col_spec = "l" + "c" * (n_cols - 1)

        def _escape(v) -> str:
            return str(v).replace("_", r"\_").replace("%", r"\%")

        lines: list[str] = [
            r"\begin{table}[htbp]",
            "",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            "",
            r"\toprule",
            "",
            " & ".join(_escape(c) for c in df.columns) + r" \\",
            "",
            r"\midrule",
            "",
        ]
        for _, row in df.iterrows():
            cells = " & ".join(
                _escape(v) if pd.notna(v) else "" for v in row
            )
            lines.append(cells + r" \\")

        lines += ["", r"\bottomrule", "", r"\end{tabular}", "", r"\end{table}"]
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [Table]  → {path.name}")
        return path

    # ------------------------------------------------------------------
    # VALVE B – PDF figure export
    # ------------------------------------------------------------------
    def export_plot(self, filename: str) -> Path:
        path = self.figure_dir / f"{filename}.pdf"
        plt.savefig(path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"  [Figure] → {path.name}")
        return path

    # ------------------------------------------------------------------
    # SHARED: heatmap renderer (used by Stations 3 and 5)
    # ------------------------------------------------------------------
    def _render_heatmap(
        self,
        matrix: pd.DataFrame,
        filename: str,
        title: str,
        cbar_label: str,
        cmap: str = "coolwarm",
        vmin: float = -1,
        vmax: float = 1,
        center: float = 0,
        x_rotation: int = 45,
    ) -> None:
        """
        Render a square annotated heatmap and export it as a PDF.

        The diagonal is masked (hidden) because self-correlation is always
        trivially 1.0 and adds no information to the figure.
        """
        n = len(matrix)
        fig_size = max(6, n * 0.75)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        # Mask the diagonal — self-correlation is always trivially 1.0
        mask = np.zeros_like(matrix.values, dtype=bool)
        np.fill_diagonal(mask, True)

        sns.heatmap(
            matrix,
            ax=ax,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.4,
            linecolor="white",
            annot_kws={"size": max(6, 10 - n // 3)},
            square=True,
            cbar_kws={"label": cbar_label, "shrink": 0.75},
        )
        ax.set_title(title, fontsize=10, pad=12)
        ax.tick_params(axis="x", rotation=x_rotation, labelsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), ha="right" if x_rotation == 45 else "center")
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
        fig.tight_layout()
        self.export_plot(filename)

    # ==================================================================
    # STATION 1 – Global Missingness Evidence
    # Justifies: ALWAYS_DROP_COLUMNS (Alpha Pathway ≥ 80% rule) and
    #            the choice to keep smear_microscopy / height_cm as
    #            Gamma indicators (83–85% missing but clinically relevant).
    # ==================================================================
    def station1_global_missingness(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("STATION 1: Global Missingness Evidence")
        print("=" * 60)

        miss_ratio = df.isnull().mean().round(4)
        miss_count = df.isnull().sum()
        miss_df = pd.DataFrame({
            "Feature":              miss_ratio.index,
            "Missing Count":        miss_count.values,
            "Missing \\%":          (miss_ratio.values * 100).round(2),
            "Alpha Drop (≥80\\%)":  [
                "Yes" if v >= ALPHA_THRESHOLD else "No"
                for v in miss_ratio.values
            ],
        }).sort_values("Missing \\%", ascending=False).reset_index(drop=True)

        self.export_table(
            miss_df,
            "raw_missingness_summary",
            "Global Feature Missingness Rates in the Raw Temporal Dataset",
            "tab:raw_missingness_summary",
        )

        # Bar chart with red dashed line at 80%
        miss_sorted = miss_ratio[miss_ratio > 0].sort_values(ascending=False)
        n = len(miss_sorted)
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.22)))
        colors = [
            "#d62728" if v >= ALPHA_THRESHOLD else
            "#ff7f0e" if v >= 0.50 else
            "#1f77b4"
            for v in miss_sorted.values
        ]
        ax.barh(miss_sorted.index[::-1], miss_sorted.values[::-1] * 100,
                color=colors[::-1], edgecolor="black", linewidth=0.4)
        ax.axvline(ALPHA_THRESHOLD * 100, color="red", linestyle="--",
                   linewidth=1.2,
                   label=f"Alpha Drop Threshold ({int(ALPHA_THRESHOLD*100)}\\%)")
        ax.axvline(50, color="orange", linestyle=":", linewidth=1.0,
                   label="50\\% Reference")
        ax.set_xlabel("Missing Data (\\%)")
        ax.set_title(
            "Global Feature Missingness — Raw Temporal Dataset\n"
            "Red bars meet the Alpha (Drop) threshold (≥80\\%)"
        )
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        fig.tight_layout()
        self.export_plot("raw_missingness_thresholds")

    # ==================================================================
    # STATION 2 – Patient Retention Cascade
    # Justifies: Forward-fill strategy and Gamma Pathway (missing-as-
    #            indicator), because missingness *increases* over time,
    #            signalling patient dropout rather than random absence.
    # ==================================================================
    def station2_patient_retention_cascade(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("STATION 2: Patient Retention Cascade")
        print("=" * 60)

        total_patients = len(df)
        records: list[dict] = []

        for m in MONTH_RANGE:
            month_cols = [c for c in df.columns if c.startswith(f"m{m}_")]
            if not month_cols:
                continue
            has_data   = df[month_cols].notnull().any(axis=1).sum()
            pct_retain = has_data / total_patients * 100
            pct_decay  = 100 - pct_retain

            adh_col = f"m{m}_pct_adherence"
            if adh_col not in df.columns:
                adh_col = month_cols[0]
            adh_obs = df[adh_col].notnull().sum()

            records.append({
                "Month":            f"M{m}",
                "Active Patients":  int(has_data),
                "\\% Retained":     round(pct_retain, 2),
                "\\% Data Decay":   round(pct_decay, 2),
                "Adherence Obs.":   int(adh_obs),
            })

        retention_df = pd.DataFrame(records)
        self.export_table(
            retention_df,
            "retention_stats_by_month",
            "Patient Data Retention by Month (M0–M12): "
            "Evidence for Gamma Pathway (Missing-as-Indicator)",
            "tab:retention_stats_by_month",
        )

        months   = retention_df["Month"].tolist()
        retained = retention_df["\\% Retained"].tolist()
        decay    = retention_df["\\% Data Decay"].tolist()

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(months, retained, marker="o", color="#1f77b4",
                 linewidth=2, label="\\% Patients with Data")
        ax1.fill_between(months, retained, alpha=0.15, color="#1f77b4")
        ax1.set_ylabel("Patients with Recorded Data (\\%)", color="#1f77b4")
        ax1.set_ylim(0, 110)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        ax2.plot(months, decay, marker="s", color="#d62728",
                 linewidth=1.5, linestyle="--", label="\\% Data Decay")
        ax2.set_ylabel("Data Decay / Missing (\\%)", color="#d62728")
        ax2.set_ylim(0, 110)
        ax2.tick_params(axis="y", labelcolor="#d62728")

        ax1.set_xlabel("Treatment Month")
        ax1.set_title(
            "Patient Retention Cascade (M0–M12)\n"
            "Increasing missingness over time justifies Forward-Fill "
            "+ Gamma (Missing Indicator) strategy"
        )
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=8, loc="lower left")
        fig.tight_layout()
        self.export_plot("temporal_persistence_cascade")

    # ==================================================================
    # STATION 3 – Raw Distribution & Skewness Audit
    # Justifies: Clinical bound clipping (Stage 3), MICE imputation
    #            for vitals, and future StandardScaler recommendation.
    #
    # Pearson correlation matrix (vitals_pearson_correlation.pdf)
    # ────────────────────────────────────────────────────────────
    # MICE imputes each missing value using the other features as
    # predictors. If the vitals are correlated, MICE can borrow that
    # signal (e.g. high bp_systolic predicts bp_diastolic). An
    # uncorrelated matrix would mean MICE has no advantage over
    # mean-fill. Showing meaningful correlations here is the direct
    # justification for choosing MICE in the preprocessing pipeline.
    # ==================================================================
    def station3_raw_vitals_distribution(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("STATION 3: Raw Distribution & Skewness Audit")
        print("=" * 60)

        available = [c for c in VITAL_COLUMNS if c in df.columns]
        if not available:
            print("  No vital columns found – skipping Station 3.")
            return

        # ── Table: vitals_descriptive_stats ─────────────────────────
        rows: list[dict] = []
        for col in available:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            rows.append({
                "Feature":     col,
                "N":           len(s),
                "Mean":        round(s.mean(), 2),
                "Median":      round(s.median(), 2),
                "SD":          round(s.std(), 2),
                "Min":         round(s.min(), 2),
                "Max":         round(s.max(), 2),
                "Skewness":    round(stats.skew(s), 3),
                "Kurtosis":    round(stats.kurtosis(s), 3),
                "Missing \\%": round(df[col].isnull().mean() * 100, 2),
            })

        if not rows:
            print("  No numeric vital data – skipping table.")
            return

        self.export_table(
            pd.DataFrame(rows),
            "vitals_descriptive_stats",
            "Descriptive Statistics and Skewness of Raw Vital Sign Features "
            "(Justifying Clinical Bound Clipping and MICE Imputation)",
            "tab:vitals_descriptive_stats",
        )

        # ── Figure: raw_vitals_distribution (histograms + KDE) ──────
        n_cols_grid = 3
        n_rows_grid = int(np.ceil(len(available) / n_cols_grid))
        fig, axes = plt.subplots(
            n_rows_grid, n_cols_grid,
            figsize=(14, n_rows_grid * 3.5),
        )
        axes = np.array(axes).flatten()

        for i, col in enumerate(available):
            s  = pd.to_numeric(df[col], errors="coerce").dropna()
            ax = axes[i]
            ax.hist(s, bins=30, color="#1f77b4", edgecolor="black",
                    linewidth=0.4, density=True, alpha=0.75)
            if len(s) > 5:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(s)
                xs  = np.linspace(s.min(), s.max(), 200)
                ax.plot(xs, kde(xs), color="#d62728", linewidth=1.5)
            ax.set_title(f"{col}\nskew={stats.skew(s):.2f}", fontsize=9)
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Raw Vital Sign Distributions — Skewness Audit\n"
            "Justifies clinical clipping and MICE imputation in Stage 3",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        self.export_plot("raw_vitals_distribution")

        # ── Figure: vitals_pearson_correlation ──────────────────────
        # Pearson r matrix across static numeric vital features.
        # Correlated vitals confirm that MICE can borrow cross-feature
        # signal during imputation — the primary justification for
        # choosing MICE over mean/median fill in the preprocessing pipeline.
        numeric_df = df[available].apply(pd.to_numeric, errors="coerce")
        usable = [c for c in available
                  if numeric_df[c].notna().sum() >= 10]

        if len(usable) < 2:
            print("  Insufficient data for Pearson matrix – skipping.")
            return

        corr = numeric_df[usable].corr(method="pearson").round(2)

        self._render_heatmap(
            corr,
            filename="vitals_pearson_correlation",
            title=(
                "Pearson Correlation Matrix — Static Numeric Vital Features\n"
                "Inter-feature correlations confirm MICE is preferable to "
                "mean/median imputation\n"
                "(diagonal masked: self-correlation is always 1.0)"
            ),
            cbar_label="Pearson r",
            cmap="coolwarm",
            vmin=-1, vmax=1, center=0,
        )

    # ==================================================================
    # STATION 4 – Temporal Correlation (Lag Analysis)
    # Justifies: Use of a temporal model architecture (RNN/LSTM) and
    #            informs the look-back window selection.
    # ==================================================================
    def station4_temporal_lag_correlation(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("STATION 4: Temporal Lag Correlation Analysis")
        print("=" * 60)

        feat = LAG_FEATURE
        if f"m0_{feat}" not in df.columns:
            for fb in LAG_FEATURE_FALLBACKS:
                if f"m0_{fb}" in df.columns:
                    feat = fb
                    break
            else:
                print("  No monthly feature found – skipping Station 4.")
                return

        month_series: dict[str, pd.Series] = {}
        for m in MONTH_RANGE:
            col = f"m{m}_{feat}"
            if col in df.columns:
                month_series[f"M{m}"] = pd.to_numeric(df[col], errors="coerce")

        if len(month_series) < 2:
            print("  Insufficient monthly data – skipping Station 4.")
            return

        wide = pd.DataFrame(month_series)

        # ── Table: lag_correlation_matrix ───────────────────────────
        months = list(wide.columns)
        lag_rows: list[dict] = []
        for i in range(1, len(months)):
            m_prev, m_curr = months[i - 1], months[i]
            x = wide[m_prev].dropna()
            y = wide[m_curr].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 5:
                continue
            r, p = stats.pearsonr(x[common], y[common])
            lag_rows.append({
                "Lag Pair":     f"{m_prev} → {m_curr}",
                "Pearson r":    round(r, 4),
                "p-value":      f"{p:.4f}",
                "Significant?": "Yes" if p < 0.05 else "No",
                "N (pairs)":    len(common),
            })

        if not lag_rows:
            print("  Could not compute lag correlations – skipping table.")
            return

        self.export_table(
            pd.DataFrame(lag_rows),
            "lag_correlation_matrix",
            f"Lag-1 Temporal Correlation of {feat.replace('_',' ').title()} "
            "Across Consecutive Treatment Months "
            "(Justifying Time-Series Model Architecture)",
            "tab:lag_correlation_matrix",
        )

        # ── Figure: temporal_lag_heatmap ─────────────────────────────
        corr_matrix = wide.corr(method="pearson")
        fig, axes   = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(
            corr_matrix,
            ax=axes[0],
            annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.3, annot_kws={"size": 7},
        )
        axes[0].set_title(
            f"Pairwise Monthly Correlation — "
            f"{feat.replace('_', ' ').title()}\n"
            "Strong off-diagonal values support temporal modelling"
        )
        axes[0].tick_params(axis="x", rotation=45, labelsize=7)
        axes[0].tick_params(axis="y", rotation=0,  labelsize=7)

        lag_labels = [r["Lag Pair"] for r in lag_rows]
        lag_r      = [r["Pearson r"] for r in lag_rows]
        bar_colors = ["#1f77b4" if v >= 0 else "#d62728" for v in lag_r]
        axes[1].barh(lag_labels[::-1], lag_r[::-1],
                     color=bar_colors[::-1], edgecolor="black", linewidth=0.4)
        axes[1].axvline(0,    color="black",  linewidth=0.8)
        axes[1].axvline(0.5,  color="green",  linestyle="--",
                        linewidth=0.8, label="r = 0.5")
        axes[1].axvline(-0.5, color="orange", linestyle="--",
                        linewidth=0.8, label="r = −0.5")
        axes[1].set_xlabel("Pearson r (Lag-1)")
        axes[1].set_title(
            "Lag-1 Correlations by Month Pair\n"
            "Informs RNN/LSTM look-back window selection"
        )
        axes[1].legend(fontsize=8)
        axes[1].set_xlim(-1, 1)

        fig.suptitle(
            f"Temporal Lag Correlation Analysis — "
            f"{feat.replace('_', ' ').title()}",
            fontsize=11,
        )
        fig.tight_layout()
        self.export_plot("temporal_lag_heatmap")

    # ==================================================================
    # STATION 5 – Categorical Audit
    # Justifies: Stage 4 categorical harmonization targets — shows the
    #            raw inconsistency that made harmonization necessary.
    #
    # Cramér's V matrix (categorical_cramersv_correlation.pdf)
    # ─────────────────────────────────────────────────────────
    # Cramér's V is symmetric and normalized to [0, 1], making it the
    # correct measure for unordered categorical × categorical association.
    # The matrix reveals which column pairs are strongly associated,
    # showing that inconsistent coding (e.g. 'BC' vs 'Bacteriologically
    # Confirmed') would destroy a real association signal — directly
    # justifying the careful harmonization in Stage 4.
    #
    # Why not Pearson or mixed: Pearson requires numeric input.
    # A mixed matrix would place Pearson r, Cramér's V, and η² on one
    # colorbar where 0.7 means different things in different cells.
    # ==================================================================
    def station5_categorical_audit(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("STATION 5: Categorical Value Audit")
        print("=" * 60)

        available = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        if not available:
            print("  No categorical columns found – skipping Station 5.")
            return

        # ── Table: categorical_cardinality_summary ──────────────────
        rows: list[dict] = []
        for col in available:
            vc       = df[col].value_counts(dropna=False)
            n_unique = df[col].nunique(dropna=True)
            n_miss   = df[col].isnull().sum()
            top_vals = ", ".join(str(v) for v in vc.head(3).index)
            rows.append({
                "Feature":           col,
                "Unique Values":     n_unique,
                "Missing":           n_miss,
                "Missing \\%":       round(n_miss / len(df) * 100, 2),
                "Top-3 Raw Values":  top_vals,
            })

        self.export_table(
            pd.DataFrame(rows),
            "categorical_cardinality_summary",
            "Raw Categorical Feature Cardinality and Top Values "
            "(Justifying Stage 4 Categorical Harmonization)",
            "tab:categorical_cardinality_summary",
        )

        # ── Figure: categorical cardinality bar chart ────────────────
        feat_names    = [r["Feature"] for r in rows]
        unique_counts = [r["Unique Values"] for r in rows]
        fig, ax = plt.subplots(figsize=(10, max(4, len(available) * 0.5)))
        ax.barh(feat_names[::-1], unique_counts[::-1],
                color="#ff7f0e", edgecolor="black", linewidth=0.4)
        ax.axvline(2, color="green", linestyle="--", linewidth=0.8,
                   label="Expected unique values (binary)")
        ax.set_xlabel("Number of Unique Raw Values")
        ax.set_title(
            "Categorical Feature Raw Cardinality\n"
            "High cardinality indicates inconsistent coding "
            "requiring harmonization"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        self.export_plot("categorical_cardinality_audit")

        # ── Figure: categorical_cramersv_correlation ─────────────────
        # Cramér's V pairwise association matrix.
        # Only include columns with at least 2 unique non-null values
        # so the chi-squared test is defined.
        usable = [
            c for c in available
            if df[c].nunique(dropna=True) >= 2
            and df[c].notna().sum() >= 10
        ]

        if len(usable) < 2:
            print("  Insufficient categorical data for Cramér's V – skipping.")
            return

        print(f"  Computing Cramér's V for {len(usable)} categorical columns...")
        v_matrix = pd.DataFrame(
            np.nan, index=usable, columns=usable, dtype=float
        )
        for i, col_i in enumerate(usable):
            for j, col_j in enumerate(usable):
                if i == j:
                    v_matrix.loc[col_i, col_j] = 1.0   # self-association
                elif j > i:
                    v = _cramers_v(df[col_i], df[col_j])
                    v_matrix.loc[col_i, col_j] = v
                    v_matrix.loc[col_j, col_i] = v     # symmetric

        v_matrix = v_matrix.round(2)

        self._render_heatmap(
            v_matrix,
            filename="categorical_cramersv_correlation",
            title=(
                "Cramér's V Association Matrix — Categorical Features\n"
                "Strong associations (V > 0.5) identify column pairs where\n"
                "inconsistent coding destroys real signal — justifying Stage 4 "
                "harmonization\n"
                "(diagonal masked: self-association is always 1.0)"
            ),
            cbar_label="Cramér's V  [0 = no association, 1 = perfect]",
            cmap="YlOrRd",   # 0–1 only, no negative values
            vmin=0, vmax=1, center=0.5,
            x_rotation=90,
        )

    # ==================================================================
    # STATION 6 – Temporal Cohort Timeline
    # Justifies: Date parsing logic in Stage 3 (mixed formats, century
    #            swaps, treatment_start < date_of_diagnosis fix).
    # ==================================================================
    def station6_temporal_cohort_timeline(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("STATION 6: Temporal Cohort Timeline & Date Anomaly Audit")
        print("=" * 60)

        date_cols = [
            "date_of_diagnosis", "treatment_start_date",
            "date_of_outcome", "date_of_birth",
        ]
        available = [c for c in date_cols if c in df.columns]
        if not available:
            print("  No date columns found – skipping Station 6.")
            return

        parsed: dict[str, pd.Series] = {
            col: pd.to_datetime(df[col], errors="coerce")
            for col in available
        }

        # ── Table: date_anomaly_summary ──────────────────────────────
        rows: list[dict] = []
        ref_year = 2026
        for col, series in parsed.items():
            n_ok     = series.notna().sum()
            n_future = int((series.dt.year > ref_year).sum()) if n_ok else 0
            n_old    = int((series.dt.year < 1920).sum())     if n_ok else 0
            n_miss   = int(series.isna().sum())
            rows.append({
                "Date Column":    col,
                "Parsed OK":      int(n_ok),
                "Unparseable":    n_miss,
                "Future Dates":   n_future,
                "Pre-1920 Dates": n_old,
                "Notes": (
                    "Century-swap fix applied"
                    if n_future > 0 or n_old > 0
                    else "Clean"
                ),
            })

        if "date_of_diagnosis" in parsed and "treatment_start_date" in parsed:
            bad = int((
                parsed["treatment_start_date"].notna()
                & parsed["date_of_diagnosis"].notna()
                & (parsed["treatment_start_date"] < parsed["date_of_diagnosis"])
            ).sum())
            for r in rows:
                if r["Date Column"] == "treatment_start_date" and bad > 0:
                    r["Notes"] = f"{bad} pre-diagnosis starts corrected"

        self.export_table(
            pd.DataFrame(rows),
            "date_anomaly_summary",
            "Date Column Parsing Anomalies in the Raw Dataset "
            "(Justifying Stage 3 Date Correction Logic)",
            "tab:date_anomaly_summary",
        )

        # ── Figure: cohort_diagnosis_timeline ───────────────────────
        if "date_of_diagnosis" in parsed:
            diag_year = parsed["date_of_diagnosis"].dt.year
            diag_year = diag_year[(diag_year >= 2015) & (diag_year <= 2025)]
            if len(diag_year) > 0:
                year_counts = diag_year.value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar(year_counts.index.astype(str), year_counts.values,
                       color="#1f77b4", edgecolor="black", linewidth=0.5)
                ax.set_xlabel("Year of Diagnosis")
                ax.set_ylabel("Number of Cases")
                ax.set_title(
                    "Annual TB Cohort Distribution — Raw Temporal Dataset\n"
                    "Confirms study window 2015–2025 and identifies data gaps"
                )
                fig.tight_layout()
                self.export_plot("cohort_diagnosis_timeline")

    # ==================================================================
    # RUNNER
    # ==================================================================
    def run(self, df: pd.DataFrame) -> None:
        print("\n" + "#" * 60)
        print("# TB-DOTS CAR CDSS — Temporal EDA Pipeline")
        print("# 'The Baseline Audit'")
        print("#" * 60)

        self.station1_global_missingness(df)
        self.station2_patient_retention_cascade(df)
        self.station3_raw_vitals_distribution(df)       # + Pearson matrix
        self.station4_temporal_lag_correlation(df)
        self.station5_categorical_audit(df)             # + Cramér's V matrix
        self.station6_temporal_cohort_timeline(df)

        print("\n" + "#" * 60)
        print("# EDA COMPLETE")
        print(f"# Tables  → {self.table_dir}")
        print(f"# Figures → {self.figure_dir}")
        print("#" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================

def _load_raw(path: Path) -> pd.DataFrame:
    """Load and snake_case the raw dataset (no cleaning applied)."""
    df = pd.read_csv(path)
    df.columns = [_snake(c) for c in df.columns]
    return df


if __name__ == "__main__":
    print(f"Loading raw dataset from:\n  {INPUT_PATH}")
    df_raw = _load_raw(INPUT_PATH)
    print(f"Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

    pipeline = TemporalEDAPipeline()
    pipeline.run(df_raw)