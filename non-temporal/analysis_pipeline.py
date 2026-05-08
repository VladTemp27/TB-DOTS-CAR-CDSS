from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator


@dataclass
class TBAnalysisPipeline:
    output_dir: Path

    def __init__(self, output_dir: Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent  # go up from non-temporal/ to project root
        self.output_dir = output_dir or (base_dir / "paper" / "apa" / "figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tables go to paper/apa/tables/
        self.table_dir = base_dir / "paper" / "apa" / "tables"
        self.table_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        return df.replace("No Data", np.nan)

    def compute_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = pd.DataFrame({
            "Column": df.columns,
            "Missing_Count": df.isnull().sum(),
            "Missing_Percentage": (df.isnull().sum() / len(df) * 100).round(2),
            "Data_Type": df.dtypes.astype(str),
        })
        return summary.sort_values("Missing_Percentage", ascending=False).reset_index(drop=True)

    def compute_missing_by_year(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        if "Year" not in df.columns:
            return pd.DataFrame()
        available = [c for c in columns if c in df.columns]
        if not available:
            return pd.DataFrame()
        return pd.DataFrame({
            col: df.groupby("Year")[col].apply(lambda x: (x.isnull().sum() / len(x) * 100).round(2))
            for col in available
        })

    def compute_yearly_case_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Year" not in df.columns:
            return pd.DataFrame(columns=["Year", "Case_Count"])
        return df.groupby("Year").size().reset_index(name="Case_Count")

    def compute_value_counts(self, df: pd.DataFrame, column: str, top_n: int | None = None) -> pd.DataFrame:
        if column not in df.columns:
            return pd.DataFrame(columns=[column, "Count", "Percent"])
        counts = df[column].value_counts(dropna=False)
        if top_n:
            counts = counts.head(top_n)
        total = counts.sum()
        return pd.DataFrame({
            column: counts.index.astype(str),
            "Count": counts.values,
            "Percent": (counts.values / total * 100).round(2),
        })

    # ------------------------------------------------------------------ #
    #  LaTeX table output                                                  #
    # ------------------------------------------------------------------ #
    def save_table_latex(
        self,
        df: pd.DataFrame,
        name: str,
        caption: str = "",
        label: str = "",
    ) -> Path:
        """Render *df* as a publication-style LaTeX table and write it to the
        tables directory.  Returns the path to the written .tex file.

        The generated table uses booktabs rules (\\toprule / \\midrule /
        \\bottomrule) and no vertical rules, matching the project style.
        """
        output_path = self.table_dir / f"{name}.tex"

        n_cols = len(df.columns)

        # Column alignment: first column left-aligned, rest centred
        col_spec = "l" + "c" * (n_cols - 1)

        caption_str = caption or name.replace("_", " ").title()
        label_str = label or f"tab:{name}"

        lines: list[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append("")
        lines.append(rf"\caption{{{caption_str}}}")
        lines.append(rf"\label{{{label_str}}}")
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append("")
        lines.append(r"\toprule")
        lines.append("")

        # Header row
        header = " & ".join(str(c).replace("_", r"\_") for c in df.columns) + r" \\"
        lines.append(header)
        lines.append("")
        lines.append(r"\midrule")
        lines.append("")

        # Data rows
        for _, row in df.iterrows():
            cells = " & ".join(
                str(v).replace("_", r"\_") if pd.notna(v) else ""
                for v in row
            )
            lines.append(cells + r" \\")
        lines.append("")
        lines.append(r"\bottomrule")
        lines.append("")
        lines.append(r"\end{tabular}")
        lines.append("")
        lines.append(r"\end{table}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------ #
    #  Publication-style PDF table (kept for internal / figure use)       #
    # ------------------------------------------------------------------ #
    def save_table_pdf(self, df: pd.DataFrame, name: str, width: float = 8.0) -> Path:
        n_rows = len(df)
        n_cols = len(df.columns)

        row_height = 0.28
        header_height = 0.38
        fig_height = max(1.6, n_rows * row_height + header_height + 0.3)

        fig, ax = plt.subplots(figsize=(width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=df.fillna("").values,
            colLabels=list(df.columns),
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.auto_set_column_width(list(range(n_cols)))

        last_data_row = n_rows

        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.set_linewidth(0.7)

            if row == 0:
                cell.visible_edges = "TB"
                cell.set_text_props(fontweight="bold")
                cell.PAD = 0.06
            elif row == last_data_row:
                cell.visible_edges = "B"
                cell.PAD = 0.06
            else:
                cell.visible_edges = ""
                cell.PAD = 0.06

        fig.tight_layout(pad=0.3)
        return self._save_fig(fig, name)

    def save_bar_pdf(self, df: pd.DataFrame, x: str, y: str, name: str, title: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df[x].fillna("").astype(str).tolist(), df[y].tolist(), color="steelblue", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        return self._save_fig(fig, name)

    def save_barh_pdf(self, df: pd.DataFrame, y: str, x: str, name: str, title: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df[y].fillna("").astype(str).tolist(), df[x].tolist(), color="coral", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        return self._save_fig(fig, name)

    def save_line_pdf(self, df: pd.DataFrame, name: str, title: str, xlabel: str, ylabel: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in df.columns:
            ax.plot(df.index, df[col], marker="o", label=col)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_fig(fig, name)

    def save_pie_pdf(
        self,
        df: pd.DataFrame,
        labels: str,
        values: str,
        name: str,
        title: str,
        use_legend: bool = False,
    ) -> Path:
        total = df[values].sum()
        pct = (df[values] / total * 100).round(1)

        if use_legend:
            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, _ = ax.pie(df[values], startangle=90)
            legend_labels = [
                f"{lbl}  ({p:.1f}%)"
                for lbl, p in zip(df[labels].astype(str), pct)
            ]
            ax.legend(
                wedges,
                legend_labels,
                title=labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
                title_fontsize=9,
            )
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(df[values], labels=df[labels].astype(str), autopct="%1.1f%%", startangle=90)

        ax.set_title(title)
        fig.tight_layout()
        return self._save_fig(fig, name)

    def save_hist_pdf(self, series: pd.Series, name: str, title: str, xlabel: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series.dropna(), bins=30, color="steelblue", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        return self._save_fig(fig, name)

    def save_box_pdf(self, series: pd.Series, name: str, title: str, ylabel: str) -> Path:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(series.dropna(), vert=True)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return self._save_fig(fig, name)

    def save_age_sex_pyramid_pdf(self, df: pd.DataFrame, name: str, title: str) -> Path:
        if "Age" not in df.columns or "Sex" not in df.columns:
            return self.output_dir / f"{name}.pdf"

        df = df.copy()
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df[df["Age"].between(0, 120)]
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
        labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"]
        df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

        pivot = df.groupby(["Age_Group", "Sex"]).size().unstack(fill_value=0)
        if "M" not in pivot.columns or "F" not in pivot.columns:
            return self.output_dir / f"{name}.pdf"

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(pivot.index, -pivot["M"], color="steelblue", edgecolor="black", label="Male")
        ax.barh(pivot.index, pivot["F"], color="salmon", edgecolor="black", label="Female")
        ax.set_xlabel("Number of Cases")
        ax.set_ylabel("Age Group")
        ax.set_title(title)
        ax.legend()
        ticks = ax.get_xticks()
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([str(abs(int(t))) for t in ticks])
        fig.tight_layout()
        return self._save_fig(fig, name)

    def build_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = {
            "Total Cases": len(df),
            "Years Covered": f"{df['Year'].min()} - {df['Year'].max()}" if "Year" in df.columns else "N/A",
            "Average Cases per Year": f"{len(df) / df['Year'].nunique():.0f}" if "Year" in df.columns else "N/A",
            "Male Cases": df["Sex"].eq("M").sum() if "Sex" in df.columns else "N/A",
            "Female Cases": df["Sex"].eq("F").sum() if "Sex" in df.columns else "N/A",
            "Median Age": f"{pd.to_numeric(df['Age'], errors='coerce').median():.1f} years" if "Age" in df.columns else "N/A",
            "Pulmonary TB": df["Anatomical Site"].eq("P").sum() if "Anatomical Site" in df.columns else "N/A",
            "Extrapulmonary TB": df["Anatomical Site"].eq("EP").sum() if "Anatomical Site" in df.columns else "N/A",
            "Bacteriologically Confirmed": df["Bacteriologic Status"].str.contains("Bacteriologically", na=False).sum() if "Bacteriologic Status" in df.columns else "N/A",
            "Treatment Success": df["Outcome/Status"].isin(["CURED", "TREATMENT COMPLETED"]).sum() if "Outcome/Status" in df.columns else "N/A",
            "Deaths": df["Outcome/Status"].eq("DIED").sum() if "Outcome/Status" in df.columns else "N/A",
            "Lost to Follow-up": df["Outcome/Status"].eq("LOST TO FF-UP").sum() if "Outcome/Status" in df.columns else "N/A",
        }
        return pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])

    def build_quality_table(self, df: pd.DataFrame) -> pd.DataFrame:
        critical_fields = [
            "Age",
            "Sex",
            "Outcome/Status",
            "Anatomical Site",
            "Registration Group",
            "Bacteriologic Status",
        ]
        available = [c for c in critical_fields if c in df.columns]
        complete_records = df[available].notnull().all(axis=1).sum() if available else 0
        quality = {
            "Total Records": len(df),
            "Complete Records (critical fields)": complete_records,
            "Completeness Rate": f"{(complete_records / len(df) * 100):.1f}%" if len(df) else "N/A",
            "Records with Missing Age": pd.to_numeric(df["Age"], errors="coerce").isna().sum() if "Age" in df.columns else "N/A",
            "Records with Missing Sex": df["Sex"].isna().sum() if "Sex" in df.columns else "N/A",
            "Records with Missing Outcome": df["Outcome/Status"].isna().sum() if "Outcome/Status" in df.columns else "N/A",
            "Average Completeness": f"{(1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%" if len(df) else "N/A",
        }
        return pd.DataFrame(list(quality.items()), columns=["Quality Metric", "Value"])

    def export_all_figures(self, df: pd.DataFrame) -> dict[str, Path]:
        outputs: dict[str, Path] = {}

        # --- Missing data summary ---
        missing_summary = self.compute_missing_summary(df)
        outputs["missing_data_summary"] = self.save_table_latex(
            missing_summary,
            "missing_data_summary",
            caption="Missing Data Summary by Column",
            label="tab:missing_data_summary",
        )

        key_columns = [
            "Days_Screening_To_Diagnosis",
            "Days_To_Treatment",
            "Days_To_Microscopy_Result",
            "Days_To_RDT_Result",
            "Microscopy Result",
            "RDT Result",
            "Outcome/Status",
        ]
        missing_by_year = self.compute_missing_by_year(df, key_columns)
        if not missing_by_year.empty:
            outputs["missing_data_by_year"] = self.save_table_latex(
                missing_by_year.reset_index(),
                "missing_data_by_year",
                caption="Missing Data Percentage by Year for Key Variables",
                label="tab:missing_data_by_year",
            )
            outputs["missing_data_trends"] = self.save_line_pdf(
                missing_by_year,
                "missing_data_trends",
                "Missing Data Trends Over Time",
                "Year",
                "Missing Data (%)",
            )

        yearly_counts = self.compute_yearly_case_counts(df)
        if not yearly_counts.empty:
            outputs["yearly_case_counts"] = self.save_bar_pdf(
                yearly_counts,
                "Year",
                "Case_Count",
                "yearly_case_counts",
                "TB Case Notifications by Year",
            )

        if "Type" in df.columns and "Year" in df.columns:
            type_by_year = pd.crosstab(df["Year"], df["Type"], normalize="index") * 100
            outputs["type_by_year_table"] = self.save_table_latex(
                type_by_year.reset_index(),
                "type_by_year_table",
                caption="Distribution of TB Case Types by Year (\\%)",
                label="tab:type_by_year",
            )

        if "Diagnosis_Month" in df.columns:
            month_counts = df["Diagnosis_Month"].value_counts().sort_index().reset_index()
            month_counts.columns = ["Month", "Count"]
            outputs["monthly_notifications"] = self.save_bar_pdf(
                month_counts,
                "Month",
                "Count",
                "monthly_notifications",
                "Seasonal Pattern of TB Notifications",
            )

        if "Age" in df.columns:
            age_series = pd.to_numeric(df["Age"], errors="coerce")
            outputs["age_distribution_hist"] = self.save_hist_pdf(
                age_series,
                "age_distribution_hist",
                "Age Distribution of TB Cases",
                "Age",
            )
            outputs["age_distribution_box"] = self.save_box_pdf(
                age_series,
                "age_distribution_box",
                "Age Distribution (Box Plot)",
                "Age",
            )

        if "Sex" in df.columns:
            sex_counts = self.compute_value_counts(df, "Sex")
            outputs["sex_distribution"] = self.save_bar_pdf(
                sex_counts,
                "Sex",
                "Count",
                "sex_distribution",
                "TB Cases by Sex",
            )
            outputs["sex_distribution_pie"] = self.save_pie_pdf(
                sex_counts,
                "Sex",
                "Count",
                "sex_distribution_pie",
                "Sex Distribution",
            )

        outputs["age_sex_pyramid"] = self.save_age_sex_pyramid_pdf(
            df,
            "age_sex_pyramid",
            "Age-Sex Pyramid of TB Cases",
        )

        if "City/Municipality" in df.columns:
            city_counts = self.compute_value_counts(df, "City/Municipality", top_n=15)
            outputs["top_cities"] = self.save_barh_pdf(
                city_counts,
                "City/Municipality",
                "Count",
                "top_cities",
                "Top 15 Cities/Municipalities by TB Case Count",
            )

        if "Anatomical Site" in df.columns:
            anatomical_counts = self.compute_value_counts(df, "Anatomical Site")
            outputs["anatomical_site"] = self.save_bar_pdf(
                anatomical_counts,
                "Anatomical Site",
                "Count",
                "anatomical_site",
                "TB Cases by Anatomical Site",
            )

        if "Bacteriologic Status" in df.columns:
            bact_counts = self.compute_value_counts(df, "Bacteriologic Status")
            outputs["bacteriologic_status"] = self.save_barh_pdf(
                bact_counts,
                "Bacteriologic Status",
                "Count",
                "bacteriologic_status",
                "Bacteriologic Status of TB Cases",
            )

        if "Registration Group" in df.columns:
            reg_counts = self.compute_value_counts(df, "Registration Group", top_n=10)
            outputs["registration_group"] = self.save_barh_pdf(
                reg_counts,
                "Registration Group",
                "Count",
                "registration_group",
                "Registration Group Distribution",
            )

        if "Source of Patient" in df.columns:
            source_counts = self.compute_value_counts(df, "Source of Patient")
            outputs["source_of_patient"] = self.save_barh_pdf(
                source_counts,
                "Source of Patient",
                "Count",
                "source_of_patient",
                "Source of Patients",
            )

        if "Outcome/Status" in df.columns:
            outcome_counts = self.compute_value_counts(df, "Outcome/Status")
            outputs["treatment_outcomes"] = self.save_barh_pdf(
                outcome_counts,
                "Outcome/Status",
                "Count",
                "treatment_outcomes",
                "Treatment Outcomes",
            )
            outputs["treatment_outcomes_pie"] = self.save_pie_pdf(
                outcome_counts,
                "Outcome/Status",
                "Count",
                "treatment_outcomes_pie",
                "Outcome Distribution",
                use_legend=True,
            )

            outcome_category = df["Outcome/Status"].apply(
                lambda x: "Success" if x in ["CURED", "TREATMENT COMPLETED"] else
                "Died" if x == "DIED" else
                "Lost to Follow-up" if x == "LOST TO FF-UP" else
                "Other"
            )
            if "Year" in df.columns:
                outcome_by_year = pd.crosstab(df["Year"], outcome_category, normalize="index") * 100
                outputs["outcome_trends_by_year"] = self.save_table_latex(
                    outcome_by_year.reset_index(),
                    "outcome_trends_by_year",
                    caption="Treatment Outcome Distribution by Year (\\%)",
                    label="tab:outcome_trends_by_year",
                )

        interval_cols = [
            "Days_Screening_To_Diagnosis",
            "Days_To_Treatment",
            "Days_To_Microscopy_Result",
            "Days_To_RDT_Result",
        ]
        interval_stats = []
        for col in interval_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                outputs[f"{col}_hist"] = self.save_hist_pdf(
                    series,
                    f"{col}_hist",
                    f"{col.replace('_', ' ')} Distribution",
                    "Days",
                )
                interval_stats.append({
                    "Interval": col,
                    "Count": series.dropna().shape[0],
                    "Median": round(series.median(), 2),
                    "Q1": round(series.quantile(0.25), 2),
                    "Q3": round(series.quantile(0.75), 2),
                })

        if interval_stats:
            interval_df = pd.DataFrame(interval_stats)
            outputs["time_interval_summary"] = self.save_table_latex(
                interval_df,
                "time_interval_summary",
                caption="Summary Statistics for Time Intervals (Days)",
                label="tab:time_interval_summary",
            )

        summary_df = self.build_summary_table(df)
        outputs["summary_statistics"] = self.save_table_latex(
            summary_df,
            "summary_statistics",
            caption="Summary Statistics of the TB Dataset",
            label="tab:summary_statistics",
        )

        quality_df = self.build_quality_table(df)
        outputs["data_quality"] = self.save_table_latex(
            quality_df,
            "data_quality",
            caption="Data Quality Assessment",
            label="tab:data_quality",
        )

        return outputs

    def _save_fig(self, fig: plt.Figure, name: str) -> Path:
        output_path = self.output_dir / f"{name}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return output_path