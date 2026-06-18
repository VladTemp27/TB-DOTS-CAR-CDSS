"""Generate the full treatment-outcome distribution figure for the 7,389-record
non-temporal TB dataset.

Unlike treatment_outcomes.pdf (which groups outcomes into success / death / lost),
this renders all 11 distinct ``Outcome/Status`` categories with counts and percentages,
matching the table reported in distribution_statistics.md.

Output: paper/apa/pharma_figures/treatment_outcome_distribution.pdf (+ .png)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "dataset" / "non-temporal" / "2015-2025-nontemporal-7389.csv"


def main() -> None:
    df = pd.read_csv(DATA, low_memory=False)
    counts = df["Outcome/Status"].value_counts(dropna=False)
    total = int(counts.sum())

    labels = counts.index.astype(str).tolist()
    values = counts.values.tolist()

    # Horizontal bars, largest at the top.
    labels = labels[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, values, color="steelblue", edgecolor="black")

    for bar, val in zip(bars, values):
        pct = 100.0 * val / total
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Number of Cases")
    ax.set_title(f"Treatment Outcome Distribution (N = {total:,})")
    ax.set_xlim(0, max(values) * 1.18)
    ax.margins(y=0.02)
    fig.tight_layout()

    pdf_path = HERE / "treatment_outcome_distribution.pdf"
    png_path = HERE / "treatment_outcome_distribution.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Companion LaTeX table (largest outcome first), mirroring the figure values.
    rows = "\n".join(
        f"{lab} & {val:,} & {100.0 * val / total:.2f}\\% \\\\"
        for lab, val in zip(labels[::-1], values[::-1])
    )
    tex = (
        "\\begin{table}[htbp]\n\n"
        "\\caption{Treatment Outcome Distribution (7{,}389-Record Non-Temporal Dataset)}\n"
        "\\label{tab:treatment_outcome_distribution}\n"
        "\\begin{tabular}{lrr}\n\n\\toprule\n\n"
        "Outcome/Status & Count & Percent \\\\\n\n\\midrule\n\n"
        f"{rows}\n\n\\midrule\n\n"
        f"Total & {total:,} & 100.00\\% \\\\\n\n\\bottomrule\n\n"
        "\\end{tabular}\n\n\\end{table}\n"
    )
    tex_path = HERE / "treatment_outcome_distribution.tex"
    tex_path.write_text(tex)
    print(f"Wrote {tex_path}")

    print(f"N = {total}")
    for lab, val in zip(labels[::-1], values[::-1]):
        print(f"  {lab}: {val} ({100.0 * val / total:.2f}%)")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
