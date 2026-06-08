"""Per-feature permutation importance for the current Temporal Hybrid Bi-LSTM.

WHY THIS EXISTS
---------------
The saved checkpoints only hold the network's layer weight matrices, which are NOT
per-feature importance scores (each feature feeds a *column* of weights that then passes
through attention, normalization, and fusion). Unlike the tree models in this folder
(xgboost/lightgbm/random_forest), an LSTM has no native `feature_importances_`. To get one
interpretable score per feature we use permutation importance: shuffle a single feature
across patients on the held-out test split and measure how much performance drops. A larger
drop => the model relies more on that feature.

WHICH MODEL
-----------
This targets the CURRENT trained model produced by `Hybrid_Bi-LSTM_temporal.ipynb`:
    output/hybrid_lstm_failure_positive/best_model.pt   (16 temporal + 22 static features)
NOT the stale `output/hybrid_lstm/best_model.pt` (94 one-hot features), which predates the
May-2026 feature-policy refactor and no longer matches `model_utils`.

The architecture (TB_LSTM_CDSS / TemporalAttention) and the data prep (selective scaling,
`1=Failure` / P(Failure) convention) are replicated verbatim from the notebook so the model
loads exactly as trained. Baseline AUC should reproduce
`model_config.json -> metrics.test_m12_full_sequence.roc_auc_failure` (~0.906).

OUTPUTS (next to the model, in output/hybrid_lstm_failure_positive/)
    permutation_importance.csv  : one row per feature, ranked by AUC drop
    permutation_importance.png  : horizontal bar chart of the features

Run:
    python models/Temporal/v2/feature_permutation_importance.py
    python models/Temporal/v2/feature_permutation_importance.py --n-repeats 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score


# --------------------------------------------------------------------------- #
# Architecture — copied verbatim from Hybrid_Bi-LSTM_temporal.ipynb (cell 10). #
# --------------------------------------------------------------------------- #
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // 2, 8)),
            nn.Tanh(),
            nn.Linear(max(hidden_size // 2, 8), 1, bias=False),
        )

    def forward(self, lstm_out, seq_lens=None):
        scores = self.attn(lstm_out).squeeze(-1)
        if seq_lens is not None:
            max_len = lstm_out.size(1)
            time_idx = torch.arange(max_len, device=lstm_out.device).unsqueeze(0)
            mask = time_idx >= seq_lens.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class TB_LSTM_CDSS(nn.Module):
    def __init__(
        self,
        temporal_features,
        static_features,
        lstm_hidden=32,
        lstm_layers=1,
        dropout=0.4,
        static_hidden=32,
        static_out=16,
        classifier_hidden=32,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=temporal_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        temporal_out = lstm_hidden * 2
        self.temporal_attention = TemporalAttention(temporal_out)
        self.temporal_norm = nn.LayerNorm(temporal_out)
        self.static_net = nn.Sequential(
            nn.Linear(static_features, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out + static_out, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x_temporal, x_static, seq_lens=None):
        if seq_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_temporal, seq_lens.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=x_temporal.size(1),
            )
        else:
            lstm_out, _ = self.lstm(x_temporal)
        context, attn_weights = self.temporal_attention(lstm_out, seq_lens)
        context = self.temporal_norm(context)
        static_repr = self.static_net(x_static)
        fused = torch.cat([context, static_repr], dim=1)
        logit = self.classifier(fused).squeeze(-1)
        return logit, attn_weights


# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    # <repo>/models/Temporal/v2/feature_permutation_importance.py
    return Path(__file__).resolve().parents[3]


def _build_test_arrays(seed: int):
    """Replicate notebook cell 4: data prep + SELECTIVE scaling, on the test split.

    Returns scaled test arrays plus y_failure (1=Failure) and feature names.
    """
    root = _repo_root()
    sys.path.insert(0, str(root / "models" / "Temporal"))
    import model_utils as mu  # type: ignore

    csv_path = root / "dataset" / "temporal" / "output" / "cleaned_human_readable.csv"
    model_data = mu.prepare_default_model_inputs(
        str(csv_path),
        train_frac=0.70, val_frac=0.20, test_frac=0.10,
        random_state=seed,
        drop_feature_cols=mu.get_temporal_v2_drop_feature_cols(),
        imputation_policy="deployment_simple",
    )

    X_static_raw = np.asarray(model_data["X_static"], dtype=np.float32)
    X_temporal_raw = np.asarray(model_data["X_temporal"], dtype=np.float32)
    y_success = np.asarray(model_data["y"], dtype=np.int64)
    y_failure = (1 - y_success).astype(np.int64)  # 1=Failure, matches checkpoint
    train_idx = np.asarray(model_data["train_idx"], dtype=np.int64)
    test_idx = np.asarray(model_data["test_idx"], dtype=np.int64)
    static_names = list(model_data["static_feature_names"])
    temporal_names = list(model_data["temporal_feature_names"])

    # Scale ONLY continuous features (keep one-hots / missing flags as 0/1) — cell 4.
    scaled = mu.scale_full_arrays_selective(
        X_static_raw, X_temporal_raw, train_idx,
        static_feature_names=static_names,
        temporal_feature_names=temporal_names,
    )
    X_static = np.asarray(scaled["X_static_scaled"], dtype=np.float32)
    X_temporal = np.asarray(scaled["X_temporal_scaled"], dtype=np.float32)

    return {
        "Xs": X_static[test_idx], "Xt": X_temporal[test_idx], "y": y_failure[test_idx],
        "static_names": static_names, "temporal_names": temporal_names,
        "n_timesteps": X_temporal.shape[1],
    }


# Human-readable feature labels (snake_case -> Title Case, with clinical acronyms kept).
_LABEL_OVERRIDES = {
    "xpert_mtb_rif": "Xpert MTB/RIF",
    "smear_tb_lamp": "Smear TB-LAMP",
    "smear_microscopy": "Smear Microscopy",
    "bp_systolic": "BP Systolic",
    "bp_diastolic": "BP Diastolic",
    "o2_sat": "O₂ Saturation",
    "heart_rate": "Heart Rate",
    "pct_adherence": "% Adherence",
    "weight_kg": "Weight (kg)",
    "height_cm": "Height (cm)",
    "cumulative_doses_taken": "Cumulative Doses Taken",
    "monthly_doses_taken": "Monthly Doses Taken",
    "monthly_missed_doses": "Monthly Missed Doses",
    "date_of_diagnosis": "Date of Diagnosis",
    "date_of_notification": "Date of Notification",
    "treatment_start_date": "Treatment Start Date",
    "intensive_phase_start_date": "Intensive Phase Start Date",
    "name_of_diagnosing_facility": "Name of Diagnosing Facility",
    "name_of_treatment_unit": "Name of Treatment Unit",
    "age": "Age",
    "weight": "Weight",
    "height": "Height",
}


def _prettify(name: str) -> str:
    """Turn a raw feature name into a readable Title-Case label.

    `is_missing_<x>` -> "<X> (Missing)"; underscores -> spaces; clinical acronyms preserved.
    """
    missing = name.startswith("is_missing_")
    base = name[len("is_missing_"):] if missing else name
    label = _LABEL_OVERRIDES.get(base)
    if label is None:
        label = base.replace("_", " ").title()
    if missing:
        label = f"{label} (Missing)"
    return label


def _predict_failure_proba(model: nn.Module, Xs: np.ndarray, Xt: np.ndarray, seq_len: int) -> np.ndarray:
    """P(Failure) for every patient at full sequence length (sigmoid(logit))."""
    x_s = torch.from_numpy(np.ascontiguousarray(Xs, dtype=np.float32))
    x_t = torch.from_numpy(np.ascontiguousarray(Xt, dtype=np.float32))
    seq = torch.full((Xs.shape[0],), seq_len, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x_t, x_s, seq_lens=seq)
        return torch.sigmoid(logits).cpu().numpy().reshape(-1).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="hybrid_lstm_failure_positive",
                        help="Subdir of output/ holding best_model.pt (default: hybrid_lstm_failure_positive)")
    parser.add_argument("--n-repeats", type=int, default=10, help="Permutation repeats per feature (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42, matches training)")
    args = parser.parse_args()

    # Reproducibility — mirror notebook cell 2.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = _repo_root()
    out_dir = root / "models" / "Temporal" / "v2" / "output" / args.model_dir
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    spec = ckpt["model_spec"]
    threshold = float(ckpt.get("threshold", 0.5))

    data = _build_test_arrays(args.seed)
    Xs, Xt, y = data["Xs"], data["Xt"], data["y"]
    static_names, temporal_names = data["static_names"], data["temporal_names"]
    seq_len = data["n_timesteps"]

    model = TB_LSTM_CDSS(
        temporal_features=Xt.shape[2],
        static_features=Xs.shape[1],
        lstm_hidden=int(spec["lstm_hidden"]),
        lstm_layers=int(spec["lstm_layers"]),
        dropout=float(spec["dropout"]),
        static_hidden=int(spec["static_hidden"]),
        static_out=int(spec["static_out"]),
        classifier_hidden=int(spec["classifier_hidden"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def score(prob: np.ndarray) -> tuple[float, float]:
        auc = float(roc_auc_score(y, prob))
        f1 = float(f1_score(y, (prob >= threshold).astype(int), pos_label=1, zero_division=0))
        return auc, f1

    base_prob = _predict_failure_proba(model, Xs, Xt, seq_len)
    base_auc, base_f1 = score(base_prob)
    print(f"Model: {out_dir.name}/best_model.pt  spec={spec['name']}")
    print(f"Test patients: {len(y)} (failures={int(y.sum())})  | "
          f"baseline AUC={base_auc:.4f}  failure-F1={base_f1:.4f}  (thr={threshold:.2f})")
    print("(compare baseline AUC to model_config.json metrics.test_m12_full_sequence.roc_auc_failure)\n")

    rng = np.random.default_rng(args.seed)
    n = Xs.shape[0]

    def permute_feature(group: str, j: int):
        """Average metric drop over n_repeats when feature j of `group` is shuffled across patients."""
        auc_drops, f1_drops = [], []
        for _ in range(args.n_repeats):
            perm = rng.permutation(n)
            if group == "static":
                Xs_p = Xs.copy()
                Xs_p[:, j] = Xs[perm, j]
                prob = _predict_failure_proba(model, Xs_p, Xt, seq_len)
            else:  # temporal: shuffle the patient order of this feature's full (13,) trajectory
                Xt_p = Xt.copy()
                Xt_p[:, :, j] = Xt[perm, :, j]
                prob = _predict_failure_proba(model, Xs, Xt_p, seq_len)
            auc, f1 = score(prob)
            auc_drops.append(base_auc - auc)
            f1_drops.append(base_f1 - f1)
        return float(np.mean(auc_drops)), float(np.std(auc_drops)), float(np.mean(f1_drops))

    rows = []
    for j, name in enumerate(temporal_names):
        a_mean, a_std, f_mean = permute_feature("temporal", j)
        rows.append(("temporal", name, a_mean, a_std, f_mean))
    for j, name in enumerate(static_names):
        a_mean, a_std, f_mean = permute_feature("static", j)
        rows.append(("static", name, a_mean, a_std, f_mean))

    df = pd.DataFrame(rows, columns=["group", "feature", "auc_drop_mean", "auc_drop_std", "f1_drop_mean"])
    df = df.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)

    csv_path = out_dir / "permutation_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} features)\n")
    print("Ranked by mean AUC drop (P(Failure) on test split):")
    with pd.option_context("display.max_colwidth", 60):
        print(df.to_string(index=False))

    # Plot — styled to match paper/apa/figures (seaborn whitegrid, steel-blue / salmon).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STATIC_BLUE = "#5B8FB9"
    TEMPORAL_SALMON = "#F58C6B"

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "font.size": 10,
    })

    top = df.iloc[::-1]  # largest at top of horizontal bar
    labels = [_prettify(f) for f in top["feature"]]
    colors = [TEMPORAL_SALMON if g == "temporal" else STATIC_BLUE for g in top["group"]]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.34 * len(top))))
    bars = ax.barh(
        labels, top["auc_drop_mean"],
        xerr=top["auc_drop_std"], color=colors,
        error_kw={"ecolor": "#555555", "elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
    )
    ax.axvline(0.0, color="#888888", lw=0.8, zorder=0)
    ax.set_xlabel("Mean ROC-AUC Drop When Permuted")
    ax.set_title("Permutation Feature Importances — Hybrid Bi-LSTM (Best Model)")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.01)

    # Value labels at each bar end (offset past the error cap; sign-aware placement).
    vals = top["auc_drop_mean"].to_numpy()
    errs = top["auc_drop_std"].to_numpy()
    xspan = float(max(vals.max(), 0.0) - min(vals.min(), 0.0)) or 1.0
    pad = 0.012 * xspan
    bbox = dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.4)
    for bar, v, e in zip(bars, vals, errs):
        y = bar.get_y() + bar.get_height() / 2
        if v >= 0:
            ax.text(v + e + pad, y, f"{v:.3f}", va="center", ha="left", fontsize=7, color="#333333", bbox=bbox)
        else:
            ax.text(v - e - pad, y, f"{v:.3f}", va="center", ha="right", fontsize=7, color="#333333", bbox=bbox)
    ax.set_xlim(vals.min() - errs.max() - 0.06 * xspan, vals.max() + errs.max() + 0.10 * xspan)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TEMPORAL_SALMON),
               plt.Rectangle((0, 0), 1, 1, color=STATIC_BLUE)]
    ax.legend(handles, ["Temporal", "Static"], loc="lower right", frameon=True)
    fig.tight_layout()

    png_path = out_dir / "permutation_importance.png"
    pdf_path = out_dir / "permutation_importance.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)  # vector copy, matching the paper figure format
    print(f"\nWrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
