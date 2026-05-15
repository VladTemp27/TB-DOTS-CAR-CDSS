"""Test inference for the Temporal Hybrid Bi-LSTM "best" model.

This script runs the PyTorch checkpoint (`best_model.pt`) on real data prepared
by `models/Temporal/model_utils.py` and prints progressive predictions at
M0/M3/M6/M9/M12.

It also searches the held-out test split for:
- strict flips: M0 predicted Failure/Risk -> M12 predicted Success
- biggest directional improvements: drop in failure probability M0 -> M12

Run:
  python models/Temporal/v2/test_hybrid_lstm_best_inference.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class MonthPred:
    month: int
    prob_success: float
    prob_failure: float
    derived_label: int


def _repo_root() -> Path:
    # <repo>/models/Temporal/v2/test_hybrid_lstm_best_inference.py
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TemporalAttention(nn.Module):
    """Soft attention over LSTM time-step outputs with optional masking."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor, seq_lens: torch.Tensor | None = None):
        # lstm_out: (batch, seq_len, hidden_size)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)

        if seq_lens is not None:
            max_len = lstm_out.size(1)
            time_idx = torch.arange(max_len, device=lstm_out.device).unsqueeze(0)
            mask = time_idx >= seq_lens.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class TB_LSTM_CDSS(nn.Module):
    """Hybrid Bi-LSTM + masked attention + static FC."""

    def __init__(
        self,
        temporal_features: int,
        static_features: int,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
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
        self.temporal_attention = TemporalAttention(lstm_hidden * 2)
        self.temporal_norm = nn.LayerNorm(lstm_hidden * 2)

        self.static_net = nn.Sequential(
            nn.Linear(static_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_dim = lstm_hidden * 2 + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_static: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ):
        if seq_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_temporal,
                seq_lens.cpu().clamp(min=1),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x_temporal.size(1),
            )
        else:
            lstm_out, _ = self.lstm(x_temporal)

        context, attn_weights = self.temporal_attention(lstm_out, seq_lens)
        context = self.temporal_norm(context)
        static_repr = self.static_net(x_static)
        fused = torch.cat([context, static_repr], dim=1)
        logit = self.classifier(fused).squeeze(-1)
        return logit, attn_weights


def _pad_progressive(x_temporal_single: np.ndarray, up_to_month: int) -> tuple[np.ndarray, int]:
    # x_temporal_single: (13, n_temporal_features)
    x_t = np.zeros_like(x_temporal_single, dtype=np.float32)
    x_t[: up_to_month + 1] = x_temporal_single[: up_to_month + 1]
    return x_t, up_to_month + 1


def _predict_at_month(
    model: nn.Module,
    *,
    x_temporal_single: np.ndarray,
    x_static_single: np.ndarray,
    up_to_month: int,
    threshold: float,
) -> MonthPred:
    x_t, seq_len = _pad_progressive(x_temporal_single, up_to_month)
    x_t_t = torch.from_numpy(x_t).unsqueeze(0)
    x_s_t = torch.from_numpy(x_static_single).unsqueeze(0)
    seq = torch.tensor([seq_len], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logit, _ = model(x_t_t, x_s_t, seq_lens=seq)
        prob_success = torch.sigmoid(logit).item()
    prob_failure = 1.0 - prob_success
    derived_label = 1 if prob_success >= threshold else 0
    return MonthPred(
        month=up_to_month,
        prob_success=float(prob_success),
        prob_failure=float(prob_failure),
        derived_label=int(derived_label),
    )


def _load_state_dict(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj
    raise RuntimeError(f"Unrecognized checkpoint format: {path.name}")


def main() -> int:
    root = _repo_root()
    out_dir = root / "models" / "Temporal" / "v2" / "output" / "hybrid_lstm"
    cfg = _load_json(out_dir / "model_config.json")

    # "best_model.pt" corresponds to the augmented model in this output folder.
    threshold = float(cfg["variants"]["augmented"]["threshold"])
    n_temporal = int(cfg["temporal_features"])
    n_static = int(cfg["static_features"])
    lstm_hidden = int(cfg.get("lstm_hidden", 64))
    lstm_layers = int(cfg.get("lstm_layers", 2))
    dropout = float(cfg.get("dropout", 0.3))

    # Prepare arrays exactly as training did.
    import sys

    sys.path.append(str(root / "models" / "Temporal"))
    import model_utils as mu  # type: ignore

    csv_path = root / "dataset" / "temporal" / "output" / "cleaned_human_readable.csv"
    model_data = mu.prepare_default_model_inputs(
        str(csv_path),
        train_frac=0.70,
        val_frac=0.20,
        test_frac=0.10,
        random_state=int(cfg.get("seed", 42)),
        drop_feature_cols=mu.get_temporal_v2_drop_feature_cols(),
    )

    # Match notebook training: scale using train split only.
    train_idx = np.asarray(model_data["train_idx"], dtype=np.int64)
    scaled = mu.scale_full_arrays(model_data["X_static"], model_data["X_temporal"], train_idx)
    X_static = np.asarray(scaled["X_static_scaled"], dtype=np.float32)
    X_temporal = np.asarray(scaled["X_temporal_scaled"], dtype=np.float32)
    y = np.asarray(model_data["y"], dtype=np.int64)
    test_idx = np.asarray(model_data["test_idx"], dtype=np.int64)

    if X_temporal.shape[1] != 13:
        raise RuntimeError(f"Expected 13 timesteps, got {X_temporal.shape[1]}")
    if X_temporal.shape[2] != n_temporal:
        raise RuntimeError(f"Temporal feature mismatch: config={n_temporal} data={X_temporal.shape[2]}")
    if X_static.shape[1] != n_static:
        raise RuntimeError(f"Static feature mismatch: config={n_static} data={X_static.shape[1]}")

    model = TB_LSTM_CDSS(
        temporal_features=n_temporal,
        static_features=n_static,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )
    state = _load_state_dict(out_dir / "best_model.pt")
    model.load_state_dict(state)
    model.eval()

    print("Hybrid Bi-LSTM (best/augmented) inference test")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")
    print(f"Threshold (augmented): {threshold:.2f}")
    print(f"X_temporal: {tuple(X_temporal.shape)}  X_static: {tuple(X_static.shape)}")
    print(f"Test patients: {len(test_idx)}")

    # Quick logit sanity check on a small batch at full sequence length.
    sample_ids = test_idx[:8].astype(int)
    x_t = torch.from_numpy(X_temporal[sample_ids])
    x_s = torch.from_numpy(X_static[sample_ids])
    seq = torch.full((len(sample_ids),), 13, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x_t, x_s, seq_lens=seq)
        lg = logits.detach().cpu().numpy().reshape(-1)
    print(f"Logits sanity (M12 batch of {len(sample_ids)}): min/mean/max {lg.min():.4f} / {lg.mean():.4f} / {lg.max():.4f}")

    months = (0, 3, 6, 9, 12)

    # Show a couple of concrete patients.
    for pid in test_idx[:2].tolist():
        actual = int(y[pid])
        print(f"\nPatient index {pid} | actual={'Success' if actual==1 else 'Failure'}")
        for m in months:
            p = _predict_at_month(
                model,
                x_temporal_single=X_temporal[pid],
                x_static_single=X_static[pid],
                up_to_month=m,
                threshold=threshold,
            )
            print(
                f"  M{m}: P(S)={p.prob_success:.4f} P(F)={p.prob_failure:.4f} "
                f"label={'Success' if p.derived_label==1 else 'Failure/Risk'}"
            )

    # Search for strict flips.
    flips: list[int] = []
    improvements: list[tuple[int, float, MonthPred, MonthPred]] = []
    m12_success = []
    m0_success = []
    for pid in test_idx.tolist():
        p0 = _predict_at_month(
            model,
            x_temporal_single=X_temporal[pid],
            x_static_single=X_static[pid],
            up_to_month=0,
            threshold=threshold,
        )
        p12 = _predict_at_month(
            model,
            x_temporal_single=X_temporal[pid],
            x_static_single=X_static[pid],
            up_to_month=12,
            threshold=threshold,
        )
        m0_success.append(p0.prob_success)
        m12_success.append(p12.prob_success)
        if p0.derived_label == 0 and p12.derived_label == 1:
            flips.append(pid)
        improvements.append((pid, p0.prob_failure - p12.prob_failure, p0, p12))

    print("\nFlip matches (M0 risk -> M12 success):", len(flips))
    for pid in flips[:5]:
        actual = int(y[pid])
        print(f"  patient {pid} actual={'Success' if actual==1 else 'Failure'}")

    improvements.sort(key=lambda t: t[1], reverse=True)
    print("\nTop 5 directional improvers (largest failure drop M0->M12):")
    for pid, delta, p0, p12 in improvements[:5]:
        actual = int(y[pid])
        print(
            f"  patient {pid} actual={'Success' if actual==1 else 'Failure'} | "
            f"M0 P(F)={p0.prob_failure:.4f} -> M12 P(F)={p12.prob_failure:.4f} (delta={delta:+.4f})"
        )

    # Quick sanity summary.
    m0_arr = np.asarray(m0_success, dtype=np.float32)
    m12_arr = np.asarray(m12_success, dtype=np.float32)
    print("\nProbability summary (test set):")
    print(f"  M0  P(S) min/mean/max: {float(m0_arr.min()):.6f} / {float(m0_arr.mean()):.6f} / {float(m0_arr.max()):.6f}")
    print(f"  M12 P(S) min/mean/max: {float(m12_arr.min()):.6f} / {float(m12_arr.mean()):.6f} / {float(m12_arr.max()):.6f}")
    print(f"  M12 count >= threshold({threshold:.2f}): {int((m12_arr >= threshold).sum())} / {len(m12_arr)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
