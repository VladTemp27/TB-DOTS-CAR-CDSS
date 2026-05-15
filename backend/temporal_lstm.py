from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TemporalLstmArtifacts:
    threshold: float
    static_feature_names: list[str]
    temporal_feature_names: list[str]
    scaler_static: Any
    scaler_temporal: Any
    model: nn.Module


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor, seq_lens: torch.Tensor | None = None):
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


_ARTIFACTS: TemporalLstmArtifacts | None = None


def _repo_root() -> Path:
    # backend/temporal_lstm.py -> <repo>
    return Path(__file__).resolve().parents[1]


def _load_matching_scaler(paths: list[Path], expected_features: int, label: str) -> Any:
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            scaler = joblib.load(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        n_features = getattr(scaler, "n_features_in_", expected_features)
        if int(n_features) == expected_features:
            return scaler
        errors.append(f"{path}: expected {expected_features} features, got {n_features}")
    detail = "; ".join(errors) if errors else "no candidate files found"
    raise FileNotFoundError(f"Missing compatible {label} scaler: {detail}")


def load_artifacts() -> TemporalLstmArtifacts:
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    root = _repo_root()
    out_dir = root / "models" / "Temporal" / "v2" / "output" / "hybrid_lstm"
    cfg = json.loads((out_dir / "model_config.json").read_text(encoding="utf-8"))

    threshold = float(cfg["variants"]["augmented"]["threshold"])
    static_feature_names = list(cfg["static_feature_names"])
    temporal_feature_names = list(cfg["temporal_feature_names"])
    n_static = int(cfg["static_features"])
    n_temporal = int(cfg["temporal_features"])

    scaler_static = _load_matching_scaler(
        [
            root / "dataset" / "temporal" / "scaler_static.pkl",
            root / "dataset" / "temporal" / "output" / "scaler_static.pkl",
        ],
        n_static,
        "static",
    )
    scaler_temporal = _load_matching_scaler(
        [
            root / "dataset" / "temporal" / "scaler_temporal.pkl",
            root / "dataset" / "temporal" / "output" / "scaler_temporal.pkl",
        ],
        n_temporal,
        "temporal",
    )

    model = TB_LSTM_CDSS(
        temporal_features=n_temporal,
        static_features=n_static,
        lstm_hidden=int(cfg.get("lstm_hidden", 64)),
        lstm_layers=int(cfg.get("lstm_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )

    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    state_dict = ckpt.get("model_state_dict") if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unexpected best_model.pt format")
    model.load_state_dict(state_dict)
    model.eval()

    _ARTIFACTS = TemporalLstmArtifacts(
        threshold=threshold,
        static_feature_names=static_feature_names,
        temporal_feature_names=temporal_feature_names,
        scaler_static=scaler_static,
        scaler_temporal=scaler_temporal,
        model=model,
    )
    return _ARTIFACTS


def predict_success_probability(
    *,
    x_temporal_unscaled: np.ndarray,
    x_static_unscaled: np.ndarray,
    seq_len: int,
) -> float:
    art = load_artifacts()

    # Scale: static (94) and temporal (13x16)
    xs = art.scaler_static.transform(x_static_unscaled.reshape(1, -1)).astype(np.float32)
    xt2d = x_temporal_unscaled.reshape(-1, x_temporal_unscaled.shape[1])
    xt_scaled = art.scaler_temporal.transform(xt2d).reshape(1, 13, -1).astype(np.float32)

    with torch.no_grad():
        x_t = torch.from_numpy(xt_scaled)
        x_s = torch.from_numpy(xs)
        seq = torch.tensor([seq_len], dtype=torch.long)
        logit, _ = art.model(x_t, x_s, seq_lens=seq)
        prob_success = torch.sigmoid(logit).item()
    return float(prob_success)
