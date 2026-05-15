from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn

from backend.temporal_lstm import TB_LSTM_CDSS


class ExportWrapper(nn.Module):
    def __init__(self, model: TB_LSTM_CDSS):
        super().__init__()
        self.model = model

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        logit, _ = self.model(x_temporal, x_static, seq_lens=None)
        return logit


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_state_dict(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unexpected checkpoint format: {path}")


def _load_matching_scaler(paths: list[Path], expected_features: int):
    for path in paths:
        if not path.exists():
            continue
        scaler = joblib.load(path)
        if int(getattr(scaler, "n_features_in_", expected_features)) == expected_features:
            return scaler, path
    raise FileNotFoundError(f"No scaler found for {expected_features} features")


def main() -> int:
    root = _repo_root()
    model_dir = root / "models" / "Temporal" / "v2" / "output" / "hybrid_lstm"
    public_dir = root / "web-app" / "public" / "model"
    public_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    n_static = int(cfg["static_features"])
    n_temporal = int(cfg["temporal_features"])
    threshold = float(cfg["variants"]["augmented"]["threshold"])

    model = TB_LSTM_CDSS(
        temporal_features=n_temporal,
        static_features=n_static,
        lstm_hidden=int(cfg.get("lstm_hidden", 64)),
        lstm_layers=int(cfg.get("lstm_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(_load_state_dict(model_dir / "best_model.pt"))
    model.eval()

    wrapper = ExportWrapper(model).eval()
    x_temporal = torch.zeros(1, 13, n_temporal, dtype=torch.float32)
    x_static = torch.zeros(1, n_static, dtype=torch.float32)

    onnx_path = public_dir / "hybrid_lstm_temporal.onnx"
    torch.onnx.export(
        wrapper,
        (x_temporal, x_static),
        onnx_path,
        input_names=["x_temporal", "x_static"],
        output_names=["logit"],
        dynamic_axes={
            "x_temporal": {0: "batch", 1: "months"},
            "x_static": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    scaler_static, scaler_static_path = _load_matching_scaler(
        [
            root / "dataset" / "temporal" / "scaler_static.pkl",
            root / "dataset" / "temporal" / "output" / "scaler_static.pkl",
        ],
        n_static,
    )
    scaler_temporal, scaler_temporal_path = _load_matching_scaler(
        [
            root / "dataset" / "temporal" / "scaler_temporal.pkl",
            root / "dataset" / "temporal" / "output" / "scaler_temporal.pkl",
        ],
        n_temporal,
    )

    metadata = {
        "modelName": "hybrid_bi_lstm_best_onnx",
        "threshold": threshold,
        "staticFeatureNames": cfg["static_feature_names"],
        "temporalFeatureNames": cfg["temporal_feature_names"],
        "staticScaler": {
            "mean": scaler_static.mean_.tolist(),
            "scale": scaler_static.scale_.tolist(),
            "source": str(scaler_static_path.relative_to(root)),
        },
        "temporalScaler": {
            "mean": scaler_temporal.mean_.tolist(),
            "scale": scaler_temporal.scale_.tolist(),
            "source": str(scaler_temporal_path.relative_to(root)),
        },
    }
    metadata_path = public_dir / "hybrid_lstm_temporal_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {onnx_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
