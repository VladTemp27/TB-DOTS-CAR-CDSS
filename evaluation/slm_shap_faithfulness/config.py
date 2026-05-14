from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    top_k: int = 5
    threshold_feature_f1: float = 0.60
    threshold_sign_accuracy: float = 0.60
    threshold_magnitude_accuracy: float = 0.50
    threshold_composite: float = 0.60
    weight_feature_f1: float = 0.45
    weight_sign_accuracy: float = 0.35
    weight_magnitude_accuracy: float = 0.20
    parser_coverage_min: float = 0.95
    allowed_modes: tuple[str, ...] = ("artifact", "runtime")
