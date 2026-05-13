from evaluation.slm_shap_faithfulness.config import BenchmarkConfig


def score_case(
    truth_rows: list[dict],
    claims: list[dict],
    cfg: BenchmarkConfig | None = None,
) -> dict:
    """Score SLM explanation claims against SHAP truth rows.

    truth_rows: list of dicts with keys: feature, sign, magnitude_band
    claims: list of dicts with keys: feature, direction, magnitude
    Returns: feature_f1, sign_accuracy, magnitude_accuracy, composite, passed, failure_tags
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    truth_by_feature = {r["feature"]: r for r in truth_rows}
    claims_by_feature = {c["feature"]: c for c in claims}

    truth_features = set(truth_by_feature)
    claim_features = set(claims_by_feature)

    # Feature F1
    tp = len(truth_features & claim_features)
    precision = tp / len(claim_features) if claim_features else 0.0
    recall = tp / len(truth_features) if truth_features else 0.0
    feature_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Sign accuracy and magnitude accuracy on matched features
    matched = truth_features & claim_features
    if matched:
        sign_correct = sum(
            1 for f in matched
            if claims_by_feature[f]["direction"] == truth_by_feature[f]["sign"]
        )
        mag_correct = sum(
            1 for f in matched
            if claims_by_feature[f]["magnitude"] == truth_by_feature[f]["magnitude_band"]
        )
        sign_accuracy = sign_correct / len(matched)
        magnitude_accuracy = mag_correct / len(matched)
    else:
        sign_accuracy = 0.0
        magnitude_accuracy = 0.0

    composite = round(
        cfg.weight_feature_f1 * feature_f1
        + cfg.weight_sign_accuracy * sign_accuracy
        + cfg.weight_magnitude_accuracy * magnitude_accuracy,
        6,
    )

    failure_tags = []
    if feature_f1 < cfg.threshold_feature_f1:
        failure_tags.append("low_feature_f1")
    if sign_accuracy < cfg.threshold_sign_accuracy:
        failure_tags.append("low_sign_accuracy")
    if magnitude_accuracy < cfg.threshold_magnitude_accuracy:
        failure_tags.append("low_magnitude_accuracy")
    if composite < cfg.threshold_composite:
        failure_tags.append("low_composite")

    return {
        "feature_f1": round(feature_f1, 6),
        "sign_accuracy": round(sign_accuracy, 6),
        "magnitude_accuracy": round(magnitude_accuracy, 6),
        "composite": composite,
        "passed": composite >= cfg.threshold_composite and not failure_tags,
        "failure_tags": failure_tags,
    }
