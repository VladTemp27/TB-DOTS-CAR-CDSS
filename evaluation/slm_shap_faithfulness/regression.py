"""Regression comparison between benchmark runs."""


def compare_runs(base: dict, current: dict) -> dict[str, float]:
    """Compute metric deltas between a baseline and current run summary.

    Args:
        base: Summary dict from a previous benchmark run (e.g. {"pass_rate": 0.8}).
        current: Summary dict from the current benchmark run.

    Returns:
        Dict of {metric_delta: value} for each numeric metric present in both.
        Positive delta means improvement.
    """
    deltas = {}
    for key in base:
        if key in current:
            try:
                delta = round(float(current[key]) - float(base[key]), 6)
                deltas[f"{key}_delta"] = delta
            except (TypeError, ValueError):
                pass
    return deltas
