"""Regression comparison between benchmark runs."""


def compare_runs(base: dict, current: dict) -> dict[str, float]:
    """Compute metric deltas between a baseline and current run summary.

    Compares all numeric keys present in either base or current.
    Keys present only in current get a delta equal to the current value (base assumed 0).
    Keys present only in base get a delta equal to -base value (current assumed 0).
    Positive delta means improvement.
    """
    all_keys = set(base) | set(current)
    deltas = {}
    for key in all_keys:
        base_val = base.get(key)
        curr_val = current.get(key)
        try:
            delta = round(float(curr_val or 0.0) - float(base_val or 0.0), 6)
            deltas[f"{key}_delta"] = delta
        except (TypeError, ValueError):
            pass
    return deltas
