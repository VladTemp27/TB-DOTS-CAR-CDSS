import re
from pathlib import Path

# Direction keywords — ordered tuples for deterministic matching (first match wins)
_INCREASE_WORDS = (
    "strongly increases", "significantly increases", "increases", "increase",
    "raises", "raise", "higher", "elevates", "elevate",
)
_DECREASE_WORDS = (
    "strongly reduces", "significantly reduces", "reduces", "reduce",
    "decreases", "decrease", "lowers", "lower", "declines", "decline",
)

# Magnitude keywords mapped to canonical form (ordered: more specific first)
_MAGNITUDE_MAP = (
    ("strongly", "strong"),
    ("significantly", "strong"),
    ("strong", "strong"),
    ("moderately", "moderate"),
    ("moderate", "moderate"),
    ("slightly", "weak"),
    ("weakly", "weak"),
    ("weak", "weak"),
)

# Feature names to recognize — multi-word first to avoid partial matches
_KNOWN_FEATURES: tuple[str, ...] = (
    # Multi-word first to prevent partial matches on substrings
    "Drug Resistance Status",
    "Missing Indicators",
    "Days To Treatment",
    "Days to Treatment",
    "Registration Group",
    "Case Registration Group",
    "Bacteriologic Status",
    "Bacteriological Status",
    "Microscopy Result",
    "Smear Microscopy",
    "Treatment Adherence",
    "Treatment Regimen",
    "Body Weight",
    "Smear Result",
    "Smear/TB Lamp",
    "TB Lamp",
    "Xpert Result",
    "Xpert MTB/RIF",
    "Monthly Doses",
    "Cumulative Doses",
    "Missed Doses",
    "Months Available",
    "Vital Signs",
    "Site Factors",
    "Chest X-Ray",
    "Chest Xray",
    "Chest X Ray",
    "Civil Status",
    "Diagnosis Type",
    # Single-word last
    "Comorbidities",
    "Nationality",
    "Dates",
    "Age",
    "Sex",
    "Height",
)

_MIXED_WORDS = (
    "mixed effect",    # more specific first
    "mixed",
    "both increased and decreased",
    "varied",
    "variable",
    "inconsistent",
)


def parse_explanation(
    text: str,
    alias_map: dict[str, str] | None = None,
) -> dict:
    """Parse an SLM explanation text and extract feature+direction+magnitude claims.

    Args:
        text: Raw SLM explanation text.
        alias_map: When provided, only features whose canonical name appears as a value
            in alias_map are extracted. When None, all features in _KNOWN_FEATURES are searched.

    Returns:
        dict with keys:
            status: "ok" or "parse_failed"
            claims: list of {"feature": str, "direction": "increase"|"decrease",
                             "magnitude": str|"unspecified"}
            coverage: fraction of known features that appear in the text with a direction
    """
    # Determine which features to search for
    if alias_map is not None:
        canonical_names = set(alias_map.values())
        features_to_search = tuple(f for f in _KNOWN_FEATURES if f in canonical_names)
    else:
        features_to_search = _KNOWN_FEATURES

    raw_claims: list[tuple[int, dict]] = []  # (position, claim)

    for feature in features_to_search:
        pattern = re.compile(re.escape(feature), re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue

        # Context window: ±250 chars — SLM explanations use long sentences where
        # direction/magnitude words appear well after the feature name.
        start = max(0, match.start() - 250)
        end = min(len(text), match.end() + 250)
        context = text[start:end]
        context_lower = context.lower()

        # Find direction word (first match wins; longer phrases checked first)
        direction: str | None = None
        for phrase in _INCREASE_WORDS:
            if re.search(r"\b" + re.escape(phrase) + r"\b", context_lower):
                direction = "increase"
                break
        if direction is None:
            for phrase in _DECREASE_WORDS:
                if re.search(r"\b" + re.escape(phrase) + r"\b", context_lower):
                    direction = "decrease"
                    break
        if direction is None:
            for phrase in _MIXED_WORDS:
                if re.search(r"\b" + re.escape(phrase) + r"\b", context_lower):
                    direction = "mixed"
                    break

        if direction is None:
            continue  # Feature mentioned but no direction word found

        # Find magnitude word (first match wins; more specific entries are first)
        magnitude = "unspecified"
        for word, canonical in _MAGNITUDE_MAP:
            if re.search(r"\b" + re.escape(word) + r"\b", context_lower):
                magnitude = canonical
                break

        raw_claims.append((match.start(), {
            "feature": feature,
            "direction": direction,
            "magnitude": magnitude,
        }))

    # Sort by appearance order in text
    raw_claims.sort(key=lambda x: x[0])
    claims = [claim for _, claim in raw_claims]

    status = "ok" if claims else "parse_failed"
    coverage = len(claims) / len(features_to_search) if features_to_search else 0.0

    return {
        "status": status,
        "claims": claims,
        "coverage": coverage,
    }
