import re

# Direction keywords mapped to canonical form
_INCREASE_WORDS = {"increase", "increases", "raise", "raises", "higher", "elevate", "elevates"}
_DECREASE_WORDS = {"decrease", "decreases", "reduce", "reduces", "lower", "lowers", "decline", "declines"}

# Magnitude keywords mapped to canonical form
_MAGNITUDE_MAP = {
    "strongly": "strong",
    "strong": "strong",
    "significantly": "strong",
    "moderately": "moderate",
    "moderate": "moderate",
    "slightly": "weak",
    "weakly": "weak",
    "weak": "weak",
}

# Feature names to recognize (multi-word come first to avoid partial matches)
_KNOWN_FEATURES = [
    "Days To Treatment",
    "Treatment Category",
    "Patient Category",
    "Smear Result",
    "Age",
    "Sex",
    "BMI",
    "Province",
    "Weight",
    "Height",
]


def parse_explanation(text: str, alias_map: dict[str, str] | None = None) -> dict:
    """Parse an SLM explanation text and extract feature+direction+magnitude claims.

    Returns a dict with keys:
        status: "ok" or "parse_failed"
        claims: list of {"feature": str, "direction": "increase"|"decrease", "magnitude": str}
        coverage: float (fraction of known features mentioned)
    """
    # Collect (match_start, feature, direction, magnitude) tuples, then sort by position
    raw_claims = []

    # For each known feature, check if it appears in the text
    for feature in _KNOWN_FEATURES:
        pattern = re.compile(re.escape(feature), re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue

        # Find direction word near the feature (within 60 chars before/after)
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 60)
        context = text[start:end].lower()

        direction = None
        for word in _INCREASE_WORDS:
            if re.search(r'\b' + word + r'\b', context):
                direction = "increase"
                break
        if direction is None:
            for word in _DECREASE_WORDS:
                if re.search(r'\b' + word + r'\b', context):
                    direction = "decrease"
                    break

        if direction is None:
            continue

        # Find magnitude word near the feature
        magnitude = "moderate"  # default
        for word, canonical in _MAGNITUDE_MAP.items():
            if re.search(r'\b' + word + r'\b', context):
                magnitude = canonical
                break

        raw_claims.append((match.start(), feature, direction, magnitude))

    # Sort claims by their position in the text so claims[0] is the first mention
    raw_claims.sort(key=lambda x: x[0])
    claims = [
        {"feature": feat, "direction": direc, "magnitude": mag}
        for _, feat, direc, mag in raw_claims
    ]

    status = "ok" if claims else "parse_failed"
    mentioned_features = len(claims)
    coverage = mentioned_features / len(_KNOWN_FEATURES) if _KNOWN_FEATURES else 0.0

    return {
        "status": status,
        "claims": claims,
        "coverage": coverage,
    }
