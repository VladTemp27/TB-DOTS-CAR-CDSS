## TB-DOTS CAR CDSS

The temporal-model work is now split into versioned folders so the repository is easier to learn from:

### Folder Map

- [v1](v1) contains the earlier 205-patient temporal pipeline and baseline results.
- [v2](v2) contains the improved 599-patient temporal pipeline and final comparison report.
- [TEMPORAL_MODEL_V1_V2_COMPARISON.md](TEMPORAL_MODEL_V1_V2_COMPARISON.md) explains why v2 is stronger than v1.

### How to read it

Start with [v1/TEMPORAL_MODEL_RESULTS.md](v1/TEMPORAL_MODEL_RESULTS.md) to understand the baseline pipeline, then read [v2/results.md](v2/results.md) to see the refined pipeline, and finally open [TEMPORAL_MODEL_V1_V2_COMPARISON.md](TEMPORAL_MODEL_V1_V2_COMPARISON.md) for the side-by-side explanation.

### What changed between versions

- v1 is the smaller, earlier implementation.
- v2 has the larger dataset, better preprocessing, richer temporal features, and stronger validation.
- The version split is intentional: v1 is the archive, v2 is the active reference point.

