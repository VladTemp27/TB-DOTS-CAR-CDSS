# Permutation Feature Importance — Interpretation

**Model:** Temporal Hybrid Bi-LSTM (`hybrid_lstm_failure_positive/best_model.pt`, spec `h48_l1_d04_balanced_bce`)
**Task:** predict P(Failure) at full sequence (M0–M12); label convention `1 = Failure`, `0 = Success`
**Baseline on test split:** ROC-AUC = **0.906**, failure-F1 = 0.522 (threshold 0.05)
**Test split:** 45 patients (7 failures, 38 successes)
**Method:** each feature is shuffled across patients 10×; `auc_drop_mean` = mean fall in ROC-AUC vs. baseline. **Higher = the model relies on it more.** `f1_drop_mean` is the matching fall in failure-F1 at the deployed 0.05 threshold.

> ⚠️ **Read these as ranks, not precise effects.** With only 7 failures in the test set, AUC moves in coarse steps (~0.02 per patient pair). Small values (|drop| < 0.01) are within noise, and *negative* drops do not mean a feature "hurts" — they are sampling noise around zero. Treat the top ~3–4 features as the trustworthy signal.

---

## 1. Primary drivers (clear, above-noise importance)

| Feature | Group | AUC drop | F1 drop | Reading |
|---|---|---|---|---|
| **Monthly Doses Taken (Missing)** | temporal | **0.111** | 0.122 | The single most important input. The *presence/absence* of a monthly dose record drives both ranking (AUC) and the operating-point decision (F1). A gap in monthly dosing records is the model's strongest failure signal — consistent with TB, where interrupted/unrecorded DOTS visits flag treatment default. |
| **Cumulative Doses Taken** | temporal | **0.046** | ≈0 (−0.002) | Total adherence-to-date is a strong *ranking* feature (it separates failures from successes) but, on its own, doesn't move the 0.05-threshold decisions — its contribution overlaps with the missingness flag and monthly counts. |
| **Monthly Doses Taken** | temporal | **0.042** | 0.028 | The actual monthly dose count matters for both ranking and thresholded decisions. Together with the two rows above, **the adherence/dosing block dominates the model** — clinically the expected driver of TB treatment outcome. |

**Takeaway:** the model is, as intended, an *adherence-driven* failure predictor. The three dosing features (and their missingness) account for the bulk of its discriminative power.

---

## 2. Secondary contributors (small but consistent positive signal)

These sit just above the noise floor (~0.01–0.02). Individually weak; collectively they sharpen the model.

| Feature | Group | AUC drop | Reading |
|---|---|---|---|
| Height (cm) | static | 0.024 | Baseline anthropometric; likely a proxy for body size/age-sex. Modest, stable (low std). |
| BP Systolic | static | 0.015 | Baseline vital; weak comorbidity/severity proxy. |
| % Adherence (Missing) | temporal | 0.012 | Reinforces the adherence-missingness theme from §1. |
| Monthly Missed Doses (Missing) | temporal | 0.012 | Same theme — *whether* missed-dose data exists carries signal. |
| BP Diastolic | static | 0.010 | Weak vital-sign contribution. |
| Weight (Missing, temporal) | temporal | 0.009 | Mild — missing monthly weight tracking. |

---

## 3. Missingness indicators worth noting

A recurring pattern: **the `is_missing_*` flags often matter more than the underlying values.** This is informative-missingness (MNAR) — *that* a measurement is absent is itself predictive, typically because missing records co-occur with disengagement from care.

- `Monthly Doses Taken (Missing)` (0.111) ≫ `Monthly Doses Taken` (0.042)
- `% Adherence (Missing)` (0.012) > `% Adherence` (0.004)
- `Xpert MTB/RIF (Missing)` (0.007) while `Xpert MTB/RIF` value = 0.000

**Caution for deployment:** if missingness in the live system is driven by data-entry workflow rather than patient behavior, this dependence could shift. Worth monitoring.

---

## 4. Near-zero / inert features (|drop| ≲ 0.005)

These do not measurably change AUC when shuffled — the model effectively ignores them on this test set:

`Weight (kg) (Missing)`, `Height (Missing)`, `Cumulative Doses Taken (Missing)`, `Age`, `Smear TB-LAMP (Missing)`, `Intensive Phase Start Date`, `Treatment Start Date`, `Monthly Missed Doses`, `Date of Diagnosis`, `Date of Notification`, and the **exactly 0.000** group: `Smear Microscopy`, `Xpert MTB/RIF` (both static and temporal value channels), `Smear TB-LAMP`, `Height (cm) (Missing)`, `Name of Treatment Unit (Missing)`.

Notable: the **diagnostic-test result channels** (`Smear Microscopy`, `Xpert MTB/RIF`, `Smear TB-LAMP` values) contribute ~0 — the model leans on *adherence dynamics over time*, not baseline diagnostics, to predict outcome.

---

## 5. Negative drops (noise, read as ≈ 0)

| Feature | Group | AUC drop |
|---|---|---|
| O₂ Saturation | static | −0.006 |
| Weight (kg) | static | −0.011 |
| Heart Rate | static | −0.022 |
| Height | temporal | −0.028 |
| Weight | temporal | −0.050 |

A negative value means shuffling the feature *slightly raised* test AUC. With 7 failures this is **sampling noise, not evidence the feature is harmful** — permuting a near-irrelevant feature can randomly nudge a few borderline patients across the decision boundary in a favorable direction. The honest interpretation: these features carry **no usable signal** on this test set. (`weight`/`height` temporal channels are the largest negatives, but their *missingness* flags showed mild positive signal — so it's the raw monthly weight/height values, not the act of recording them, that are uninformative.)

---

## Bottom line

1. **Adherence is the model.** Monthly dosing records — especially whether they exist — plus cumulative doses explain almost all of the model's discrimination. This is clinically the right thing to key on for TB treatment-failure prediction.
2. **Missingness is signal.** The model exploits informative-missingness in the adherence block; flag this for monitoring if data-entry behavior changes in deployment.
3. **Baseline diagnostics and vitals are minor-to-inert.** Smear/Xpert results and most static vitals contribute little once temporal adherence is present.
4. **Statistical caveat.** n=45 (7 failures) makes everything below ~0.01 indistinguishable from zero. For publication-grade importances, re-estimate with `--n-repeats 20` and ideally report on a larger evaluation set or via PR-AUC (better suited to this 1:5 imbalance).

*Source: `permutation_importance.csv` · regenerate with `python models/Temporal/v2/feature_permutation_importance.py`*
