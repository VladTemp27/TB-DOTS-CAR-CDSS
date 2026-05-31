# Citation reconciliation guide — merging PR #135 (`humanize/method-rewrite`) into `main`

The humanization branch was built from the citation-stripped base, so its prose carries
` .` gaps where citations belong. `main` later restored **33** real `\citep{}` citations.
To merge: **keep the branch's humanized prose, and re-attach each citation below at the
matching sentence in that prose** (the wording differs from `main`, but every claim is
preserved, so match on the claim, not the exact words).

Verification after applying: the file's `\citep{}` multiset must exactly equal `main`'s
(33 total). Quick check:

```
diff <(git show main:paper/apa/thesis_apa.tex | grep -oE '\\citep\{[^}]*\}' | sort) \
     <(grep -oE '\\citep\{[^}]*\}' paper/apa/thesis_apa.tex | sort)
```
(empty diff = every citation restored, none added or dropped)

## The 33 citations and the claim each attaches to (document order)

| # | Citation | Place at the sentence in the humanized prose ending on this claim |
|---|----------|------------------------------------------------------------------|
| 1 | `\citep{ref23, ref24}` | output stays legible to providers (multi-layer framework para) |
| 2 | `\citep{ref6}` | private practitioners in the Public-Private Mix DOTS (PPMD) scheme |
| 3 | `\citep{ref31}` | rather than forcing one rule across the board (temporal threshold triage) |
| 4 | `\citep{ref27}` | nominal or ordinal categories whose spacing is uneven (Cramér's V, temporal EDA) |
| 5 | `\citep{ref10, ref12}` | only strong correlation justifies the extra complexity of an RNN or LSTM (lag-1) |
| 6 | `\citep{ref31, ref32}` | ...Chained Equations (MICE) (Figure~\ref{fig:missing_data_pipeline}) (Missing Data intro) |
| 7 | `\citep{ref32}` | Deleting cases wholesale is off the table (Little's MCAR result) |
| 8 | `\citep{ref31, ref32}` | lets each handling strategy match the statistics of its own variable (column-level mechanism) |
| 9 | `\citep{ref35}` | those above the median get priority for stochastic imputation (RF importance ranking) |
| 10 | `\citep{ref37}` | the longitudinal design of TB monitoring showing through, not a data quality fault (thresholds) |
| 11 | `\citep{ref32}` | No variable here cleared all three conditions at once, so the pathway never fired (listwise) |
| 12 | `\citep{ref34, ref36, ref33}` | leak information the model has no legitimate access to at inference (backward-fill barred) |
| 13 | `\citep{ref35}` | without standardized variables, much like the missForest framework (ExtraTreesRegressor) |
| 14 | `\citep{ref38}` | the number of imputations should roughly match the percentage of missing information (Von Hippel) |
| 15 | `\citep{ref1}` | instead of collapsing them onto conditional means (stochastic vs deterministic) |
| 16 | `\citep{lundberg2017unified}` | measures each variable's marginal contribution to the model's output (SHAP, Identification of Variables) |
| 17 | `\citep{ref26, ref27}` | the line between computational attribution and medical judgment is kept deliberately clear |
| 18 | `\citep{ref11}` | model long-term dependencies without the vanishing gradients of a plain RNN (Bi-LSTM choice) |
| 19 | `\citep{ref12}` | the strong month-to-month adherence dependence the lag-1 analysis turned up (Bi-LSTM) |
| 20 | `\citep{ref10}` | strong static learners like XGBoost, which holds up better across the different patient subgroups |
| 21 | `\citep{hastie2009elements}` | holdout validation (10\%), with the class proportions held across all three (Data Partitioning) |
| 22 | `\citep{chawla2002smote}` | mid-sentence, immediately after `SMOTE` (before "interpolates synthetic minority-class points") |
| 23 | `\citep{batista2004study,nishat2022smoteenn}` | mid-sentence, immediately after `SMOTE-ENN` (before "runs SMOTE, then clears...") |
| 24 | `\citep{batista2004study}` | mid-sentence, immediately after `SMOTE-Tomek` (before "runs SMOTE, then removes Tomek links") |
| 25 | `\citep{salmi2024imbalanced}` | they keep their original distributions, so the evaluation stays unbiased |
| 26 | `\citep{hastie2009elements}` | a mean and standard deviation of ROC-AUC across folds (Cross-Validation) |
| 27 | `\citep{chicco2020advantages}` | $+1$ for perfect prediction, $0$ for random, $-1$ for complete inversion (MCC supplementary) |
| 28 | `\citep{davis2006relationship}` | PR-AUC tells you more than ROC-AUC ... when the positive class is rare (PR-AUC definition) |
| 29 | `\citep{chicco2020advantages}` | $+1$ perfect, $0$ random, $-1$ fully inverted (MCC formula definition) |
| 30 | `\citep{ref39, ref41}` | not a predictor, not a diagnostic engine, not a source of fresh clinical inference (SLM layer) |
| 31 | `\citep{ref40, ref42, ref44}` | keeps interpretability tied to the original attribution signal (SLM confined) |
| 32 | `\citep{ref42, ref43}` | a generative output is easy to over-read past what actually supports it (Safety Scope 1) |
| 33 | `\citep{ref44}` | transparency, bounded use, and human oversight are still the core requirements (Safety Scope 2) |

## Also do on merge
- **References + Appendices A–D:** comment out (do not delete) for now, per the team's call.
- Most ` .` gaps in the Intro/lit-review are *not* in this list — `main` never restored them
  either, so leave them as gaps (they match `main`'s current state).
