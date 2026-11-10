# Version 0.1.0 contract

The typed API owns immutable observation snapshots; public NumPy arrays are detached readonly copies. Python objects with hostile custom behavior and direct mutation of private attributes are outside this cooperative local API contract.

## Data and splits

160–4096 rows; 1–16 rectangular features, finite magnitude ≤1e6 or missing `None`/JSON `null`. Labels are integer 0/1, excluding bool. Unique IDs have 1–64 ASCII characters 33–126. Optional group IDs and signed64-bit timestamps are all present or all absent and never features. JSON metadata, when present, cannot be null. The CLI rejects duplicate keys, unknown fields, nonfinite numbers and inputs above 4,194,304 bytes. File leaves use NONBLOCK/NOFOLLOW and must fstat as regular before reading. Stalled inherited streams remain incomplete under the outer owner.

Canonical order is ASCII row ID. Allocation units use floor(n/2), floor(n/5), floor(3n/20), remainder, in train/calibration/policy/test order. Stratification uses canonical order and sklearn with seed,seed+1,seed+2 modulo2^31. Whole groups use a seeded permutation of sorted group IDs. Chronological uses sorted whole timestamp buckets; embargo0–16 removes that many buckets between adjacent roles before allocation. No seed retries or class-support repair; every role needs at least eight of both labels.

FrozenSplitPlan metadata identity excludes labels and features. A supplied plan is independently checked for complete disjoint membership, current support and requested assertions. Its historical support is labelled historical. Test-label mutation checks reuse the same plan. Explicit config split settings must match it; stricter requested assurances are checked. Group isolation and time order do not prove independence or absence of future-information features.

## Fitting and metrics

Only train fits median imputation, logistic scaling, a64-tree depth6 single-worker forest, logistic regression(max_iter200) and prior baseline. Calibration fits sklearn sigmoid CalibratedClassifierCV over FrozenEstimator on calibration only. Threshold selection uses policy only; final scores use test. All native work has threadpool limit one. No estimator search, serialization, executable loading or runtime service exists.

Brier is mean squared binary probability error without clipping. Log loss alone clips valid probabilities to [2^-52,1-2^-52], uses float64 and records changed values. Out-of-range or nonfinite probabilities are rejected. AUC assigns exact-score ties half credit. Default confusion uses p≥0.5; undefined precision/recall are null. Reliability bins are float64 uniform, right-closed, with zero included in the first bin; empty bins remain with zero count and null means. Sparse bins and one holdout do not certify probability quality. Calibration can worsen scores.

Threshold candidates are all_negative, distinct observed cutoffs descending, all_positive. Extremes have null cutoff, never infinity. Cutoff decisions use p≥cutoff and tied groups move together. Integer penalty is FPcost·FP+FNcost·FN, each cost0–1000 with positive sum. First minimum wins, so the tie order is deterministic. These hypothetical units are not currency or deployment advice. At most4098 candidates.

## Output and publication

Fixed files: report.json(CTL1 envelope),reliability.csv,thresholds.csv,report.html,manifest.json. JSON includes aligned test/policy labels and probabilities for independent replay; no model objects. Runtime scalar/pair/count replay precedes rendering. CSV is aggregate numeric data with fixed model/policy names. HTML escapes strings, uses inline SVG and no scripts/external assets. Manifest hashes the other four files; it does not hash itself. All five files together must fit the caller budget, at most4,194,304 bytes.

Publication supports owned cooperative local Linux directories, stable input leases and same-filesystem staging. Files/stage are synced before real renameat2(RENAME_NOREPLACE). Existing destinations are preserved. After rename, parent-sync failure leaves the committed directory visible and reports uncertainty. There is no overwrite fallback, hostile-parent snapshot, power-loss, universal-cleanup or real-time guarantee. Stdout is not atomic; require its complete JSON frame and exit0 plus valid artifacts.

| Outcome | Exit | Meaning |
|---|---:|---|
| COMPLETE | 0 | Verified complete artifacts committed and parent sync returned successfully |
| INVALID_INPUT | 2 | Fixed schema/domain/option violation |
| INVALID_SPLIT | 2 | Unsupported membership/support/assurance |
| FIT_FAILED | 2 | Verified convergence, finite-result or replay failure |
| UNKNOWN_BUDGET | 2 | Supported data exceeds a lower caller admission/output limit |
| ALREADY_EXISTS | 3 | Actual no-replace collision; previous destination preserved |
| PRECOMMIT_FAILED / IO_FAILED | 3 | No successful report commit in this invocation |
| COMMITTED_UNCERTAIN | 3 | Report visible; later durability confirmation failed |
| INCOMPLETE | 3 or outer execution | Dependency/native failure, timeout, OOM or unfinished execution |

Native failure tests use labelled injection seams, not claims of observed storage failure. No high-stakes decisions, causal conclusion, general accuracy, commercial delivery or production-readiness certification is offered.
