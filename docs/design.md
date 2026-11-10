# Design

Scikit-learn owns fitting, imputation and calibration. The original layer owns bounded snapshots, canonical split roles, evidence, sorted threshold selection, independent replay and publication.

Dataset metadata and features remain separate. A label-independent frozen membership plan makes test-label mutation a useful leakage regression without incorrectly rerunning stratification. Train preprocessing evidence and forbidden-refit tests check calibration ownership. Four roles keep threshold selection apart from both calibration fitting and final scoring.

Metrics use explicit finite float64 conventions. Small Fraction/Decimal/pair-count goldens and brute policy tests are independent of the main implementations. Reports retain prior/logistic/raw/calibrated values and empty bins; they do not manufacture improvement. The CLI performs a separate replay from aligned labels and predictions before writing anything visible.

Reports are complete private artifacts before a fixed native no-replace directory commit. Precommit failures and postcommit uncertainty are different states. The library has no arbitrary command, endpoint, model-file or expression capability. Shared native work is limited to one thread; the qualified environment is Linux amd64/CPython3.13 with content-hashed dependencies.

Tests cover domain versus lower admission limits, mutation isolation, group/time constraints, max shape, actual sklearn calibration, JSON/file adapters, real no-replace collision, cooperative publication races and labelled sync failure seams. Actual execution and source/artifact hashes remain necessary evidence; test source alone is not proof.
