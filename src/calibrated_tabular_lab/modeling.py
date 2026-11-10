"""Train-only baseline fitting, FrozenEstimator sigmoid calibration, and provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
import warnings
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from calibrated_tabular_lab.metrics import validate_binary_labels, validate_probability_vector
from calibrated_tabular_lab.status import (
    DOMAIN_MAX_FEATURE_MAGNITUDE, DOMAIN_MAX_FEATURES, DOMAIN_MAX_ROWS,
    LabError, LabStatus, validate_seed,
)

MODEL_NAMES: tuple[str, str, str, str] = (
    "prior",
    "logistic",
    "raw_forest",
    "calibrated_forest",
)
FOREST_N_ESTIMATORS: int = 64
FOREST_MAX_DEPTH: int = 6
LOGISTIC_MAX_ITER: int = 200


@dataclass(frozen=True, slots=True)
class TrainingEvidence:
    """Bounded train-only preprocessing statistics and fixed model hyperparameters."""

    seed: int
    n_train_rows: int
    n_input_features: int
    n_imputed_features: int
    forest_imputer_medians: tuple[float, ...]
    forest_missing_indicator_columns: tuple[int, ...]
    logistic_imputer_medians: tuple[float, ...]
    logistic_scaler_mean: tuple[float, ...]
    logistic_scaler_scale: tuple[float, ...]
    logistic_n_iter: int
    prior_positive_rate: float
    forest_params: Mapping[str, Any]
    logistic_params: Mapping[str, Any]
    raw_forest_train_probabilities: tuple[float, ...]
    threadpool_max_threads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_train_rows": self.n_train_rows,
            "n_input_features": self.n_input_features,
            "n_imputed_features": self.n_imputed_features,
            "forest_imputer_medians": list(self.forest_imputer_medians),
            "forest_missing_indicator_columns": list(self.forest_missing_indicator_columns),
            "logistic_imputer_medians": list(self.logistic_imputer_medians),
            "logistic_scaler_mean": list(self.logistic_scaler_mean),
            "logistic_scaler_scale": list(self.logistic_scaler_scale),
            "logistic_n_iter": self.logistic_n_iter,
            "prior_positive_rate": self.prior_positive_rate,
            "forest_params": dict(self.forest_params),
            "logistic_params": dict(self.logistic_params),
            "raw_forest_train_probabilities": list(self.raw_forest_train_probabilities),
            "threadpool_max_threads": self.threadpool_max_threads,
        }


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Audit evidence proving base forest pipeline remained frozen during calibration."""

    method: str
    ensemble: bool
    n_calibration_rows: int
    base_imputer_medians_unchanged: bool
    base_train_predictions_unchanged: bool
    post_calibration_forest_imputer_medians: tuple[float, ...]
    post_calibration_raw_forest_train_probabilities: tuple[float, ...]
    calibrated_forest_train_probabilities: tuple[float, ...]
    threadpool_max_threads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "ensemble": self.ensemble,
            "n_calibration_rows": self.n_calibration_rows,
            "base_imputer_medians_unchanged": self.base_imputer_medians_unchanged,
            "base_train_predictions_unchanged": self.base_train_predictions_unchanged,
            "post_calibration_forest_imputer_medians": list(
                self.post_calibration_forest_imputer_medians
            ),
            "post_calibration_raw_forest_train_probabilities": list(
                self.post_calibration_raw_forest_train_probabilities
            ),
            "calibrated_forest_train_probabilities": list(
                self.calibrated_forest_train_probabilities
            ),
            "threadpool_max_threads": self.threadpool_max_threads,
        }


@dataclass(frozen=True, slots=True)
class FittedBaseModels:
    """In-memory train-fitted base models and their training evidence."""

    prior_pipeline: Pipeline
    logistic_pipeline: Pipeline
    forest_pipeline: Pipeline
    training_evidence: TrainingEvidence


@dataclass(frozen=True, slots=True)
class FittedExperimentModels:
    """In-memory fitted base and calibrated models plus provenance."""

    prior_pipeline: Pipeline
    logistic_pipeline: Pipeline
    forest_pipeline: Pipeline
    calibrated_forest: CalibratedClassifierCV
    training_evidence: TrainingEvidence
    calibration_provenance: CalibrationProvenance


def _validate_feature_matrix_input(X: np.ndarray, *, role_name: str) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype != np.float64:
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"{role_name} feature matrix must be a 2D float64 ndarray.",
            code="invalid_feature_matrix",
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"{role_name} feature matrix must be non-empty.",
            code="empty_feature_matrix",
        )
    if X.shape[0] > DOMAIN_MAX_ROWS or X.shape[1] > DOMAIN_MAX_FEATURES:
        raise LabError(
            LabStatus.INVALID_INPUT,
            "feature matrix exceeds fixed row or feature dimensions.",
            code="feature_matrix_domain_shape",
        )
    # Check for inf/-inf (NaN is allowed for missing values).
    if np.isinf(X).any():
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"{role_name} feature matrix contains infinite values.",
            code="infinite_feature_value",
        )
    if np.any(np.abs(X) > DOMAIN_MAX_FEATURE_MAGNITUDE):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "feature magnitude exceeds the fixed domain bound.",
            code="feature_magnitude_exceeded",
        )
    return X


def _check_no_all_missing_columns(X_train: np.ndarray) -> None:
    all_missing = np.all(np.isnan(X_train), axis=0)
    if bool( np.any(all_missing) ):
        bad_cols = [int(i) for i in np.flatnonzero(all_missing)]
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"training set contains all-missing feature column(s): {bad_cols}.",
            code="all_missing_training_feature",
            details={"columns": bad_cols},
        )


def _extract_validated_positive_probabilities(
    estimator: Any, X: np.ndarray, *, model_name: str
) -> tuple[float, ...]:
    try:
        raw_proba = estimator.predict_proba(X)
    except (ValueError, FloatingPointError) as exc:
        raise LabError(
            LabStatus.FIT_FAILED,
            f"{model_name} predict_proba failed: {exc}",
            code="predict_proba_failed",
        ) from exc
    if (
        not isinstance(raw_proba, np.ndarray)
        or raw_proba.ndim != 2
        or raw_proba.shape != (X.shape[0], 2)
    ):
        raise LabError(
            LabStatus.FIT_FAILED,
            f"{model_name} produced malformed probability array shape {getattr(raw_proba, 'shape', None)}.",
            code="malformed_proba_shape",
        )
    if not bool(np.all(np.isfinite(raw_proba))):
        raise LabError(
            LabStatus.FIT_FAILED,
            f"{model_name} produced non-finite probabilities.",
            code="nonfinite_estimator_output",
        )
    if not bool(np.all((raw_proba >= 0.0) & (raw_proba <= 1.0))):
        raise LabError(
            LabStatus.FIT_FAILED,
            f"{model_name} produced probabilities outside [0, 1].",
            code="out_of_bounds_estimator_output",
        )
    return validate_probability_vector(
        raw_proba[:, 1],
        expected_length=X.shape[0],
        error_status=LabStatus.FIT_FAILED,
    )


def _observed_max_threads() -> int:
    info = threadpool_info()
    if not info:
        return 1
    return max(int(entry.get("num_threads", 1)) for entry in info)


def fit_base_models(
    X_train: np.ndarray,
    y_train: Sequence[int] | np.ndarray,
    *,
    seed: int,
) -> FittedBaseModels:
    """Fit prior, logistic, and raw random-forest pipelines exclusively on train."""

    valid_seed = validate_seed(seed)
    valid_X = _validate_feature_matrix_input(X_train, role_name="train")
    valid_y = validate_binary_labels(y_train)
    if valid_X.shape[0] != len(valid_y):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "X_train row count does not match y_train length.",
            code="train_length_mismatch",
        )
    _check_no_all_missing_columns(valid_X)
    y_arr = np.asarray(valid_y, dtype=np.int64)
    if len(set(valid_y)) < 2:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "train role must contain both class 0 and class 1.",
            code="single_class_train",
        )

    prior_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    logistic_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=valid_seed),
            ),
        ]
    )
    forest_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=FOREST_N_ESTIMATORS,
                    max_depth=FOREST_MAX_DEPTH,
                    n_jobs=1,
                    random_state=valid_seed,
                ),
            ),
        ]
    )

    with threadpool_limits(limits=1):
        max_threads = _observed_max_threads()
        try:
            prior_pipeline.fit(valid_X, y_arr)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                logistic_pipeline.fit(valid_X, y_arr)
            forest_pipeline.fit(valid_X, y_arr)
        except ConvergenceWarning as exc:
            raise LabError(
                LabStatus.FIT_FAILED,
                f"logistic regression failed to converge within {LOGISTIC_MAX_ITER} iterations.",
                code="logistic_convergence_failed",
            ) from exc
        except (ValueError, FloatingPointError) as exc:
            raise LabError(
                LabStatus.FIT_FAILED,
                f"base estimator fitting failed: {exc}",
                code="base_fit_failed",
            ) from exc

        raw_train_probs = _extract_validated_positive_probabilities(
            forest_pipeline, valid_X, model_name="raw_forest"
        )
        # Validate prior and logistic train outputs as well.
        _extract_validated_positive_probabilities(
            prior_pipeline, valid_X, model_name="prior"
        )
        _extract_validated_positive_probabilities(
            logistic_pipeline, valid_X, model_name="logistic"
        )

    forest_imputer: SimpleImputer = forest_pipeline.named_steps["imputer"]
    logistic_imputer: SimpleImputer = logistic_pipeline.named_steps["imputer"]
    logistic_scaler: StandardScaler = logistic_pipeline.named_steps["scaler"]
    logistic_clf: LogisticRegression = logistic_pipeline.named_steps["classifier"]

    forest_medians = tuple(float(v) for v in forest_imputer.statistics_)
    if not all(np.isfinite(forest_medians)):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "non-finite imputer median statistic on training data.",
            code="nonfinite_imputer_median",
        )
    indicator = getattr(forest_imputer, "indicator_", None)
    indicator_cols = (
        tuple(int(i) for i in indicator.features_)
        if indicator is not None and hasattr(indicator, "features_")
        else ()
    )
    n_imputed_features = len(forest_medians) + len(indicator_cols)
    logistic_medians = tuple(float(v) for v in logistic_imputer.statistics_)
    scaler_mean = tuple(float(v) for v in logistic_scaler.mean_)
    scaler_scale = tuple(float(v) for v in logistic_scaler.scale_)
    n_iter_val = int(np.max(logistic_clf.n_iter_))
    prior_pos_rate = float(np.mean(y_arr))

    evidence = TrainingEvidence(
        seed=valid_seed,
        n_train_rows=int(valid_X.shape[0]),
        n_input_features=int(valid_X.shape[1]),
        n_imputed_features=n_imputed_features,
        forest_imputer_medians=forest_medians,
        forest_missing_indicator_columns=indicator_cols,
        logistic_imputer_medians=logistic_medians,
        logistic_scaler_mean=scaler_mean,
        logistic_scaler_scale=scaler_scale,
        logistic_n_iter=n_iter_val,
        prior_positive_rate=prior_pos_rate,
        forest_params=MappingProxyType(
            {
                "n_estimators": FOREST_N_ESTIMATORS,
                "max_depth": FOREST_MAX_DEPTH,
                "n_jobs": 1,
                "random_state": valid_seed,
            }
        ),
        logistic_params=MappingProxyType(
            {
                "max_iter": LOGISTIC_MAX_ITER,
                "random_state": valid_seed,
            }
        ),
        raw_forest_train_probabilities=raw_train_probs,
        threadpool_max_threads=max_threads,
    )

    return FittedBaseModels(
        prior_pipeline=prior_pipeline,
        logistic_pipeline=logistic_pipeline,
        forest_pipeline=forest_pipeline,
        training_evidence=evidence,
    )


def calibrate_forest_model(
    base_models: FittedBaseModels,
    X_calibration: np.ndarray,
    y_calibration: Sequence[int] | np.ndarray,
    *,
    X_train: np.ndarray,
) -> FittedExperimentModels:
    """Fit sigmoid CalibratedClassifierCV over FrozenEstimator(forest_pipeline) on calibration only."""

    if not isinstance(base_models, FittedBaseModels):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "base_models must be a FittedBaseModels instance.",
            code="invalid_base_models_type",
        )
    valid_X_cal = _validate_feature_matrix_input(X_calibration, role_name="calibration")
    valid_X_train = _validate_feature_matrix_input(X_train, role_name="train")
    valid_y_cal = validate_binary_labels(y_calibration)
    if valid_X_cal.shape[0] != len(valid_y_cal):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "X_calibration row count does not match y_calibration length.",
            code="calibration_length_mismatch",
        )
    if len(set(valid_y_cal)) < 2:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "calibration role must contain both class 0 and class 1.",
            code="single_class_calibration",
        )

    forest_pipeline = base_models.forest_pipeline

    y_cal_arr = np.asarray(valid_y_cal, dtype=np.int64)
    frozen_forest = FrozenEstimator(forest_pipeline)
    calibrator = CalibratedClassifierCV(
        frozen_forest,
        method="sigmoid",
        ensemble=False,
        n_jobs=1,
    )

    with threadpool_limits(limits=1):
        max_threads = _observed_max_threads()
        try:
            calibrator.fit(valid_X_cal, y_cal_arr)
        except (ValueError, FloatingPointError) as exc:
            raise LabError(
                LabStatus.FIT_FAILED,
                f"sigmoid probability calibration failed: {exc}",
                code="calibration_fit_failed",
            ) from exc

        post_raw_train_probs = _extract_validated_positive_probabilities(
            forest_pipeline, valid_X_train, model_name="raw_forest"
        )
        cal_train_probs = _extract_validated_positive_probabilities(
            calibrator, valid_X_train, model_name="calibrated_forest"
        )

    forest_imputer: SimpleImputer = forest_pipeline.named_steps["imputer"]
    post_medians = tuple(float(v) for v in forest_imputer.statistics_)
    medians_unchanged = post_medians == base_models.training_evidence.forest_imputer_medians
    preds_unchanged = (
        post_raw_train_probs == base_models.training_evidence.raw_forest_train_probabilities
    )
    if not medians_unchanged or not preds_unchanged:
        raise LabError(
            LabStatus.FIT_FAILED,
            "base forest pipeline state changed during calibration.",
            code="frozen_estimator_mutated",
        )

    provenance = CalibrationProvenance(
        method="sigmoid",
        ensemble=False,
        n_calibration_rows=int(valid_X_cal.shape[0]),
        base_imputer_medians_unchanged=medians_unchanged,
        base_train_predictions_unchanged=preds_unchanged,
        post_calibration_forest_imputer_medians=post_medians,
        post_calibration_raw_forest_train_probabilities=post_raw_train_probs,
        calibrated_forest_train_probabilities=cal_train_probs,
        threadpool_max_threads=max_threads,
    )
    return FittedExperimentModels(
        prior_pipeline=base_models.prior_pipeline,
        logistic_pipeline=base_models.logistic_pipeline,
        forest_pipeline=forest_pipeline,
        calibrated_forest=calibrator,
        training_evidence=base_models.training_evidence,
        calibration_provenance=provenance,
    )


def predict_role_probabilities(
    models: FittedExperimentModels,
    X: np.ndarray,
    *,
    role_name: str,
) -> Mapping[str, tuple[float, ...]]:
    """Predict validated positive-class probabilities for all four models on X."""

    valid_X = _validate_feature_matrix_input(X, role_name=role_name)
    with threadpool_limits(limits=1):
        prior_probs = _extract_validated_positive_probabilities(
            models.prior_pipeline, valid_X, model_name="prior"
        )
        logistic_probs = _extract_validated_positive_probabilities(
            models.logistic_pipeline, valid_X, model_name="logistic"
        )
        raw_forest_probs = _extract_validated_positive_probabilities(
            models.forest_pipeline, valid_X, model_name="raw_forest"
        )
        cal_forest_probs = _extract_validated_positive_probabilities(
            models.calibrated_forest, valid_X, model_name="calibrated_forest"
        )
    return MappingProxyType(
        {
            "prior": prior_probs,
            "logistic": logistic_probs,
            "raw_forest": raw_forest_probs,
            "calibrated_forest": cal_forest_probs,
        }
    )
