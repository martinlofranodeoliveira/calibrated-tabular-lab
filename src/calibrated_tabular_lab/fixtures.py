"""Deterministic synthetic process-quality fixtures for testing and experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from calibrated_tabular_lab.dataset import Observation, TabularDataset
from calibrated_tabular_lab.status import (
    DOMAIN_MAX_FEATURES, DOMAIN_MAX_ROWS, DOMAIN_MIN_ROWS,
    AdmissionBudget, LabError, LabStatus, coerce_budget, is_strict_int,
    to_finite_float64, validate_seed,
)

DEFAULT_PROCESS_FEATURE_NAMES: tuple[str, ...] = (
    "chamber_delta_c",
    "flow_ratio",
    "pressure_ripple_kpa",
    "residence_index",
    "viscosity_proxy",
    "agitation_harmonic",
    "thermal_skew",
    "reagent_balance",
    "jacket_gradient",
    "purge_interval",
    "nozzle_impedance",
    "condensate_load",
    "torque_variance",
    "baffle_shear",
    "feed_pulsation",
    "quench_symmetry",
)


@dataclass(frozen=True, slots=True)
class ThresholdGoldenFixture:
    """Contract golden fixture for threshold-cost sweep verification."""

    y_true: tuple[int, ...]
    y_prob: tuple[float, ...]
    fp_cost: int
    fn_cost: int
    expected_costs: tuple[int, ...]
    expected_cutoff: float
    expected_min_cost: int


@dataclass(frozen=True, slots=True)
class MetricGoldenFixture:
    """Small hand-checkable golden fixture for Brier, log loss, AUC, and reliability."""

    y_true: tuple[int, ...]
    y_prob: tuple[float, ...]
    boundary_probs: tuple[float, ...]
    boundary_labels: tuple[int, ...]
    expected_ten_bin_indices: tuple[int, ...]


def make_threshold_golden_fixture() -> ThresholdGoldenFixture:
    """Return the exact threshold sweep golden specified in the frozen contract."""

    return ThresholdGoldenFixture(
        y_true=(0, 1, 0, 1, 1, 0),
        y_prob=(0.0, 0.2, 0.2, 0.6, 1.0, 1.0),
        fp_cost=1,
        fn_cost=4,
        expected_costs=(12, 9, 5, 2, 3, 3),
        expected_cutoff=0.2,
        expected_min_cost=2,
    )


def make_metric_golden_fixture() -> MetricGoldenFixture:
    """Return small hand-checkable metric and bin-boundary golden fixture."""

    return MetricGoldenFixture(
        y_true=(0, 1, 0, 1, 1, 0, 1, 0),
        y_prob=(0.0, 0.125, 0.25, 0.25, 0.5, 0.75, 0.875, 1.0),
        boundary_probs=(0.0, 0.1, 0.2, 0.5, 1.0),
        boundary_labels=(0, 1, 0, 1, 1),
        expected_ten_bin_indices=(0, 0, 1, 4, 9),
    )


def make_synthetic_process_dataset(
    *,
    n_rows: int = 320,
    n_features: int = 8,
    seed: int = 42,
    include_groups: bool = True,
    include_timestamps: bool = True,
    n_groups: int = 24,
    n_time_buckets: int = 40,
    missing_rate: float = 0.05,
    budget: AdmissionBudget | None = None,
) -> TabularDataset:
    """Generate a deterministic synthetic laboratory process-deviation dataset."""

    valid_seed = validate_seed(seed)
    active_budget = coerce_budget(budget)
    if not is_strict_int(n_rows) or not DOMAIN_MIN_ROWS <= n_rows <= DOMAIN_MAX_ROWS:
        raise LabError(LabStatus.INVALID_INPUT, "n_rows must be an integer in160..4096.", code="invalid_row_count")
    if not is_strict_int(n_features) or not 1 <= n_features <= DOMAIN_MAX_FEATURES:
        raise LabError(LabStatus.INVALID_INPUT, "n_features must be an integer in1..16.", code="invalid_feature_count")
    if not isinstance(include_groups, bool) or not isinstance(include_timestamps, bool):
        raise LabError(LabStatus.INVALID_INPUT, "metadata flags must be booleans.", code="invalid_metadata_flags")
    if not is_strict_int(n_groups) or not 8 <= n_groups <= n_rows:
        raise LabError(LabStatus.INVALID_INPUT, "n_groups must be an integer in8..n_rows.", code="invalid_n_groups")
    if not is_strict_int(n_time_buckets) or not 8 <= n_time_buckets <= n_rows:
        raise LabError(LabStatus.INVALID_INPUT, "n_time_buckets must be an integer in8..n_rows.", code="invalid_n_time_buckets")
    valid_missing_rate = to_finite_float64(missing_rate, field_name="missing_rate")
    if not 0.0 <= valid_missing_rate <= 0.20:
        raise LabError(LabStatus.INVALID_INPUT, "missing_rate must be in[0,0.20].", code="invalid_missing_rate")
    if n_rows > active_budget.max_rows or n_features > active_budget.max_features:
        raise LabError(LabStatus.UNKNOWN_BUDGET, "fixture dimensions exceed caller admission budget.", code="fixture_admission_budget")
    rng = np.random.default_rng(valid_seed)

    # Generate raw continuous process signals in [-3.5, 3.5].
    raw_x = rng.normal(loc=0.0, scale=1.2, size=(n_rows, n_features))
    raw_x = np.clip(raw_x, -250.0, 250.0)

    # Nonlinear process-deviation latent response.
    c0 = raw_x[:, 0]
    c1 = raw_x[:, 1 % n_features]
    c2 = raw_x[:, 2 % n_features]
    c3 = raw_x[:, 3 % n_features]
    nonlinear_signal = (
        0.85 * np.sin(c0)
        + 0.65 * (c1 * c2)
        - 0.45 * (c3 * c3 - 1.0)
        + 0.30 * np.cos(c0 - c1)
    )
    noise = rng.normal(loc=0.0, scale=0.75, size=n_rows)
    latent = nonlinear_signal + noise
    median_latent = float(np.median(latent))

    # Inject deterministic missing values while keeping at least 75% non-missing per column.
    missing_mask = rng.random(size=(n_rows, n_features)) < valid_missing_rate
    for col in range(n_features):
        if int(np.sum(~missing_mask[:, col])) < max(16, n_rows // 2):
            missing_mask[: max(16, n_rows // 2), col] = False

    base_timestamp = 1_700_000_000
    observations: list[Observation] = []
    for i in range(n_rows):
        row_id = f"proc-{i:04d}"
        prob = 1.0 / (1.0 + math.exp(-float(latent[i] - median_latent)))
        # Blend parity and a uniform draw without guaranteeing per-role label support.
        u = float(rng.random())
        label = 1 if (0.65 * prob + 0.35 * ((i % 2) * 0.6 + 0.2)) > u else 0

        feats: list[float | None] = []
        for j in range(n_features):
            if bool(missing_mask[i, j]):
                feats.append(None)
            else:
                feats.append(round(float(raw_x[i, j]), 6))

        group_id = f"batch-{(i % n_groups):03d}" if include_groups else None
        bucket = (i * n_time_buckets) // n_rows
        timestamp = (base_timestamp + bucket * 600) if include_timestamps else None

        observations.append(
            Observation(
                row_id=row_id,
                features=feats,
                label=label,
                group_id=group_id,
                timestamp=timestamp,
            )
        )

    feature_names = DEFAULT_PROCESS_FEATURE_NAMES[:n_features]
    return TabularDataset(
        observations,
        feature_names=feature_names,
        budget=budget,
    )
