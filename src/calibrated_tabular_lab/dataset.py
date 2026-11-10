"""Immutable tabular dataset snapshots and canonical identity fingerprinting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any
import numpy as np

from calibrated_tabular_lab.status import (
    DOMAIN_MAX_FEATURE_MAGNITUDE,
    DOMAIN_MAX_FEATURES,
    DOMAIN_MAX_ROWS,
    DOMAIN_MIN_FEATURES,
    DOMAIN_MIN_ROWS,
    INT64_MAX,
    INT64_MIN,
    AdmissionBudget,
    LabError,
    LabStatus,
    coerce_budget,
    is_strict_int,
    to_finite_float64,
    validate_ascii_identifier,
)

METADATA_FINGERPRINT_VERSION: str = "ctl-meta-v1"


def _validate_label(value: object) -> int:
    if not is_strict_int(value) or value not in (0, 1):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "label must be Python integer 0 or 1 excluding bool.",
            code="invalid_label",
        )
    return value


def _validate_timestamp(value: object) -> int | None:
    if value is None:
        return None
    if not is_strict_int(value) or not (INT64_MIN <= value <= INT64_MAX):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "timestamp must be a Python integer in signed 64-bit range excluding bool.",
            code="invalid_timestamp",
        )
    return value


def _validate_feature_sequence(features: object) -> tuple[float | None, ...]:
    if isinstance(features, (str, bytes, bytearray, Mapping)) or not isinstance(
        features, (Sequence, np.ndarray)
    ):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "features must be a sequence of numeric values or None.",
            code="invalid_features_container",
        )
    if isinstance(features, np.ndarray) and features.ndim != 1:
        raise LabError(
            LabStatus.INVALID_INPUT,
            "observation feature array must be one-dimensional.",
            code="invalid_features_ndim",
        )
    if not (DOMAIN_MIN_FEATURES <= len(features) <= DOMAIN_MAX_FEATURES):
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"feature count must be between {DOMAIN_MIN_FEATURES} and {DOMAIN_MAX_FEATURES}.",
            code="invalid_feature_count",
        )
    raw_items = list(features)
    validated: list[float | None] = []
    for idx, val in enumerate(raw_items):
        if val is None:
            validated.append(None)
            continue
        fval = to_finite_float64(val, field_name=f"feature[{idx}]")
        if abs(fval) > DOMAIN_MAX_FEATURE_MAGNITUDE:
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"feature[{idx}] magnitude exceeds {DOMAIN_MAX_FEATURE_MAGNITUDE}.",
                code="feature_magnitude_exceeded",
            )
        validated.append(fval)
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class Observation:
    """Owned immutable single-row observation."""

    row_id: str
    features: tuple[float | None, ...]
    label: int
    group_id: str | None = None
    timestamp: int | None = None

    def __init__(
        self,
        row_id: str,
        features: Sequence[float | int | None] | np.ndarray,
        label: int,
        group_id: str | None = None,
        timestamp: int | None = None,
    ) -> None:
        valid_row_id = validate_ascii_identifier(row_id, field_name="row_id")
        valid_features = _validate_feature_sequence(features)
        valid_label = _validate_label(label)
        valid_group_id = (
            validate_ascii_identifier(group_id, field_name="group_id")
            if group_id is not None
            else None
        )
        valid_timestamp = _validate_timestamp(timestamp)
        object.__setattr__(self, "row_id", valid_row_id)
        object.__setattr__(self, "features", valid_features)
        object.__setattr__(self, "label", valid_label)
        object.__setattr__(self, "group_id", valid_group_id)
        object.__setattr__(self, "timestamp", valid_timestamp)


def _compute_metadata_fingerprint(
    row_ids: tuple[str, ...],
    group_ids: tuple[str, ...] | None,
    timestamps: tuple[int, ...] | None,
) -> str:
    rows_payload: list[list[str | int | None]] = []
    for idx, row_id in enumerate(row_ids):
        grp = group_ids[idx] if group_ids is not None else None
        ts = timestamps[idx] if timestamps is not None else None
        rows_payload.append([row_id, grp, ts])
    canonical_json = json.dumps(
        {
            "version": METADATA_FINGERPRINT_VERSION,
            "has_groups": group_ids is not None,
            "has_timestamps": timestamps is not None,
            "rows": rows_payload,
        },
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"{METADATA_FINGERPRINT_VERSION}:{digest}"


class TabularDataset:
    """Owned immutable tabular dataset canonicalized by ascending ASCII row ID."""

    __slots__ = (
        "_observations",
        "_row_ids",
        "_labels",
        "_group_ids",
        "_timestamps",
        "_feature_names",
        "_feature_matrix",
        "_missing_mask",
        "_row_index",
        "_metadata_fingerprint",
    )

    def __init__(
        self,
        observations: Sequence[Observation | Mapping[str, Any]],
        *,
        feature_names: Sequence[str] | None = None,
        budget: AdmissionBudget | None = None,
    ) -> None:
        active_budget = coerce_budget(budget)
        if isinstance(observations, (str, bytes, bytearray, Mapping)) or not isinstance(
            observations, Sequence
        ):
            raise LabError(
                LabStatus.INVALID_INPUT,
                "observations must be a sequence of Observation objects or mappings.",
                code="invalid_observations_container",
            )
        n_rows = len(observations)
        if not (DOMAIN_MIN_ROWS <= n_rows <= DOMAIN_MAX_ROWS):
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"observation count must be between {DOMAIN_MIN_ROWS} and {DOMAIN_MAX_ROWS}; got {n_rows}.",
                code="invalid_row_count",
            )
        raw_list = list(observations)

        parsed_obs: list[Observation] = []
        seen_ids: set[str] = set()
        expected_n_features: int | None = None
        has_group_any = False
        has_group_none = False
        has_ts_any = False
        has_ts_none = False

        for item in raw_list:
            if isinstance(item, Observation):
                obs = item
            elif isinstance(item, Mapping):
                allowed_keys = {"row_id", "features", "label", "group_id", "timestamp"}
                extra_keys = set(item.keys()) - allowed_keys
                if extra_keys or "row_id" not in item or "features" not in item or "label" not in item:
                    raise LabError(
                        LabStatus.INVALID_INPUT,
                        "observation mapping must contain row_id, features, label and optional group_id, timestamp.",
                        code="invalid_observation_mapping",
                    )
                obs = Observation(
                    row_id=item["row_id"],
                    features=item["features"],
                    label=item["label"],
                    group_id=item.get("group_id"),
                    timestamp=item.get("timestamp"),
                )
            else:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    "each observation must be an Observation or mapping.",
                    code="invalid_observation_type",
                )

            if obs.row_id in seen_ids:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    f"duplicate row_id '{obs.row_id}' is not allowed.",
                    code="duplicate_row_id",
                )
            seen_ids.add(obs.row_id)

            d = len(obs.features)
            if expected_n_features is None:
                expected_n_features = d
            elif d != expected_n_features:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    "all observations must have identical rectangular feature dimensions.",
                    code="ragged_features",
                )

            if obs.group_id is None:
                has_group_none = True
            else:
                has_group_any = True

            if obs.timestamp is None:
                has_ts_none = True
            else:
                has_ts_any = True

            parsed_obs.append(obs)

        assert expected_n_features is not None
        if has_group_any and has_group_none:
            raise LabError(
                LabStatus.INVALID_INPUT,
                "group_id must be provided for all observations or omitted for all observations.",
                code="partial_group_ids",
            )
        if has_ts_any and has_ts_none:
            raise LabError(
                LabStatus.INVALID_INPUT,
                "timestamp must be provided for all observations or omitted for all observations.",
                code="partial_timestamps",
            )

        if feature_names is None:
            validated_names = tuple(f"f{i}" for i in range(expected_n_features))
        else:
            if isinstance(feature_names, (str, bytes, bytearray)) or not isinstance(
                feature_names, Sequence
            ):
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    "feature_names must be a sequence of ASCII identifiers.",
                    code="invalid_feature_names_type",
                )
            if len(feature_names) != expected_n_features:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    "feature_names must match feature count.",
                    code="invalid_feature_names",
                )
            names_list = [
                validate_ascii_identifier(name, field_name="feature_name")
                for name in feature_names
            ]
            if len(names_list) != expected_n_features or len(set(names_list)) != len(names_list):
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    "feature_names must match feature count and be unique.",
                    code="invalid_feature_names",
                )
            validated_names = tuple(names_list)

        # Check caller admission budgets after domain validation and before float64 matrix construction.
        if n_rows > active_budget.max_rows:
            raise LabError(
                LabStatus.UNKNOWN_BUDGET,
                f"observation count {n_rows} exceeds caller budget max_rows={active_budget.max_rows}.",
                code="max_rows_budget_exceeded",
                details={"n_rows": n_rows, "max_rows": active_budget.max_rows},
            )
        if expected_n_features > active_budget.max_features:
            raise LabError(
                LabStatus.UNKNOWN_BUDGET,
                f"feature count {expected_n_features} exceeds caller budget max_features={active_budget.max_features}.",
                code="max_features_budget_exceeded",
                details={
                    "n_features": expected_n_features,
                    "max_features": active_budget.max_features,
                },
            )

        # Canonicalize by ascending ASCII row_id.
        parsed_obs.sort(key=lambda o: o.row_id)
        canonical_obs = tuple(parsed_obs)
        row_ids = tuple(o.row_id for o in canonical_obs)
        labels = tuple(o.label for o in canonical_obs)
        group_ids = (
            tuple(o.group_id for o in canonical_obs if o.group_id is not None)
            if has_group_any
            else None
        )
        timestamps = (
            tuple(o.timestamp for o in canonical_obs if o.timestamp is not None)
            if has_ts_any
            else None
        )

        matrix = np.empty((n_rows, expected_n_features), dtype=np.float64)
        missing = np.zeros((n_rows, expected_n_features), dtype=np.bool_)
        for r_idx, obs in enumerate(canonical_obs):
            for c_idx, fval in enumerate(obs.features):
                if fval is None:
                    matrix[r_idx, c_idx] = np.nan
                    missing[r_idx, c_idx] = True
                else:
                    matrix[r_idx, c_idx] = fval
        matrix.setflags(write=False)
        missing.setflags(write=False)

        row_index = MappingProxyType({rid: i for i, rid in enumerate(row_ids)})
        fingerprint = _compute_metadata_fingerprint(row_ids, group_ids, timestamps)

        self._observations = canonical_obs
        self._row_ids = row_ids
        self._labels = labels
        self._group_ids = group_ids
        self._timestamps = timestamps
        self._feature_names = validated_names
        self._feature_matrix = matrix
        self._missing_mask = missing
        self._row_index = row_index
        self._metadata_fingerprint = fingerprint

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[Observation | Mapping[str, Any]],
        *,
        feature_names: Sequence[str] | None = None,
        budget: AdmissionBudget | None = None,
    ) -> TabularDataset:
        return cls(observations, feature_names=feature_names, budget=budget)

    @classmethod
    def from_arrays(
        cls,
        row_ids: Sequence[str],
        features: Sequence[Sequence[float | int | None]] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        *,
        group_ids: Sequence[str] | None = None,
        timestamps: Sequence[int] | np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        budget: AdmissionBudget | None = None,
    ) -> TabularDataset:
        """Admit ordinary sequence sizes and shape before retaining or normalizing values."""
        active_budget = coerce_budget(budget)

        def invalid(message: str, code: str) -> None:
            raise LabError(LabStatus.INVALID_INPUT, message, code=code)

        if isinstance(row_ids, (str, bytes, bytearray)) or not isinstance(row_ids, Sequence):
            invalid("row_ids must be a sequence of strings.", "invalid_row_ids_container")
        n = len(row_ids)
        if not DOMAIN_MIN_ROWS <= n <= DOMAIN_MAX_ROWS:
            invalid("observation count must be between 160 and 4096.", "invalid_row_count")
        if isinstance(features, (str, bytes, bytearray, Mapping)) or not isinstance(features, (Sequence, np.ndarray)):
            invalid("features must be a 2D sequence or ndarray.", "invalid_features_matrix")
        if isinstance(features, np.ndarray) and features.ndim != 2:
            invalid("features ndarray must be 2-dimensional.", "invalid_features_ndim")
        if isinstance(labels, (str, bytes, bytearray, Mapping)) or not isinstance(labels, (Sequence, np.ndarray)):
            invalid("labels must be a one-dimensional sequence.", "invalid_labels_container")
        if isinstance(labels, np.ndarray) and labels.ndim != 1:
            invalid("labels ndarray must be one-dimensional.", "invalid_labels_ndim")
        if len(features) != n or len(labels) != n:
            invalid("row_ids, features, and labels must have identical length.", "length_mismatch")

        # Inspect metadata sizes and dimensions without copying payloads.
        if group_ids is not None:
            if isinstance(group_ids, (str, bytes, bytearray)) or not isinstance(group_ids, Sequence):
                invalid("group_ids must be a sequence of strings.", "invalid_group_ids_container")
            if len(group_ids) != n:
                invalid("group_ids length must match row_ids length.", "group_ids_length_mismatch")
        if timestamps is not None:
            if isinstance(timestamps, (str, bytes, bytearray, Mapping)) or not isinstance(timestamps, (Sequence, np.ndarray)):
                invalid("timestamps must be a one-dimensional sequence.", "invalid_timestamps_container")
            if isinstance(timestamps, np.ndarray) and timestamps.ndim != 1:
                invalid("timestamps ndarray must be one-dimensional.", "invalid_timestamps_ndim")
            if len(timestamps) != n:
                invalid("timestamps length must match row_ids length.", "timestamps_length_mismatch")

        if n > active_budget.max_rows:
            raise LabError(LabStatus.UNKNOWN_BUDGET, "Rows exceed caller admission budget.", code="max_rows_budget_exceeded")

        # Walk admitted references only: no list growth or scalar conversion.
        width: int | None = None
        for index in range(n):
            row = features[index]
            if isinstance(row, (str, bytes, bytearray, Mapping)) or not isinstance(row, (Sequence, np.ndarray)):
                invalid("Each feature row must be a numeric sequence.", "invalid_features_container")
            if isinstance(row, np.ndarray) and row.ndim != 1:
                invalid("Feature rows must be one-dimensional.", "invalid_features_ndim")
            row_width = len(row)
            if not DOMAIN_MIN_FEATURES <= row_width <= DOMAIN_MAX_FEATURES:
                invalid("Feature count must be between 1 and 16.", "invalid_feature_count")
            if width is None:
                width = row_width
            elif row_width != width:
                invalid("All feature rows must have equal width.", "ragged_features")
        assert width is not None
        if feature_names is not None:
            if isinstance(feature_names, (str, bytes, bytearray)) or not isinstance(feature_names, Sequence):
                invalid("feature_names must be a sequence of strings.", "invalid_feature_names_type")
            if len(feature_names) != width:
                invalid("feature_names must match feature count.", "invalid_feature_names")
        if width > active_budget.max_features:
            raise LabError(LabStatus.UNKNOWN_BUDGET, "Features exceed caller admission budget.", code="max_features_budget_exceeded")

        # Size, shape and caller admission checks passed; normalization may begin.
        row_id_list = list(row_ids)
        feat_list = list(features)
        label_list = [int(v) if isinstance(v, np.integer) and not isinstance(v, (bool, np.bool_)) else v for v in labels]
        grp_list = [None] * n if group_ids is None else list(group_ids)
        ts_list = [None] * n if timestamps is None else [
            int(v) if isinstance(v, np.integer) and not isinstance(v, (bool, np.bool_)) else v for v in timestamps
        ]
        observations = [Observation(row_id_list[i], feat_list[i], label_list[i], grp_list[i], ts_list[i]) for i in range(n)]
        return cls(observations, feature_names=feature_names, budget=active_budget)

    @property
    def n_rows(self) -> int:
        return len(self._observations)

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def observations(self) -> tuple[Observation, ...]:
        return self._observations

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self._row_ids

    @property
    def labels(self) -> tuple[int, ...]:
        return self._labels

    @property
    def group_ids(self) -> tuple[str, ...] | None:
        return self._group_ids

    @property
    def timestamps(self) -> tuple[int, ...] | None:
        return self._timestamps

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def metadata_fingerprint(self) -> str:
        return self._metadata_fingerprint

    @property
    def feature_matrix(self) -> np.ndarray:
        # A readonly view still exposes its owning ndarray through .base.
        # A detached copy cannot write through to the owned dataset snapshot.
        detached = np.array(self._feature_matrix, dtype=np.float64, copy=True)
        detached.setflags(write=False)
        return detached

    @property
    def missing_mask(self) -> np.ndarray:
        detached = np.array(self._missing_mask, dtype=np.bool_, copy=True)
        detached.setflags(write=False)
        return detached

    def indices_for_row_ids(self, row_ids: Sequence[str]) -> np.ndarray:
        """Return canonical integer row indices for the given sequence of row_ids."""

        try:
            indices = [self._row_index[rid] for rid in row_ids]
        except KeyError as exc:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                f"unknown row_id '{exc.args[0]}' in split role.",
                code="unknown_row_id_in_plan",
            ) from exc
        return np.asarray(indices, dtype=np.intp)

    def slice_features(self, row_ids: Sequence[str]) -> np.ndarray:
        """Return an owned read-only float64 feature matrix for the requested row_ids."""

        idx = self.indices_for_row_ids(row_ids)
        sub = np.array(self._feature_matrix[idx], dtype=np.float64, copy=True)
        sub.setflags(write=False)
        return sub

    def slice_labels(self, row_ids: Sequence[str]) -> tuple[int, ...]:
        """Return canonical binary integer labels for the requested row_ids."""

        idx = self.indices_for_row_ids(row_ids)
        return tuple(self._labels[int(i)] for i in idx)

    def with_replaced_labels(
        self,
        label_updates: Mapping[str, int],
        *,
        budget: AdmissionBudget | None = None,
    ) -> TabularDataset:
        """Return a new TabularDataset with updated labels for specified row_ids."""

        if not isinstance(label_updates, Mapping):
            raise LabError(
                LabStatus.INVALID_INPUT,
                "label_updates must be a mapping from row_id to 0/1 label.",
                code="invalid_label_updates",
            )
        unknown = set(label_updates.keys()) - set(self._row_ids)
        if unknown:
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"unknown row_ids in label_updates: {sorted(unknown)[:5]}.",
                code="unknown_row_id_in_label_updates",
            )
        new_obs = [
            Observation(
                row_id=obs.row_id,
                features=obs.features,
                label=label_updates.get(obs.row_id, obs.label),
                group_id=obs.group_id,
                timestamp=obs.timestamp,
            )
            for obs in self._observations
        ]
        return TabularDataset(new_obs, feature_names=self._feature_names, budget=budget)
