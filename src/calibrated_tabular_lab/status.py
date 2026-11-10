"""Status codes, typed domain errors, and caller admission budgets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Any, TypeGuard
import numpy as np

DOMAIN_MIN_ROWS: int = 160
DOMAIN_MAX_ROWS: int = 4096
DOMAIN_MIN_FEATURES: int = 1
DOMAIN_MAX_FEATURES: int = 16
DOMAIN_MIN_CLASS_SUPPORT: int = 8
DOMAIN_MAX_FEATURE_MAGNITUDE: float = 1_000_000.0
DOMAIN_MAX_ID_BYTES: int = 64
DOMAIN_MIN_SEED: int = 0
DOMAIN_MAX_SEED: int = 2_147_483_647
DOMAIN_MIN_EMBARGO: int = 0
DOMAIN_MAX_EMBARGO: int = 16
DOMAIN_MIN_BINS: int = 2
DOMAIN_MAX_BINS: int = 20
DOMAIN_MIN_COST: int = 0
DOMAIN_MAX_COST: int = 1000
DOMAIN_MAX_THRESHOLD_CANDIDATES: int = 4098
DOMAIN_MAX_OUTPUT_BYTES: int = 4_194_304
INT64_MIN: int = -(1 << 63)
INT64_MAX: int = (1 << 63) - 1


class LabStatus(StrEnum):
    """Distinct experiment and validation outcome statuses."""

    COMPLETE = "COMPLETE"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_SPLIT = "INVALID_SPLIT"
    FIT_FAILED = "FIT_FAILED"
    UNKNOWN_BUDGET = "UNKNOWN_BUDGET"
    INCOMPLETE = "INCOMPLETE"


class LabError(Exception):
    """Typed error carrying a non-COMPLETE LabStatus and diagnostic code."""

    def __init__(
        self,
        status: LabStatus,
        message: str,
        *,
        code: str = "lab_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(status, LabStatus) or status == LabStatus.COMPLETE:
            raise ValueError("LabError requires a non-COMPLETE LabStatus.")
        super().__init__(message)
        self.status: LabStatus = status
        self.message: str = message
        self.code: str = code
        self.details: Mapping[str, Any] = MappingProxyType(
            dict(details) if details is not None else {}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


def is_strict_int(value: object) -> TypeGuard[int]:
    """Return True iff value is a Python int and not a bool."""

    return isinstance(value, int) and not isinstance(value, bool)


def is_strict_real_number(value: object) -> bool:
    """Return True iff value is a real int/float (including NumPy scalars) and not bool."""

    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def to_finite_float64(value: object, *, field_name: str, status: LabStatus = LabStatus.INVALID_INPUT) -> float:
    """Convert a non-bool numeric value to a finite Python float, rejecting overflow."""

    if not is_strict_real_number(value):
        raise LabError(
            status,
            f"{field_name} must be a numeric int or float excluding bool.",
            code=f"invalid_{field_name}_type",
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise LabError(
            status,
            f"{field_name} overflowed float64 conversion.",
            code=f"{field_name}_overflow",
        ) from exc
    if not math.isfinite(result):
        raise LabError(
            status,
            f"{field_name} must be finite.",
            code=f"nonfinite_{field_name}",
        )
    return result


def validate_ascii_identifier(value: object, *, field_name: str) -> str:
    """Validate a printable non-whitespace ASCII identifier of 1..64 bytes (33..126)."""

    if not isinstance(value, str):
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"{field_name} must be a string.",
            code=f"invalid_{field_name}_type",
        )
    if not (1 <= len(value) <= DOMAIN_MAX_ID_BYTES):
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"{field_name} must have length 1 through {DOMAIN_MAX_ID_BYTES}.",
            code=f"invalid_{field_name}_length",
        )
    for ch in value:
        codepoint = ord(ch)
        if codepoint < 33 or codepoint > 126:
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"{field_name} must contain only printable non-whitespace ASCII characters (33..126).",
                code=f"invalid_{field_name}_chars",
            )
    return value


def validate_seed(seed: object) -> int:
    """Validate a strict integer seed in [0, 2147483647]."""

    if not is_strict_int(seed) or not (DOMAIN_MIN_SEED <= seed <= DOMAIN_MAX_SEED):
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"seed must be a strict integer in [{DOMAIN_MIN_SEED}, {DOMAIN_MAX_SEED}].",
            code="invalid_seed",
        )
    return seed


def validate_embargo(embargo: object) -> int:
    """Validate a strict integer embargo in [0, 16]."""

    if not is_strict_int(embargo) or not (DOMAIN_MIN_EMBARGO <= embargo <= DOMAIN_MAX_EMBARGO):
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"embargo must be a strict integer in [{DOMAIN_MIN_EMBARGO}, {DOMAIN_MAX_EMBARGO}].",
            code="invalid_embargo",
        )
    return embargo


def validate_n_bins(n_bins: object) -> int:
    """Validate a strict integer reliability bin count in [2, 20]."""

    if not is_strict_int(n_bins) or not (DOMAIN_MIN_BINS <= n_bins <= DOMAIN_MAX_BINS):
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"n_bins must be a strict integer in [{DOMAIN_MIN_BINS}, {DOMAIN_MAX_BINS}].",
            code="invalid_n_bins",
        )
    return n_bins


@dataclass(frozen=True, slots=True)
class AdmissionBudget:
    """Caller admission and output budget limits bounded by fixed domain maxima."""

    max_rows: int = DOMAIN_MAX_ROWS
    max_features: int = DOMAIN_MAX_FEATURES
    max_threshold_candidates: int = DOMAIN_MAX_THRESHOLD_CANDIDATES
    max_output_bytes: int = DOMAIN_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        _validate_budget_int(
            self.max_rows,
            field_name="max_rows",
            upper=DOMAIN_MAX_ROWS,
        )
        _validate_budget_int(
            self.max_features,
            field_name="max_features",
            upper=DOMAIN_MAX_FEATURES,
        )
        _validate_budget_int(
            self.max_threshold_candidates,
            field_name="max_threshold_candidates",
            upper=DOMAIN_MAX_THRESHOLD_CANDIDATES,
        )
        _validate_budget_int(
            self.max_output_bytes,
            field_name="max_output_bytes",
            upper=DOMAIN_MAX_OUTPUT_BYTES,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_rows": self.max_rows,
            "max_features": self.max_features,
            "max_threshold_candidates": self.max_threshold_candidates,
            "max_output_bytes": self.max_output_bytes,
        }


def _validate_budget_int(value: object, *, field_name: str, upper: int) -> int:
    if not is_strict_int(value) or value <= 0 or value > upper:
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"{field_name} must be a positive strict integer at most {upper}.",
            code=f"invalid_budget_{field_name}",
        )
    return value


def coerce_budget(budget: AdmissionBudget | None) -> AdmissionBudget:
    """Return validated AdmissionBudget or default domain-maximum budget."""

    if budget is None:
        return AdmissionBudget()
    if not isinstance(budget, AdmissionBudget):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "budget must be an AdmissionBudget instance or None.",
            code="invalid_budget_type",
        )
    return budget
