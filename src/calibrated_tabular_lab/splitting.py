"""Deterministic four-role split planning, freezing, and independent audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from calibrated_tabular_lab.dataset import TabularDataset
from calibrated_tabular_lab.status import (
    DOMAIN_MIN_CLASS_SUPPORT,
    LabError,
    LabStatus,
    validate_ascii_identifier,
    validate_embargo,
    validate_seed,
)

ROLE_ORDER: tuple[str, str, str, str] = ("train", "calibration", "policy", "test")
TARGET_ROLE_FRACTIONS: Mapping[str, float] = MappingProxyType(
    {"train": 0.50, "calibration": 0.20, "policy": 0.15, "test": 0.15}
)
_SEED_MODULUS: int = 1 << 31


class SplitStrategy(StrEnum):
    """Supported deterministic split strategies."""

    STRATIFIED = "stratified"
    GROUP = "group"
    CHRONOLOGICAL = "chronological"


def _coerce_strategy(strategy: SplitStrategy | str) -> SplitStrategy:
    if isinstance(strategy, SplitStrategy):
        return strategy
    if isinstance(strategy, str):
        try:
            return SplitStrategy(strategy)
        except ValueError as exc:
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"unsupported split strategy '{strategy}'.",
                code="invalid_split_strategy",
            ) from exc
    raise LabError(
        LabStatus.INVALID_INPUT,
        "strategy must be a SplitStrategy or string.",
        code="invalid_split_strategy_type",
    )


def compute_role_counts(n_units: int) -> tuple[int, int, int, int]:
    """Compute (train, calibration, policy, test) unit counts using floor/remainder rule."""

    if n_units <= 0:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            f"allocatable unit count must be positive; got {n_units}.",
            code="nonpositive_allocatable_units",
        )
    n_train = n_units // 2
    n_calibration = n_units // 5
    n_policy = (3 * n_units) // 20
    n_test = n_units - n_train - n_calibration - n_policy
    if min(n_train, n_calibration, n_policy, n_test) <= 0:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            f"insufficient allocation units ({n_units}) caused a zero-sized role: "
            f"{(n_train, n_calibration, n_policy, n_test)}.",
            code="zero_sized_role",
        )
    return n_train, n_calibration, n_policy, n_test


@dataclass(frozen=True, slots=True)
class RoleClassSupport:
    """Class support counts within a single split role."""

    total: int
    negative_count: int
    positive_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
        }


@dataclass(frozen=True, slots=True)
class FrozenSplitPlan:
    """Immutable frozen split plan storing canonical row IDs and metadata identity."""

    strategy: SplitStrategy
    seed: int
    embargo: int
    train_row_ids: tuple[str, ...]
    calibration_row_ids: tuple[str, ...]
    policy_row_ids: tuple[str, ...]
    test_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    require_group_disjoint: bool
    require_chronological_order: bool
    metadata_fingerprint: str
    historical_class_support: Mapping[str, RoleClassSupport] | None = None

    def __post_init__(self) -> None:
        strat = _coerce_strategy(self.strategy)
        object.__setattr__(self, "strategy", strat)
        validate_seed(self.seed)
        validate_embargo(self.embargo)
        if not isinstance(self.require_group_disjoint, bool) or not isinstance(
            self.require_chronological_order, bool
        ):
            raise LabError(
                LabStatus.INVALID_INPUT,
                "require_group_disjoint and require_chronological_order must be booleans.",
                code="invalid_assertion_flag_type",
            )
        if not isinstance(self.metadata_fingerprint, str) or not self.metadata_fingerprint:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                "metadata_fingerprint must be a non-empty string.",
                code="invalid_metadata_fingerprint",
            )
        for attr_name in (
            "train_row_ids",
            "calibration_row_ids",
            "policy_row_ids",
            "test_row_ids",
            "excluded_row_ids",
        ):
            raw_val = getattr(self, attr_name)
            if isinstance(raw_val, (str, bytes, bytearray)) or not isinstance(
                raw_val, (tuple, list)
            ):
                raise LabError(
                    LabStatus.INVALID_SPLIT,
                    f"{attr_name} must be a sequence of row_id strings.",
                    code="invalid_plan_row_ids",
                )
            tup = tuple(
                validate_ascii_identifier(rid, field_name=attr_name) for rid in raw_val
            )
            object.__setattr__(self, attr_name, tup)
        if self.historical_class_support is not None:
            object.__setattr__(
                self,
                "historical_class_support",
                MappingProxyType(dict(self.historical_class_support)),
            )

    def role_row_ids(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                "train": self.train_row_ids,
                "calibration": self.calibration_row_ids,
                "policy": self.policy_row_ids,
                "test": self.test_row_ids,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "seed": self.seed,
            "embargo": self.embargo,
            "train_row_ids": list(self.train_row_ids),
            "calibration_row_ids": list(self.calibration_row_ids),
            "policy_row_ids": list(self.policy_row_ids),
            "test_row_ids": list(self.test_row_ids),
            "excluded_row_ids": list(self.excluded_row_ids),
            "require_group_disjoint": self.require_group_disjoint,
            "require_chronological_order": self.require_chronological_order,
            "metadata_fingerprint": self.metadata_fingerprint,
            "historical_class_support": (
                {k: v.to_dict() for k, v in self.historical_class_support.items()}
                if self.historical_class_support is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SplitAuditReport:
    """Independent verification audit of a FrozenSplitPlan against a TabularDataset."""

    strategy: SplitStrategy
    seed: int
    embargo: int
    total_rows: int
    allocated_rows: int
    excluded_rows: int
    role_counts: Mapping[str, int]
    target_fractions: Mapping[str, float]
    achieved_row_fractions: Mapping[str, float]
    class_support: Mapping[str, RoleClassSupport]
    row_disjoint: bool
    group_disjoint: bool | None
    chronological_ordered: bool | None
    role_group_counts: Mapping[str, int] | None
    role_timestamp_bounds: Mapping[str, tuple[int, int]] | None
    metadata_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "seed": self.seed,
            "embargo": self.embargo,
            "total_rows": self.total_rows,
            "allocated_rows": self.allocated_rows,
            "excluded_rows": self.excluded_rows,
            "role_counts": dict(self.role_counts),
            "target_fractions": dict(self.target_fractions),
            "achieved_row_fractions": dict(self.achieved_row_fractions),
            "class_support": {k: v.to_dict() for k, v in self.class_support.items()},
            "row_disjoint": self.row_disjoint,
            "group_disjoint": self.group_disjoint,
            "chronological_ordered": self.chronological_ordered,
            "role_group_counts": (
                dict(self.role_group_counts) if self.role_group_counts is not None else None
            ),
            "role_timestamp_bounds": (
                {k: list(v) for k, v in self.role_timestamp_bounds.items()}
                if self.role_timestamp_bounds is not None
                else None
            ),
            "metadata_fingerprint": self.metadata_fingerprint,
        }


def create_split_plan(
    dataset: TabularDataset,
    *,
    strategy: SplitStrategy | str = SplitStrategy.STRATIFIED,
    seed: int = 0,
    embargo: int = 0,
    require_group_disjoint: bool = False,
    require_chronological_order: bool = False,
) -> FrozenSplitPlan:
    """Create and independently validate a deterministic FrozenSplitPlan."""

    if not isinstance(dataset, TabularDataset):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "dataset must be a TabularDataset instance.",
            code="invalid_dataset_type",
        )
    strat = _coerce_strategy(strategy)
    valid_seed = validate_seed(seed)
    valid_embargo = validate_embargo(embargo)
    if not isinstance(require_group_disjoint, bool) or not isinstance(
        require_chronological_order, bool
    ):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "require_group_disjoint and require_chronological_order must be booleans.",
            code="invalid_assertion_flag_type",
        )
    if strat != SplitStrategy.CHRONOLOGICAL and valid_embargo != 0:
        raise LabError(
            LabStatus.INVALID_INPUT,
            "nonzero embargo is only supported for chronological split strategy.",
            code="embargo_not_supported_for_strategy",
        )

    if strat == SplitStrategy.STRATIFIED:
        train_ids, cal_ids, pol_ids, test_ids = _plan_stratified(dataset, valid_seed)
        excl_ids: tuple[str, ...] = ()
    elif strat == SplitStrategy.GROUP:
        train_ids, cal_ids, pol_ids, test_ids = _plan_group(dataset, valid_seed)
        excl_ids = ()
    else:
        train_ids, cal_ids, pol_ids, test_ids, excl_ids = _plan_chronological(
            dataset, valid_Embargo=valid_embargo
        )

    initial_support = _compute_current_support(
        dataset,
        {
            "train": train_ids,
            "calibration": cal_ids,
            "policy": pol_ids,
            "test": test_ids,
        },
    )
    plan = FrozenSplitPlan(
        strategy=strat,
        seed=valid_seed,
        embargo=valid_embargo,
        train_row_ids=train_ids,
        calibration_row_ids=cal_ids,
        policy_row_ids=pol_ids,
        test_row_ids=test_ids,
        excluded_row_ids=excl_ids,
        require_group_disjoint=require_group_disjoint or (strat == SplitStrategy.GROUP),
        require_chronological_order=require_chronological_order
        or (strat == SplitStrategy.CHRONOLOGICAL),
        metadata_fingerprint=dataset.metadata_fingerprint,
        historical_class_support=initial_support,
    )
    validate_split_plan(dataset, plan)
    return plan


def _plan_stratified(
    dataset: TabularDataset, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    n = dataset.n_rows
    n_train, n_cal, n_pol, n_test = compute_role_counts(n)
    y_all = np.asarray(dataset.labels, dtype=np.int64)
    row_ids = dataset.row_ids

    try:
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_train,
            test_size=n - n_train,
            random_state=seed,
        )
        train_idx, rem1_idx = next(sss1.split(np.zeros(n), y_all))
        rem1_sorted = np.sort(rem1_idx)

        seed2 = (seed + 1) % _SEED_MODULUS
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_cal,
            test_size=n_pol + n_test,
            random_state=seed2,
        )
        cal_sub_idx, rem2_sub_idx = next(
            sss2.split(np.zeros(len(rem1_sorted)), y_all[rem1_sorted])
        )
        cal_idx = rem1_sorted[cal_sub_idx]
        rem2_sorted = np.sort(rem1_sorted[rem2_sub_idx])

        seed3 = (seed + 2) % _SEED_MODULUS
        sss3 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_pol,
            test_size=n_test,
            random_state=seed3,
        )
        pol_sub_idx, test_sub_idx = next(
            sss3.split(np.zeros(len(rem2_sorted)), y_all[rem2_sorted])
        )
        pol_idx = rem2_sorted[pol_sub_idx]
        test_idx = rem2_sorted[test_sub_idx]
    except ValueError as exc:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            f"stratified split failed due to insufficient class support: {exc}",
            code="stratified_split_failed",
        ) from exc

    return (
        tuple(row_ids[int(i)] for i in np.sort(train_idx)),
        tuple(row_ids[int(i)] for i in np.sort(cal_idx)),
        tuple(row_ids[int(i)] for i in np.sort(pol_idx)),
        tuple(row_ids[int(i)] for i in np.sort(test_idx)),
    )


def _plan_group(
    dataset: TabularDataset, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if dataset.group_ids is None:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "group_ids are mandatory for group split strategy.",
            code="missing_group_ids",
        )
    unique_groups = sorted(set(dataset.group_ids))
    n_groups = len(unique_groups)
    g_train, g_cal, g_pol, _ = compute_role_counts(n_groups)

    rng = np.random.default_rng(seed)
    permuted = [str(g) for g in rng.permutation(unique_groups)]
    train_groups = set(permuted[:g_train])
    cal_groups = set(permuted[g_train : g_train + g_cal])
    pol_groups = set(permuted[g_train + g_cal : g_train + g_cal + g_pol])
    test_groups = set(permuted[g_train + g_cal + g_pol :])

    train_ids: list[str] = []
    cal_ids: list[str] = []
    pol_ids: list[str] = []
    test_ids: list[str] = []
    for rid, gid in zip(dataset.row_ids, dataset.group_ids):
        if gid in train_groups:
            train_ids.append(rid)
        elif gid in cal_groups:
            cal_ids.append(rid)
        elif gid in pol_groups:
            pol_ids.append(rid)
        elif gid in test_groups:
            test_ids.append(rid)
        else:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                f"unassigned group_id '{gid}'.",
                code="unassigned_group",
            )
    return tuple(train_ids), tuple(cal_ids), tuple(pol_ids), tuple(test_ids)


def _plan_chronological(
    dataset: TabularDataset, *, valid_Embargo: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if dataset.timestamps is None:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "timestamps are mandatory for chronological split strategy.",
            code="missing_timestamps",
        )
    buckets = sorted(set(dataset.timestamps))
    allocatable = len(buckets) - 3 * valid_Embargo
    if allocatable <= 0:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            f"nonpositive allocatable timestamp bucket count ({allocatable}) for {len(buckets)} buckets and embargo {valid_Embargo}.",
            code="insufficient_timestamp_buckets",
        )
    b_train, b_cal, b_pol, b_test = compute_role_counts(allocatable)

    cursor = 0
    train_b = set(buckets[cursor : cursor + b_train])
    cursor += b_train
    emb1_b = set(buckets[cursor : cursor + valid_Embargo])
    cursor += valid_Embargo
    cal_b = set(buckets[cursor : cursor + b_cal])
    cursor += b_cal
    emb2_b = set(buckets[cursor : cursor + valid_Embargo])
    cursor += valid_Embargo
    pol_b = set(buckets[cursor : cursor + b_pol])
    cursor += b_pol
    emb3_b = set(buckets[cursor : cursor + valid_Embargo])
    cursor += valid_Embargo
    test_b = set(buckets[cursor : cursor + b_test])
    excluded_b = emb1_b | emb2_b | emb3_b

    train_ids: list[str] = []
    cal_ids: list[str] = []
    pol_ids: list[str] = []
    test_ids: list[str] = []
    excl_ids: list[str] = []
    for rid, ts in zip(dataset.row_ids, dataset.timestamps):
        if ts in train_b:
            train_ids.append(rid)
        elif ts in cal_b:
            cal_ids.append(rid)
        elif ts in pol_b:
            pol_ids.append(rid)
        elif ts in test_b:
            test_ids.append(rid)
        elif ts in excluded_b:
            excl_ids.append(rid)
        else:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                f"unassigned timestamp bucket {ts}.",
                code="unassigned_timestamp_bucket",
            )
    return (
        tuple(train_ids),
        tuple(cal_ids),
        tuple(pol_ids),
        tuple(test_ids),
        tuple(excl_ids),
    )


def _compute_current_support(
    dataset: TabularDataset, role_map: Mapping[str, tuple[str, ...]]
) -> Mapping[str, RoleClassSupport]:
    support: dict[str, RoleClassSupport] = {}
    for role in ROLE_ORDER:
        rids = role_map[role]
        labels = dataset.slice_labels(rids)
        pos = sum(1 for y in labels if y == 1)
        neg = len(labels) - pos
        support[role] = RoleClassSupport(
            total=len(labels),
            negative_count=neg,
            positive_count=pos,
        )
    return MappingProxyType(support)


def validate_split_plan(dataset: TabularDataset, plan: FrozenSplitPlan) -> SplitAuditReport:
    """Independently validate a FrozenSplitPlan against a TabularDataset."""

    if not isinstance(dataset, TabularDataset):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "dataset must be a TabularDataset instance.",
            code="invalid_dataset_type",
        )
    if not isinstance(plan, FrozenSplitPlan):
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "plan must be a FrozenSplitPlan instance.",
            code="invalid_plan_type",
        )

    if plan.metadata_fingerprint != dataset.metadata_fingerprint:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "plan metadata_fingerprint does not match dataset metadata identity.",
            code="metadata_fingerprint_mismatch",
        )

    if plan.strategy != SplitStrategy.CHRONOLOGICAL and (
        plan.embargo != 0 or len(plan.excluded_row_ids) > 0
    ):
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "non-chronological plan cannot contain embargo or excluded rows.",
            code="unexpected_exclusions_in_plan",
        )

    role_map = plan.role_row_ids()
    all_segments: list[tuple[str, tuple[str, ...]]] = [
        ("train", plan.train_row_ids),
        ("calibration", plan.calibration_row_ids),
        ("policy", plan.policy_row_ids),
        ("test", plan.test_row_ids),
        ("excluded", plan.excluded_row_ids),
    ]

    seen_total: set[str] = set()
    total_listed = 0
    for seg_name, rids in all_segments:
        if seg_name != "excluded" and len(rids) == 0:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                f"role '{seg_name}' is empty.",
                code="empty_role",
            )
        # Verify canonical ascending order and absence of duplicates inside segment.
        for idx in range(len(rids) - 1):
            if rids[idx] >= rids[idx + 1]:
                raise LabError(
                    LabStatus.INVALID_SPLIT,
                    f"segment '{seg_name}' row_ids are not strictly ascending canonical ASCII.",
                    code="non_canonical_plan_order",
                )
        overlap = seen_total.intersection(rids)
        if overlap:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                f"segment '{seg_name}' overlaps with another role or exclusion.",
                code="overlapping_roles",
            )
        seen_total.update(rids)
        total_listed += len(rids)

    dataset_id_set = set(dataset.row_ids)
    if total_listed != dataset.n_rows or seen_total != dataset_id_set:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "plan roles and exclusions do not form an exact partition of dataset row_ids.",
            code="incomplete_or_unknown_plan_coverage",
        )

    current_support = _compute_current_support(dataset, role_map)
    for role in ROLE_ORDER:
        sup = current_support[role]
        if (
            sup.negative_count < DOMAIN_MIN_CLASS_SUPPORT
            or sup.positive_count < DOMAIN_MIN_CLASS_SUPPORT
        ):
            raise LabError(
                LabStatus.INVALID_SPLIT,
                f"role '{role}' fails minimum class support ({DOMAIN_MIN_CLASS_SUPPORT}): "
                f"neg={sup.negative_count}, pos={sup.positive_count}.",
                code="insufficient_class_support",
                details={"role": role, **sup.to_dict()},
            )

    # Audit group disjointness.
    need_group_disjoint = plan.require_group_disjoint or (plan.strategy == SplitStrategy.GROUP)
    group_disjoint: bool | None = None
    role_group_counts: Mapping[str, int] | None = None
    if dataset.group_ids is None:
        if need_group_disjoint:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                "group isolation was requested or required by strategy, but dataset has no group_ids.",
                code="missing_group_ids_for_assertion",
            )
    else:
        row_to_group = dict(zip(dataset.row_ids, dataset.group_ids))
        role_groups: dict[str, set[str]] = {
            role: {row_to_group[rid] for rid in role_map[role]} for role in ROLE_ORDER
        }
        excl_groups: set[str] = {row_to_group[rid] for rid in plan.excluded_row_ids}
        is_disjoint = True
        for i, r1 in enumerate(ROLE_ORDER):
            for r2 in ROLE_ORDER[i + 1 :]:
                if role_groups[r1].intersection(role_groups[r2]):
                    is_disjoint = False
            if excl_groups and role_groups[r1].intersection(excl_groups):
                is_disjoint = False
        group_disjoint = is_disjoint
        role_group_counts = MappingProxyType({r: len(role_groups[r]) for r in ROLE_ORDER})
        if need_group_disjoint and not is_disjoint:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                "group disjointness assertion failed across roles.",
                code="group_disjointness_violated",
            )

    # Audit chronological ordering and timestamp bucket integrity.
    need_chrono = plan.require_chronological_order or (
        plan.strategy == SplitStrategy.CHRONOLOGICAL
    )
    chrono_ok: bool | None = None
    role_ts_bounds: Mapping[str, tuple[int, int]] | None = None
    if dataset.timestamps is None:
        if need_chrono:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                "chronological ordering was requested or required by strategy, but dataset has no timestamps.",
                code="missing_timestamps_for_assertion",
            )
    else:
        row_to_ts = dict(zip(dataset.row_ids, dataset.timestamps))
        role_buckets: dict[str, set[int]] = {
            role: {row_to_ts[rid] for rid in role_map[role]} for role in ROLE_ORDER
        }
        excl_buckets: set[int] = {row_to_ts[rid] for rid in plan.excluded_row_ids}
        bounds: dict[str, tuple[int, int]] = {
            role: (min(role_buckets[role]), max(role_buckets[role])) for role in ROLE_ORDER
        }
        role_ts_bounds = MappingProxyType(bounds)

        # Check strict ordering max(earlier) < min(later) and unsplit timestamp buckets.
        strict_order = (
            bounds["train"][1] < bounds["calibration"][0]
            and bounds["calibration"][1] < bounds["policy"][0]
            and bounds["policy"][1] < bounds["test"][0]
        )
        bucket_unsplit = True
        for i, r1 in enumerate(ROLE_ORDER):
            for r2 in ROLE_ORDER[i + 1 :]:
                if role_buckets[r1].intersection(role_buckets[r2]):
                    bucket_unsplit = False
            if role_buckets[r1].intersection(excl_buckets):
                bucket_unsplit = False

        chrono_structure_valid = strict_order and bucket_unsplit
        if plan.strategy == SplitStrategy.CHRONOLOGICAL and chrono_structure_valid:
            chrono_structure_valid = _verify_chronological_bucket_layout(
                all_buckets=sorted(set(dataset.timestamps)),
                role_buckets=role_buckets,
                excl_buckets=excl_buckets,
                embargo=plan.embargo,
            )
        chrono_ok = chrono_structure_valid
        if need_chrono and not chrono_structure_valid:
            raise LabError(
                LabStatus.INVALID_SPLIT,
                "chronological ordering or timestamp bucket embargo integrity failed.",
                code="chronological_order_violated",
            )

    allocated_rows = sum(len(role_map[r]) for r in ROLE_ORDER)
    role_counts = MappingProxyType({r: len(role_map[r]) for r in ROLE_ORDER})
    achieved_fractions = MappingProxyType(
        {r: len(role_map[r]) / dataset.n_rows for r in ROLE_ORDER}
    )

    return SplitAuditReport(
        strategy=plan.strategy,
        seed=plan.seed,
        embargo=plan.embargo,
        total_rows=dataset.n_rows,
        allocated_rows=allocated_rows,
        excluded_rows=len(plan.excluded_row_ids),
        role_counts=role_counts,
        target_fractions=TARGET_ROLE_FRACTIONS,
        achieved_row_fractions=achieved_fractions,
        class_support=current_support,
        row_disjoint=True,
        group_disjoint=group_disjoint,
        chronological_ordered=chrono_ok,
        role_group_counts=role_group_counts,
        role_timestamp_bounds=role_ts_bounds,
        metadata_fingerprint=dataset.metadata_fingerprint,
    )


def _verify_chronological_bucket_layout(
    *,
    all_buckets: list[int],
    role_buckets: Mapping[str, set[int]],
    excl_buckets: set[int],
    embargo: int,
) -> bool:
    allocatable = len(all_buckets) - 3 * embargo
    if allocatable <= 0:
        return False
    try:
        b_train, b_cal, b_pol, b_test = compute_role_counts(allocatable)
    except LabError:
        return False
    cursor = 0
    exp_train = set(all_buckets[cursor : cursor + b_train])
    cursor += b_train
    exp_emb1 = set(all_buckets[cursor : cursor + embargo])
    cursor += embargo
    exp_cal = set(all_buckets[cursor : cursor + b_cal])
    cursor += b_cal
    exp_emb2 = set(all_buckets[cursor : cursor + embargo])
    cursor += embargo
    exp_pol = set(all_buckets[cursor : cursor + b_pol])
    cursor += b_pol
    exp_emb3 = set(all_buckets[cursor : cursor + embargo])
    cursor += embargo
    exp_test = set(all_buckets[cursor : cursor + b_test])
    exp_excl = exp_emb1 | exp_emb2 | exp_emb3
    return (
        role_buckets["train"] == exp_train
        and role_buckets["calibration"] == exp_cal
        and role_buckets["policy"] == exp_pol
        and role_buckets["test"] == exp_test
        and excl_buckets == exp_excl
    )
