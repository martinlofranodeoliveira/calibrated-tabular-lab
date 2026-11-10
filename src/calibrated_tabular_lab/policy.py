"""Sorted threshold-cost sweep and independent direct verification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
import numpy as np

from calibrated_tabular_lab.metrics import (
    ConfusionSummary,
    confusion_from_Counts,
    validate_binary_labels,
    validate_probability_vector,
)
from calibrated_tabular_lab.status import (
    DOMAIN_MAX_COST,
    DOMAIN_MAX_THRESHOLD_CANDIDATES,
    DOMAIN_MIN_COST,
    AdmissionBudget,
    LabError,
    LabStatus,
    coerce_budget,
    is_strict_int,
    to_finite_float64,
)


class PolicyKind(StrEnum):
    """Decision policy kinds without infinite or NaN sentinels."""

    ALL_NEGATIVE = "all_negative"
    CUTOFF = "cutoff"
    ALL_POSITIVE = "all_positive"


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """Immutable decision policy representation."""

    kind: PolicyKind
    cutoff: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str) and not isinstance(self.kind, PolicyKind):
            try:
                object.__setattr__(self, "kind", PolicyKind(self.kind))
            except ValueError as exc:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    f"invalid policy kind '{self.kind}'.",
                    code="invalid_policy_kind",
                ) from exc
        elif not isinstance(self.kind, PolicyKind):
            raise LabError(
                LabStatus.INVALID_INPUT,
                "kind must be a PolicyKind.",
                code="invalid_policy_kind_type",
            )

        if self.kind == PolicyKind.CUTOFF:
            if self.cutoff is None:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    "cutoff policy requires a finite cutoff in [0, 1].",
                    code="missing_policy_cutoff",
                )
            cval = to_finite_float64(self.cutoff, field_name="cutoff")
            if cval < 0.0 or cval > 1.0:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    "policy cutoff must be within [0, 1].",
                    code="policy_cutoff_out_of_bounds",
                )
            object.__setattr__(self, "cutoff", float(np.float64(cval)))
        else:
            if self.cutoff is not None:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    f"{self.kind} policy must have cutoff=None.",
                    code="unexpected_policy_cutoff",
                )

    def predict_one(self, probability: float) -> int:
        if self.kind == PolicyKind.ALL_NEGATIVE:
            return 0
        if self.kind == PolicyKind.ALL_POSITIVE:
            return 1
        assert self.cutoff is not None
        return 1 if probability >= self.cutoff else 0

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "kind": str(self.kind),
            "cutoff": self.cutoff,
        }


@dataclass(frozen=True, slots=True)
class CostModel:
    """Integer penalty weights for false positives and false negatives."""

    fp_cost: int
    fn_cost: int

    def __post_init__(self) -> None:
        if not is_strict_int(self.fp_cost) or not (
            DOMAIN_MIN_COST <= self.fp_cost <= DOMAIN_MAX_COST
        ):
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"fp_cost must be a strict integer in [{DOMAIN_MIN_COST}, {DOMAIN_MAX_COST}].",
                code="invalid_fp_cost",
            )
        if not is_strict_int(self.fn_cost) or not (
            DOMAIN_MIN_COST <= self.fn_cost <= DOMAIN_MAX_COST
        ):
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"fn_cost must be a strict integer in [{DOMAIN_MIN_COST}, {DOMAIN_MAX_COST}].",
                code="invalid_fn_cost",
            )
        if self.fp_cost + self.fn_cost <= 0:
            raise LabError(
                LabStatus.INVALID_INPUT,
                "fp_cost + fn_cost must be strictly positive.",
                code="nonpositive_cost_sum",
            )

    def compute_cost(self, *, fp: int, fn: int) -> int:
        return int(self.fp_cost * fp + self.fn_cost * fn)

    def to_dict(self) -> dict[str, int]:
        return {"fp_cost": self.fp_cost, "fn_cost": self.fn_cost}


@dataclass(frozen=True, slots=True)
class ThresholdCandidateResult:
    """Evaluation of a single candidate policy on a probability vector."""

    candidate_index: int
    policy: ThresholdPolicy
    cost: int
    confusion: ConfusionSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "policy": self.policy.to_dict(),
            "cost": self.cost,
            "confusion": self.confusion.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ThresholdSweepResult:
    """Complete threshold sweep output with independently verified selected policy."""

    cost_model: CostModel
    n_observations: int
    n_candidates: int
    selected: ThresholdCandidateResult
    candidates: tuple[ThresholdCandidateResult, ...]
    independently_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_model": self.cost_model.to_dict(),
            "n_observations": self.n_observations,
            "n_candidates": self.n_candidates,
            "selected": self.selected.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "independently_verified": self.independently_verified,
        }


def evaluate_policy_direct(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
    policy: ThresholdPolicy,
    *,
    fp_cost: int,
    fn_cost: int,
    candidate_index: int = 0,
) -> ThresholdCandidateResult:
    """Directly evaluate a ThresholdPolicy by an independent pass over observations."""

    if not isinstance(policy, ThresholdPolicy):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "policy must be a ThresholdPolicy instance.",
            code="invalid_policy_type",
        )
    cost_model = CostModel(fp_cost=fp_cost, fn_cost=fn_cost)
    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))

    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for y, p in zip(labels, probs):
        pred = policy.predict_one(p)
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    conf = confusion_from_Counts(tn, fp, fn, tp)
    total_cost = cost_model.compute_cost(fp=fp, fn=fn)
    return ThresholdCandidateResult(
        candidate_index=candidate_index,
        policy=policy,
        cost=total_cost,
        confusion=conf,
    )


def sweep_threshold_policies(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
    *,
    fp_cost: int,
    fn_cost: int,
    budget: AdmissionBudget | None = None,
) -> ThresholdSweepResult:
    """Run the original sorted probability threshold-cost sweep with independent check."""

    active_budget = coerce_budget(budget)
    cost_model = CostModel(fp_cost=fp_cost, fn_cost=fn_cost)
    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))

    # Group observations by exact float64 probability.
    score_groups: dict[float, list[int]] = {}
    total_pos = 0
    total_neg = 0
    for y, p in zip(labels, probs):
        if y == 1:
            total_pos += 1
        else:
            total_neg += 1
        grp = score_groups.get(p)
        if grp is None:
            score_groups[p] = [1 - y, y]
        else:
            grp[0] += 1 - y
            grp[1] += y

    distinct_desc = sorted(score_groups.keys(), reverse=True)
    prospective_candidates = len(distinct_desc) + 2
    if prospective_candidates > DOMAIN_MAX_THRESHOLD_CANDIDATES:
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"candidate count {prospective_candidates} exceeds domain maximum {DOMAIN_MAX_THRESHOLD_CANDIDATES}.",
            code="domain_threshold_candidates_exceeded",
        )
    if prospective_candidates > active_budget.max_threshold_candidates:
        raise LabError(
            LabStatus.UNKNOWN_BUDGET,
            f"prospective threshold candidate count {prospective_candidates} exceeds caller budget "
            f"max_threshold_candidates={active_budget.max_threshold_candidates}.",
            code="max_threshold_candidates_budget_exceeded",
            details={
                "prospective_candidates": prospective_candidates,
                "max_threshold_candidates": active_budget.max_threshold_candidates,
            },
        )

    candidates: list[ThresholdCandidateResult] = []

    # Candidate 0: all_negative (predict 0 for every observation)
    tn = total_neg
    fp = 0
    fn = total_pos
    tp = 0
    all_neg_cand = ThresholdCandidateResult(
        candidate_index=0,
        policy=ThresholdPolicy(kind=PolicyKind.ALL_NEGATIVE, cutoff=None),
        cost=cost_model.compute_cost(fp=fp, fn=fn),
        confusion=confusion_from_Counts(tn, fp, fn, tp),
    )
    candidates.append(all_neg_cand)
    best_candidate = all_neg_cand

    # Candidates 1..k: cutoff at each distinct probability descending, moving tied groups atomically.
    for idx, score in enumerate(distinct_desc, start=1):
        neg_in_grp, pos_in_grp = score_groups[score]
        tp += pos_in_grp
        fn -= pos_in_grp
        fp += neg_in_grp
        tn -= neg_in_grp
        cand = ThresholdCandidateResult(
            candidate_index=idx,
            policy=ThresholdPolicy(kind=PolicyKind.CUTOFF, cutoff=score),
            cost=cost_model.compute_cost(fp=fp, fn=fn),
            confusion=confusion_from_Counts(tn, fp, fn, tp),
        )
        candidates.append(cand)
        if cand.cost < best_candidate.cost:
            best_candidate = cand

    # Candidate k+1: all_positive (predict 1 for every observation)
    all_pos_cand = ThresholdCandidateResult(
        candidate_index=len(distinct_desc) + 1,
        policy=ThresholdPolicy(kind=PolicyKind.ALL_POSITIVE, cutoff=None),
        cost=cost_model.compute_cost(fp=total_neg, fn=0),
        confusion=confusion_from_Counts(0, total_neg, 0, total_pos),
    )
    candidates.append(all_pos_cand)
    if all_pos_cand.cost < best_candidate.cost:
        best_candidate = all_pos_cand

    # Independently recompute the chosen policy confusion and cost via a direct pass.
    direct_check = evaluate_policy_direct(
        labels,
        probs,
        best_candidate.policy,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        candidate_index=best_candidate.candidate_index,
    )
    if direct_check != best_candidate:
        raise LabError(
            LabStatus.FIT_FAILED,
            "independent threshold policy verification failed to match sweep selection.",
            code="threshold_sweep_verification_failed",
        )

    return ThresholdSweepResult(
        cost_model=cost_model,
        n_observations=len(labels),
        n_candidates=len(candidates),
        selected=best_candidate,
        candidates=tuple(candidates),
        independently_verified=True,
    )


def exhaustive_policy_search(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
    *,
    fp_cost: int,
    fn_cost: int,
) -> ThresholdSweepResult:
    """Exhaustively evaluate every candidate policy by direct passes for golden testing."""

    cost_model = CostModel(fp_cost=fp_cost, fn_cost=fn_cost)
    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))

    distinct_desc = sorted(set(probs), reverse=True)
    policies: list[ThresholdPolicy] = [
        ThresholdPolicy(kind=PolicyKind.ALL_NEGATIVE, cutoff=None),
        *[
            ThresholdPolicy(kind=PolicyKind.CUTOFF, cutoff=score)
            for score in distinct_desc
        ],
        ThresholdPolicy(kind=PolicyKind.ALL_POSITIVE, cutoff=None),
    ]
    candidates: list[ThresholdCandidateResult] = []
    best_cand: ThresholdCandidateResult | None = None
    for idx, pol in enumerate(policies):
        cand = evaluate_policy_direct(
            labels,
            probs,
            pol,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            candidate_index=idx,
        )
        candidates.append(cand)
        if best_cand is None or cand.cost < best_cand.cost:
            best_cand = cand
    assert best_cand is not None
    return ThresholdSweepResult(
        cost_model=cost_model,
        n_observations=len(labels),
        n_candidates=len(candidates),
        selected=best_cand,
        candidates=tuple(candidates),
        independently_verified=True,
    )
