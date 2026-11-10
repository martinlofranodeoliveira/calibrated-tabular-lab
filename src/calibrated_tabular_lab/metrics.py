"""Probability validation and exact evaluation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from calibrated_tabular_lab.status import (
    DOMAIN_MAX_ROWS,
    LabError,
    LabStatus,
    is_strict_int,
    to_finite_float64,
    validate_n_bins,
)

FLOAT64_EPSILON: float = float(np.finfo(np.float64).eps)
LOG_LOSS_LOWER_BOUND: float = FLOAT64_EPSILON
LOG_LOSS_UPPER_BOUND: float = 1.0 - FLOAT64_EPSILON


def validate_binary_labels(y_true: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    """Validate a non-empty 1D sequence of strict binary integer labels 0 or 1."""

    if isinstance(y_true, (str, bytes, bytearray, Mapping)) or not isinstance(
        y_true, (Sequence, np.ndarray)
    ):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "y_true must be a 1D sequence of 0/1 integers.",
            code="invalid_y_true_container",
        )
    if isinstance(y_true, np.ndarray) and y_true.ndim != 1:
        raise LabError(
            LabStatus.INVALID_INPUT,
            "y_true array must be one-dimensional.",
            code="invalid_y_true_ndim",
        )
    if len(y_true) == 0 or len(y_true) > DOMAIN_MAX_ROWS:
        raise LabError(
            LabStatus.INVALID_INPUT,
            f"y_true length must be between 1 and {DOMAIN_MAX_ROWS}.",
            code="invalid_y_true_length",
        )
    items = list(y_true)
    out: list[int] = []
    for idx, val in enumerate(items):
        if isinstance(val, np.integer) and not isinstance(val, (bool, np.bool_)):
            ival = int(val)
        elif is_strict_int(val):
            ival = val
        else:
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"y_true[{idx}] must be integer 0 or 1 excluding bool.",
                code="invalid_label_type",
            )
        if ival not in (0, 1):
            raise LabError(
                LabStatus.INVALID_INPUT,
                f"y_true[{idx}] must be 0 or 1; got {ival}.",
                code="invalid_label_value",
            )
        out.append(ival)
    return tuple(out)


def validate_probability_vector(
    y_prob: Sequence[float | int] | np.ndarray,
    *,
    expected_length: int | None = None,
    error_status: LabStatus = LabStatus.INVALID_INPUT,
) -> tuple[float, ...]:
    """Validate and copy a probability vector into an owned finite float64 tuple in [0, 1]."""

    if isinstance(y_prob, (str, bytes, bytearray, Mapping)) or not isinstance(
        y_prob, (Sequence, np.ndarray)
    ):
        raise LabError(
            error_status,
            "y_prob must be a 1D sequence of finite probabilities in [0, 1].",
            code="invalid_y_prob_container",
        )
    if isinstance(y_prob, np.ndarray) and y_prob.ndim != 1:
        raise LabError(
            error_status,
            "y_prob array must be one-dimensional.",
            code="invalid_y_prob_ndim",
        )
    if len(y_prob) == 0 or len(y_prob) > DOMAIN_MAX_ROWS:
        raise LabError(
            error_status,
            f"y_prob length must be between 1 and {DOMAIN_MAX_ROWS}.",
            code="invalid_y_prob_length",
        )
    items = list(y_prob)
    if expected_length is not None and len(items) != expected_length:
        raise LabError(
            error_status,
            f"y_prob length ({len(items)}) does not match y_true length ({expected_length}).",
            code="probability_length_mismatch",
        )
    probs: list[float] = []
    for idx, val in enumerate(items):
        pval = to_finite_float64(val, field_name=f"y_prob[{idx}]", status=error_status)
        if pval < 0.0 or pval > 1.0:
            raise LabError(
                error_status,
                f"y_prob[{idx}]={pval!r} is outside [0, 1].",
                code="probability_out_of_bounds",
            )
        probs.append(float(np.float64(pval)))
    return tuple(probs)


@dataclass(frozen=True, slots=True)
class ClippedLogLossResult:
    """Binary log-loss value under float64 epsilon clipping and clipped observation count."""

    log_loss: float
    clipped_count: int
    epsilon: float = FLOAT64_EPSILON
    upper_bound: float = LOG_LOSS_UPPER_BOUND

    def to_dict(self) -> dict[str, float | int]:
        return {
            "log_loss": self.log_loss,
            "clipped_count": self.clipped_count,
            "epsilon": self.epsilon,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True, slots=True)
class AucPairAudit:
    """Exact positive-negative pair counts and ROC AUC value."""

    roc_auc: float | None
    total_pairs: int
    concordant_pairs: int
    tied_pairs: int
    discordant_pairs: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "roc_auc": self.roc_auc,
            "total_pairs": self.total_pairs,
            "concordant_pairs": self.concordant_pairs,
            "tied_pairs": self.tied_pairs,
            "discordant_pairs": self.discordant_pairs,
        }


@dataclass(frozen=True, slots=True)
class ConfusionSummary:
    """Confusion matrix counts and precision/recall with None for undefined ratios."""

    tn: int
    fp: int
    fn: int
    tp: int
    precision: float | None
    recall: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """Single uniform reliability bin under right-closed convention."""

    bin_index: int
    lower_edge: float
    upper_edge: float
    left_closed: bool
    right_closed: bool
    count: int
    positive_count: int
    mean_probability: float | None
    observed_fraction: float | None

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "bin_index": self.bin_index,
            "lower_edge": self.lower_edge,
            "upper_edge": self.upper_edge,
            "left_closed": self.left_closed,
            "right_closed": self.right_closed,
            "count": self.count,
            "positive_count": self.positive_count,
            "mean_probability": self.mean_probability,
            "observed_fraction": self.observed_fraction,
        }


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Complete evaluation metrics for a single model on a split role."""

    model_name: str
    n_observations: int
    brier_loss: float
    log_loss: float
    log_loss_clipped_count: int
    roc_auc: float | None
    auc_pairs: AucPairAudit
    default_cutoff: float
    confusion_at_default_cutoff: ConfusionSummary
    reliability_bins: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_observations": self.n_observations,
            "brier_loss": self.brier_loss,
            "log_loss": self.log_loss,
            "log_loss_clipped_count": self.log_loss_clipped_count,
            "roc_auc": self.roc_auc,
            "auc_pairs": self.auc_pairs.to_dict(),
            "default_cutoff": self.default_cutoff,
            "confusion_at_default_cutoff": self.confusion_at_default_cutoff.to_dict(),
            "reliability_bins": [b.to_dict() for b in self.reliability_bins],
        }


def brier_score(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
) -> float:
    """Compute unclipped binary Brier loss math.fsum((p - y)^2) / n."""

    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))
    terms = [(p - float(y)) * (p - float(y)) for y, p in zip(labels, probs)]
    return math.fsum(terms) / len(labels)


def clipped_log_loss(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
) -> ClippedLogLossResult:
    """Compute binary log loss with float64 epsilon clipping and exact clip count."""

    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))
    terms: list[float] = []
    clipped_count = 0
    for y, p in zip(labels, probs):
        q = min(LOG_LOSS_UPPER_BOUND, max(LOG_LOSS_LOWER_BOUND, p))
        if q != p:
            clipped_count += 1
        if y == 1:
            terms.append(-math.log(q))
        else:
            terms.append(-math.log1p(-q))
    return ClippedLogLossResult(
        log_loss=math.fsum(terms) / len(labels),
        clipped_count=clipped_count,
    )


def roc_auc_score_exact(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
) -> AucPairAudit:
    """Compute exact pair-based ROC AUC using float64 equality for ties."""

    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return AucPairAudit(
            roc_auc=None,
            total_pairs=0,
            concordant_pairs=0,
            tied_pairs=0,
            discordant_pairs=0,
        )

    score_groups: dict[float, list[int]] = {}
    for y, p in zip(labels, probs):
        counts = score_groups.get(p)
        if counts is None:
            score_groups[p] = [1 - y, y]
        else:
            counts[0] += 1 - y
            counts[1] += y

    seen_neg = 0
    concordant = 0
    tied = 0
    for score in sorted(score_groups.keys()):
        neg_c, pos_c = score_groups[score]
        concordant += pos_c * seen_neg
        tied += pos_c * neg_c
        seen_neg += neg_c

    total_pairs = n_pos * n_neg
    discordant = total_pairs - concordant - tied
    auc = (2 * concordant + tied) / (2 * total_pairs)
    return AucPairAudit(
        roc_auc=auc,
        total_pairs=total_pairs,
        concordant_pairs=concordant,
        tied_pairs=tied,
        discordant_pairs=discordant,
    )


def exhaustive_auc_pairs(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
) -> AucPairAudit:
    """Exhaustively enumerate all positive-negative pairs for independent AUC verification."""

    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))
    pos_scores = [p for y, p in zip(labels, probs) if y == 1]
    neg_scores = [p for y, p in zip(labels, probs) if y == 0]
    if not pos_scores or not neg_scores:
        return AucPairAudit(
            roc_auc=None,
            total_pairs=0,
            concordant_pairs=0,
            tied_pairs=0,
            discordant_pairs=0,
        )
    concordant = 0
    tied = 0
    discordant = 0
    for p_pos in pos_scores:
        for p_neg in neg_scores:
            if p_pos > p_neg:
                concordant += 1
            elif p_pos == p_neg:
                tied += 1
            else:
                discordant += 1
    total_pairs = len(pos_scores) * len(neg_scores)
    auc = (2 * concordant + tied) / (2 * total_pairs)
    return AucPairAudit(
        roc_auc=auc,
        total_pairs=total_pairs,
        concordant_pairs=concordant,
        tied_pairs=tied,
        discordant_pairs=discordant,
    )


def confusion_from_Counts(tn: int, fp: int, fn: int, tp: int) -> ConfusionSummary:
    """Build ConfusionSummary with explicit None for zero-denominator precision/recall."""

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    return ConfusionSummary(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        precision=precision,
        recall=recall,
    )


def confusion_at_cutoff(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
    *,
    cutoff: float = 0.5,
) -> ConfusionSummary:
    """Compute confusion counts and precision/recall for decision p >= cutoff."""

    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))
    cval = to_finite_float64(cutoff, field_name="cutoff")
    if cval < 0.0 or cval > 1.0:
        raise LabError(
            LabStatus.INVALID_INPUT,
            "cutoff must be in [0, 1].",
            code="cutoff_out_of_bounds",
        )
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for y, p in zip(labels, probs):
        pred = 1 if p >= cval else 0
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    return confusion_from_Counts(tn, fp, fn, tp)


def reliability_bins(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    """Compute uniform right-closed reliability bins matching contract searchsorted rule."""

    valid_bins = validate_n_bins(n_bins)
    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))

    edges = np.linspace(0.0, 1.0, valid_bins + 1, dtype=np.float64)
    prob_arr = np.asarray(probs, dtype=np.float64)
    bin_indices = np.searchsorted(edges[1:-1], prob_arr, side="left")

    per_bin_probs: list[list[float]] = [[] for _ in range(valid_bins)]
    per_bin_pos: list[int] = [0] * valid_bins
    for idx, y, p in zip(bin_indices, labels, probs):
        b = int(idx)
        per_bin_probs[b].append(p)
        per_bin_pos[b] += y

    out: list[ReliabilityBin] = []
    for b in range(valid_bins):
        b_probs = per_bin_probs[b]
        count = len(b_probs)
        pos_count = per_bin_pos[b]
        mean_prob = (math.fsum(b_probs) / count) if count > 0 else None
        obs_frac = (pos_count / count) if count > 0 else None
        out.append(
            ReliabilityBin(
                bin_index=b,
                lower_edge=float(edges[b]),
                upper_edge=float(edges[b + 1]),
                left_closed=(b == 0),
                right_closed=True,
                count=count,
                positive_count=pos_count,
                mean_probability=mean_prob,
                observed_fraction=obs_frac,
            )
        )
    return tuple(out)


def evaluate_predictions(
    model_name: str,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float | int] | np.ndarray,
    *,
    n_bins: int = 10,
    default_cutoff: float = 0.5,
) -> ModelEvaluation:
    """Compute complete ModelEvaluation and verify AUC pair totals independently."""

    labels = validate_binary_labels(y_true)
    probs = validate_probability_vector(y_prob, expected_length=len(labels))
    brier = brier_score(labels, probs)
    ll = clipped_log_loss(labels, probs)
    auc_audit = roc_auc_score_exact(labels, probs)
    conf = confusion_at_cutoff(labels, probs, cutoff=default_cutoff)
    bins = reliability_bins(labels, probs, n_bins=n_bins)

    if sum(b.count for b in bins) != len(labels) or sum(b.positive_count for b in bins) != sum(labels):
        raise LabError(
            LabStatus.FIT_FAILED,
            "reliability bin support totals failed independent consistency check.",
            code="reliability_bin_audit_failed",
        )
    if conf.tn + conf.fp + conf.fn + conf.tp != len(labels):
        raise LabError(
            LabStatus.FIT_FAILED,
            "confusion matrix total failed independent consistency check.",
            code="confusion_audit_failed",
        )

    return ModelEvaluation(
        model_name=model_name,
        n_observations=len(labels),
        brier_loss=brier,
        log_loss=ll.log_loss,
        log_loss_clipped_count=ll.clipped_count,
        roc_auc=auc_audit.roc_auc,
        auc_pairs=auc_audit,
        default_cutoff=default_cutoff,
        confusion_at_default_cutoff=conf,
        reliability_bins=bins,
    )
