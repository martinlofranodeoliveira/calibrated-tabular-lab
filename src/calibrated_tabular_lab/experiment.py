"""End-to-end four-role tabular calibration experiment execution and serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any

from calibrated_tabular_lab.dataset import TabularDataset
from calibrated_tabular_lab.metrics import (
    ConfusionSummary,
    ModelEvaluation,
    evaluate_predictions,
)
from calibrated_tabular_lab.modeling import (
    MODEL_NAMES,
    CalibrationProvenance,
    TrainingEvidence,
    calibrate_forest_model,
    fit_base_models,
    predict_role_probabilities,
)
from calibrated_tabular_lab.policy import (
    CostModel,
    ThresholdPolicy,
    ThresholdSweepResult,
    evaluate_policy_direct,
    sweep_threshold_policies,
)
from calibrated_tabular_lab.splitting import (
    FrozenSplitPlan,
    SplitAuditReport,
    SplitStrategy,
    create_split_plan,
    validate_split_plan,
)
from calibrated_tabular_lab.status import (
    AdmissionBudget,
    LabError,
    LabStatus,
    coerce_budget,
    validate_embargo,
    validate_n_bins,
    validate_seed,
)

_MIN_SERIALIZED_REPORT_BYTES_FLOOR: int = 512


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a four-role tabular calibration experiment."""

    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    seed: int = 0
    embargo: int = 0
    require_group_disjoint: bool = False
    require_chronological_order: bool = False
    fp_cost: int = 1
    fn_cost: int = 1
    n_bins: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str) and not isinstance(self.strategy, SplitStrategy):
            try:
                object.__setattr__(self, "strategy", SplitStrategy(self.strategy))
            except ValueError as exc:
                raise LabError(
                    LabStatus.INVALID_INPUT,
                    f"invalid split strategy '{self.strategy}'.",
                    code="invalid_split_strategy",
                ) from exc
        elif not isinstance(self.strategy, SplitStrategy):
            raise LabError(
                LabStatus.INVALID_INPUT,
                "strategy must be a SplitStrategy or string.",
                code="invalid_split_strategy_type",
            )
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
        if self.strategy != SplitStrategy.CHRONOLOGICAL and self.embargo != 0:
            raise LabError(
                LabStatus.INVALID_INPUT,
                "nonzero embargo is only valid for chronological strategy.",
                code="embargo_not_supported_for_strategy",
            )
        CostModel(fp_cost=self.fp_cost, fn_cost=self.fn_cost)
        validate_n_bins(self.n_bins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "seed": self.seed,
            "embargo": self.embargo,
            "require_group_disjoint": self.require_group_disjoint,
            "require_chronological_order": self.require_chronological_order,
            "fp_cost": self.fp_cost,
            "fn_cost": self.fn_cost,
            "n_bins": self.n_bins,
        }


@dataclass(frozen=True, slots=True)
class SelectedPolicyTestEvaluation:
    """Final-test evaluation of the policy selected on the policy-threshold role."""

    policy: ThresholdPolicy
    cost_model: CostModel
    test_cost: int
    test_confusion: ConfusionSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "cost_model": self.cost_model.to_dict(),
            "test_cost": self.test_cost,
            "test_confusion": self.test_confusion.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Immutable bounded evidence and metrics from a completed experiment."""

    status: LabStatus
    config: ExperimentConfig
    budget: AdmissionBudget
    plan: FrozenSplitPlan
    split_audit: SplitAuditReport
    training_evidence: TrainingEvidence
    calibration_provenance: CalibrationProvenance
    policy_probabilities: tuple[float, ...]
    policy_sweep: ThresholdSweepResult
    test_probabilities: Mapping[str, tuple[float, ...]]
    test_evaluations: Mapping[str, ModelEvaluation]
    selected_policy_test_evaluation: SelectedPolicyTestEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "config": self.config.to_dict(),
            "budget": self.budget.to_dict(),
            "plan": self.plan.to_dict(),
            "split_audit": self.split_audit.to_dict(),
            "training_evidence": self.training_evidence.to_dict(),
            "calibration_provenance": self.calibration_provenance.to_dict(),
            "policy_probabilities": list(self.policy_probabilities),
            "policy_sweep": self.policy_sweep.to_dict(),
            "test_probabilities": {
                k: list(v) for k, v in self.test_probabilities.items()
            },
            "test_evaluations": {
                k: v.to_dict() for k, v in self.test_evaluations.items()
            },
            "selected_policy_test_evaluation": self.selected_policy_test_evaluation.to_dict(),
        }


def serialize_experiment_result(
    result: ExperimentResult,
    *,
    budget: AdmissionBudget | None = None,
) -> str:
    """Serialize ExperimentResult to strict finite JSON and enforce max_output_bytes."""

    if not isinstance(result, ExperimentResult):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "result must be an ExperimentResult instance.",
            code="invalid_result_type",
        )
    active_budget = coerce_budget(budget) if budget is not None else result.budget
    try:
        payload = json.dumps(
            result.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except ValueError as exc:
        raise LabError(
            LabStatus.FIT_FAILED,
            f"strict finite JSON serialization failed: {exc}",
            code="nonfinite_serialization",
        ) from exc

    byte_len = len(payload.encode("utf-8"))
    if byte_len > active_budget.max_output_bytes:
        raise LabError(
            LabStatus.UNKNOWN_BUDGET,
            f"serialized experiment output ({byte_len} bytes) exceeds caller budget "
            f"max_output_bytes={active_budget.max_output_bytes}.",
            code="max_output_bytes_budget_exceeded",
            details={
                "output_bytes": byte_len,
                "max_output_bytes": active_budget.max_output_bytes,
            },
        )
    return payload


def run_experiment(
    dataset: TabularDataset,
    *,
    config: ExperimentConfig | None = None,
    plan: FrozenSplitPlan | None = None,
    budget: AdmissionBudget | None = None,
) -> ExperimentResult:
    """Execute the four-role training, calibration, policy selection, and test evaluation."""

    if not isinstance(dataset, TabularDataset):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "dataset must be a TabularDataset instance.",
            code="invalid_dataset_type",
        )
    if config is None and isinstance(plan, FrozenSplitPlan):
        active_config = ExperimentConfig(
            strategy=plan.strategy,
            seed=plan.seed,
            embargo=plan.embargo,
            require_group_disjoint=plan.require_group_disjoint,
            require_chronological_order=plan.require_chronological_order,
        )
    else:
        active_config = config if config is not None else ExperimentConfig()
    if not isinstance(active_config, ExperimentConfig):
        raise LabError(
            LabStatus.INVALID_INPUT,
            "config must be an ExperimentConfig instance or None.",
            code="invalid_config_type",
        )
    active_budget = coerce_budget(budget)

    # Check dataset against caller admission budget before work.
    if dataset.n_rows > active_budget.max_rows:
        raise LabError(
            LabStatus.UNKNOWN_BUDGET,
            f"dataset row count {dataset.n_rows} exceeds caller budget max_rows={active_budget.max_rows}.",
            code="max_rows_budget_exceeded",
            details={"n_rows": dataset.n_rows, "max_rows": active_budget.max_rows},
        )
    if dataset.n_features > active_budget.max_features:
        raise LabError(
            LabStatus.UNKNOWN_BUDGET,
            f"dataset feature count {dataset.n_features} exceeds caller budget max_features={active_budget.max_features}.",
            code="max_features_budget_exceeded",
            details={
                "n_features": dataset.n_features,
                "max_features": active_budget.max_features,
            },
        )
    if active_budget.max_output_bytes < _MIN_SERIALIZED_REPORT_BYTES_FLOOR:
        raise LabError(
            LabStatus.UNKNOWN_BUDGET,
            f"max_output_bytes={active_budget.max_output_bytes} is below minimum report floor {_MIN_SERIALIZED_REPORT_BYTES_FLOOR}.",
            code="max_output_bytes_budget_exceeded",
        )

    # Create or independently validate supplied FrozenSplitPlan.
    if plan is None:
        active_plan = create_split_plan(
            dataset,
            strategy=active_config.strategy,
            seed=active_config.seed,
            embargo=active_config.embargo,
            require_group_disjoint=active_config.require_group_disjoint,
            require_chronological_order=active_config.require_chronological_order,
        )
    else:
        active_plan = plan
    split_audit = validate_split_plan(dataset, active_plan)
    if (
        active_config.strategy != active_plan.strategy
        or active_config.seed != active_plan.seed
        or active_config.embargo != active_plan.embargo
    ):
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "explicit configuration strategy, seed and embargo must match the frozen plan.",
            code="config_plan_mismatch",
        )
    if active_config.require_group_disjoint and split_audit.group_disjoint is not True:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "configured group isolation is not satisfied by the frozen plan.",
            code="config_group_disjointness_violated",
        )
    if active_config.require_chronological_order and split_audit.chronological_ordered is not True:
        raise LabError(
            LabStatus.INVALID_SPLIT,
            "configured chronological ordering is not satisfied by the frozen plan.",
            code="config_chronological_order_violated",
        )

    # Extract role feature matrices without metadata columns.
    X_train = dataset.slice_features(active_plan.train_row_ids)
    y_train = dataset.slice_labels(active_plan.train_row_ids)
    X_cal = dataset.slice_features(active_plan.calibration_row_ids)
    y_cal = dataset.slice_labels(active_plan.calibration_row_ids)
    X_policy = dataset.slice_features(active_plan.policy_row_ids)
    y_policy = dataset.slice_labels(active_plan.policy_row_ids)
    X_test = dataset.slice_features(active_plan.test_row_ids)

    # Fit base models exclusively on train.
    base_models = fit_base_models(X_train, y_train, seed=active_plan.seed)

    # Calibrate forest model exclusively on calibration using FrozenEstimator.
    fitted_models = calibrate_forest_model(
        base_models,
        X_cal,
        y_cal,
        X_train=X_train,
    )

    # Predict policy-role probabilities and select threshold policy on policy role only.
    policy_role_probs = predict_role_probabilities(
        fitted_models, X_policy, role_name="policy"
    )
    cal_policy_probs = policy_role_probs["calibrated_forest"]
    policy_sweep = sweep_threshold_policies(
        y_policy,
        cal_policy_probs,
        fp_cost=active_config.fp_cost,
        fn_cost=active_config.fn_cost,
        budget=active_budget,
    )

    # Produce final test probabilities for all 4 models without consulting y_test.
    test_probs = predict_role_probabilities(fitted_models, X_test, role_name="test")

    # Consult y_test exclusively for final scoring.
    y_test = dataset.slice_labels(active_plan.test_row_ids)
    test_evals: dict[str, ModelEvaluation] = {}
    for model_name in MODEL_NAMES:
        test_evals[model_name] = evaluate_predictions(
            model_name,
            y_test,
            test_probs[model_name],
            n_bins=active_config.n_bins,
            default_cutoff=0.5,
        )

    selected_test_cand = evaluate_policy_direct(
        y_test,
        test_probs["calibrated_forest"],
        policy_sweep.selected.policy,
        fp_cost=active_config.fp_cost,
        fn_cost=active_config.fn_cost,
        candidate_index=policy_sweep.selected.candidate_index,
    )
    selected_test_eval = SelectedPolicyTestEvaluation(
        policy=policy_sweep.selected.policy,
        cost_model=policy_sweep.cost_model,
        test_cost=selected_test_cand.cost,
        test_confusion=selected_test_cand.confusion,
    )

    result = ExperimentResult(
        status=LabStatus.COMPLETE,
        config=active_config,
        budget=active_budget,
        plan=active_plan,
        split_audit=split_audit,
        training_evidence=fitted_models.training_evidence,
        calibration_provenance=fitted_models.calibration_provenance,
        policy_probabilities=cal_policy_probs,
        policy_sweep=policy_sweep,
        test_probabilities=test_probs,
        test_evaluations=MappingProxyType(test_evals),
        selected_policy_test_evaluation=selected_test_eval,
    )
    # Enforce serialization output budget before returning a complete result.
    serialize_experiment_result(result, budget=active_budget)
    return result
