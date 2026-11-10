"""Altered public DTO regressions for the independent report-verification boundary.

These cases model future benign producer mistakes, not observed invalid results
from the normal experiment pipeline. No fit, metric or split implementation is
patched or duplicated.
"""
from dataclasses import replace
import csv
import io
import json
import math
import unittest

from calibrated_tabular_lab import (
    AdmissionBudget, ExperimentConfig, LabError, LabStatus,
    make_synthetic_process_dataset, run_experiment,
)
from calibrated_tabular_lab import cli


class ReportReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = make_synthetic_process_dataset(seed=42)
        cls.result = run_experiment(cls.dataset, config=ExperimentConfig(seed=42))

    def artifacts(self, result):
        return cli.make_artifacts(self.dataset, result, {"kind": "synthetic-demo"}, AdmissionBudget())

    def rejected(self, result):
        with self.assertRaises(LabError) as caught:
            self.artifacts(result)
        self.assertEqual(caught.exception.status, LabStatus.FIT_FAILED)
        self.assertEqual(caught.exception.code, "report_replay_failed")

    def test_policy_probabilities_must_align_before_zip(self):
        probabilities = self.result.policy_probabilities
        for changed in [probabilities + (probabilities[0],), probabilities[:-1], ()]:
            with self.subTest(length=len(changed)):
                self.rejected(replace(self.result, policy_probabilities=changed))

    def test_policy_probabilities_must_be_finite_and_in_unit_interval(self):
        probabilities = self.result.policy_probabilities
        for value in [math.nan, math.inf, -math.inf, -0.01, 1.01, True, "0.5", 10 ** 1000]:
            with self.subTest(value=repr(value)[:30]):
                self.rejected(replace(self.result, policy_probabilities=(value,) + probabilities[1:]))

    def test_candidate_indices_and_declared_counts_must_match_actual_records(self):
        sweep = self.result.policy_sweep
        changed = [replace(sweep, n_candidates=0), replace(sweep, n_observations=sweep.n_observations + 1),
                   replace(sweep, candidates=tuple(replace(item, candidate_index=99) for item in sweep.candidates)),
                   replace(sweep, candidates=(replace(sweep.candidates[0], candidate_index=False),) + sweep.candidates[1:])]
        for index, invalid in enumerate(changed):
            with self.subTest(case=index):
                self.rejected(replace(self.result, policy_sweep=invalid))
        evaluation = self.result.test_evaluations["prior"]
        evaluations = dict(self.result.test_evaluations)
        evaluations["prior"] = replace(evaluation, n_observations=evaluation.n_observations + 1)
        self.rejected(replace(self.result, test_evaluations=evaluations))

    def test_selected_record_must_equal_first_minimum_candidate(self):
        sweep = self.result.policy_sweep
        selected = sweep.selected
        changed = [replace(selected, cost=selected.cost + 1),
                   replace(selected, confusion=replace(selected.confusion, precision=0.123456)),
                   replace(selected, policy=sweep.candidates[0].policy)]
        for index, invalid in enumerate(changed):
            with self.subTest(case=index):
                self.rejected(replace(self.result, policy_sweep=replace(sweep, selected=invalid)))

    def test_every_exported_confusion_ratio_is_replayed(self):
        result = self.result
        for key in ["precision", "recall"]:
            for invalid in [0.123456, math.nan, math.inf, True, None]:
                with self.subTest(location="test", key=key, value=invalid):
                    evaluations = dict(result.test_evaluations)
                    evaluation = evaluations["prior"]
                    evaluations["prior"] = replace(evaluation, confusion_at_default_cutoff=replace(evaluation.confusion_at_default_cutoff, **{key: invalid}))
                    self.rejected(replace(result, test_evaluations=evaluations))
            with self.subTest(location="candidate", key=key):
                sweep = result.policy_sweep
                candidate = sweep.candidates[-1]
                altered = replace(candidate, confusion=replace(candidate.confusion, **{key: 0.123456}))
                self.rejected(replace(result, policy_sweep=replace(sweep, candidates=sweep.candidates[:-1] + (altered,))))
            with self.subTest(location="selected_test", key=key):
                selected = result.selected_policy_test_evaluation
                self.rejected(replace(result, selected_policy_test_evaluation=replace(selected, test_confusion=replace(selected.test_confusion, **{key: 0.123456}))))

    def test_zero_denominator_precision_requires_none(self):
        sweep = self.result.policy_sweep
        candidate = sweep.candidates[0]
        self.assertEqual(candidate.confusion.tp + candidate.confusion.fp, 0)
        self.assertIsNone(candidate.confusion.precision)
        for invalid in [0.0, math.nan]:
            with self.subTest(value=invalid):
                changed = replace(candidate, confusion=replace(candidate.confusion, precision=invalid))
                self.rejected(replace(self.result, policy_sweep=replace(sweep, candidates=(changed,) + sweep.candidates[1:])))

    def test_selected_test_cost_model_must_match_policy_selection(self):
        selected = self.result.selected_policy_test_evaluation
        altered_cost = replace(selected.cost_model, fp_cost=selected.cost_model.fp_cost + 1)
        self.rejected(replace(self.result, selected_policy_test_evaluation=replace(selected, cost_model=altered_cost)))

    def test_unmodified_result_keeps_aligned_json_and_one_csv_selection(self):
        artifacts = self.artifacts(self.result)
        envelope = json.loads(artifacts["report.json"])
        result = envelope["experiment"]
        self.assertEqual(len(envelope["policy_labels"]), len(result["policy_probabilities"]))
        candidates = list(csv.DictReader(io.StringIO(artifacts["thresholds.csv"].decode())))
        self.assertEqual(len(candidates), result["policy_sweep"]["n_candidates"])
        self.assertEqual(sum(row["selected"] == "1" for row in candidates), 1)
        self.assertEqual([int(row["candidate"]) for row in candidates], list(range(len(candidates))))


if __name__ == "__main__":
    unittest.main()
