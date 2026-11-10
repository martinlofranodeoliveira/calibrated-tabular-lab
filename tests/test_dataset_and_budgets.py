"""Tests for immutable dataset snapshots, canonical IDs, and domain vs caller budgets."""

from __future__ import annotations

import unittest
import numpy as np

from calibrated_tabular_lab import (
    AdmissionBudget,
    LabError,
    LabStatus,
    Observation,
    TabularDataset,
    make_synthetic_process_dataset,
)


class DatasetAndBudgetTests(unittest.TestCase):
    """Verify dataset immutability, canonical row-ID ordering, and admission boundaries."""

    def test_snapshot_immutability_and_caller_mutation_isolation(self) -> None:
        base = make_synthetic_process_dataset(n_rows=160, n_features=4, seed=11)
        caller_features = [[1.0, 2.0, None, 4.0] for _ in range(160)]
        caller_obs = [
            Observation(
                row_id=f"r-{i:04d}",
                features=caller_features[i],
                label=base.labels[i],
                group_id=f"g-{i % 10:02d}",
                timestamp=1000 + i,
            )
            for i in range(160)
        ]
        ds = TabularDataset(caller_obs, feature_names=("a", "b", "c", "d"))
        before_matrix = ds.feature_matrix.copy()

        # Mutate caller lists after construction.
        caller_features[0][0] = 999.0
        caller_obs.reverse()

        np.testing.assert_allclose(
            ds.feature_matrix,
            before_matrix,
            equal_nan=True,
        )
        self.assertFalse(ds.feature_matrix.flags.writeable)
        self.assertFalse(ds.missing_mask.flags.writeable)
        with self.assertRaises(ValueError):
            ds.feature_matrix[0, 0] = -123.0
        with self.assertRaises(ValueError):
            ds.missing_mask[0, 0] = False

        sliced = ds.slice_features(ds.row_ids[:10])
        self.assertFalse(sliced.flags.writeable)
        with self.assertRaises(ValueError):
            sliced[0, 0] = 77.0

    def test_from_arrays_copies_caller_numpy_buffer(self) -> None:
        n = 160
        row_ids = [f"id-{i:03d}" for i in range(n)]
        arr = np.ones((n, 3), dtype=np.float64)
        labels = np.array([i % 2 for i in range(n)], dtype=np.int64)
        ds = TabularDataset.from_arrays(row_ids, arr, labels)
        arr[:, :] = -500.0
        labels[:] = 0
        self.assertEqual(float(ds.feature_matrix[0, 0]), 1.0)
        self.assertEqual(ds.labels[1], 1)

    def test_row_permutation_invariance(self) -> None:
        ds_canonical = make_synthetic_process_dataset(n_rows=180, n_features=5, seed=19)
        rng = np.random.default_rng(99)
        perm = rng.permutation(ds_canonical.n_rows)
        shuffled_obs = [ds_canonical.observations[int(i)] for i in perm]
        ds_permuted = TabularDataset(
            shuffled_obs, feature_names=ds_canonical.feature_names
        )

        self.assertEqual(ds_canonical.row_ids, ds_permuted.row_ids)
        self.assertEqual(ds_canonical.labels, ds_permuted.labels)
        self.assertEqual(ds_canonical.group_ids, ds_permuted.group_ids)
        self.assertEqual(ds_canonical.timestamps, ds_permuted.timestamps)
        self.assertEqual(
            ds_canonical.metadata_fingerprint, ds_permuted.metadata_fingerprint
        )
        np.testing.assert_allclose(
            ds_canonical.feature_matrix,
            ds_permuted.feature_matrix,
            equal_nan=True,
        )

    def test_domain_violations_raise_invalid_input(self) -> None:
        valid_ds = make_synthetic_process_dataset(n_rows=160, n_features=3, seed=5)
        obs_list = list(valid_ds.observations)

        # Fewer than 160 observations -> INVALID_INPUT (even if budget max_rows is lower).
        with self.assertRaises(LabError) as ctx:
            TabularDataset(obs_list[:159], budget=AdmissionBudget(max_rows=100))
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_INPUT)

        # Duplicate row_id -> INVALID_INPUT.
        dup_obs = list(obs_list)
        dup_obs[1] = Observation(
            row_id=dup_obs[0].row_id,
            features=dup_obs[1].features,
            label=dup_obs[1].label,
            group_id=dup_obs[1].group_id,
            timestamp=dup_obs[1].timestamp,
        )
        with self.assertRaises(LabError) as ctx:
            TabularDataset(dup_obs)
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_INPUT)

        # Ragged feature dimensions -> INVALID_INPUT.
        ragged_obs = list(obs_list)
        ragged_obs[5] = Observation(
            row_id=ragged_obs[5].row_id,
            features=(1.0, 2.0),
            label=ragged_obs[5].label,
            group_id=ragged_obs[5].group_id,
            timestamp=ragged_obs[5].timestamp,
        )
        with self.assertRaises(LabError) as ctx:
            TabularDataset(ragged_obs)
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_INPUT)

        # Bool label, bool feature, bool timestamp, overflow int, >1e6 feature, whitespace ID.
        bad_cases = [
            {"row_id": "bad id", "features": [1.0, 2.0], "label": 0},
            {"row_id": "ok-1", "features": [True, 2.0], "label": 0},
            {"row_id": "ok-2", "features": [1.0, 2.0], "label": True},
            {"row_id": "ok-3", "features": [1.0, 2.0], "label": 2},
            {"row_id": "ok-4", "features": [10**400, 2.0], "label": 1},
            {"row_id": "ok-5", "features": [1_000_000.1, 2.0], "label": 1},
            {"row_id": "ok-6", "features": [float("nan"), 2.0], "label": 1},
            {"row_id": "ok-7", "features": [1.0, 2.0], "label": 1, "timestamp": True},
            {"row_id": "ok-8", "features": [1.0, 2.0], "label": 1, "timestamp": 1 << 63},
        ]
        for case in bad_cases:
            with self.subTest(case=case):
                with self.assertRaises(LabError) as err:
                    Observation(**case)
                self.assertEqual(err.exception.status, LabStatus.INVALID_INPUT)

    def test_caller_admission_budget_raises_unknown_budget(self) -> None:
        ds_200 = make_synthetic_process_dataset(n_rows=200, n_features=6, seed=7)

        # Supported domain (200 rows in 160..4096), but exceeds caller max_rows=180.
        with self.assertRaises(LabError) as ctx:
            TabularDataset(
                ds_200.observations,
                budget=AdmissionBudget(max_rows=180),
            )
        self.assertEqual(ctx.exception.status, LabStatus.UNKNOWN_BUDGET)

        # Supported domain (6 features in 1..16), but exceeds caller max_features=4.
        with self.assertRaises(LabError) as ctx:
            TabularDataset(
                ds_200.observations,
                budget=AdmissionBudget(max_features=4),
            )
        self.assertEqual(ctx.exception.status, LabStatus.UNKNOWN_BUDGET)

        # Invalid budget values themselves (bool, <=0, >domain max) are INVALID_INPUT.
        for invalid_kwargs in (
            {"max_rows": True},
            {"max_rows": 0},
            {"max_rows": 5000},
            {"max_features": 17},
            {"max_threshold_candidates": 4099},
            {"max_output_bytes": 0},
        ):
            with self.subTest(invalid_kwargs=invalid_kwargs):
                with self.assertRaises(LabError) as err:
                    AdmissionBudget(**invalid_kwargs)
                self.assertEqual(err.exception.status, LabStatus.INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()
