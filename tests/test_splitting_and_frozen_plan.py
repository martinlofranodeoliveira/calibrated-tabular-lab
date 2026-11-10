"""Tests for four-role split strategies, embargo rules, and FrozenSplitPlan validation."""

from __future__ import annotations

import unittest

from calibrated_tabular_lab import (
    FrozenSplitPlan,
    LabError,
    LabStatus,
    Observation,
    SplitStrategy,
    TabularDataset,
    compute_role_counts,
    create_split_plan,
    make_synthetic_process_dataset,
    validate_split_plan,
)


class SplittingAndFrozenPlanTests(unittest.TestCase):
    """Verify stratified, group, and chronological splits and independent plan validation."""

    def test_role_count_allocation_rule(self) -> None:
        self.assertEqual(compute_role_counts(800), (400, 160, 120, 120))
        self.assertEqual(compute_role_counts(160), (80, 32, 24, 24))
        self.assertEqual(compute_role_counts(163), (81, 32, 24, 26))

        with self.assertRaises(LabError) as ctx:
            compute_role_counts(6)
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_SPLIT)

    def test_stratified_group_and_chronological_splits(self) -> None:
        ds = make_synthetic_process_dataset(
            n_rows=320,
            n_features=6,
            seed=42,
            include_groups=True,
            include_timestamps=True,
            n_groups=24,
            n_time_buckets=40,
        )

        # 1. Stratified
        strat_plan = create_split_plan(
            ds, strategy=SplitStrategy.STRATIFIED, seed=42
        )
        strat_audit = validate_split_plan(ds, strat_plan)
        self.assertEqual(dict(strat_audit.role_counts), {"train": 160, "calibration": 64, "policy": 48, "test": 48})
        self.assertEqual(strat_audit.excluded_rows, 0)
        self.assertTrue(strat_audit.row_disjoint)

        # 2. Group
        group_plan = create_split_plan(
            ds, strategy=SplitStrategy.GROUP, seed=42
        )
        group_audit = validate_split_plan(ds, group_plan)
        self.assertTrue(group_audit.group_disjoint)
        # Verify group sets are pairwise disjoint independently.
        row_to_grp = dict(zip(ds.row_ids, ds.group_ids or ()))
        grp_sets = [
            {row_to_grp[r] for r in rids}
            for rids in (
                group_plan.train_row_ids,
                group_plan.calibration_row_ids,
                group_plan.policy_row_ids,
                group_plan.test_row_ids,
            )
        ]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertTrue(grp_sets[i].isdisjoint(grp_sets[j]))

        # 3. Chronological with embargo=2
        chrono_plan = create_split_plan(
            ds, strategy=SplitStrategy.CHRONOLOGICAL, seed=42, embargo=2
        )
        chrono_audit = validate_split_plan(ds, chrono_plan)
        self.assertTrue(chrono_audit.chronological_ordered)
        self.assertGreater(chrono_audit.excluded_rows, 0)
        bounds = chrono_audit.role_timestamp_bounds
        assert bounds is not None
        self.assertLess(bounds["train"][1], bounds["calibration"][0])
        self.assertLess(bounds["calibration"][1], bounds["policy"][0])
        self.assertLess(bounds["policy"][1], bounds["test"][0])

    def test_incompatible_simultaneous_assertions_rejected(self) -> None:
        # Periodic group IDs repeat across time buckets, so group and chronological cannot both hold.
        ds = make_synthetic_process_dataset(
            n_rows=320,
            n_features=4,
            seed=42,
            include_groups=True,
            include_timestamps=True,
            n_groups=24,
            n_time_buckets=40,
        )
        with self.assertRaises(LabError) as ctx:
            create_split_plan(
                ds,
                strategy=SplitStrategy.GROUP,
                seed=42,
                require_chronological_order=True,
            )
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_SPLIT)

        with self.assertRaises(LabError) as ctx:
            create_split_plan(
                ds,
                strategy=SplitStrategy.CHRONOLOGICAL,
                seed=42,
                require_group_disjoint=True,
            )
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_SPLIT)

    def test_insufficient_class_support_and_no_seed_retry(self) -> None:
        # Only 10 positives total across 160 rows -> impossible to have >=8 positives in every role.
        obs = [
            Observation(
                row_id=f"row-{i:03d}",
                features=(float(i % 7), float((i * 3) % 11)),
                label=1 if i < 10 else 0,
                group_id=f"g-{i % 20:02d}",
                timestamp=1000 + (i // 4) * 60,
            )
            for i in range(160)
        ]
        ds = TabularDataset(obs)
        for strat in (SplitStrategy.STRATIFIED, SplitStrategy.GROUP, SplitStrategy.CHRONOLOGICAL):
            with self.subTest(strategy=strat):
                with self.assertRaises(LabError) as ctx:
                    create_split_plan(ds, strategy=strat, seed=7)
                self.assertEqual(ctx.exception.status, LabStatus.INVALID_SPLIT)

    def test_hand_crafted_malformed_or_tampered_plans_rejected(self) -> None:
        ds = make_synthetic_process_dataset(n_rows=200, n_features=4, seed=12)
        valid_plan = create_split_plan(ds, strategy=SplitStrategy.GROUP, seed=12)

        # 1. Swap one row between train and test in a group plan -> breaks group disjointness.
        tampered_group_plan = FrozenSplitPlan(
            strategy=valid_plan.strategy,
            seed=valid_plan.seed,
            embargo=valid_plan.embargo,
            train_row_ids=tuple(sorted((*valid_plan.train_row_ids[1:], valid_plan.test_row_ids[0]))),
            calibration_row_ids=valid_plan.calibration_row_ids,
            policy_row_ids=valid_plan.policy_row_ids,
            test_row_ids=tuple(sorted((*valid_plan.test_row_ids[1:], valid_plan.train_row_ids[0]))),
            excluded_row_ids=valid_plan.excluded_row_ids,
            require_group_disjoint=True,
            require_chronological_order=False,
            metadata_fingerprint=valid_plan.metadata_fingerprint,
        )
        with self.assertRaises(LabError) as ctx:
            validate_split_plan(ds, tampered_group_plan)
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_SPLIT)

        # 2. Duplicate row across train and calibration.
        dup_plan = FrozenSplitPlan(
            strategy=SplitStrategy.STRATIFIED,
            seed=valid_plan.seed,
            embargo=0,
            train_row_ids=valid_plan.train_row_ids,
            calibration_row_ids=tuple(sorted((*valid_plan.calibration_row_ids, valid_plan.train_row_ids[0]))),
            policy_row_ids=valid_plan.policy_row_ids,
            test_row_ids=valid_plan.test_row_ids[1:],
            excluded_row_ids=(),
            require_group_disjoint=False,
            require_chronological_order=False,
            metadata_fingerprint=valid_plan.metadata_fingerprint,
        )
        with self.assertRaises(LabError) as ctx:
            validate_split_plan(ds, dup_plan)
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_SPLIT)

        # 3. Mismatched metadata fingerprint.
        mismatch_plan = FrozenSplitPlan(
            strategy=valid_plan.strategy,
            seed=valid_plan.seed,
            embargo=valid_plan.embargo,
            train_row_ids=valid_plan.train_row_ids,
            calibration_row_ids=valid_plan.calibration_row_ids,
            policy_row_ids=valid_plan.policy_row_ids,
            test_row_ids=valid_plan.test_row_ids,
            excluded_row_ids=valid_plan.excluded_row_ids,
            require_group_disjoint=valid_plan.require_group_disjoint,
            require_chronological_order=valid_plan.require_chronological_order,
            metadata_fingerprint="ctl-meta-v1:0000000000000000",
        )
        with self.assertRaises(LabError) as ctx:
            validate_split_plan(ds, mismatch_plan)
        self.assertEqual(ctx.exception.status, LabStatus.INVALID_SPLIT)


if __name__ == "__main__":
    unittest.main()
