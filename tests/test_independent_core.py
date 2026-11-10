"""Independent pre-source goldens and native regressions added by Codex support."""
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.calibration import calibration_curve

from calibrated_tabular_lab import (
    AdmissionBudget, ExperimentConfig, LabError, LabStatus,
    Observation, TabularDataset, brier_score, clipped_log_loss,
    calibrate_forest_model, create_split_plan, evaluate_predictions, fit_base_models,
    make_synthetic_process_dataset, reliability_bins, roc_auc_score_exact,
    run_experiment, serialize_experiment_result, sweep_threshold_policies,
)


class OversizedSequence(Sequence):
    """A length-known seam which records any attempted retention/iteration."""
    def __init__(self, size): self.size=size; self.reads=0
    def __len__(self): return self.size
    def __getitem__(self, index):
        self.reads+=1
        raise AssertionError('SEQUENCE_READ_BEFORE_FIXED_SIZE_CHECK')


class IndependentCoreTests(unittest.TestCase):
    def assert_status(self, expected, fn):
        with self.assertRaises(LabError) as caught: fn()
        self.assertEqual(caught.exception.status,expected)

    def test_public_array_base_cannot_write_through_dataset(self):
        ds=make_synthetic_process_dataset(n_rows=160,n_features=2,seed=11)
        before=ds.feature_matrix.copy()
        public=ds.feature_matrix
        # Follow the ordinary NumPy .base chain; no dataset private attribute is accessed.
        root=public
        while isinstance(root.base,np.ndarray): root=root.base
        try:
            root.setflags(write=True)
            root.flat[0]=12345.
        except ValueError:
            pass
        np.testing.assert_array_equal(ds.feature_matrix,before)
        mask_before=ds.missing_mask.copy()
        mask=ds.missing_mask
        while isinstance(mask.base,np.ndarray): mask=mask.base
        try:
            mask.setflags(write=True);mask.flat[0]=not mask.flat[0]
        except ValueError:
            pass
        np.testing.assert_array_equal(ds.missing_mask,mask_before)

    def test_fixed_sequence_size_rejected_before_retention(self):
        observations=OversizedSequence(4097)
        self.assert_status(LabStatus.INVALID_INPUT,lambda:TabularDataset(observations))
        self.assertEqual(observations.reads,0)
        features=OversizedSequence(17)
        self.assert_status(LabStatus.INVALID_INPUT,lambda:Observation('id',features,0))
        self.assertEqual(features.reads,0)

    def test_fraction_decimal_auc_and_reliability_goldens(self):
        y=[0,1,1,1];r=[Fraction(1,10),Fraction(7,10),Fraction(1,5),Fraction(9,10)]
        exact=sum((p-a)**2 for a,p in zip(y,r))/4
        self.assertEqual(exact,Fraction(3,16))
        self.assertAlmostEqual(brier_score(y,[float(p) for p in r]),float(exact),delta=1e-15)
        eps=2.**-52
        labels=[0,1,0,1,0,1]
        probabilities=[0.,1.,1.,0.,eps/2,1-eps/2]
        with localcontext() as ctx:
            ctx.prec=80
            q=[min(1-eps,max(eps,p)) for p in probabilities]
            terms=[-Decimal.from_float(p).ln() if a else -(1-Decimal.from_float(p)).ln() for a,p in zip(labels,q)]
            expected=sum(terms)/len(terms)
        loss=clipped_log_loss(labels,probabilities)
        self.assertEqual(loss.clipped_count,6)
        self.assertAlmostEqual(loss.log_loss,float(expected),delta=1e-14)
        self.assertEqual(roc_auc_score_exact([0,0,1,1],[.1,.5,.5,.9]).roc_auc,.875)
        bins=reliability_bins([0,0,1,1,1],[0,.1,.2,.5,1],n_bins=10)
        self.assertEqual([b.count for b in bins],[2,1,0,0,1,0,0,0,0,1])
        self.assertEqual(sum(b.positive_count for b in bins),3)
        observed,predicted=calibration_curve([0,0,1,1,1],[0,.1,.2,.5,1],n_bins=10)
        np.testing.assert_allclose([b.observed_fraction for b in bins if b.count],observed)
        np.testing.assert_allclose([b.mean_probability for b in bins if b.count],predicted)
        self.assertTrue(all(b.mean_probability is None and b.observed_fraction is None for b in bins if not b.count))

    def test_threshold_sweep_against_independent_policy_counts(self):
        rng=random.Random(901)
        fixtures=[([0,1,0,1,1,0],[0,.2,.2,.6,1,1],1,4)]
        for _ in range(45):
            n=rng.randrange(1,17)
            fixtures.append(([rng.randrange(2) for _ in range(n)], [rng.choice([0.,.1,.2,.5,1.]) for _ in range(n)], rng.randrange(5),rng.randrange(1,5)))
        for y,p,fp_cost,fn_cost in fixtures:
            thresholds=[None]+sorted(set(p),reverse=True)+['positive']
            expected=[]
            for t in thresholds:
                pred=[0]*len(y) if t is None else [1]*len(y) if t=='positive' else [int(v>=t) for v in p]
                fp=sum(a==0 and b==1 for a,b in zip(y,pred))
                fn=sum(a==1 and b==0 for a,b in zip(y,pred))
                expected.append(fp_cost*fp+fn_cost*fn)
            actual=sweep_threshold_policies(y,p,fp_cost=fp_cost,fn_cost=fn_cost)
            self.assertEqual([a.cost for a in actual.candidates],expected)
            self.assertEqual(actual.selected.candidate_index,expected.index(min(expected)))
        golden=sweep_threshold_policies(*fixtures[0][:2],fp_cost=1,fn_cost=4)
        self.assertEqual([a.cost for a in golden.candidates],[12,9,5,2,3,3])
        self.assertEqual(golden.selected.policy.cutoff,.2)
        self.assert_status(LabStatus.UNKNOWN_BUDGET,lambda:sweep_threshold_policies([0,1],[.1,.2],fp_cost=1,fn_cost=1,budget=AdmissionBudget(max_threshold_candidates=3)))

    def test_incompatible_config_assertion_is_checked_with_frozen_plan(self):
        ds=make_synthetic_process_dataset(n_rows=320,seed=42)
        plan=create_split_plan(ds,seed=42)
        config=ExperimentConfig(seed=42,require_group_disjoint=True)
        self.assert_status(LabStatus.INVALID_SPLIT,lambda:run_experiment(ds,plan=plan,config=config))

    def test_frozen_plan_test_labels_do_not_change_models_or_policy(self):
        ds=make_synthetic_process_dataset(n_rows=320,seed=42)
        plan=create_split_plan(ds,seed=42)
        config=ExperimentConfig(seed=42,fp_cost=1,fn_cost=4)
        first=run_experiment(ds,config=config,plan=plan)
        updates={rid:1-y for rid,y in zip(plan.test_row_ids,ds.slice_labels(plan.test_row_ids))}
        changed=ds.with_replaced_labels(updates)
        second=run_experiment(changed,config=config,plan=plan)
        self.assertEqual(first.training_evidence,second.training_evidence)
        self.assertEqual(first.calibration_provenance,second.calibration_provenance)
        self.assertEqual(first.policy_probabilities,second.policy_probabilities)
        self.assertEqual(first.policy_sweep,second.policy_sweep)
        self.assertEqual(first.test_probabilities,second.test_probabilities)
        self.assertNotEqual(first.test_evaluations['raw_forest'].brier_loss,second.test_evaluations['raw_forest'].brier_loss)
        medians=np.nanmedian(ds.slice_features(plan.train_row_ids),axis=0)
        np.testing.assert_allclose(first.training_evidence.forest_imputer_medians,medians)
        payload=serialize_experiment_result(first)
        self.assertEqual(json.loads(payload)['status'],'COMPLETE')
        self.assert_status(LabStatus.UNKNOWN_BUDGET,lambda:serialize_experiment_result(first,budget=AdmissionBudget(max_output_bytes=len(payload.encode())-1)))

    def test_calibration_never_refits_returned_base_model(self):
        ds=make_synthetic_process_dataset(n_rows=320,seed=42)
        plan=create_split_plan(ds,seed=42)
        x_train=ds.slice_features(plan.train_row_ids)
        base=fit_base_models(x_train,ds.slice_labels(plan.train_row_ids),seed=42)
        calls=[]
        def forbidden(*args,**kwargs):
            calls.append(True)
            raise AssertionError('FROZEN_BASE_REFIT')
        base.forest_pipeline.fit=forbidden
        fitted=calibrate_forest_model(base,ds.slice_features(plan.calibration_row_ids),
                                    ds.slice_labels(plan.calibration_row_ids),X_train=x_train)
        self.assertEqual(calls,[])
        self.assertTrue(fitted.calibration_provenance.base_imputer_medians_unchanged)
        self.assertTrue(fitted.calibration_provenance.base_train_predictions_unchanged)

    def test_environment_oserror_is_not_a_fit_status(self):
        x=np.ones((20,2),dtype=np.float64);y=[0,1]*10
        with patch('sklearn.pipeline.Pipeline.fit',side_effect=OSError('controlled environment seam')):
            with self.assertRaises(OSError): fit_base_models(x,y,seed=0)

    def test_fixture_domain_and_budget_checked_before_rng_allocation(self):
        for kwargs,status in [
            ({'n_rows':4097},LabStatus.INVALID_INPUT),
            ({'n_features':17},LabStatus.INVALID_INPUT),
            ({'n_features':0},LabStatus.INVALID_INPUT),
            ({'missing_rate':float('nan')},LabStatus.INVALID_INPUT),
            ({'n_rows':320,'budget':AdmissionBudget(max_rows=160)},LabStatus.UNKNOWN_BUDGET),
        ]:
            with self.subTest(kwargs=kwargs):
                with patch('calibrated_tabular_lab.fixtures.np.random.default_rng',side_effect=AssertionError('RNG_BEFORE_ADMISSION')):
                    self.assert_status(status,lambda:make_synthetic_process_dataset(**kwargs))

    def test_metric_size_checked_before_retention(self):
        labels=OversizedSequence(4097)
        self.assert_status(LabStatus.INVALID_INPUT,lambda:brier_score(labels,labels))
        self.assertEqual(labels.reads,0)

    def test_exported_model_matrix_domain_checked_before_fit(self):
        for x in [np.ones((20,17)),np.full((20,2),1_000_001.)]:
            with patch('sklearn.pipeline.Pipeline.fit',side_effect=AssertionError('FIT_BEFORE_DOMAIN')):
                self.assert_status(LabStatus.INVALID_INPUT,lambda:fit_base_models(x,[0,1]*10,seed=0))

    def test_chronological_group_assurance_and_embargo_hand_golden(self):
        rows=[Observation(f'r-{i:04}',[float(i%7),float(i%3)],i%2,group_id=f'g-{i//2:03}',timestamp=i//2) for i in range(400)]
        ds=TabularDataset(rows)
        plan=create_split_plan(ds,strategy='chronological',embargo=1,require_group_disjoint=True)
        self.assertEqual([len(v) for v in plan.role_row_ids().values()],[196,78,58,62])
        self.assertEqual(plan.excluded_row_ids,tuple(f'r-{i:04}' for i in [196,197,276,277,336,337]))
        for strategy in ['stratified','group','chronological']:
            original=create_split_plan(ds,strategy=strategy,seed=2147483647)
            reordered=create_split_plan(TabularDataset(list(reversed(rows))),strategy=strategy,seed=2147483647)
            self.assertEqual(original,reordered)

    def test_max_supported_shape_runs_real_native_fit(self):
        ds=make_synthetic_process_dataset(n_rows=4096,n_features=16,seed=11,include_groups=False,include_timestamps=False)
        result=run_experiment(ds,config=ExperimentConfig(seed=11))
        self.assertEqual(result.status,LabStatus.COMPLETE)
        self.assertEqual(result.training_evidence.n_input_features,16)
        self.assertEqual(result.training_evidence.n_train_rows,2048)
        self.assertEqual(result.training_evidence.threadpool_max_threads,1)
        self.assertEqual(sum(result.split_audit.role_counts.values()),4096)


if __name__=='__main__': unittest.main()
