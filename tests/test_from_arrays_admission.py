"""Independent regressions for admission before retention and normalization."""
from collections.abc import Sequence
import unittest
import numpy as np
from calibrated_tabular_lab import AdmissionBudget, LabError, LabStatus, TabularDataset


class TrapSequence(Sequence):
    def __init__(self, size): self.size=size; self.reads=0
    def __len__(self): return self.size
    def __getitem__(self, index):
        self.reads+=1
        raise AssertionError('ITERATION_BEFORE_ADMISSION')


class FloatTrap(float):
    reads=0
    def __float__(self):
        type(self).reads+=1
        raise AssertionError('CONVERSION_BEFORE_ADMISSION')


class FromArraysAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.ids=[f'r-{i:03}' for i in range(160)]
        self.features=[[1.,2.] for _ in self.ids]
        self.labels=[0,1]*80

    def expect(self,status,operation):
        with self.assertRaises(LabError) as caught:operation()
        self.assertEqual(caught.exception.status,status)

    def test_timestamps_size_checked_before_retention(self):
        for size in [161,100_000_000]:
            trap=TrapSequence(size)
            with self.subTest(size=size):
                self.expect(LabStatus.INVALID_INPUT,lambda:TabularDataset.from_arrays(self.ids,self.features,self.labels,timestamps=trap))
                self.assertEqual(trap.reads,0)

    def test_row_admission_precedes_features_and_labels_copy(self):
        features=TrapSequence(160);labels=TrapSequence(160)
        self.expect(LabStatus.UNKNOWN_BUDGET,lambda:TabularDataset.from_arrays(self.ids,features,labels,budget=AdmissionBudget(max_rows=159)))
        self.assertEqual(features.reads,0);self.assertEqual(labels.reads,0)

    def test_feature_admission_precedes_scalar_conversion(self):
        FloatTrap.reads=0
        features=[[FloatTrap(1.),FloatTrap(2.)] for _ in self.ids]
        self.expect(LabStatus.UNKNOWN_BUDGET,lambda:TabularDataset.from_arrays(self.ids,features,self.labels,budget=AdmissionBudget(max_features=1)))
        self.assertEqual(FloatTrap.reads,0)

    def test_scalar_ndarrays_are_typed_input_errors(self):
        for field in ['labels','timestamps']:
            kwargs={'timestamps':np.array(123)} if field=='timestamps' else {}
            labels=np.array(0) if field=='labels' else self.labels
            with self.subTest(field=field):
                self.expect(LabStatus.INVALID_INPUT,lambda:TabularDataset.from_arrays(self.ids,self.features,labels,**kwargs))

    def test_shapes_are_checked_before_conversion(self):
        for features in [np.ones((160,0)),np.ones((160,17)),np.ones((160,2,1)),[[1.],[1.,2.]]+[[1.] for _ in range(158)]]:
            with self.subTest(shape=getattr(features,'shape',None)):
                self.expect(LabStatus.INVALID_INPUT,lambda:TabularDataset.from_arrays(self.ids,features,self.labels))

    def test_admitted_arrays_preserve_owned_snapshot_and_metadata(self):
        values=np.arange(320,dtype=np.float64).reshape(160,2)
        labels=np.array(self.labels,dtype=np.int64)
        timestamps=np.arange(160,dtype=np.int64)
        dataset=TabularDataset.from_arrays(self.ids,values,labels,timestamps=timestamps,budget=AdmissionBudget(max_rows=160,max_features=2))
        values[:]=-1;labels[:]=0;timestamps[:]=0
        self.assertEqual(dataset.feature_matrix[0].tolist(),[0.,1.])
        self.assertEqual(dataset.labels,tuple(self.labels))
        self.assertEqual(dataset.timestamps,tuple(range(160)))


if __name__=='__main__':unittest.main()
