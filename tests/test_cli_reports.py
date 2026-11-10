"""Real filesystem/CLI checks and independent report replay; fault seams are labelled."""
from concurrent.futures import ThreadPoolExecutor
import csv
import errno
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibrated_tabular_lab import AdmissionBudget, ExperimentConfig, LabError, LabStatus, TabularDataset, make_synthetic_process_dataset, run_experiment
from calibrated_tabular_lab import cli


class CliReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = make_synthetic_process_dataset(seed=42)
        cls.result = run_experiment(cls.dataset, config=ExperimentConfig(seed=42))
        cls.artifacts = cli.make_artifacts(cls.dataset, cls.result, {'kind':'synthetic-demo'}, AdmissionBudget())
        cls.input = json.dumps({'rows':[{'row_id':row.row_id,'features':list(row.features),'label':row.label,
                                       'group_id':row.group_id,'timestamp':row.timestamp}
                                      for row in cls.dataset.observations],
                                'feature_names':list(cls.dataset.feature_names)}).encode()

    def invoke(self, arguments, stdin=None):
        output=io.StringIO()
        if stdin is None:
            with patch('sys.stdout',output): code=cli.main(arguments)
        else:
            stream=io.TextIOWrapper(io.BytesIO(stdin),encoding='utf-8')
            with patch('sys.stdin',stream),patch('sys.stdout',output): code=cli.main(arguments)
        return code,json.loads(output.getvalue())

    def test_input_file_and_stdin_are_bounded_and_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'input.json';source.write_bytes(self.input)
            dataset,digest=cli.read_dataset(str(source),AdmissionBudget())
            self.assertEqual(dataset.row_ids,self.dataset.row_ids)
            self.assertEqual(digest,hashlib.sha256(self.input).hexdigest())
            stream=io.TextIOWrapper(io.BytesIO(self.input),encoding='utf-8')
            with patch('sys.stdin',stream): other,_=cli.read_dataset('-',AdmissionBudget())
            self.assertEqual(other.metadata_fingerprint,dataset.metadata_fingerprint)

    def test_invalid_closed_json_never_fits(self):
        bad=[b'{"rows":[],"rows":[]}',b'{"rows":[],"extra":1}',b'{"rows":NaN}',
             b'{"rows":Infinity}',b'{"rows":[],"feature_names":null}',b'\xff',b'[]',b'{']
        parsed=json.loads(self.input)
        for change in [{'label':True},{'label':0.0},{'extra':1},{'features':[1e309]*8},{'timestamp':None}]:
            modified=json.loads(self.input);modified['rows'][0].update(change)
            bad.append(json.dumps(modified).encode())
        for raw in bad:
            with self.subTest(raw=raw[:60]),tempfile.TemporaryDirectory() as directory:
                with patch.object(cli,'run_experiment',side_effect=AssertionError('FIT_ON_REJECTED_INPUT')):
                    code,outcome=self.invoke(['analyze','--input','-','--output',str(Path(directory)/'report')],raw)
                self.assertEqual(code,2)
                self.assertEqual(outcome['status'],'INVALID_INPUT')
                self.assertEqual(list(Path(directory).iterdir()),[])

    def test_oversize_fifo_and_symlink_have_exact_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);large=root/'large';large.write_bytes(b' '*(cli.INPUT_LIMIT+1))
            fifo=root/'fifo';os.mkfifo(fifo)
            regular=root/'regular';regular.write_bytes(self.input)
            link=root/'link';link.symlink_to(regular)
            for path,code in [(large,'input_bytes'),(fifo,'input_not_regular'),(link,'input_symlink')]:
                with self.subTest(path=path),self.assertRaises(cli.AdapterError) as caught:
                    cli.read_dataset(str(path),AdmissionBudget())
                self.assertEqual(caught.exception.outcome['status'],'INVALID_INPUT')
                self.assertEqual(caught.exception.outcome['code'],code)

    def test_manifest_csv_and_metrics_replay_actual_values(self):
        manifest=json.loads(self.artifacts['manifest.json'])
        self.assertEqual(len(manifest['files']),4)
        for entry in manifest['files']:
            body=self.artifacts[entry['name']]
            self.assertEqual(entry['bytes'],len(body))
            self.assertEqual(entry['sha256'],hashlib.sha256(body).hexdigest())
        envelope=json.loads(self.artifacts['report.json']);result=envelope['experiment']
        for name,p in result['test_probabilities'].items():
            y=envelope['test_labels']
            expected=math.fsum((value-label)**2 for value,label in zip(p,y))/len(y)
            self.assertAlmostEqual(expected,result['test_evaluations'][name]['brier_loss'],delta=1e-14)
        bins=list(csv.DictReader(io.StringIO(self.artifacts['reliability.csv'].decode())))
        self.assertEqual(len(bins),40)
        for name in cli.MODEL_NAMES:
            self.assertEqual(sum(int(row['count']) for row in bins if row['model']==name),len(envelope['test_labels']))
        self.assertTrue(any(row['count']=='0' and row['mean_probability']=='' for row in bins))
        candidates=list(csv.DictReader(io.StringIO(self.artifacts['thresholds.csv'].decode())))
        self.assertEqual(len(candidates),result['policy_sweep']['n_candidates'])
        self.assertEqual(sum(row['selected']=='1' for row in candidates),1)

    def test_html_escapes_metadata_and_uses_no_external_assets(self):
        attack='<svg/onload=alert(1)>'
        names=[attack]+list(self.dataset.feature_names[1:])
        dataset=TabularDataset(self.dataset.observations,feature_names=names)
        text=cli.make_artifacts(dataset,self.result,{'kind':'synthetic-demo'},AdmissionBudget())['report.html'].decode()
        self.assertNotIn(attack,text)
        self.assertIn(html.escape(attack),text)
        self.assertNotIn('<script',text)
        self.assertNotIn('https://',text)
        self.assertIn('Calibration can worsen',text)
        self.assertIn('raw_forest',text);self.assertIn('calibrated_forest',text)

    def test_aggregate_budget_and_tampering_cannot_publish(self):
        with self.assertRaises(LabError) as caught:
            cli.make_artifacts(self.dataset,self.result,{'kind':'demo'},AdmissionBudget(max_output_bytes=100))
        self.assertEqual(caught.exception.status,LabStatus.UNKNOWN_BUDGET)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'report'
            tampered=dict(self.artifacts);tampered['report.json']+=b' '
            with self.assertRaises(cli.AdapterError) as caught: cli.publish_artifacts(tampered,str(target))
            self.assertEqual(caught.exception.outcome['code'],'artifact_manifest')
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_real_noreplace_collision_preserves_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            prior=Path(directory)/'report';prior.mkdir();(prior/'canary').write_bytes(b'prior')
            with self.assertRaises(cli.AdapterError) as caught:cli.publish_artifacts(self.artifacts,str(prior))
            self.assertEqual(caught.exception.outcome['status'],'ALREADY_EXISTS')
            self.assertEqual(caught.exception.outcome['errno'],errno.EEXIST)
            self.assertEqual((prior/'canary').read_bytes(),b'prior')
            self.assertEqual(set(p.name for p in Path(directory).iterdir()),{'report'})

    def test_precommit_failure_seam_preserves_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            prior=Path(directory)/'report';prior.mkdir();(prior/'canary').write_bytes(b'prior')
            with patch.object(cli,'_sync_directory',side_effect=OSError(errno.EIO,'labelled precommit seam')):
                with self.assertRaises(cli.AdapterError) as caught:cli.publish_artifacts(self.artifacts,str(prior))
            self.assertEqual(caught.exception.outcome['status'],'PRECOMMIT_FAILED')
            self.assertFalse(caught.exception.outcome['committed'])
            self.assertEqual((prior/'canary').read_bytes(),b'prior')

    def test_postcommit_parent_sync_seam_keeps_visible_report(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);target=parent/'report';original=cli._sync_directory
            def seam(path):
                if path==parent:raise OSError(errno.EINVAL,'labelled invalid parent-sync seam, not observed storage failure')
                original(path)
            with patch.object(cli,'_sync_directory',side_effect=seam):
                with self.assertRaises(cli.AdapterError) as caught:cli.publish_artifacts(self.artifacts,str(target))
            self.assertEqual(caught.exception.outcome['status'],'COMMITTED_UNCERTAIN')
            self.assertTrue(caught.exception.outcome['committed'])
            self.assertEqual((target/'report.json').read_bytes(),self.artifacts['report.json'])

    def test_concurrent_cooperative_publish_has_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'report'
            def publish(_):
                try:return cli.publish_artifacts(self.artifacts,str(target))['status']
                except cli.AdapterError as error:return error.outcome['status']
            with ThreadPoolExecutor(max_workers=2) as pool: outcomes=list(pool.map(publish,[0,1]))
            self.assertEqual(sorted(outcomes),['ALREADY_EXISTS','COMPLETE'])
            self.assertEqual((target/'manifest.json').read_bytes(),self.artifacts['manifest.json'])

    def test_main_demo_and_analyze_real_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            code,outcome=self.invoke(['demo','--assert','--preset','chronological','--output',str(root/'demo')])
            self.assertEqual(code,0);self.assertEqual(outcome['status'],'COMPLETE')
            self.assertEqual(len(list((root/'demo').iterdir())),5)
            code,outcome=self.invoke(['analyze','--input','-','--seed','42','--output',str(root/'analyze')],self.input)
            self.assertEqual(code,0)
            report=json.loads((root/'analyze'/'report.json').read_bytes())
            self.assertEqual(report['source']['bounded_input_sha256'],hashlib.sha256(self.input).hexdigest())
            with patch.object(cli,'publish_artifacts',side_effect=AssertionError('PUBLISH_ON_BUDGET_FAILURE')):
                code,outcome=self.invoke(['demo','--output',str(root/'budget'),'--max-output-bytes','100'])
            self.assertEqual(code,2);self.assertEqual(outcome['status'],'UNKNOWN_BUDGET')
            self.assertFalse((root/'budget').exists())

    def test_unexpected_runtime_does_not_leak_raw_exception(self):
        with patch.object(cli,'run_experiment',side_effect=RuntimeError('PRIVATE-CANARY-DO-NOT-PRINT')):
            code,outcome=self.invoke(['demo','--output','unused'])
        self.assertEqual(code,3);self.assertEqual(outcome['status'],'INCOMPLETE')
        self.assertNotIn('PRIVATE-CANARY',json.dumps(outcome))

    def test_known_sdk_fit_failure_has_stable_public_message(self):
        with patch('sklearn.pipeline.Pipeline.fit',side_effect=ValueError('PRIVATE-SDK-CANARY')):
            code,outcome=self.invoke(['demo','--output','unused'])
        self.assertEqual(code,2)
        self.assertEqual(outcome['status'],'FIT_FAILED')
        self.assertEqual(outcome['code'],'base_fit_failed')
        self.assertNotIn('PRIVATE-SDK-CANARY',json.dumps(outcome))


if __name__=='__main__':unittest.main()
