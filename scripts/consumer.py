"""Independent installed-artifact API/CLI consumer; no checkout or build package."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import calibrated_tabular_lab as ctl

assert ctl.__file__.startswith('/work/venv/')
assert not Path('/opt/project').exists()
assert importlib.util.find_spec('build') is None
assert abs(ctl.brier_score([0,1,1,1],[.1,.7,.2,.9])-.1875)<1e-15
dataset=ctl.make_synthetic_process_dataset()
plan=ctl.create_split_plan(dataset,strategy='chronological',seed=7,embargo=1)
result=ctl.run_experiment(dataset,plan=plan)
assert result.status==ctl.LabStatus.COMPLETE and result.config.seed==7

executable=str(Path(sys.executable).parent/'calibrated-tabular-lab')
completed=subprocess.run([executable,'demo','--assert','--preset','chronological','--output','/work/report'],capture_output=True,text=True,check=True)
outcome=json.loads(completed.stdout)
assert outcome['status']=='COMPLETE' and outcome['committed'] is True
root=Path('/work/report');manifest=json.loads((root/'manifest.json').read_bytes())
for entry in manifest['files']:
    body=(root/entry['name']).read_bytes()
    assert len(body)==entry['bytes'] and hashlib.sha256(body).hexdigest()==entry['sha256']
report=json.loads((root/'report.json').read_bytes());labels=report['test_labels']
for name,p in report['experiment']['test_probabilities'].items():
    brier=math.fsum((x-y)**2 for x,y in zip(p,labels))/len(labels)
    assert abs(brier-report['experiment']['test_evaluations'][name]['brier_loss'])<1e-12
again=subprocess.run([executable,'demo','--output','/work/report'],capture_output=True,text=True)
assert again.returncode==3 and json.loads(again.stdout)['status']=='ALREADY_EXISTS'
print(json.dumps({'status':'FRESH_INSTALLED_ARTIFACT_CONSUMER_COMPLETE','library':ctl.__file__,
                  'artifact_count':len(list(root.iterdir())),'test_rows':len(labels),
                  'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}))
