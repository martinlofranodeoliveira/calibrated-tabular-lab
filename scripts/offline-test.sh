#!/bin/sh
set -eu
python -m venv /work/venv
/work/venv/bin/python -m pip install --no-index --find-links /opt/wheelhouse --require-hashes -r /opt/requirements.lock
cp -R /opt/project /work/project
cd /work/project
/work/venv/bin/python -m build --no-isolation --sdist --wheel
/work/venv/bin/python -m pip install --no-index --no-deps dist/*.whl
/work/venv/bin/python -m pip check
/work/venv/bin/python -m unittest discover -s tests -v
cp dist/* /work/distributions/
/work/venv/bin/calibrated-tabular-lab demo --assert --output /work/distributions/report-demo
/work/venv/bin/python -c 'import calibrated_tabular_lab; assert calibrated_tabular_lab.__file__.startswith("/work/venv/"); print("INSTALLED_PACKAGE_GATE_COMPLETE")'
