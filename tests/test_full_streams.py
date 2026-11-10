"""Real installed CLI output failures, including argparse's help exit path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(Path("/dev/full").exists(), "Linux /dev/full required")
class FullOutputStreams(unittest.TestCase):
    def test_help_output_failure_retains_io_exit(self):
        with open("/dev/full", "wb", buffering=0) as full:
            completed = subprocess.run([sys.executable, "-m", "calibrated_tabular_lab", "--help"],
                                       stdout=full, stderr=full, timeout=20)
        self.assertEqual(completed.returncode, 3)

    def test_complete_report_with_failed_outcome_is_not_successful_exit(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "report"
            with open("/dev/full", "wb", buffering=0) as full:
                completed = subprocess.run([sys.executable, "-m", "calibrated_tabular_lab", "demo", "--output", str(target)],
                                           stdout=full, stderr=full, timeout=30)
            self.assertEqual(completed.returncode, 3)
            # The model, replay and directory commit actually succeeded before
            # the outcome write failed. Do not treat exit3 as a rollback claim.
            manifest = json.loads((target / "manifest.json").read_bytes())
            self.assertEqual(len(manifest["files"]), 4)
            for entry in manifest["files"]:
                payload = (target / entry["name"]).read_bytes()
                self.assertEqual(len(payload), entry["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
