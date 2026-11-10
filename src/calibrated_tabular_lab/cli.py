"""Closed local JSON adapter, replayed reports, and Linux no-replace publication."""
from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import errno
import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import sys
import tempfile
from typing import Any

from . import (
    AdmissionBudget, ExperimentConfig, LabError, LabStatus, TabularDataset,
    brier_score, make_synthetic_process_dataset, run_experiment,
)

INPUT_LIMIT = 4_194_304
PRESETS = {"stratified": ("stratified", 42, 0), "group": ("group", 42, 0),
           "chronological": ("chronological", 7, 1)}
MODEL_NAMES = ("prior", "logistic", "raw_forest", "calibrated_forest")


class AdapterError(Exception):
    def __init__(self, status: str, code: str, message: str, **details: Any):
        super().__init__(message)
        self.outcome = {"status": status, "code": code, "message": message, **details}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AdapterError("INVALID_INPUT", "arguments", "Invalid command-line arguments; see --help.")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            self._print_message(message, sys.stderr)
        raise SystemExit(status if _flush_standard_streams() else 3)


def _silence_failed_stream(stream: Any) -> None:
    """Prevent CPython's shutdown flush from replacing the observed I/O exit."""
    try:
        descriptor = stream.fileno()
        null = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
        try:
            os.dup2(null, descriptor)
        finally:
            os.close(null)
    except (OSError, ValueError, AttributeError):
        pass  # Best effort only; no file or user report is removed.


def _flush_standard_streams() -> bool:
    complete = True
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            complete = False
            _silence_failed_stream(stream)
    return complete


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise AdapterError("INVALID_INPUT", "duplicate_key", "Duplicate JSON object key.")
        out[key] = value
    return out


def _constant(value: str) -> None:
    raise AdapterError("INVALID_INPUT", "nonfinite_json", "Nonfinite JSON numbers are unsupported.")


def read_dataset(name: str, budget: AdmissionBudget) -> tuple[TabularDataset, str]:
    """Bound retention; a file leaf is opened nonblocking before regular-file validation."""
    if name == "-":
        raw = sys.stdin.buffer.read(INPUT_LIMIT + 1)
    else:
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise AdapterError("INVALID_INPUT", "input_symlink", "A symbolic-link input leaf is unsupported.") from exc
            raise
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AdapterError("INVALID_INPUT", "input_not_regular", "Input must be a regular file.")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(INPUT_LIMIT + 1)
        finally:
            os.close(descriptor)
    if len(raw) > INPUT_LIMIT:
        raise AdapterError("INVALID_INPUT", "input_bytes", "Input exceeds the fixed byte limit.")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AdapterError("INVALID_INPUT", "json_syntax", "Input is not supported strict UTF-8 JSON.") from exc
    if not isinstance(value, dict) or set(value) - {"rows", "feature_names"} or "rows" not in value:
        raise AdapterError("INVALID_INPUT", "input_schema", "Expected rows and optional feature_names only.")
    if not isinstance(value["rows"], list):
        raise AdapterError("INVALID_INPUT", "input_schema", "rows must be an array.")
    if "feature_names" in value and not isinstance(value["feature_names"], list):
        raise AdapterError("INVALID_INPUT", "input_schema", "feature_names must be an array when present.")
    for row in value["rows"]:
        if isinstance(row, dict) and any(key in row and row[key] is None for key in ["group_id", "timestamp"]):
            raise AdapterError("INVALID_INPUT", "input_schema", "Optional metadata must be omitted or have its declared type.")
    dataset = TabularDataset(value["rows"], feature_names=value.get("feature_names"), budget=budget)
    return dataset, hashlib.sha256(raw).hexdigest()


def _decisions(probabilities: list[float], policy: dict[str, Any]) -> list[int]:
    if policy["kind"] == "all_negative":
        return [0] * len(probabilities)
    if policy["kind"] == "all_positive":
        return [1] * len(probabilities)
    return [int(p >= policy["cutoff"]) for p in probabilities]


def _counts(labels: list[int], predictions: list[int]) -> tuple[int, int, int, int]:
    return tuple(sum(a == y and b == pred for y, pred in zip(labels, predictions))
                 for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)])


def _replay(result: dict[str, Any], test_labels: list[int], policy_labels: list[int]) -> None:
    """Separate scalar/pair/count replay before exposing any report artifact."""
    def require(condition: bool) -> None:
        if not condition:
            raise LabError(LabStatus.FIT_FAILED, "Independent report replay failed.", code="report_replay_failed")

    def require_probabilities(probabilities: list[float], labels: list[int]) -> None:
        require(len(probabilities) == len(labels) and len(labels) > 0)
        require(all(isinstance(p, (int, float)) and not isinstance(p, bool)
                    and 0 <= p <= 1 and math.isfinite(p) for p in probabilities))

    def require_confusion(record: dict[str, Any], counts: tuple[int, int, int, int]) -> None:
        keys = ("tn", "fp", "fn", "tp")
        require(all(type(record[key]) is int for key in keys))
        require(tuple(record[key] for key in keys) == counts)
        tn, fp, fn, tp = counts
        ratios = {"precision": tp / (tp + fp) if tp + fp else None,
                  "recall": tp / (tp + fn) if tp + fn else None}
        for key, expected in ratios.items():
            actual = record[key]
            if expected is None:
                require(actual is None)
            else:
                require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                        and 0 <= actual <= 1 and math.isfinite(actual)
                        and abs(actual - expected) <= 1e-12)

    epsilon = 2.0 ** -52
    for name in MODEL_NAMES:
        probabilities = result["test_probabilities"][name]
        metrics = result["test_evaluations"][name]
        require_probabilities(probabilities, test_labels)
        require(type(metrics["n_observations"]) is int and metrics["n_observations"] == len(test_labels))
        brier = math.fsum((p - y) ** 2 for p, y in zip(probabilities, test_labels)) / len(test_labels)
        q = [min(1 - epsilon, max(epsilon, p)) for p in probabilities]
        loss = math.fsum(-math.log(p) if y else -math.log1p(-p) for p, y in zip(q, test_labels)) / len(test_labels)
        require(abs(metrics["brier_loss"] - brier) <= 1e-12)
        require(abs(metrics["log_loss"] - loss) <= 1e-12)
        counts = _counts(test_labels, [int(p >= .5) for p in probabilities])
        require_confusion(metrics["confusion_at_default_cutoff"], counts)
        positive = [p for p, y in zip(probabilities, test_labels) if y]
        negative = [p for p, y in zip(probabilities, test_labels) if not y]
        twice_favorable = sum(2 * (p > n) + (p == n) for p in positive for n in negative)
        require(abs(metrics["roc_auc"] - twice_favorable / (2 * len(positive) * len(negative))) <= 1e-12)
        bins = metrics["reliability_bins"]
        boundaries = [b["upper_edge"] for b in bins[:-1]]
        for index, record in enumerate(bins):
            selected = [(p, y) for p, y in zip(probabilities, test_labels) if bisect.bisect_left(boundaries, p) == index]
            require(record["count"] == len(selected))
            require(record["positive_count"] == sum(y for _, y in selected))
            if selected:
                require(abs(record["mean_probability"] - math.fsum(p for p, _ in selected) / len(selected)) <= 1e-12)
                require(abs(record["observed_fraction"] - sum(y for _, y in selected) / len(selected)) <= 1e-12)
            else:
                require(record["mean_probability"] is None and record["observed_fraction"] is None)
    sweep = result["policy_sweep"]
    probabilities = result["policy_probabilities"]
    require_probabilities(probabilities, policy_labels)
    require(type(sweep["n_observations"]) is int and sweep["n_observations"] == len(policy_labels))
    cost_model = sweep["cost_model"]
    expected_policies = [{"kind": "all_negative", "cutoff": None}]
    expected_policies += [{"kind": "cutoff", "cutoff": p} for p in sorted(set(probabilities), reverse=True)]
    expected_policies += [{"kind": "all_positive", "cutoff": None}]
    require(len(sweep["candidates"]) == len(expected_policies))
    require(type(sweep["n_candidates"]) is int and sweep["n_candidates"] == len(sweep["candidates"]))
    costs = []
    for index, (candidate, expected) in enumerate(zip(sweep["candidates"], expected_policies)):
        require(type(candidate["candidate_index"]) is int and candidate["candidate_index"] == index)
        require(candidate["policy"] == expected)
        tn, fp, fn, tp = _counts(policy_labels, _decisions(probabilities, expected))
        cost = cost_model["fp_cost"] * fp + cost_model["fn_cost"] * fn
        costs.append(cost)
        require(candidate["cost"] == cost)
        require_confusion(candidate["confusion"], (tn, fp, fn, tp))
    require(sweep["selected"] == sweep["candidates"][costs.index(min(costs))])
    selected_test = result["selected_policy_test_evaluation"]
    require(selected_test["policy"] == sweep["selected"]["policy"])
    require(selected_test["cost_model"] == cost_model)
    tn, fp, fn, tp = _counts(test_labels, _decisions(result["test_probabilities"]["calibrated_forest"], selected_test["policy"]))
    require_confusion(selected_test["test_confusion"], (tn, fp, fn, tp))
    require(selected_test["test_cost"] == cost_model["fp_cost"] * fp + cost_model["fn_cost"] * fn)


def _csv_bytes(rows: list[list[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _html(envelope: dict[str, Any]) -> bytes:
    esc = lambda value: html.escape(str(value), quote=True)
    number = lambda value: "undefined" if value is None else f"{value:.6g}"
    result = envelope["experiment"]
    table = []
    confusion_rows = []
    circles = []
    colors = ["#777", "#1565c0", "#934e00", "#00735c"]
    for name, color in zip(MODEL_NAMES, colors):
        metrics = result["test_evaluations"][name]
        table.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [name, number(metrics['brier_loss']), number(metrics['log_loss']), number(metrics['roc_auc'])]) + "</tr>")
        confusion = metrics['confusion_at_default_cutoff']
        confusion_rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [name] + [confusion[k] for k in ['tn', 'fp', 'fn', 'tp']] + [number(confusion[k]) for k in ['precision', 'recall']]) + "</tr>")
        for b in metrics["reliability_bins"]:
            if b["count"]:
                x, y = 50 + 300 * b["mean_probability"], 350 - 300 * b["observed_fraction"]
                circles.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="4" fill="{color}"><title>{esc(name)} bin {b["bin_index"]}: n={b["count"]}</title></circle>')
    rows = []
    for name in MODEL_NAMES:
        for b in result["test_evaluations"][name]["reliability_bins"]:
            interval = ("[" if b["left_closed"] else "(") + f'{b["lower_edge"]:.3g}, {b["upper_edge"]:.3g}]'
            rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [name, interval, b['count'], b['positive_count'], number(b['mean_probability']), number(b['observed_fraction'])]) + "</tr>")
    audit = result["split_audit"]
    role_rows = "".join(f"<tr><th>{esc(role)}</th><td>{support['total']}</td><td>{support['negative_count']}</td><td>{support['positive_count']}</td></tr>" for role, support in audit['class_support'].items())
    selected = result["selected_policy_test_evaluation"]
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Calibrated Tabular Lab — local experiment</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#17232c;background:#fff}}h1,h2{{line-height:1.2}}small,p{{max-width:80ch}}table{{border-collapse:collapse;width:100%;margin:1.5rem 0;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #d9e1e7}}thead{{background:#f0f4f6}}.scroll{{overflow:auto}}svg{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style>
<header><p>CALIBRATED TABULAR LAB · DEVELOPMENT 0.1.0</p><h1>Local experiment evidence</h1><p>Four separate roles. Actual fitted baselines, raw forest and held-out sigmoid calibration. Calibration can worsen scores.</p></header>
<h2>Data roles and checks</h2><p>Source: {esc(envelope['source']['kind'])}. Features: {esc(', '.join(envelope['feature_names']))}.</p><table><thead><tr><th>Role</th><th>Rows</th><th>Label 0</th><th>Label 1</th></tr></thead><tbody>{role_rows}</tbody></table>
<p>Excluded rows: {audit['excluded_rows']}. Row disjoint: {esc(audit['row_disjoint'])}; group disjoint: {esc(audit['group_disjoint'])}; time ordered: {esc(audit['chronological_ordered'])}. These checks do not establish independence or detect all future-information features.</p>
<h2>Same-test-row model comparison</h2><p>Lower Brier and log loss are better on these rows. Both assess calibration, discrimination and uncertainty. ROC AUC gives exact-score ties half credit. Default confusion cutoff is 0.5, separate from policy selection.</p><table><thead><tr><th>Model</th><th>Brier</th><th>Log loss</th><th>ROC AUC</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<div class="scroll"><table><thead><tr><th>Model at0.5</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{''.join(confusion_rows)}</tbody></table></div>
<h2>Reliability with support</h2><svg viewBox="0 0 440 400" role="img" aria-label="Nonempty reliability-bin means; diagonal is an ideal reference"><path d="M50 50V350H350" fill="none" stroke="#333"/><path d="M50 350L350 50" fill="none" stroke="#aaa" stroke-dasharray="5 5"/>{''.join(circles)}<text x="50" y="375">0</text><text x="340" y="375">1</text><text x="120" y="395">Mean predicted probability</text><text x="4" y="35">Observed fraction</text></svg>
<p>Dots show nonempty bin means; no interpolation is implied. Gray: prior; blue: logistic; brown: raw forest; green: calibrated forest. Sparse bins are not certification. Bins are right-closed; the first includes zero. Empty bins remain in the table.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Bin</th><th>Rows</th><th>Positive</th><th>Mean p</th><th>Observed</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Policy selected on separate policy rows</h2><p>Policy: <code>{esc(json.dumps(selected['policy'],allow_nan=False))}</code>. Hypothetical penalties: FP={selected['cost_model']['fp_cost']}, FN={selected['cost_model']['fn_cost']}. Final-test penalty: {selected['test_cost']}. Selected-policy final-test counts: TN={selected['test_confusion']['tn']}, FP={selected['test_confusion']['fp']}, FN={selected['test_confusion']['fn']}, TP={selected['test_confusion']['tp']}. This is not currency or a deployment recommendation; all candidates and counts are in thresholds.csv.</p>
<footer><h2>Scope and replay</h2><p>Fixed synthetic examples are laboratory process labels. No medical, credit, hiring or safety-critical use is evaluated. A single holdout does not certify real-world probability quality, causality or production suitability. JSON carries aligned labels/probabilities for independent replay; manifest.json hashes every other artifact. Stdout/exit and artifact checks are required. No power-loss durability guarantee.</p><p>Dataset metadata identity: <code>{esc(envelope['dataset_metadata_fingerprint'])}</code></p></footer></html>'''
    return document.encode("utf-8")


def make_artifacts(dataset: TabularDataset, result: Any, source: dict[str, Any], budget: AdmissionBudget) -> dict[str, bytes]:
    data = result.to_dict()
    test_labels = list(dataset.slice_labels(result.plan.test_row_ids))
    policy_labels = list(dataset.slice_labels(result.plan.policy_row_ids))
    _replay(data, test_labels, policy_labels)
    envelope = {"format": "CTL1", "schema_version": 1, "status": "COMPLETE", "source": source,
                "dataset_metadata_fingerprint": dataset.metadata_fingerprint,
                "feature_names": list(dataset.feature_names), "test_labels": test_labels,
                "policy_labels": policy_labels, "experiment": data,
                "versions": {name: importlib.metadata.version(name) for name in ["calibrated-tabular-lab", "numpy", "scipy", "scikit-learn", "threadpoolctl"]},
                "python": platform.python_version()}
    reliability = [["model", "bin", "lower", "upper", "count", "positive_count", "mean_probability", "observed_fraction"]]
    for name in MODEL_NAMES:
        for b in data["test_evaluations"][name]["reliability_bins"]:
            reliability.append([name] + [b[k] for k in ["bin_index", "lower_edge", "upper_edge", "count", "positive_count", "mean_probability", "observed_fraction"]])
    thresholds = [["candidate", "kind", "cutoff", "penalty", "tn", "fp", "fn", "tp", "selected"]]
    selected_index = data["policy_sweep"]["selected"]["candidate_index"]
    for candidate in data["policy_sweep"]["candidates"]:
        thresholds.append([candidate["candidate_index"], candidate["policy"]["kind"], candidate["policy"]["cutoff"], candidate["cost"]] + [candidate["confusion"][k] for k in ["tn", "fp", "fn", "tp"]] + [int(candidate["candidate_index"] == selected_index)])
    artifacts = {"report.json": _json_bytes(envelope), "reliability.csv": _csv_bytes(reliability),
                 "thresholds.csv": _csv_bytes(thresholds), "report.html": _html(envelope)}
    manifest = {"format": "CTL1-MANIFEST", "status": "COMPLETE", "files": [
        {"name": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        for name, body in sorted(artifacts.items())]}
    artifacts["manifest.json"] = _json_bytes(manifest)
    if sum(len(body) for body in artifacts.values()) > budget.max_output_bytes:
        raise LabError(LabStatus.UNKNOWN_BUDGET, "Complete report exceeds the aggregate byte budget.", code="report_output_budget")
    return artifacts


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(stage: Path, target: Path) -> None:
    library = ctypes.CDLL("libc.so.6", use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "No supported no-replace primitive.") from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
        raise OSError(ctypes.get_errno(), "Directory no-replace commit failed.")


def publish_artifacts(artifacts: dict[str, bytes], destination: str) -> dict[str, Any]:
    """Only a successful no-replace rename changes publication state."""
    names = {"report.json", "reliability.csv", "thresholds.csv", "report.html", "manifest.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != names or not all(isinstance(b, bytes) for b in artifacts.values()):
        raise AdapterError("INVALID_INPUT", "artifact_set", "A complete fixed artifact set is required.")
    if sum(len(body) for body in artifacts.values()) > INPUT_LIMIT:
        raise LabError(LabStatus.UNKNOWN_BUDGET, "Report exceeds the fixed publication bound.", code="report_output_budget")
    try:
        manifest = json.loads(artifacts["manifest.json"], object_pairs_hook=_object, parse_constant=_constant)
        expected = [{"name": name, "bytes": len(artifacts[name]), "sha256": hashlib.sha256(artifacts[name]).hexdigest()} for name in sorted(names - {"manifest.json"})]
        valid = manifest == {"format": "CTL1-MANIFEST", "status": "COMPLETE", "files": expected}
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AdapterError("INVALID_INPUT", "artifact_manifest", "Invalid report manifest.") from exc
    if not valid:
        raise AdapterError("INVALID_INPUT", "artifact_manifest", "Report manifest does not match artifacts.")
    if not destination or "\0" in destination or len(os.fsencode(destination)) > 4096:
        raise AdapterError("INVALID_INPUT", "output_path", "Unsupported output path.")
    target = Path(os.path.abspath(destination))
    if not target.name:
        raise AdapterError("INVALID_INPUT", "output_path", "Output needs a directory name.")
    stage: Path | None = None
    committed = False
    try:
        stage = Path(tempfile.mkdtemp(prefix=".ctl-stage-", dir=target.parent))
        for name, body in artifacts.items():
            if name not in {"report.json", "reliability.csv", "thresholds.csv", "report.html", "manifest.json"}:
                raise AdapterError("INVALID_INPUT", "artifact_name", "Unexpected artifact name.")
            with (stage / name).open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        _sync_directory(stage)
        _rename_noreplace(stage, target)
        committed = True
        _sync_directory(target.parent)
    except OSError as exc:
        if committed:
            raise AdapterError("COMMITTED_UNCERTAIN", "parent_sync", "Report is visible; durability confirmation failed.", committed=True, path=str(target), errno=exc.errno) from exc
        status = "ALREADY_EXISTS" if exc.errno == errno.EEXIST else "PRECOMMIT_FAILED"
        raise AdapterError(status, "directory_commit", "Report was not committed; any previous destination is preserved.", committed=False, errno=exc.errno) from exc
    finally:
        if stage is not None and not committed:
            try:
                shutil.rmtree(stage)
            except OSError:
                pass  # Cleanup is best effort; no universal cleanup claim is made.
    return {"status": "COMPLETE", "committed": True, "path": str(target), "artifact_count": len(artifacts),
            "artifact_bytes": sum(len(value) for value in artifacts.values()),
            "manifest_sha256": hashlib.sha256(artifacts["manifest.json"]).hexdigest()}


def _parser() -> _Parser:
    parser = _Parser(prog="calibrated-tabular-lab", description="Bounded offline tabular calibration experiments.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    for command in ["demo", "analyze"]:
        sub = commands.add_parser(command)
        sub.add_argument("--output", required=True)
        sub.add_argument("--max-output-bytes", type=int, default=INPUT_LIMIT)
        if command == "demo":
            sub.add_argument("--preset", choices=PRESETS, default="stratified")
            sub.add_argument("--assert", dest="assert_demo", action="store_true")
        else:
            sub.add_argument("--input", required=True)
            sub.add_argument("--strategy", choices=PRESETS, default="stratified")
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--embargo", type=int, default=0)
            sub.add_argument("--require-group-disjoint", action="store_true")
            sub.add_argument("--require-chronological-order", action="store_true")
            sub.add_argument("--fp-cost", type=int, default=1)
            sub.add_argument("--fn-cost", type=int, default=1)
            sub.add_argument("--bins", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        budget = AdmissionBudget(max_output_bytes=arguments.max_output_bytes)
        if arguments.command == "demo":
            strategy, seed, embargo = PRESETS[arguments.preset]
            dataset = make_synthetic_process_dataset(budget=budget)
            config = ExperimentConfig(strategy=strategy, seed=seed, embargo=embargo)
            source = {"kind": "synthetic-demo", "preset": arguments.preset, "generator_seed": 42}
            if arguments.assert_demo and abs(brier_score([0, 1, 1, 1], [.1, .7, .2, .9]) - .1875) > 1e-15:
                raise LabError(LabStatus.FIT_FAILED, "Demo golden verification failed.", code="demo_assertion")
        else:
            config = ExperimentConfig(strategy=arguments.strategy, seed=arguments.seed, embargo=arguments.embargo,
                                      require_group_disjoint=arguments.require_group_disjoint,
                                      require_chronological_order=arguments.require_chronological_order,
                                      fp_cost=arguments.fp_cost, fn_cost=arguments.fn_cost, n_bins=arguments.bins)
            dataset, digest = read_dataset(arguments.input, budget)
            source = {"kind": "input", "bounded_input_sha256": digest}
        result = run_experiment(dataset, config=config, budget=budget)
        outcome = publish_artifacts(make_artifacts(dataset, result, source, budget), arguments.output)
        exit_code = 0
    except LabError as exc:
        messages = {
            LabStatus.INVALID_INPUT: "Input or options violate the supported schema.",
            LabStatus.INVALID_SPLIT: "Dataset cannot satisfy the requested split.",
            LabStatus.FIT_FAILED: "A fitting, finite-result or verification check failed.",
            LabStatus.UNKNOWN_BUDGET: "Work or complete output exceeds the caller budget.",
            LabStatus.INCOMPLETE: "An operational dependency or native execution failed.",
        }
        outcome = {"status": str(exc.status), "code": exc.code, "message": messages.get(exc.status, "Experiment did not complete.")}
        exit_code = 3 if exc.status == LabStatus.INCOMPLETE else 2
    except AdapterError as exc:
        outcome = exc.outcome
        exit_code = 2 if outcome["status"] == "INVALID_INPUT" else 3
    except OSError as exc:
        outcome, exit_code = {"status": "IO_FAILED", "code": "local_io", "message": "Local input/output operation failed.", "errno": exc.errno, "committed": False}, 3
    except Exception:
        outcome, exit_code = {"status": "INCOMPLETE", "code": "runtime_failure", "message": "An operational dependency or native execution failed."}, 3
    try:
        sys.stdout.write(_json_bytes(outcome).decode("utf-8"))
    except (OSError, ValueError):
        _silence_failed_stream(sys.stdout)
        _flush_standard_streams()
        return 3  # Output can be truncated; never emit a second outcome frame.
    return exit_code if _flush_standard_streams() else 3
