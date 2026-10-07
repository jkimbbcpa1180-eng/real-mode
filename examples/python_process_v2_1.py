#!/usr/bin/env python3
"""Python Process v2.1 — specification validation and recorded code checks.

Intent -> shape -> necessary math -> implementation -> verification.
No equations or dataclasses are required for a task that does not need them.
This is not a code generator, a proof of correctness, or an agent service.
Specifications can be developed collaboratively with an assistant.
An empirical metric such as Brier score evaluates logged predictions against
recorded outcomes; a forecast probability remains a model estimate.

CLI (Python 3.9+, standard library):
  python python_process_v2_1.py --demo
  python python_process_v2_1.py --spec specification.json --json
  python python_process_v2_1.py --check-file module.py --json
  python python_process_v2_1.py --check-file module.py --run-checks --json
  python python_process_v2_1.py --self-test --json

--check-file compiles without executing the target. --run-checks explicitly
executes the trusted target with --help and --demo; --test-json also runs
--demo --json. The target runs in its own folder, may write files or use the
network. --help must exit 0 and print "usage:" on stdout; --demo must exit 0
with non-empty stdout. JSON checks accept any JSON value with finite numbers.
Each command has a 10-second timeout. Use execution options
only for code you intend to run. Checks cannot establish mathematical accuracy,
scientific validity, security, or source authenticity. Those remain manual.
Exit 0 means requested validation/checks passed, not all release criteria passed.
Exit 1 means validation/check failure; argparse usage errors exit 2.
--self-test runs offline unit tests using temporary files only. Timeouts cover
the direct target process; descendant processes are not isolated or sandboxed.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import tokenize
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field, fields
from io import StringIO
from pathlib import Path
from unittest.mock import patch

VERSION = "2.1"
TIMEOUT_SECONDS = 10
DETAIL_LIMIT = 1000
MANUAL_CHECKS = (
    "Equations match computations and units",
    "Assumptions and domain limits are justified",
    "Behavior matches intended inputs and outputs",
    "Synthetic data and uncertainty are labeled",
    "Source provenance and outcome evidence are valid",
    "Live inputs inspected: field names, sort order, one value hand-checked",
)


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def text_list(value, label):
    if not isinstance(value, list) or any(not nonblank(x) for x in value):
        raise ValueError(f"{label} must be a list of nonblank strings")


@dataclass
class Intent:
    one_sentence: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class Shape:
    dataclasses: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    state: list[str] = field(default_factory=list)
    cli_commands: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)


@dataclass
class Equations:
    equations: list[dict] = field(default_factory=list)
    required: bool = False


@dataclass
class Implementation:
    files: list[str] = field(default_factory=list)
    line_limit: int = 2000  # Review guideline; not an automatic correctness rule.


@dataclass
class Verification:
    checks: list[str] = field(default_factory=lambda: list(MANUAL_CHECKS))


@dataclass
class Process:
    intent: Intent
    shape: Shape = field(default_factory=Shape)
    equations: Equations = field(default_factory=Equations)
    implementation: Implementation = field(default_factory=Implementation)
    verification: Verification = field(default_factory=Verification)

    def validate(self):
        if not nonblank(self.intent.one_sentence):
            raise ValueError("intent.one_sentence cannot be blank")
        for name in ("inputs", "outputs", "constraints"):
            text_list(getattr(self.intent, name), f"intent.{name}")
        if not self.intent.outputs:
            raise ValueError("specify at least one output")
        for name in ("dataclasses", "functions", "state", "cli_commands", "invariants"):
            text_list(getattr(self.shape, name), f"shape.{name}")
        text_list(self.implementation.files, "implementation.files")
        text_list(self.verification.checks, "verification.checks")
        if isinstance(self.implementation.line_limit, bool) or not isinstance(self.implementation.line_limit, int) or self.implementation.line_limit <= 0:
            raise ValueError("line_limit must be a positive integer")
        if not isinstance(self.equations.required, bool):
            raise ValueError("equations.required must be boolean")
        if not isinstance(self.equations.equations, list):
            raise ValueError("equations.equations must be a list")
        if self.equations.required and not self.equations.equations:
            raise ValueError("math is required but no equations were supplied")
        for equation in self.equations.equations:
            if not isinstance(equation, dict) or any(not nonblank(equation.get(k)) for k in ("name", "expression", "units", "domain")):
                raise ValueError("each equation needs name, expression, units and domain")

    def is_ready(self):
        """Specification completeness only; never claims executed verification."""
        try:
            self.validate()
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    def report(self):
        self.validate()
        return {"version": VERSION, "spec_ready": True, "release_ready": None,
                "specification": asdict(self), "verification": [
                    {"name": name, "status": "NOT_RUN"} for name in self.verification.checks]}

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("specification must be a JSON object")
        allowed = {"intent", "shape", "equations", "implementation", "verification"}
        if set(raw) - allowed:
            raise ValueError("unknown specification fields: " + ", ".join(sorted(set(raw) - allowed)))
        if "intent" not in raw:
            raise ValueError("intent is required")
        constructors = {"intent": Intent, "shape": Shape, "equations": Equations,
                        "implementation": Implementation, "verification": Verification}
        values = {}
        for name, constructor in constructors.items():
            if name in raw:
                if not isinstance(raw[name], dict):
                    raise ValueError(f"{name} must be an object")
                unknown = set(raw[name]) - {f.name for f in fields(constructor)}
                if unknown:
                    raise ValueError(f"unknown {name} fields: " + ", ".join(sorted(unknown)))
                try:
                    values[name] = constructor(**raw[name])
                except TypeError as exc:
                    raise ValueError(f"invalid {name} fields: {exc}") from None
        result = cls(**values)
        result.validate()
        return result


def strict_json(text):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON float exceeds finite range: " + value)
        return number
    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float)


def command_check(path, args, name, require_json=False):
    path = Path(path).resolve()
    command = [sys.executable, str(path), *args]
    try:
        result = subprocess.run(command, cwd=path.parent, capture_output=True,
                                text=True, encoding="utf-8", timeout=TIMEOUT_SECONDS)
        if result.returncode != 0:
            return {"name": name, "status": "FAIL", "exit_code": result.returncode,
                    "detail": (result.stderr or result.stdout)[-DETAIL_LIMIT:]}
        if args == ["--help"] and "usage:" not in result.stdout.lower():
            raise ValueError("--help stdout must include usage:")
        if "--demo" in args and not result.stdout.strip():
            raise ValueError("--demo stdout must not be empty")
        if require_json:
            strict_json(result.stdout)
        return {"name": name, "status": "PASS", "exit_code": 0}
    except (subprocess.TimeoutExpired, OSError, UnicodeError, ValueError) as exc:
        return {"name": name, "status": "FAIL", "detail": str(exc)[:DETAIL_LIMIT]}


def check_file(filename, run_checks=False, test_json=False):
    path = Path(filename).resolve(strict=True)
    if not path.is_file():
        raise ValueError("target must be a file")
    with tokenize.open(path) as handle:
        source = handle.read()
    checks = []
    try:
        compile(source, str(path), "exec")
        checks.append({"name": "Python syntax", "status": "PASS"})
    except SyntaxError as exc:
        checks.append({"name": "Python syntax", "status": "FAIL", "detail": str(exc)})
    if run_checks and checks[0]["status"] == "PASS":
        checks += [command_check(path, ["--help"], "--help exits 0 and prints usage"),
                   command_check(path, ["--demo"], "--demo exits 0 and prints output")]
        if test_json:
            checks.append(command_check(path, ["--demo", "--json"], "--demo --json emits JSON", True))
    checks += [{"name": name, "status": "MANUAL_REQUIRED"} for name in MANUAL_CHECKS]
    return {"version": VERSION, "file": str(path), "release_ready": None,
            "requested_checks_passed": not any(c["status"] == "FAIL" for c in checks),
            "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--spec", type=Path)
    mode.add_argument("--check-file", type=Path)
    mode.add_argument("--self-test", action="store_true", help="Run offline regression tests")
    parser.add_argument("--run-checks", action="store_true", help="Execute target --help and --demo")
    parser.add_argument("--test-json", action="store_true", help="Also execute target --demo --json")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report")
    args = parser.parse_args(argv)
    if (args.run_checks or args.test_json) and args.check_file is None:
        parser.error("execution flags require --check-file")
    if args.test_json and not args.run_checks:
        parser.error("--test-json requires --run-checks")
    try:
        if args.self_test:
            report = self_test()
        elif args.demo:
            report = Process(Intent("Print a greeting", outputs=["Greeting on stdout"]),
                Shape(functions=["main"], cli_commands=["--demo", "--help"])).report()
        elif args.spec:
            report = Process.from_dict(strict_json(args.spec.read_text(encoding="utf-8"))).report()
        else:
            report = check_file(args.check_file, args.run_checks, args.test_json)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        else:
            print("🚂 Python Process v" + VERSION)
            print("Specification ready:", report.get("spec_ready", "not assessed"))
            if "tests_run" in report:
                print(f"Self-test: {report['tests_run']} tests, {report['failures']} failures, {report['errors']} errors")
            for check in report.get("checks", report.get("verification", [])):
                print(f"{check['status']}: {check['name']}")
            print("Release readiness: not established; manual review remains.")
        return 0 if report.get("requested_checks_passed", True) else 1
    except (OSError, ValueError, TypeError, UnicodeError, SyntaxError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "version": VERSION}, ensure_ascii=False))
        else:
            print("Error: " + str(exc), file=sys.stderr)
        return 1


class SelfTests(unittest.TestCase):
    def target_check(self, source, args, require_json=False):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.py"
            path.write_text(source, encoding="utf-8")
            return command_check(path, args, "test", require_json)

    def test_whitespace_intent(self):
        self.assertFalse(Process(Intent("  ", outputs=["text"])).is_ready())

    def test_optional_math_and_dataclasses(self):
        self.assertTrue(Process(Intent("Print hello", outputs=["text"])).is_ready())

    def test_required_math(self):
        self.assertFalse(Process(Intent("Calculate", outputs=["score"]), equations=Equations(required=True)).is_ready())

    def test_bad_list_type(self):
        with self.assertRaises(ValueError):
            Process.from_dict({"intent": {"one_sentence": "Print", "outputs": "text"}})

    def test_unknown_nested_field(self):
        with self.assertRaisesRegex(ValueError, "unknown intent fields"):
            Process.from_dict({"intent": {"one_sentence": "Print", "outputs": ["text"], "typo": 1}})

    def test_missing_intent_field(self):
        with self.assertRaisesRegex(ValueError, "invalid intent fields"):
            Process.from_dict({"intent": {"outputs": ["text"]}})

    def test_manual_checks_not_passed(self):
        report = Process(Intent("Print", outputs=["text"])).report()
        self.assertIsNone(report["release_ready"])
        self.assertTrue(all(x["status"] == "NOT_RUN" for x in report["verification"]))

    def test_compile_never_executes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.py"
            path.write_text("raise RuntimeError('executed')", encoding="utf-8")
            self.assertTrue(check_file(path)["requested_checks_passed"])

    def test_syntax_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.py"
            path.write_text("def broken(:", encoding="utf-8")
            self.assertFalse(check_file(path)["requested_checks_passed"])

    def test_encoding_error_is_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.py"
            path.write_text("# coding: nonexistent\nprint(1)", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                status = main(["--check-file", str(path), "--json"])
            self.assertEqual(status, 1)
            self.assertIn("error", strict_json(output.getvalue()))

    def test_help_contract(self):
        self.assertEqual(self.target_check("print('  Usage: target')", ["--help"])["status"], "PASS")
        self.assertEqual(self.target_check("print('hello')", ["--help"])["status"], "FAIL")

    def test_empty_demo_fails(self):
        self.assertEqual(self.target_check("print('   ')", ["--demo"])["status"], "FAIL")

    def test_demo_output_passes(self):
        self.assertEqual(self.target_check("print('demo')", ["--demo"])["status"], "PASS")

    def test_nonzero_exit_fails(self):
        self.assertEqual(self.target_check("raise SystemExit(3)", ["--demo"])["status"], "FAIL")

    def test_json_contract(self):
        self.assertEqual(self.target_check("print('{\"ok\":true}')", ["--demo", "--json"], True)["status"], "PASS")
        self.assertEqual(self.target_check("print('noise before JSON {}')", ["--demo", "--json"], True)["status"], "FAIL")
        for invalid in ("NaN", "Infinity", "1e999"):
            with self.assertRaises(ValueError):
                strict_json(invalid)

    def test_timeout_fails(self):
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("test", TIMEOUT_SECONDS)):
            self.assertEqual(command_check(Path("unused.py"), ["--demo"], "timeout")["status"], "FAIL")

    def test_demo_json_stdout_clean(self):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(["--demo", "--json"])
        self.assertEqual(status, 0)
        self.assertTrue(strict_json(out.getvalue())["spec_ready"])
        self.assertEqual(err.getvalue(), "")


def self_test():
    stream = StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {"version": VERSION, "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "requested_checks_passed": result.wasSuccessful(),
            "release_ready": None, "test_output": stream.getvalue()}


if __name__ == "__main__":
    raise SystemExit(main())
