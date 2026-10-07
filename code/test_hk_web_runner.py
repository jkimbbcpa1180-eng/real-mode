import contextlib
import io
import json
import subprocess
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import hk_web_runner as hk

HERE = Path(__file__).resolve().parent
INJECTIONS = ["; rm -rf ~", "$(id)", "`id`", "https://data.gov.hk/ && id",
              "https://data.gov.hk/; rm -rf ~", "data.gov.hk | sh", "file:///etc/passwd"]


class FakeResponse:
    def __init__(self, body=b"ok", url="https://data.gov.hk/", status=200):
        self._body, self._url, self.status = body, url, status
        self.headers = {"Content-Type": "text/plain"}
    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]
    def geturl(self):
        return self._url
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = hk.main(argv)
        except SystemExit as exc:  # argparse errors
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class NoExecutionMixin:
    """Patch every way a command or request could escape during a test."""
    def setUp(self):
        self.patches = [mock.patch.object(subprocess, name, side_effect=AssertionError(f"subprocess.{name} called"))
                        for name in ("Popen", "run", "call", "check_call", "check_output")]
        self.patches += [mock.patch.object(os, "system", side_effect=AssertionError("os.system called"))]
        self.urlopen = mock.patch("urllib.request.urlopen", side_effect=AssertionError("urlopen called"))
        self.opener_open = mock.patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network called"))
        for p in self.patches + [self.urlopen, self.opener_open]:
            p.start()
            self.addCleanup(p.stop)


class InjectionTests(NoExecutionMixin, unittest.TestCase):
    def test_injection_urls_rejected(self):
        for value in INJECTIONS:
            with self.subTest(value=value), self.assertRaises(hk.UnsafeURLError):
                hk.validate_url(value)

    def test_injection_via_fetch_never_opens(self):
        opener = mock.Mock(side_effect=AssertionError("opener called"))
        for value in INJECTIONS:
            with self.subTest(value=value), self.assertRaises(hk.UnsafeURLError):
                hk.fetch(value, opener=opener)
        opener.assert_not_called()

    def test_injection_via_cli_rejected_without_execution(self):
        for value in INJECTIONS:
            with self.subTest(value=value):
                code, out, _ = run_main([value])            # bare positional (old style)
                self.assertEqual(code, 2)
                code, out, _ = run_main(["--action", "fetch", "--url", value])
                self.assertEqual(code, 2)
                code, out, _ = run_main(["--action", value])
                self.assertEqual(code, 2)

    def test_no_shell_in_source(self):
        import ast
        tree = ast.parse((HERE / "hk_web_runner.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.keyword):
                self.assertNotEqual(node.arg, "shell", "shell= keyword found")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn("/bin/bash", node.value)
        for banned in ("subprocess", "os", "pty", "shlex"):
            self.assertNotIn(banned, imported, f"{banned} is imported")

    def test_no_run_bash_wrapper(self):
        self.assertFalse(hasattr(hk, "run_bash"))
        self.assertEqual(set(hk.ACTIONS), {"status", "fetch"})


class URLRuleTests(NoExecutionMixin, unittest.TestCase):
    def test_http_rejected(self):
        with self.assertRaises(hk.UnsafeURLError):
            hk.validate_url("http://data.gov.hk/")

    def test_non_allowlisted_hosts_rejected(self):
        for value in ("https://example.com/", "https://data.gov.hk.evil.example/",
                      "https://evildata.gov.hk/", "https://user:pw@data.gov.hk/",
                      "https://data.gov.hk:8443/", "https://127.0.0.1/"):
            with self.subTest(value=value), self.assertRaises(hk.UnsafeURLError):
                hk.validate_url(value)

    def test_allowlisted_https_accepted(self):
        self.assertEqual(hk.validate_url("https://DATA.gov.hk/en/"), "https://data.gov.hk/en/")

    def test_cli_http_rejected(self):
        code, _, err = run_main(["--action", "fetch", "--url", "http://data.gov.hk/"])
        self.assertEqual(code, 2)
        self.assertIn("https", err)


class FetchTests(unittest.TestCase):
    def test_fetch_uses_timeout_and_honest_agent(self):
        seen = {}
        def opener(request, timeout):
            seen["ua"], seen["timeout"], seen["url"] = request.get_header("User-agent"), timeout, request.full_url
            return FakeResponse(b"hello")
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError):
            result = hk.fetch("https://data.gov.hk/x", opener=opener, timeout=5)
        self.assertEqual(result["text"], "hello")
        self.assertEqual(seen["timeout"], 5.0)
        self.assertNotIn("Mozilla", seen["ua"])
        self.assertIn("hk_web_runner", seen["ua"])

    def test_size_cap(self):
        with self.assertRaises(hk.ResponseTooLargeError):
            hk.fetch("https://data.gov.hk/", opener=lambda r, t: FakeResponse(b"x" * 11), max_bytes=10)

    def test_redirect_off_allowlist_rejected(self):
        with self.assertRaises(hk.UnsafeURLError):
            hk.fetch("https://data.gov.hk/", opener=lambda r, t: FakeResponse(url="https://evil.example/"))
        handler = hk._CheckedRedirectHandler()
        with self.assertRaises(hk.UnsafeURLError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://data.gov.hk/")

    def test_timeout_and_cap_bounds(self):
        for kwargs in ({"timeout": 0}, {"timeout": 999}, {"max_bytes": 0}, {"max_bytes": 10**9}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                hk.fetch("https://data.gov.hk/", opener=lambda r, t: FakeResponse(), **kwargs)

    def test_status_action_is_head_to_fixed_url(self):
        seen = {}
        def opener(request, timeout):
            seen["method"], seen["url"] = request.get_method(), request.full_url
            return FakeResponse(b"")
        hk.action_status(5, 1000, opener=opener)
        self.assertEqual((seen["method"], seen["url"]), ("HEAD", "https://data.gov.hk/"))


class HelperBehaviourTests(unittest.TestCase):
    def test_helpers_unchanged(self):
        self.assertEqual(hk.normalize_hk_url("https://例子.香港/path"), "https://xn--fsqu00a.xn--j6w193g/path")
        self.assertEqual(hk.decode_hk_bytes("香港".encode("big5hkscs")), "香港")
        self.assertEqual(hk.parse_hk_timestamp("2026年9月28日 14:05").isoformat(), "2026-09-28T14:05:00+08:00")
        self.assertEqual(hk.parse_hk_timestamp("2026-09-28T14:05:00").utcoffset().total_seconds(), 8 * 3600)


class DemoTests(NoExecutionMixin, unittest.TestCase):
    def test_demo_offline(self):
        code, out, _ = run_main(["--demo"])
        self.assertEqual(code, 0)
        self.assertIn("network_used: False", out)

    def test_demo_json_offline(self):
        code, out, _ = run_main(["--demo", "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertFalse(report["network_used"])
        self.assertFalse(report["url_checks"]["$(id)"]["allowed"])


class DemoSubprocessTest(unittest.TestCase):
    def test_demo_runs_as_script(self):
        # The test itself starts Python to prove the CLI works standalone.
        result = subprocess.run([sys.executable, str(HERE / "hk_web_runner.py"), "--demo", "--json"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["network_used"])


if __name__ == "__main__":
    unittest.main()
