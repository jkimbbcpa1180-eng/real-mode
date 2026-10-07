#!/usr/bin/env python3
"""Hong Kong web helpers with a small, allowlisted network action.

Helpers (no network):
- normalize_hk_url: encode internationalized domain names to Punycode.
- decode_hk_bytes: decode UTF-8 / Big5-HKSCS / GB18030 text, NFKC-normalized.
- parse_hk_timestamp: parse dates and pin them to Asia/Hong_Kong (UTC+8).

Network (only when you ask for it):
- --action status: HEAD request to https://data.gov.hk/ and print status/headers.
- --action fetch --url URL: GET an https URL on an allowlisted host, with a
  timeout and a response-size cap.

Safety rules: no shell, no subprocess, no arbitrary commands. URLs must be
https, on an allowlisted host, on the default port, with no embedded
credentials; redirects are re-checked against the same rules. Requests send an
honest User-Agent. --demo runs fully offline.

Examples:
    python3 hk_web_runner.py --demo
    python3 hk_web_runner.py --demo --json
    python3 hk_web_runner.py --action status
    python3 hk_web_runner.py --action fetch --url https://data.gov.hk/en/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from typing import Callable, Dict, Optional
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

try:  # IDNA 2008 if the third-party package is installed
    import idna as _idna

    def _encode_host(hostname: str) -> str:
        return _idna.encode(hostname).decode("ascii")
except ImportError:  # stdlib fallback (IDNA 2003); fine for common names
    _idna = None

    def _encode_host(hostname: str) -> str:
        return hostname.encode("idna").decode("ascii")

VERSION = "2.0"
HK_TZ = ZoneInfo("Asia/Hong_Kong")

# Honest identification instead of pretending to be a desktop Chrome browser.
USER_AGENT = f"hk_web_runner/{VERSION} (Real Mode public-domain example; Python urllib)"
DEFAULT_HK_HEADERS: Dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "zh-HK,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ALLOWED_HOSTS = frozenset({"data.gov.hk"})
STATUS_URL = "https://data.gov.hk/"
DEFAULT_TIMEOUT_S = 10.0
MAX_TIMEOUT_S = 30.0
DEFAULT_MAX_BYTES = 1_000_000
MAX_MAX_BYTES = 5_000_000
REPORTED_HEADERS = ("Content-Type", "Content-Length", "Date", "Last-Modified", "Server")


class UnsafeURLError(ValueError):
    """Raised when a URL fails the https/allowlist rules."""


class ResponseTooLargeError(ValueError):
    """Raised when a response body exceeds the size cap."""


# --------------------------------------------------------------------------
# Helpers (unchanged behaviour)
# --------------------------------------------------------------------------

def normalize_hk_url(raw_url: str) -> str:
    """Encode internationalized Chinese domain names to Punycode."""
    parts = urlsplit(raw_url)
    hostname = parts.netloc.split(":")[0]
    port = f":{parts.netloc.split(':')[1]}" if ":" in parts.netloc else ""
    encoded_host = _encode_host(hostname)
    return urlunsplit(
        (parts.scheme, f"{encoded_host}{port}", parts.path, parts.query, parts.fragment)
    )


def decode_hk_bytes(data: bytes) -> str:
    """Decode incoming payloads with HKSCS fallback and NFKC normalization."""
    for encoding in ("utf-8", "big5hkscs", "gb18030"):
        try:
            decoded = data.decode(encoding)
            return unicodedata.normalize("NFKC", decoded)
        except (UnicodeDecodeError, LookupError):
            continue
    return unicodedata.normalize("NFKC", data.decode("utf-8", errors="replace"))


def parse_hk_timestamp(raw_text: str) -> datetime:
    """Extract and pin dates to Asia/Hong_Kong (UTC+8)."""
    match = re.search(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*"
        r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?",
        raw_text,
    )
    if match:
        y, m, d, hh, mm = map(int, match.groups()[:5])
        ss = int(match.group(6)) if match.group(6) else 0
        return datetime(y, m, d, hh, mm, ss, tzinfo=HK_TZ)

    dt = datetime.fromisoformat(raw_text.strip())
    return dt.replace(tzinfo=HK_TZ) if dt.tzinfo is None else dt.astimezone(HK_TZ)


# --------------------------------------------------------------------------
# URL validation and allowlisted fetching
# --------------------------------------------------------------------------

def validate_url(raw_url: str, allowed_hosts=ALLOWED_HOSTS) -> str:
    """Return a normalized URL, or raise UnsafeURLError.

    Rules: str only; no whitespace/control characters; https scheme; no
    user:password@; default port only; host (after Punycode) exactly in the
    allowlist.
    """
    if not isinstance(raw_url, str) or not raw_url:
        raise UnsafeURLError("URL must be a non-empty string")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in raw_url):
        raise UnsafeURLError("URL contains whitespace or control characters")
    parts = urlsplit(raw_url)
    if parts.scheme.lower() != "https":
        raise UnsafeURLError("only https:// URLs are allowed")
    if "@" in parts.netloc:
        raise UnsafeURLError("credentials in URLs are not allowed")
    try:
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError("invalid port") from exc
    if port not in (None, 443):
        raise UnsafeURLError("only the default https port is allowed")
    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise UnsafeURLError("URL has no host")
    try:
        host = _encode_host(host).lower()
    except Exception as exc:  # idna raises several error types
        raise UnsafeURLError("host is not a valid domain name") from exc
    if host not in allowed_hosts:
        raise UnsafeURLError(f"host not allowlisted: {host}")
    return urlunsplit(("https", host, parts.path or "/", parts.query, ""))


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only if the new URL passes validate_url."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_open(request: urllib.request.Request, timeout: float):
    opener = urllib.request.build_opener(_CheckedRedirectHandler())
    return opener.open(request, timeout=timeout)


def fetch(
    url: str,
    *,
    method: str = "GET",
    timeout: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    opener: Optional[Callable] = None,
) -> Dict[str, object]:
    """Fetch an allowlisted https URL with a timeout and a size cap."""
    safe_url = validate_url(url)
    if method not in ("GET", "HEAD"):
        raise ValueError("method must be GET or HEAD")
    if not 0 < float(timeout) <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout must be in (0, {MAX_TIMEOUT_S}] seconds")
    if not 0 < int(max_bytes) <= MAX_MAX_BYTES:
        raise ValueError(f"max_bytes must be in (0, {MAX_MAX_BYTES}]")
    request = urllib.request.Request(safe_url, headers=dict(DEFAULT_HK_HEADERS), method=method)
    open_fn = opener or _default_open
    with open_fn(request, float(timeout)) as response:
        final_url = response.geturl() if hasattr(response, "geturl") else safe_url
        validate_url(final_url)
        body = b""
        if method == "GET":
            body = response.read(int(max_bytes) + 1)
            if len(body) > int(max_bytes):
                raise ResponseTooLargeError(f"response exceeded {int(max_bytes)} bytes")
        headers = {name: response.headers.get(name) for name in REPORTED_HEADERS
                   if response.headers.get(name) is not None}
        return {
            "url": final_url,
            "status": getattr(response, "status", None),
            "headers": headers,
            "bytes": len(body),
            "text": decode_hk_bytes(body) if body else "",
        }


# Named actions only; there is no way to run an arbitrary command.
def action_status(timeout: float, max_bytes: int, url: Optional[str] = None,
                  opener: Optional[Callable] = None) -> Dict[str, object]:
    if url is not None:
        raise ValueError("--action status does not take --url")
    result = fetch(STATUS_URL, method="HEAD", timeout=timeout, max_bytes=max_bytes, opener=opener)
    result.pop("text", None)
    return result


def action_fetch(timeout: float, max_bytes: int, url: Optional[str] = None,
                 opener: Optional[Callable] = None) -> Dict[str, object]:
    if not url:
        raise ValueError("--action fetch requires --url")
    return fetch(url, method="GET", timeout=timeout, max_bytes=max_bytes, opener=opener)


ACTIONS: Dict[str, Callable[..., Dict[str, object]]] = {
    "status": action_status,
    "fetch": action_fetch,
}


# --------------------------------------------------------------------------
# Offline demo and CLI
# --------------------------------------------------------------------------

def demo_report() -> Dict[str, object]:
    """Exercise the helpers and URL rules without any network access."""
    checks = {}
    for candidate in ("https://data.gov.hk/en/", "http://data.gov.hk/",
                      "https://example.com/", "https://data.gov.hk.evil.example/",
                      "https://user@data.gov.hk/", "https://data.gov.hk:8443/",
                      "https://data.gov.hk/; rm -rf ~", "$(id)"):
        try:
            checks[candidate] = {"allowed": True, "normalized": validate_url(candidate)}
        except UnsafeURLError as exc:
            checks[candidate] = {"allowed": False, "reason": str(exc)}
    return {
        "version": VERSION,
        "network_used": False,
        "now_hkt": datetime.now(HK_TZ).isoformat(timespec="seconds"),
        "idna_backend": "idna package" if _idna is not None else "stdlib idna codec (IDNA 2003)",
        "normalize_hk_url": normalize_hk_url("https://例子.香港/路徑"),
        "decode_hk_bytes": decode_hk_bytes("香港天文台".encode("big5hkscs")),
        "parse_hk_timestamp": parse_hk_timestamp("2026年9月28日 14:05").isoformat(),
        "user_agent": USER_AGENT,
        "allowed_hosts": sorted(ALLOWED_HOSTS),
        "url_checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hong Kong web helpers with an allowlisted, timeout-limited fetch. "
                    "No shell commands are run.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Offline demo; makes no network calls.")
    mode.add_argument("--action", choices=sorted(ACTIONS),
                      help="Named network action (live request to an allowlisted host).")
    parser.add_argument("--url", help="https URL on an allowlisted host (for --action fetch).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help=f"Seconds, max {MAX_TIMEOUT_S:g} (default {DEFAULT_TIMEOUT_S:g}).")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help=f"Response size cap, max {MAX_MAX_BYTES} (default {DEFAULT_MAX_BYTES}).")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.demo:
            report = demo_report()
        else:
            report = ACTIONS[args.action](args.timeout, args.max_bytes, url=args.url)
    except (UnsafeURLError, ResponseTooLargeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError) as exc:
        print(f"network error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
