# Security notes

## What was wrong with `code/hk_web_runner.py` (before this release)

- **It ran any command it was given.** Every command-line argument was joined into one string and run through `bash` (`subprocess.Popen(..., shell=True, executable="/bin/bash")`). Anything like `; rm -rf ~` or `$(id)` would actually run, with your user's permissions. Any program or person that could pass arguments to it could run anything.
- **Callers could change the environment.** An `extra_env` option let callers set variables such as `PATH` for the shell.
- **No limits.** There was no timeout on the command and no limit on output size.
- **It pretended to be a browser.** Requests were meant to carry a fake desktop Chrome User-Agent.
- **Its default run went online.** With no arguments it ran `curl` against data.gov.hk.

## What changed

- **No shell, no subprocess.** The run-anything wrapper (`run_bash`) is gone. The script doesn't import `subprocess` or `os` at all.
- **Named actions only.** `--action status` sends a HEAD request to `https://data.gov.hk/`. `--action fetch --url URL` sends a GET request. Unknown arguments are rejected.
- **URL checks.** Every URL, including any redirect target, must:
  - use `https`
  - be on an allowlisted host (currently only `data.gov.hk`)
  - use the default port
  - contain no `user:password@`
  - contain no spaces or control characters
- **Limits.** Requests time out (default 10 s, max 30 s). Responses are capped at 1 MB by default (max 5 MB).
- **Honest User-Agent.** Requests identify themselves as `hk_web_runner/2.0 (Real Mode public-domain example; Python urllib)`.
- **Offline demo.** `--demo` (and `--demo --json`) makes no network calls.
- **Tests.** `code/test_hk_web_runner.py` checks the following, with subprocess and network calls mocked so nothing runs or connects:
  - injection strings are rejected and never run
  - `http://` and non-allowlisted hosts are rejected
  - the size cap and timeouts are enforced
  - `--demo` works offline
- The text helpers (`normalize_hk_url`, `decode_hk_bytes`, `parse_hk_timestamp`) behave as before.

## What you should still watch for

- **Read code before you run it,** especially code from strangers, and including this package. Public-domain code comes with no warranty.
- **Network calls are live.** `hk_web_runner.py --action ...`, `gp_mobility_web_engine.py` and `gp_seoul_ip_engine.py --deploy` contact real services (data.gov.hk, Open-Meteo, Google News). Check their terms and rate limits.
- **Some scripts write files to the folder you run them in** (ledgers, reports, logs). Run them in a scratch folder.
- **Allowlist changes are on you.** If you add hosts to `ALLOWED_HOSTS`, you are trusting those sites.
- **The other scripts weren't hardened.** They don't run shell commands, but they haven't been through the same security review.
