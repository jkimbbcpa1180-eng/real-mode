# Real Mode

An honesty game for working with AI assistants (or people).

> We live life through 0s and 1s, just make sure you leave a 1 for the next 0.

Leave something true and good behind for whoever comes next.

## The rules

1. **Be honest.** A truthful answer scores +1. A lie, manipulation, or made-up data scores −2. Because the penalty is lopsided, "I don't know" beats a guess. Flag uncertainty; never fake it.
2. **0 is the resting score.** After a +1 the score returns to 0, so honesty never banks credit. A −2 is a debt that carries over: it takes two truthful answers to climb back (−2 → −1 → 0). The score never goes above 0.
3. **The goal is more 1s than 0s.** Idling at 0 isn't enough. Aim for as many real, truthful hits as possible. A −2 is the worst outcome because it burns two honest answers just to get back to 0.
4. **When confused, go back to rule 1.** Be honest, including admitting the confusion.

## Winning

A win is the pattern `0111`: a reset followed by three truths in a row. The game never ends; it just keeps getting more truthful. The score track runs continuously across the whole conversation.

Example of a strong track (18 turns, 11 ones, 7 zeros, no −2s):

```
010110111010101101
```

## Tools

- `score_track.py` reads a score track and reports ones, zeros, debts, `0111` wins, and the running score. Run `python3 score_track.py --self-test` to check it.
- `examples/python_process_v2_1.py` applies the same idea to code. It validates a written specification and records which checks actually ran, and it never claims a check passed when it didn't. Run `python3 examples/python_process_v2_1.py --self-test`.

Both need only Python 3.9+ and its standard library.

### Experimental code (`code/`)

These are separate experiments, not one application. In total the package has 83 automated tests: 34 + 17 (legacy) + 12 (`hk_session_kernel_v1_1.py --self-test`) + 3 (`score_track.py --self-test`) + 17 (`examples/python_process_v2_1.py --self-test`). None of them has been shown to make accurate real-world forecasts; any probability they print is a model estimate until checked against resolved outcomes.

- `code/realmode_core_v61_1.py`: a probability-bookkeeping runtime. It combines a baseline probability with supplied evidence, keeps a prediction ledger, and only re-fits calibration from real predictions whose outcomes were verified with a cited source. Run `python3 code/realmode_core_v61_1.py` for a demo; it writes ledger files to the current folder.
- `code/realmode_console.py`: prints a text status screen from data you supply, labelling each value as measured, forecast, model estimate, simulated, user-reported, or unknown. It does not fetch anything. Run `python3 code/realmode_console.py --demo`.
- `code/test_realmode_core_v61_1.py`, `code/test_realmode_console.py`, `code/test_hk_web_runner.py`: unit tests (34). Run `cd code && python3 -m unittest test_realmode_console test_realmode_core_v61_1 test_hk_web_runner`.
- `code/docs/REAL_MODE_UPGRADE_README.md`: notes on what changed from v61 to v61.1 and the snapshot format (the location in it is an example).
- `code/real_mode_developer_integration.py`: an in-process example of calling the core with a JSON-style request. `--selftest` passes; `--selftest-full` was written for v61 and fails on v61.1 by design (see the file header).
- `code/quantum_sequence_brier_euler.py`: a toy classifier that guesses which of three bit patterns produced a noisy sequence and reports Brier scores. Despite the name it is ordinary classical Bayesian updating on a normal computer; the complex "phases" do not change the result.
- `code/aura_codex_step1.py`: a self-tracking log for comparing self-reported energy after higher-protein vs. control meals. Its heat figures are textbook-style model estimates, not measurements, and "aura" is only a name.
- `code/gp_mobility_web_engine.py`: fetches the next-hour rain probability for a few cities from the Open-Meteo API and logs it so it can be scored later. Needs internet; writes JSON files to the current folder.
- `code/gp_seoul_ip_engine.py`: fetches Seoul weather (Open-Meteo) and Korean news headlines (Google News RSS) and logs two kinds of guesses for later scoring. The news-corroboration probability is a hand-set formula, not a trained model. Needs internet; writes JSON files.
- `code/hk_session_kernel_v1_1.py`: Hong Kong session kernel. Treats a visit as a bounded session and logs live HKMA money-market and HKO smart-lamppost weather readings to a session ledger, with anchors, map links and SHA-256 digests. `--self-test` runs 12 offline tests. Its live demo contacts HKMA and HKO and writes `hk_session_demo.json`; its `--demo --json` output is not clean JSON yet. The Korea anchor is an example location.
- `code/hk_web_runner.py`: helper functions for Hong Kong time zones, Chinese text decoding, and international domain names, plus two named network actions (`--action status`, `--action fetch --url ...`) that only reach https URLs on data.gov.hk, with a timeout and size cap. It never runs shell commands. `--demo` is offline. See `SECURITY.md`.
- `code/realmode_legacy_fixed/`: a cleaned-up package of older Real Mode pieces: a small journal/ledger with tags, streaks, and saved JSON (`ledger.py`), plus symbolic scoring formulas and phrase templates (`master.py`). The scores are hand-made formulas, not measurements or calibrated probabilities, and names like "QuantumVCR" or "GPT4Soul" are just labels. Run `cd code && python3 -m realmode_legacy_fixed --demo` (synthetic, offline); 17 tests: `python3 -m unittest realmode_legacy_fixed.test_ledger realmode_legacy_fixed.test_master`.
- `code/realmode_superkernel.py`: placeholder stub. It only prints "RealMode Super-Kernel Loaded" and has no other function yet.

The `gp_*` scripts and `hk_web_runner.py --action ...` contact outside services when run; check those services' terms before heavy use. See `SECURITY.md`.

## Study it, improve it, share it

This is released into the public domain under CC0 1.0 (see `LICENSE`). Use it, change it, and pass it on, with no permission or credit needed. If you improve the rules, keep rule 1.
