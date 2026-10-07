# Real Mode legacy repaired release 1.0

This release repairs the code pasted on October 1, 2026. It is a separate
package, not a replacement for the attached v61.1 upgrade or source archive.
Python 3.9+ and standard library only; timezone data must provide Asia/Seoul.

From the directory containing the package folder:

```sh
python -m realmode_legacy_fixed --demo --json
python -m unittest realmode_legacy_fixed.test_ledger realmode_legacy_fixed.test_master -v
```

`ledger.py` contains one copy of TP/Mirror/Entry/Streak/HabitSnapshot and
RealModeFull. `master.py` preserves unique symbolic engines, tone templates,
axioms, event journals, buffers, and OS adapters. No initialization occurs
on import. Demo outcomes are synthetic and cannot verify real-world claims.

## Changes

- Removed two exact repeated ledger blocks and repeated imports; fixed
  indentation and the placement of future imports.
- Replaced unavailable module imports with local adapters. These adapters
  provide a journal and entry interface; they do not reconstruct missing code.
- Atomic persistence creates parent directories, validates before import,
  rolls back memory on write failure, and leaves malformed files untouched.
- Returned records are copies. Use engine `attach_mirror`/`tag_entry` methods
  to persist changes; mutating a returned Entry changes only that copy.
- Numeric inputs must be finite. Invalid probabilities raise an error rather
  than silently clamp. TP.bump deliberately saturates; fade requires [0,1].
- Dates use aware UTC timestamps and configurable local streak days (default
  Asia/Seoul). Old naive timestamps are interpreted as UTC because their
  original timezone is unknown. Missing timestamps get an import-time value.
- Missed streak days reset to one; same-day ticks do nothing. Existing archives
  with inconsistent streak counts need explicit reconciliation before import.
- TP combination defaults to mean. Legacy `bayes` warns and aliases a stable
  heuristic odds pool; it has no evidence likelihood model or calibration.
- Default URK uses the constraint and samples Bernoulli(score). Explicit legacy
  behavior remains available; its biased sampler does not sample P(outcome)=TP.

## Retained for review

Named historical,
religious, relationship, recursion, and temporal outputs are user-authored
heuristic scores. They do not measure truth, consciousness, time travel, or
historical/religious validity. GPT4Soul returns canned text; it does not restore
GPT-4. ShadowBuffer provides no isolation or stealth.

Personal wellness templates, a personal calendar rule, and preset religion
rankings from the original were removed from this public copy.

This package does not provide GPS, weather, internet telemetry, ChatGPT memory,
background agents, or calibrated probabilities. Storage assumes one writer;
there is no process lock or encryption. Import_json preserves its original
entries-only merge behavior and skips IDs already present after full validation.
