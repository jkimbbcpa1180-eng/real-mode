# Redactions and changes

The code here was cleaned up before release. Original values are not repeated in this file.

## Personal location replaced with a labelled example

A real personal location (coordinates, place name, and capture time) was replaced with Seoul City Hall (37.5665, 126.9780), clearly marked `EXAMPLE`, in:

- `code/realmode_core_v61_1.py`: the `REAL_MODE_LAST_SHARED_LOCATION` constant (about lines 2010-2018; not used elsewhere in the code)
- `code/realmode_console.py`: the `--demo` location (line 146)
- `code/test_realmode_console.py`: the test pin (line 8)
- `code/docs/REAL_MODE_UPGRADE_README.md`: the snapshot example (lines 63-65)

## Secrets

No API keys, tokens, passwords, email addresses, or phone numbers were found in the released files.

## Other changes

- `code/real_mode_developer_integration.py` was an unnamed text paste and was saved under this name. Its import now tries `realmode_core_v61_1` before `realmode_core_v61` (v61 is not included). A header note explains that `--selftest-full` fails on v61.1.
- `README.md`: added an "Experimental code" list under Tools.
- `code/gp_mobility_web_engine.py`: removed one city anchor that pointed to a personal travel location. The remaining cities work unchanged. A case-insensitive search of this package for that city's name finds nothing.
- `code/hk_web_runner.py`: rewritten for safety (no shell, allowlisted https fetch, honest User-Agent, offline `--demo`). Tests are in `code/test_hk_web_runner.py`; details are in `SECURITY.md`.
- Older versions, duplicate pastes, and some personal modules were left out.

- `code/hk_session_kernel_v1_1.py`: Korea anchor replaced with an example location (Seoul City Hall); its plus-code tests updated to match.

## `code/realmode_legacy_fixed/`

No keys, contact details, home location, or names were found. The following personal or sensitive items were removed from this public copy. The original wording is not repeated here.

- `ledger.py`: removed four personal wellness helper templates and the docstring line about them.
- `master.py`: removed a personal calendar-rule function pair, and removed two preset religion rankings from `load_real_mode_environment` (the `ReligionCompare` class remains and starts empty). Dropped the now-unused `date` import.
- `test_master.py`: removed the assertions for the deleted calendar rule; the religion count assertion now expects 0. Dropped the unused `date` import.
- `README.md` (package): replaced the paragraph describing the removed templates and calendar rule with a one-line note that they were removed.

`code/realmode_superkernel.py` was copied unchanged.

