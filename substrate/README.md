<!-- SPDX-License-Identifier: CC0-1.0 -->
# Substrate v8

**Credit: humanity.** Public domain under CC0-1.0. This is a reference model of a *proposed* political economy, not a deployed system.

`substrate_v8.py` is a corrected version of Substrate v7. It uses only the Python standard library.

```
python substrate_v8.py --test          # built-in suite (v7 checks, updated where noted)
python -m unittest test_substrate_v8   # 36 regression tests
python substrate_v8.py --demo          # see demo_output_v8.txt
python substrate_v8.py --pod-report    # pod model as JSON
```

`v8.patch` is a `diff -u` from `publish/substrate_v7.py` to `substrate_v8.py`, and it applies byte-exact. The published v7 copy has had one change: the demo's real place name was replaced with "River Watershed". Nothing else in v7 was changed.

## What changed

| # | v7 problem | v8 fix |
|---|---|---|
| 1 | `['p0','p0']` counted as 2 vouches, and the demo setup relied on that | Vouches are deduplicated and self-vouching is refused. The first members can only join together through `register_genesis()` as a founding quorum of at least 3; a single person can no longer bootstrap the registry. |
| 2 | The `hour` for rate limits came from the caller, and claimers declared their own neighbors | `LedgerClock` is a monotonic, system-owned clock; the `hour` argument is ignored. Rate limits use a rolling window. Neighbors come from `NeighborGraph`, where an edge needs both registered pods to agree. There are caps per (claimer, attestor) pair and per exact attestor set in each period. |
| 3 | "Accumulation is bounded" was false: demurrage hit only idle pods, and minting had no cap | Every pod pays 10%/month on the part of its balance above 500 TC, and idle pods still decay 4% as before. Minting is capped per person per 30 days (600 TC by default), with an optional global cap. The proven bound is `B* = T + I(1-r)/r`, and a test checks it under maximum activity. |
| 4 | "Exit is always possible" had no code behind it | `Polity.exit()` leaves the polity, unbinds the pod, and pays out the full balance as a portable claim with no fee. Personhood is kept unless the person asks for deletion. Deletion erases the registry entry and crypto-shreds the person's log pseudonyms. |
| 5 | Child labor was "major" in one place and "minor" in another; any appeal text was granted automatically, and the next recompute silently undid it | Severity now comes from one place, the prohibition list. An appeal stays pending until 3 independent registered reviewers approve it; reviewers from the appealing polity are refused. An approved appeal vacates one specific violation record, so later recomputes respect it. The ladder is labeled **fictional/proposed: no such UN process exists**. |
| 6 | Auditor "signatures" were hashes of public fields, so anyone could forge them; a single slash could remove an auditor | Certificates are signed with HMAC-SHA256 using the auditor's secret key, and the registry verifies them. An auditor is removed only if (a) there are at least 20 certificates and the Wilson 95% lower bound of the slash rate is above 5%, or (b) the auditor has 3 slashes. |
| 7 | `recognize_personhood` could overwrite an existing person in registry B | It now refuses when the person ID already exists in B. The person must also present the biometric again, because each registry uses its own pepper. |
| 8 | Raw sensor variance raised the mint, so noisy or fake sensors earned more; the biometric salt was fixed and public; the log published raw care hours per person; sponsors chose their own auditors; Treasury approvals were unchecked strings; cross-polity fees vanished | Provenance is now scored on quality: variance outside a plausible band lowers the score. Biometric hashes are HMACs with a secret pepper per registry. The log publishes pseudonyms and hours aggregated per (day, domain), and cells with fewer than 3 contributors are suppressed. The registry assigns auditors at random and excludes declared conflicts. Treasury spending needs at least 3 distinct, active, registered members. Cross-polity fees go to `Federation.fee_treasury`. |
| 9 | The pod was listed at `cost_usd: 0.00` with a fixed 2,800 kcal/day | `PodModel` is a parameterized model (see below), and `sustainable` is now computed from energy, water and food. |
| + | — | `plan_calories()` is a goal-weight planner. It uses the Mifflin-St Jeor BMR times an activity factor, then applies a 300–500 kcal/day deficit or surplus (at most about 0.5 kg/week). Floors are 1,200 kcal (women) and 1,500 kcal (men), and goals with a BMI under 18.5 are refused. It also estimates weeks to goal. The result becomes the pod's calorie demand. **This is general guidance, not medical advice.** |

Changes to the v7 built-in tests are marked `CHANGED` in `run_unit_tests()`, with the reason for each:
- The max-provenance input now uses an in-band variance.
- The demurrage expectation is 914 instead of 960, because of the new holding demurrage.
- The registry is bootstrapped through genesis.
- Neighbors come from the graph.
- 2 slashes on 5 certificates no longer remove an auditor (minimum-sample rule).
- Federation requires the biometric to be presented again.

## Pod model: results (default pod)

Default pod: 2 people at 2,500 kcal each, 10 m² of LED-lit canopy (250 µmol/m²/s, 16 h/day, 2.7 µmol/J), 6 kWp of PV, a 15 kWh battery, 60 m² of roof.

| Output | Value |
|---|---|
| Indoor food (leafy greens) | **153 kcal/day = 3.0% of 5,000 kcal** (the physics ceiling for that light is 2,058) |
| Grow electricity | 17.8 kWh/day, which is about **116 kWh per 1,000 kcal** |
| Load vs PV | 25.8 kWh/day against 19.6 kWh/day annual mean and 11.4 kWh/day in December, so December needs **13.6 kWp** |
| Water balance | −24 L/day (not closed) |
| Capex | about USD 28,000 (low–high range 17,400–43,400) |
| Annualized cost | **about USD 1,380 per person per year** (range 850–2,140); with 13.6 kWp of PV it is about USD 1,600 |
| Staple crop instead of greens | 3.8% of calories (Bugbee-derived) or 7.5% (optimistic assumption) |
| Full diet indoors, best case | about **237 kWh/day, roughly 73 kWp of PV** for 2 people |
| Full diet outdoors | about **960 m² of potato** at the Korean national-average yield, for 2 people |
| `sustainable` | **False**: energy, water and food all fall short with the defaults |

Plainly: a household pod can realistically supply **vegetables, not staple calories**. Feeding people from LED light takes about 50–120 kWh per 1,000 kcal. That is an energy and land gap of one to two orders of magnitude compared with a rooftop.

### Sources (all checked 2026-10-10)
- **Solar yield.** PVGIS v5.2 (ERA5, 2005–2020) at 37.5N 127.0E, 1 kWp, 14% loss, horizontal: 3.27 kWh/kWp/day annual mean, 1.90 in December. https://re.jrc.ec.europa.eu/api/v5_2/PVcalc
- **Korean PV yield and cost.** IEA-PVPS Korea 2022: average yield 1,261 kWh/kWp/yr; residential installed cost 1,459.5 KRW/W excluding VAT. https://iea-pvps.org/wp-content/uploads/2024/01/IEA-PVPS-National-Survey-Report-KOREA-2022.pdf
- **Battery cost.** IRENA 2023: battery storage projects USD 273/kWh. https://www.irena.org/Publications/2024/Sep/Renewable-Power-Generation-Costs-in-2023 · NREL ATB 2023 residential: pack $283/kWh plus inverter $183/kWh. https://atb.nrel.gov/electricity/2023/index/residential_battery_storage
- **Plant-factory energy.** Graamans et al. 2018: 247 kWh per kg of lettuce dry weight (about 17.3 kWh/kg fresh). https://doi.org/10.1016/j.agsy.2017.11.003 · A building-energy model finds 6.2–12.0 kWh/kg fresh lettuce. https://par.nsf.gov/servlets/purl/10409935
- **LED efficacy.** Kusuma, Pattison & Bugbee 2020: 2.5–3.0 µmol/J achieved; limits of 3.4 and 4.1 µmol/J. https://doi.org/10.1038/s41438-020-0283-7
- **Crop light-use efficiency.** Bugbee & Salisbury 1988: wheat grain 60 g/m²/day at 150 mol/m²/day. https://ntrs.nasa.gov/citations/20040112090
- **Rice yield.** Statistics Korea 2024: 514 kg milled rice per 10a. https://mods.go.kr/boardDownload.es?bid=11712&list_no=434049&seq=4
- **Potato yield.** FAOSTAT via Our World in Data: Korean potato yield 24.67 t/ha (2024). https://ourworldindata.org/grapher/potato-yields
- **Calories.** USDA FoodData Central, read via a mirror: potato 77, green-leaf lettuce 15, white rice 365 kcal/100 g (FDC 170026, 169249, 169756).
- **Water use.** Ministry of Environment 2023: 303.9 L/person/day, 62.5% of it household, giving about 190 L/person/day for households (derived). https://www.waterjournal.co.kr/news/articleView.html?idxno=79553
- **BMR equation.** Mifflin et al. 1990, Am J Clin Nutr 51:241–247 (standard equation; the citation was not re-fetched).

### Assumptions (not data)
These are listed in `ASSUMPTIONS` inside the code:
- Exchange rate: 1,350 KRW/USD.
- Battery installed cost USD 500–1,000/kWh, central 700.
- Grow-system capex USD 300–1,500/m², central 800 (**no source found**).
- Water system USD 1,500–5,000.
- Component lifetimes and O&M rates.
- Photon-to-calorie yields: 1.06 kcal/mol for greens, derived using an assumed 2.5 µmol/J; 1.32 kcal/mol for staples; 2.6 kcal/mol optimistic.
- HVAC overhead 1.2×.
- 8 kWh/day of non-food electricity per household.
- 1,300 mm/yr of rain with 80% capture.
- 50% greywater reuse.
- Half the daily load falls at night.
- The 7,700 kcal/kg rule of thumb, which overstates long-run weight loss.

## Residual risks (honest)
- **Collusion:** a ring of genuinely registered, mutually linked neighbors can still mint fake activity up to the caps. The caps limit how fast fake minting can happen; they don't detect it.
- **HMAC certificates:** the registry holds the auditors' keys, so it could forge certificates, and outside parties can't verify on their own. That needs public-key signatures such as Ed25519, which requires a non-stdlib library.
- **Biometrics:** exact hashing is a toy. Real biometrics are fuzzy and need a template-protection scheme (out of scope). Erasing a person also frees their biometric, which reopens Sybil risk.
- **Balance bound:** it holds only for a bounded inflow. Sponsor funding is capped per sponsor, but not across sponsors.
- **Plausibility band:** the provenance band (variance 0.05–0.60) is an uncalibrated assumption.
- **Escalation ladder:** the "UN" ladder, recognition and the UNP codes are fictional or proposed.
- **Privacy:** pseudonyms can still be linked across events, and k-anonymity with k=3 is a weak guarantee.
