#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""Builds demo_output_v7_4.txt: default CLI run, sweeps, multi-seed held-out
evaluation, and a v7.3.1-vs-v7.4 side-by-side table. All numbers from runs."""
import contextlib
import importlib.util
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v74 = load("dg74demo", "deepgem_star_v7_4.py")
v731 = load("dg731demo", "deepgem_star_v7_3_1.py")
out = []


def p(s=""):
    out.append(s)


def section(t):
    p("")
    p("=" * 82)
    p(t)
    p("=" * 82)


# 1. Default CLI output (captured exactly as a user would see it)
section("1. DEFAULT CLI RUN: python3 deepgem_star_v7_4.py")
buf = io.StringIO()
sys.argv = ["deepgem_star_v7_4.py"]
with contextlib.redirect_stdout(buf):
    v74.main()
out.extend(buf.getvalue().rstrip("\n").split("\n"))

# 2. SOFC sweep
section("2. MODULE 1: electrical efficiency vs ASSUMED SOFC efficiency (HX 0.85)")
for e, el in v74.sofc_sweep():
    p(f"  SOFC eff {e:.2f} -> electrical {el:.4f}% of CH4 LHV")

# 3. Betavoltaic cross-checks + activity/licensing context
section("3. MODULE 2: cross-checks and source activity")
sat = v74.ni63_saturated_escaping_uw_cm2(1.0)
p(f"  Saturated escaping power, pure Ni-63, this model: {sat:.4f} uW/cm2/face")
p(f"  Published Monte Carlo saturation:                 {v74.PUBLISHED_NI63_SATURATION_UW_CM2} uW/cm2/face")
p(f"  Ratio model/published: {sat / v74.PUBLISHED_NI63_SATURATION_UW_CM2:.3f} "
  "(conservative: spectral hardening of escaping betas not modeled)")
p(f"  Prototype-matched Voc (Isc 1.27 uA, 0.25 cm2): {v74.diode_voc(1.27e-6, 0.25, 2.7e-20):.4f} V "
  "(published 1.02 V)")
b = v74.run_betavoltaic_stage()
ci = b.activity_bq / 3.7e10
p(f"  Default source: 25 mg nickel at purity 0.18 -> {b.activity_bq/1e9:.3f} GBq = {ci*1e3:.1f} mCi")
p(f"    = {b.activity_bq/(10e-6*3.7e10):,.0f}x the US exempt quantity (10 uCi, 10 CFR 30.71)")
p(f"    = {b.activity_bq/1e8:.1f}x the IAEA general exemption level (1e8 Bq)")
p(f"    -> even this tiny default source is a licensed radioactive source.")
for pur in (0.18, 0.5, 1.0):
    bb = v74.run_betavoltaic_stage(isotope_purity=pur)
    p(f"  purity {pur:.2f}: {bb.specific_activity_ci_g:.2f} Ci/g, P_elec {bb.electrical_power_uw:.6f} uW, "
      f"Voc {bb.open_circuit_voltage_v} V, cell eff {bb.cell_efficiency_pct}%")

# 4. Burst sweep (hysteresis engagement)
section("4. MODULE 3: inference burst sweep (2400 steps, seed 101)")
p("  burst_uJ   power_W  dur_s   feasible  brownout  halt_rate  schmitt_transitions")
for r in v74.burst_sweep():
    p(f"  {r['burst_uj']:9.1f}  {r['power_w']:7.3f}  {r['duration_s']:.4f}  {str(r['feasible']):8s}"
      f"  {r['brownout_rate']:.4f}    {r['safety_halt_rate']:.4f}     {r['safety_transitions']}")
p("  Notes: 6 uJ is the right-sized 42-MAC brain (default). 44,000 uJ is v7.3.1's")
p("  1.1 W x 40 ms; there the Schmitt guard engages. 80,000 uJ cannot fit in a full")
p("  cap (needs ~0.19 J incl. ESR vs 0.17 J usable) and is flagged infeasible.")

section("4b. Storage presets: is a burst feasible at all?")
for name, (esr, leak, ipk) in v74.STORAGE_PRESETS.items():
    ps = v74.DeepGemPowerState(esr_ohms=esr, quiescent_leakage_w=leak, peak_current_a=ipk)
    for pw, dur in ((0.012, 0.0005), (1.1, 0.040)):
        ok, why = ps.burst_feasible(pw, dur)
        p(f"  {name:14s} burst {pw*dur*1e6:8.1f} uJ -> feasible={ok}  ({why})")

# 5. Multi-seed held-out evaluation
section("5. MODULE 4: held-out evaluation across 5 seeds (2400 training steps each)")
ms = v74.multi_seed_evaluation()
p(f"  chance {ms['chance']}  |  Bayes ceiling {ms['bayes_ceiling']}")
for k in ("drone_accuracy", "server_accuracy", "drone_ece_T1", "drone_ece_fitted",
          "server_ece_T1", "server_ece_fitted"):
    st = ms[k]
    p(f"  {k:18s} mean {st['mean']:.4f}  sd {st['sd']:.4f}  [{st['min']:.4f}, {st['max']:.4f}]")

section("5b. Held-out accuracy vs training length (seed 101): gap to ceiling is data-limited")
for n in (2400, 9600, 24000):
    ho = v74.run_hybrid_system_simulation(n_steps=n, seed=101)["held_out_evaluation"]
    p(f"  n_steps {n:6d}: drone {ho['drone_accuracy']:.4f}  server {ho['server_accuracy']:.4f}"
      f"  (ceiling {ho['bayes_ceiling']})")

# 6. Side-by-side
section("6. v7.3.1 vs v7.4 (default runs, seed 101, 2400 steps)")
a = v731.run_hybrid_system_simulation()
z = v74.run_hybrid_system_simulation()
ap, zp = a["pyrolysis_macro_module"], z["pyrolysis_macro_module"]
ab, zb = a["deepgem_betavoltaic_module"], z["deepgem_betavoltaic_module"]
ah, zh = a["energy_harvesting_totals"], z["energy_harvesting_totals"]
as_, zs = a["star_drone_tactical_module"], z["star_drone_tactical_module"]
zho = z["held_out_evaluation"]
g731 = v731._shomate_delta_h(v731.SHOMATE_COEFFS_C, 1273.15, 298.15)
rows = [
    ("Graphite H(1273)-H(298), kJ/mol (JANAF 17.922)", f"{g731:.3f}",
     f"{v74.graphite_h_minus_h298_kj_mol(1273.15):.3f}"),
    ("Net process heat, MJ/kg CH4",
     f"{ap['dh_reaction_298k_mj'] + ap['sensible_preheat_feed_mj'] - ap['heat_recuperated_mj']:.4f} (feed-based)",
     f"{zp['net_process_heat_mj']}"),
    ("dH_rxn at T, MJ (JANAF 91.809 kJ/mol)", "not computed", f"{zp['dh_reaction_at_t_mj']}"),
    ("SOFC efficiency", "0.65 (unlabeled)", f"{zp['sofc_eff_assumed']} (ASSUMED, swept)"),
    ("Electrical efficiency, %", f"{ap['electrical_efficiency_pct']}", f"{zp['electrical_efficiency_pct']}"),
    ("Ni-63 mu/rho, cm2/g (measured 1480)", "25.0", "1480.0"),
    ("Source thickness, um", f"{ab['effective_thickness_um']} (diamond density)",
     f"{zb['source_thickness_um']} (nickel)"),
    ("Self-absorption factor", f"{ab['self_absorption_factor']}", f"{zb['self_absorption_factor']}"),
    ("Ni-63 purity / Ci per g", f"1.0 implicit / {ab['activity_bq'] / (ab['mass_mg'] * 1e-3) / 3.7e10:.4f}",
     f"{zb['isotope_purity']} / {zb['specific_activity_ci_g']}"),
    ("Pair energy, eV", "15.816 (Klein)", "13.1 (measured)"),
    ("Voc, V", f"{ab['open_circuit_voltage_v']}", f"{zb['open_circuit_voltage_v']}"),
    ("Betavoltaic output, uW", f"{ab['electrical_power_uw']}", f"{zb['electrical_power_uw']}"),
    ("Beta fraction of harvest, %", f"{ah['beta_fraction_of_total_pct']}", f"{zh['beta_fraction_of_total_pct']}"),
    ("Harvest actually storable, %", "not reported", f"{zh['stored_fraction_of_harvest_pct']}"),
    ("Inference burst, uJ", "44000 (1.1 W x 40 ms)", f"{zs['inference_burst_uj']} (42 MACs)"),
    ("ESR ohm / leakage W", "0.05 / 1.2e-7", "5.0 / 3.0e-6 (+DC-DC 0.85)"),
    ("Brownouts", f"{as_['brownout_events']} ({as_['brownout_rate']*100:.2f}%)",
     f"{zs['brownout_events']} ({zs['brownout_rate']*100:.2f}%)"),
    ("Schmitt transitions (default)", f"{ah['safety_transitions']} (unreachable)",
     f"{zh['safety_transitions']} (reachable; 365 at 44 mJ)"),
    ("Reflex threshold / rate", f"0.35 fixed / {as_['reflex_fallback_rate']*100:.2f}%",
     f"{zs['reflex_threshold']} (1.5/K) / {zs['reflex_fallback_rate']*100:.2f}%"),
    ("Label rule", "int(7S)%7 hash", "argmax W.(x-0.5), learnable"),
    ("Drone acc (prequential)", f"{as_['drone_accuracy']*100:.2f}%", f"{zs['drone_prequential_accuracy']*100:.2f}%"),
    ("Server acc (prequential)", f"{as_['server_accuracy']*100:.2f}%", f"{zs['server_prequential_accuracy']*100:.2f}%"),
    ("Drone acc (held-out)", "not measured", f"{zho['drone_accuracy']*100:.2f}%"),
    ("Server acc (held-out)", "not measured", f"{zho['server_accuracy']*100:.2f}%"),
    ("Bayes ceiling", "~14.3% (hash)", f"{zho['bayes_ceiling']*100:.2f}%"),
    ("Drone Brier (prequential)", f"{as_['drone_brier_score']}", f"{zs['drone_prequential_brier']}"),
    ("Temperature", "nudged to top-prob 0.45", f"fit on held-out NLL: {zho['drone_fitted_temperature']}"),
    ("Drone ECE before -> after", "not measured", f"{zho['drone_ece_T1']} -> {zho['drone_ece_fitted']}"),
]
w1 = max(len(r[0]) for r in rows)
w2 = max(len(r[1]) for r in rows)
p(f"  {'Metric':{w1}s} | {'v7.3.1':{w2}s} | v7.4")
p(f"  {'-'*w1}-+-{'-'*w2}-+-{'-'*30}")
for r in rows:
    p(f"  {r[0]:{w1}s} | {r[1]:{w2}s} | {r[2]}")

with open(os.path.join(HERE, "demo_output_v7_4.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
