#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""Writes demo_output_v4_2.txt: v4.1 vs v4.2 at the demo inputs, the steps between them,
and sensitivity cases. Standard library only. License: CC0 1.0 Universal."""
import contextlib
import dataclasses
import io
import os

import mountain_updraft_v4_1 as V41
import mountain_updraft_v4_2 as V42

HERE = os.path.dirname(os.path.abspath(__file__))


def v41_numbers():
    cfg = V41.FacilityConfig()
    snd = [V41.SoundingLayer(*dataclasses.astuple(s)) for s in V42.DEMO_SOUNDINGS]
    r = V41.run_system_cycle(config=cfg, soundings=snd, **V42.DEMO_INPUTS)
    th, esp = r["thermodynamics_and_updraft"], r["esp_air_purification"]
    return dict(dT=th["temperature_lift_k"], dp=th["buoyant_head_pa"], v=th["chimney_velocity_mps"],
                Q=th["volumetric_flow_m3_s"], P=th["electrical_power_mw"], esp=esp["capture_efficiency_percent"],
                w=esp["pm25_migration_velocity_mps"])


def v42_case(label, **over):
    cfg = dataclasses.replace(V42.FacilityConfig(), **{k: v for k, v in over.items() if k != "neutral"})
    s = V42.DEMO_SOUNDINGS
    dT, dp, v, m, Q = V42.AtmosphericThermodynamics.solve_draft(cfg, s[0], s[-1], V42.DEMO_INPUTS["irradiance_w_m2"],
                                                                 s, neutral=over.get("neutral", False))
    P = V42.compute_turbine_power(cfg, dp, Q)[1]
    ceiling = m * V42.CP_AIR * dT * V42.chimney_efficiency(cfg, s[0]) * cfg.turbine_aerodynamic_efficiency \
        * cfg.turbine_head_extraction_fraction * cfg.generator_efficiency / 1e6
    return f"{label:58s} dT {dT:6.2f} K  head {dp:7.1f} Pa  v {v:6.2f} m/s  Q {Q:8.0f} m3/s  P_el {P:6.3f} MW (ceiling {ceiling:5.3f})"


def main():
    out = io.StringIO()
    a = v41_numbers()
    print("v4.1 vs v4.2 at the demo inputs (780 W/m2, 50 kV, 59.78 Hz, 44 rpm, demo sounding)", file=out)
    print(f"v4.1: dT {a['dT']} K  head {a['dp']} Pa  v {a['v']} m/s  Q {a['Q']} m3/s  P_el {a['P']} MW  "
          f"ESP w {a['w']} m/s  eff {a['esp']} %", file=out)
    print("\nSteps from v4.1 to v4.2 (each row adds one change; eta_t 0.44 until the last two rows):", file=out)
    rows = [
        ("coupled draft, exit loss only (K=1), neutral, eta 0.44", dict(friction_factor=0.0, k_esp=0.0, neutral=True, turbine_aerodynamic_efficiency=0.44)),
        ("+ friction f=0.010 over L=H/sin30 + ESP K=0.5, neutral", dict(neutral=True, turbine_aerodynamic_efficiency=0.44)),
        ("+ demo sounding (stable layers), eta 0.44 [sensitivity]", dict(turbine_aerodynamic_efficiency=0.44)),
        ("v4.2 default: + eta_t 0.80 (ducted pressure-staged)", dict()),
        ("v4.2, neutral atmosphere (for the stable-layer penalty)", dict(neutral=True)),
        ("sensitivity: collector heat loss U=5 W/m2K", dict(collector_loss_w_m2k=5.0)),
    ]
    for lab, ov in rows:
        print(v42_case(lab, **ov), file=out)
    print("Note: the hand check (constant density, head = rho g H dT / T) gives ~7,640 m3/s for the first row;\n"
          "the integrated head (air thins ~7 % over 650 m, virtual temperature) gives ~3 % less flow.", file=out)
    print("\nPower invariance to the velocity guess (v4.2 default):", file=out)
    s = V42.DEMO_SOUNDINGS
    for vg in (1.0, 10.0, 60.0):
        dT, dp, v, m, Q = V42.AtmosphericThermodynamics.solve_draft(V42.FacilityConfig(), s[0], s[-1], 780.0, s, v_guess=vg)
        print(f"  v_guess {vg:5.1f} -> P_el {V42.compute_turbine_power(V42.FacilityConfig(), dp, Q)[1]:.6f} MW", file=out)
    print("\nFull v4.2 benchmark:", file=out)
    with contextlib.redirect_stdout(out):
        V42.run_self_benchmark()
    print("\nSensitivity: turbine efficiency 0.44 (v4.1 value), otherwise v4.2 default:", file=out)
    r = V42.run_system_cycle(config=V42.FacilityConfig(turbine_aerodynamic_efficiency=0.44),
                             soundings=s, daily_irradiance_w_m2=V42.default_irradiance_profile(), **V42.DEMO_INPUTS)
    print(f"  electrical {r['thermodynamics_and_updraft']['electrical_power_mw']} MW, net "
          f"{r['scale_reality']['net_electric_mw']} MW, daily {r['scale_reality']['daily_net_energy_mwh']} MWh", file=out)
    r = V42.run_system_cycle(config=V42.FacilityConfig(esp_velocity_derate=1.0), soundings=s, **V42.DEMO_INPUTS)
    print(f"Sensitivity: ESP with the theoretical migration velocity (derate 1.0): capture "
          f"{r['esp_air_purification']['capture_efficiency_percent']} %, PM2.5 {r['esp_air_purification']['pm25_removed_kg_day']} kg/day", file=out)
    txt = out.getvalue()
    open(os.path.join(HERE, "demo_output_v4_2.txt"), "w").write(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
