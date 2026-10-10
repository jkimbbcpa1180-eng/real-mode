# SPDX-License-Identifier: CC0-1.0
"""Writes demo_output_v2.txt: the v2 benchmark, v1 vs v2 fused horizontal error (v1 run
with only the crash bypassed: imu passed explicitly), and the drone MC summary if present.
License: CC0 1.0 Universal."""
import contextlib
import io
import json
import os

import gnss_budget_v2 as B
import gnss_imu_budget as V1

HERE = os.path.dirname(os.path.abspath(__file__))
V1_MAP = {"L1 only (bare)": (False, False, False), "L1 + SBAS (MSAS)": (False, True, False),
          "L1 + L5 dual-freq": (True, False, False), "L1 + L5 + SBAS": (True, True, False),
          "L1 + L5 + PPP (converged)": (True, False, True)}


def v1_vs_v2(out):
    print("v1 vs v2, fused horizontal error at a 2 s outage, suburban (m)", file=out)
    print("v1 = pr*HDOP*(1+0.4a) + IMU drift with sqrt(200 Hz) (v1 'horizontal_m', crash bypassed)", file=out)
    print("v2 = DRMS (radial RMS); per-axis = DRMS/sqrt2. GNSS-only DRMS in brackets.", file=out)
    print(f"{'scenario':<32}{'v1 quiet':>10}{'v2 quiet':>18}{'v1 storm':>10}{'v2 storm':>18}", file=out)
    for name, (l5, sb, ppp) in V1_MAP.items():
        cells = []
        for sw1, sw2 in ((V1.SpaceWeather(100, 4, 1e-9), B.QUIET), (V1.SpaceWeather(250, 120, 2e-4), B.STORM)):
            b1 = V1.compute_budget(sw1, V1.BudgetScenario(name, V1.GNSSConfig(l5, sb, ppp)), outage_s=2.0,
                                   imu=V1.IMUConfig())
            b2 = B.compute_budget(sw2, [s for s in B.SCENARIOS if s.name == name][0], outage_s=2.0)
            cells += [f"{b1['position_fused']['horizontal_m']:>10.3f}",
                      f"{b2['position_fused']['horizontal_m']:>8.3f} ({b2['position_gnss_only']['horizontal_m']:.3f})"]
        print(f"{name:<32}{cells[0]}{cells[1]:>18}{cells[2]}{cells[3]:>18}", file=out)
    b = B.compute_budget(B.QUIET, B.SCENARIOS[5], outage_s=2.0)
    print(f"{'L1 + L5 + PPP (not converged)':<32}{'n/a':>10}{b['position_fused']['horizontal_m']:>8.3f} "
          f"({b['position_gnss_only']['horizontal_m']:.3f})", file=out)


def main():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        B.main()
    print("\n" + "=" * 78, file=out)
    v1_vs_v2(out)
    p = os.path.join(HERE, "mc_gnss_heldout_v2.json")
    if os.path.exists(p):
        d = json.load(open(p))
        print("\n" + "=" * 78, file=out)
        print(f"DRONE v3 SLAM-ONLY RECON (no markers, no pad), held-out seed {d['seed']}, "
              f"{d['n_scenes']} scenes/template", file=out)
        print(f"{'mode':<26}{'bias h/v m':>13}{'tau s':>7}{'white h m':>10} | median cov / p5 / %>=90 / point err m "
              f"(lawnmower, perimeter_orbit, poi_close_pass)", file=out)
        for m, g in d["gps_models"].items():
            cells = []
            for t in ("lawnmower", "perimeter_orbit", "poi_close_pass"):
                s = d["summary"][f"{m}/{t}"]
                cells.append(f"{s['coverage_median']:.3f}/{s['coverage_p5']:.3f}/{s['pct_ge_90']:.0f}/{s['point_err_median_m']:.3f}")
            print(f"{m:<26}{g['bias_h_m']:>6.3f}/{g['bias_v_m']:<6.3f}{g['tau_s']:>7.0f}{g['white_h_m']:>10.3f} | "
                  + "  ".join(cells), file=out)
    txt = out.getvalue()
    open(os.path.join(HERE, "demo_output_v2.txt"), "w").write(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
