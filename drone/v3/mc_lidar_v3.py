# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo for feature 3 (lidar): v2 sonar-only vs sonar + lidar on the v2 recon scenes,
perimeter orbit and lawnmower, standard GPS and RTK. Per scene the facade dropout (dark /
wet surfaces) is drawn U(0.05, 0.8) and the ground dropout U(0.1, 0.6). Reports map
coverage, point error, energy used and the endurance (45 Wh battery) at 2 m/s cruise.

  python3 mc_lidar_v3.py tuning | heldout

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import common_v3 as C
from common_v3 import np
import mc_recon_v2 as MR
import mission_v3 as MV
import sensors_v3 as SE

N_SCEN = 6
TEMPLATES = ("perimeter_orbit", "lawnmower")
GPS = ("gps_standard", "gps_rtk")
CONFIGS = ("sonar_v2", "sonar_lidar_v3")


def _job(a):
    sc, tpl, gps, cfg = a
    rng = np.random.default_rng(sc["seed"] + 3)
    d_fac, d_gnd = float(rng.uniform(0.05, 0.8)), float(rng.uniform(0.1, 0.6))
    wps, scene, targets = MR.plan(sc, tpl)
    sens = [MV.SonarHead()]
    if cfg == "sonar_lidar_v3":
        a_lo, a_hi = sc["area"]
        sens.append(SE.LidarHead([sc["building"], sc["poi"]], (np.asarray(a_lo) - 20, np.asarray(a_hi) + 20), [d_fac, d_fac], d_gnd))
    f = MV.Flight(sc, scene, MR.GPS_MODES[gps], sc["seed"] + 31 * (1 + TEMPLATES.index(tpl)), wps[0].pos, sensors=sens)
    f.fly(wps)
    m = MV.map_metrics(f.rmap, scene, targets)
    p = f.power_w(2.0)
    return {"id": sc["id"], "template": tpl, "gps": gps, "config": cfg, "facade_dropout": d_fac, "ground_dropout": d_gnd,
            "coverage": m["coverage"], "acc_median_m": m["acc_median"], "acc_p95_m": m["acc_p95"],
            "mission_s": f.t, "energy_wh": f.energy_used_wh, "power_2ms_w": p, "endurance_min": C.BATTERY_WH / p * 60.0}


def main(which):
    seed = C.SEEDS["lidar"][which]
    scs = MR.make_scenes(N_SCEN, np.random.default_rng(seed))
    jobs = [(sc, t, g, c) for sc in scs for t in TEMPLATES for g in GPS for c in CONFIGS]
    jobs.sort(key=lambda j: (j[3] != "sonar_lidar_v3", j[1] != "perimeter_orbit"))   # long jobs first
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=1)
    summ = {}
    for t in TEMPLATES:
        for g in GPS:
            for c in CONFIGS:
                rr = [r for r in rows if r["template"] == t and r["gps"] == g and r["config"] == c]
                summ[f"{t}/{g}/{c}"] = {k: C.stats(r[k] for r in rr) for k in ("coverage", "acc_median_m", "acc_p95_m", "energy_wh", "endurance_min")}
    res = {"seed_set": which, "seed": seed, "summary": summ, "runtime_s": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(C.HERE, f"mc_lidar_{which}_v3.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    for k, v in summ.items():
        print(k, {kk: vv["median"] for kk, vv in v.items()})
    print("runtime_s", res["runtime_s"], "->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tuning")
