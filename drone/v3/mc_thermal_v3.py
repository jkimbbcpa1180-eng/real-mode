# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo for feature 2 (thermal hot spots): perimeter orbits at standoff 5/10/20/40 m,
standard GPS and RTK pose error. v2 has no thermal sensor (recall 0 by construction), so the
table reports v3 precision / recall by range and by target kind, plus the false-alarm sources.

  python3 mc_thermal_v3.py tuning | heldout

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
import thermal_v3 as TH

N_SCEN = 20
STANDOFFS = (5.0, 10.0, 20.0, 40.0)
GPS = {"gps_standard": 0.5, "gps_rtk": 0.03}


def _job(a):
    sc, so, g = a
    w = TH.make_world(sc, np.random.default_rng(sc["seed"] + 2))
    dets = TH.run(sc, w, so, GPS[g], np.random.default_rng(sc["seed"] + int(so) * 7 + (1 if g == "gps_rtk" else 0)))
    s = TH.score(dets, TH.truth(w))
    hot_glass = [gl for gl in w["glass"] if gl["Tr"] > 0]
    fp_glass = sum(1 for d in dets if any(np.linalg.norm(d - gl["c"]) < 2.0 for gl in hot_glass)
                   and not any(np.linalg.norm(d - v["c"]) < TH.MATCH_R_M for v in w["vents"]))
    return {"id": sc["id"], "standoff_m": so, "gps": g, **s, "fp_near_hot_glass": fp_glass}


def main(which):
    seed = C.SEEDS["thermal"][which]
    scs = MR.make_scenes(N_SCEN, np.random.default_rng(seed))
    jobs = [(sc, so, g) for sc in scs for so in STANDOFFS for g in GPS]
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=2)
    summ = {}
    for g in GPS:
        for so in STANDOFFS:
            rr = [r for r in rows if r["gps"] == g and r["standoff_m"] == so]
            tp = sum(r["tp"] for r in rr); nd = sum(r["n_det"] for r in rr)
            rec = {}
            for kind in ("vent", "person", "person_indoor"):
                h = [x[1] for r in rr for x in r["hits"] if x[0] == kind]
                rec[kind] = round(sum(h) / len(h), 3) if h else None
            summ[f"{g}/{int(so)}m"] = {"precision": round(tp / nd, 3) if nd else None, "recall_by_kind": rec,
                                       "fp_per_mission": round(sum(r["fp"] for r in rr) / len(rr), 2),
                                       "fp_near_hot_glass_share": round(sum(r["fp_near_hot_glass"] for r in rr) / max(sum(r["fp"] for r in rr), 1), 3)}
    res = {"seed_set": which, "seed": seed, "summary": summ, "runtime_s": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(C.HERE, f"mc_thermal_{which}_v3.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    for k, v in summ.items():
        print(k, v)
    print("runtime_s", res["runtime_s"], "->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tuning")
