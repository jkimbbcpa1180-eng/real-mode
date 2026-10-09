# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo for the ground-reference mode (ground_ref_v3) on the v2 recon scenes.

Templates: lawnmower, perimeter_orbit, poi_close_pass.
GPS: standard (0.5 m, tau 30 s), plus a tau 90 s sensitivity check (SLAM still assumes
tau 30 s).
Configs, run paired on the same pings: slam (v2 SLAM only), pad, pad_m3, pad_m5, and
relative (a pad-frame metric).
In every odd-id scene, marker #1 has moved 0.5-1.5 m from its surveyed spot; this is the
outlier-rejection case.

  python3 mc_ground_ref_v3.py tuning | heldout

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
import ground_ref_v3 as G
import mc_recon_v2 as MR

N_SCEN = 12
TEMPLATES = ("lawnmower", "perimeter_orbit", "poi_close_pass")
GPS = {"gps_standard": (0.5, 30.0), "gps_tau90": (0.5, 90.0)}


def _job(a):
    sc, tpl, g = a
    r = G.run(sc, tpl, GPS[g], sc["seed"] + 101 * (1 + TEMPLATES.index(tpl)), moved_marker=sc["id"] % 2 == 1)
    r["gps_mode"] = g
    return r


def summarise(rows):
    out = {}
    for g in GPS:
        for t in TEMPLATES:
            rr = [r for r in rows if r["gps_mode"] == g and r["template"] == t]
            for c in G.CONFIGS:
                v = [r[c] for r in rr if c in r]
                cov = np.array([x["coverage"] for x in v])
                d = {"coverage_median": round(float(np.median(cov)), 3), "coverage_p5": round(float(np.percentile(cov, 5)), 3),
                     "pct_ge_90": C.pct(x >= 0.90 for x in cov),
                     "point_err_median_m": round(float(np.median([x["acc_median"] for x in v])), 3),
                     "point_err_p95_median_m": round(float(np.median([x["acc_p95"] for x in v])), 3),
                     "nees_mean": round(float(np.mean([x["nees_mean"] for x in v])), 2),
                     "nees_le_7_81": round(float(np.mean([x["nees_le_7_81"] for x in v])), 3),
                     "datum_err_median_m": round(float(np.median([x["datum_err_m"] for x in v])), 3)}
                if c in ("pad_m3", "pad_m5"):
                    mv = [r for r in rr if r["moved_marker"]]
                    d["moved_marker_rejected"] = f"{sum(1 in r[c]['rejected_markers'] for r in mv)}/{len(mv)}"
                    d["good_markers_rejected"] = sum(len([m for m in r[c]["rejected_markers"] if not (m == 1 and r["moved_marker"])])
                                                     for r in rr)
                    d["coverage_median_moved_scenes"] = round(float(np.median([r[c]["coverage"] for r in mv])), 3) if mv else None
                out[f"{g}/{t}/{c}"] = d
    return out


def main(which):
    seed = C.SEEDS["ground_ref"][which]
    scs = MR.make_scenes(N_SCEN, np.random.default_rng(seed))
    jobs = [(sc, t, g) for g in GPS for t in TEMPLATES for sc in scs]
    jobs.sort(key=lambda j: j[1] == "poi_close_pass")          # long jobs first
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=1)
    res = {"seed_set": which, "seed": seed, "summary": summarise(rows), "runtime_s": round(time.time() - t0, 1),
           "settings": {k: getattr(G, k) for k in dir(G) if k.isupper() and isinstance(getattr(G, k), (int, float))},
           "rows": rows}
    path = os.path.join(C.HERE, f"mc_ground_ref_{which}_v3.json")
    json.dump(res, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    for k, v in res["summary"].items():
        print(k, v)
    print("runtime_s", res["runtime_s"], "->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tuning")
