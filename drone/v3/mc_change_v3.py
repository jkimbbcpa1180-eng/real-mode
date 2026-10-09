# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo for feature 5 (change detection): pairs of lawnmower missions with known
changes (2 added, 2 removed, 2 moved objects, sizes 0.3-2 m, moves 1-4 m) and no-change
pairs (same static objects only). v2 = differencing without alignment, v3 = aligned.

  python3 mc_change_v3.py tuning | heldout

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
import change_v3 as CH
import mc_recon_v2 as MR

N_SCEN = 8
GPS = ("gps_standard", "gps_rtk")
MODES = ("v2_noalign", "v3_aligned")


def _job(a):
    sc, gps, ch = a
    return CH.run_pair(sc, gps, ch, sc["seed"] + (500 if ch else 900) + GPS.index(gps))


def summarise(rows):
    out = {}
    for gps in GPS:
        for m in MODES:
            rc = [r for r in rows if r["gps"] == gps and r["change"]]
            rn = [r for r in rows if r["gps"] == gps and not r["change"]]
            tp = sum(r[m]["tp"] for r in rc); nd = sum(r[m]["n_det"] for r in rc)
            by = {}
            for s in CH.SIZES:
                h = [x[2] for r in rc for x in r[m]["truth_hits"] if x[1] == s]
                by[str(s)] = {"truth": len(h), "recall": round(sum(h) / len(h), 3) if h else None}
            out[f"{gps}/{m}"] = {"precision": round(tp / nd, 3) if nd else None,
                                 "recall": round(sum(sum(x[2] for x in r[m]["truth_hits"]) for r in rc)
                                                 / max(sum(r[m]["n_truth"] for r in rc), 1), 3),
                                 "recall_by_size_m": by,
                                 "false_alarms_per_nochange_pair": round(sum(r[m]["fp"] for r in rn) / max(len(rn), 1), 2),
                                 "nochange_pairs_with_any_alarm_pct": C.pct(r[m]["n_det"] > 0 for r in rn)}
        rr = [r for r in rows if r["gps"] == gps]
        out[f"{gps}/alignment"] = {"err_before_m": C.stats(r["align_err_before_m"] for r in rr),
                                   "err_after_m": C.stats(r["align_err_after_m"] for r in rr),
                                   "n_obs": C.stats(r["align_nobs"] for r in rr)}
    return out


def main(which):
    seed = C.SEEDS["change"][which]
    scs = MR.make_scenes(N_SCEN, np.random.default_rng(seed))
    jobs = [(sc, g, ch) for sc in scs for g in GPS for ch in (True, False)]
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=1)
    res = {"seed_set": which, "seed": seed, "n_scenes": N_SCEN, "summary": summarise(rows),
           "runtime_s": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(C.HERE, f"mc_change_{which}_v3.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    for k, v in res["summary"].items():
        print(k, v)
    print("runtime_s", res["runtime_s"], "->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tuning")
