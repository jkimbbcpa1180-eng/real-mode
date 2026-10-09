# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo for feature 1 (swarm). N = 1 is the v2 baseline (one drone, whole task).
Ablations at N = 4: deconfliction off, relay off.

  python3 mc_swarm_v3.py tuning | heldout

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
import swarm_v3 as SW

N_SCEN = 6
TEMPLATES = ("perimeter_orbit", "lawnmower")
CONFIGS = [dict(n=1), dict(n=2), dict(n=3), dict(n=4),
           dict(n=4, deconflict=False), dict(n=4, relay=False)]


def _job(args):
    sc, tpl, cfg = args
    return SW.run_swarm(sc, tpl, **cfg)


def _label(r):
    return f"n{r['n']}" + ("" if r["deconflict"] else "_nodeconflict") + ("" if r["relay"] else "_norelay")


def summarise(rows):
    out = {}
    for tpl in TEMPLATES:
        for lab in sorted({_label(r) for r in rows}):
            rs = [r for r in rows if r["template"] == tpl and _label(r) == lab]
            al = [a for r in rs for a in r["alignment"]]
            out[f"{tpl}/{lab}"] = {
                "runs": len(rs), "completed_pct": C.pct(r["completed"] for r in rs),
                "mission_s": C.stats(r["mission_s"] for r in rs),
                "coverage_merged_aligned": C.stats(r["coverage_merged_aligned"] for r in rs),
                "coverage_merged_unaligned": C.stats(r["coverage_merged_raw"] for r in rs),
                "near_miss_episodes_total": sum(r["near_miss_episodes"] for r in rs),
                "collision_ticks_total": sum(r["collision_ticks"] for r in rs),
                "min_sep_m": C.stats(r["min_sep_m"] for r in rs if np.isfinite(r["min_sep_m"])),
                "min_clear_m": C.stats(r["min_clear_m"] for r in rs),
                "delivered_points_frac": C.stats(x for r in rs for x in r["delivered_points_frac"]),
                "out_of_direct_range_frac": C.stats(x for r in rs for x in r["out_of_range_frac"]),
                "relayed_msgs_total": sum(r["relay_stats"]["relayed_msgs"] for r in rs),
                "relay_queue_max": max(r["relay_stats"]["relay_queue_max"] for r in rs),
                "align_ok_pct": C.pct(a.get("ok") for a in al) if al else None,
                "align_err_obs_before_m": C.stats(a.get("offset_err_obs_before_m") for a in al),
                "align_err_obs_after_m": C.stats(a.get("offset_err_obs_after_m") for a in al),
                "align_err_3d_before_m": C.stats(a.get("true_rel_offset_m") for a in al),
                "align_err_3d_after_m": C.stats(a.get("offset_err_after_m") for a in al),
                "consistency_before_m": C.stats(a.get("consistency_before_m") for a in al),
                "consistency_after_m": C.stats(a.get("consistency_after_m") for a in al),
                "energy_wh_total": C.stats(sum(r["energy_wh"]) for r in rs)}
    return out


def main(which: str) -> None:
    seed = C.SEEDS["swarm"][which]
    scs = MR.make_scenes(N_SCEN, np.random.default_rng(seed))
    jobs = [(sc, tpl, cfg) for sc in scs for tpl in TEMPLATES for cfg in CONFIGS]
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=1)
    res = {"seed_set": which, "seed": seed, "n_scenes": N_SCEN, "summary": summarise(rows),
           "runtime_s": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(C.HERE, f"mc_swarm_{which}_v3.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for k, c in res["summary"].items():
        print(k, "t", c["mission_s"].get("median"), "cov", c["coverage_merged_aligned"].get("median"),
              "raw", c["coverage_merged_unaligned"].get("median"), "nm", c["near_miss_episodes_total"],
              "coll", c["collision_ticks_total"], "deliv", c["delivered_points_frac"].get("mean"),
              "align obs", c["align_err_obs_before_m"].get("median"), "->", c["align_err_obs_after_m"].get("median"))
    print("runtime_s", res["runtime_s"], "->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tuning")
