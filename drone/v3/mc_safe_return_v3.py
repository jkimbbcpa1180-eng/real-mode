# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo for feature 6 (offline safe return), v2 baseline vs v3, paired: the same
scenario, wind realisation seed and loss time for both modes.

  python3 mc_safe_return_v3.py tuning   -> mc_safe_return_tuning_v3.json
  python3 mc_safe_return_v3.py heldout  -> mc_safe_return_heldout_v3.json  (run once, frozen settings)

Seeds: common_v3.SEEDS["safe_return"]. Development before the tuning seed used ad-hoc
scenes from numpy seed 1 (not part of any reported table).

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
import safe_return_v3 as SR

N_SCEN = 30
ONSETS = (0.3, 0.6, 0.9)
WINDS = (0.0, 3.0, 6.0)
MODES = ("v2", "v3_noslam", "v3")


def _job(args):
    sc, mode, f, w = args
    r = SR.run_return(sc, mode, f, w, sc["seed"] + int(round(f * 100)) * 1000 + int(round(w * 10)))
    d = dict(r.__dict__)
    d.pop("traj", None)
    d["scen"] = sc["id"]
    return d


def summarise(rows):
    out = {}
    def cell(rs):
        return {"runs": len(rs), "success_pct": C.pct(r["success"] for r in rs),
                "landing_err_m": C.stats(r["landing_err_m"] for r in rs),
                "collision_pct": C.pct(r["collision"] for r in rs),
                "near_miss_pct": C.pct(r["near_miss"] for r in rs),
                "land_in_place_pct": C.pct(r["landed_in_place"] for r in rs),
                "depleted_pct": C.pct(r["outcome"] == "depleted" for r in rs),
                "timeout_pct": C.pct(r["outcome"] == "timeout" for r in rs),
                "pad_seen_pct": C.pct(r["pad_seen"] for r in rs),
                "return_time_s": C.stats(r["return_time_s"] for r in rs),
                "energy_wh": C.stats(r["energy_used_wh"] for r in rs)}
    for m in MODES:
        rm = [r for r in rows if r["mode"] == m]
        out[m] = {"all": cell(rm), "soc_at_loss_ge_0.2": cell([r for r in rm if r["soc_at_loss"] >= 0.2])}
        for f in ONSETS:
            for w in WINDS:
                out[m][f"onset{f}_wind{w:g}"] = cell([r for r in rm if r["onset_frac"] == f and r["wind_mps"] == w])
    return out


def main(which: str) -> None:
    seed = C.SEEDS["safe_return"][which]
    scs = SR.make_scenarios(N_SCEN, np.random.default_rng(seed))
    jobs = [(sc, m, f, w) for sc in scs for f in ONSETS for w in WINDS for m in MODES]
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=4)
    res = {"seed_set": which, "seed": seed, "n_scenarios": N_SCEN, "onsets": ONSETS, "winds_mps": WINDS,
           "summary": summarise(rows), "runtime_s": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(C.HERE, f"mc_safe_return_{which}_v3.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    s = res["summary"]
    for m in MODES:
        a = s[m]["all"]
        print(m, "success", a["success_pct"], "% landing err", a["landing_err_m"], "collisions", a["collision_pct"],
              "% land-in-place", a["land_in_place_pct"], "% depleted", a["depleted_pct"], "%")
    print("runtime_s", res["runtime_s"], "->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tuning")
