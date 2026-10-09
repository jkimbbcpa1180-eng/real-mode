# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo for feature 4 ("go look closer" re-passes) on the v2 recon scenes.

For each scene x template x GPS mode one first pass is flown (= the v2 result). From
that exact state (deep copy) two continuations are flown:
  repass : v3 re-passes planned from the drone's own map (repass_v3)
  repeat : the whole template flown a second time (simple alternative, same drone)
Half of the scenes get a no-fly slab in front of one facade (where a closer re-pass
would want to fly); the start battery is drawn per scene so the reserve rule binds.

  python3 mc_repass_v3.py tuning | heldout

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import common_v3 as C
from common_v3 import np
import mc_recon_v2 as MR
import mission_v3 as MV
import repass_v3 as RP

N_SCEN = 12
TEMPLATES = ("perimeter_orbit", "poi_close_pass", "lawnmower")
GPS = ("gps_standard", "gps_rtk")
MAX_EXTRA_S = 300.0


def scene_extras(sc):
    rng = np.random.default_rng(sc["seed"] + 4)
    soc0 = float(rng.uniform(0.32, 1.0))
    nofly = []
    if sc["id"] % 2 == 1:
        lo, hi = sc["building"]
        nofly.append((np.array([hi[0] + 1.8, lo[1], -1.0]), np.array([hi[0] + 3.6, hi[1], 50.0])))
    return soc0, nofly


def _in_any(p, boxes):
    return any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in boxes)


def _job(args):
    sc, tpl, gm = args
    wps, scene, targets = MR.plan(sc, tpl)
    soc0, nofly = scene_extras(sc)
    f = MV.Flight(sc, scene, MR.GPS_MODES[gm], sc["seed"] + 13 * (1 + TEMPLATES.index(tpl)), wps[0].pos,
                  battery_wh=C.BATTERY_WH)
    f.energy_wh = soc0 * C.BATTERY_WH
    f.fly(wps)
    base = MV.map_metrics(f.rmap, scene, targets)
    t0, e0 = f.t, f.energy_used_wh
    out = dict(id=sc["id"], template=tpl, gps=gm, soc0=soc0, nofly=bool(nofly), base=base,
               first_pass_s=round(t0, 1), first_pass_wh=e0, min_clear_first=f.min_clear)
    # --- v3 re-pass ---
    fr = copy.deepcopy(f)
    fr.min_clear = float("inf")
    tp = time.perf_counter()
    mission = "facades" if tpl != "lawnmower" else "roofs+ground"
    focus = {"perimeter_orbit": sc["building"], "poi_close_pass": sc["poi"]}.get(tpl)
    focus = None if focus is None else (focus[0] - 2.0, focus[1] + 2.0)
    gaps, info = RP.find_gaps(fr.rmap, mission, area=sc["area"], focus=focus)
    plan = RP.plan_repasses(gaps, fr.rmap, fr.p_hat, wps[0].pos, MR.STANDOFF[tpl], MR.BEAM_HALF, fr.energy_wh,
                            fr.capacity_wh, fr.power_w(MR.SPEED), MR.SPEED, MAX_EXTRA_S, nofly=nofly)
    plan_ms = (time.perf_counter() - tp) * 1000.0
    n_before = len(fr.path)
    if plan.waypoints:
        fr.fly(plan.waypoints)
    rp = MV.map_metrics(fr.rmap, scene, targets)
    viol = sum(_in_any(p, nofly) for p in fr.path[n_before:])
    out["repass"] = dict(metrics=rp, extra_s=round(fr.t - t0, 1), extra_wh=fr.energy_used_wh - e0, gaps=len(gaps),
                         chosen=len(plan.chosen), skipped=plan.skipped, gap_info=info, plan_ms=plan_ms,
                         min_clear=fr.min_clear if plan.waypoints else None, nofly_violations=int(viol),
                         soc_end=fr.energy_wh / fr.capacity_wh)
    # --- repeat the full template ---
    fp = copy.deepcopy(f)
    fp.fly(wps)
    rr = MV.map_metrics(fp.rmap, scene, targets)
    out["repeat"] = dict(metrics=rr, extra_s=round(fp.t - t0, 1), extra_wh=fp.energy_used_wh - e0,
                         soc_end=fp.energy_wh / fp.capacity_wh)
    return out


def summarise(rows):
    s = {}
    for tpl in TEMPLATES:
        for gm in GPS:
            rs = [r for r in rows if r["template"] == tpl and r["gps"] == gm]
            def g(mode, key):
                return [r[mode]["metrics"][key] - r["base"][key] for r in rs]
            cell = {"n": len(rs), "coverage_base": C.stats(r["base"]["coverage"] for r in rs)}
            for mode in ("repass", "repeat"):
                gain = g(mode, "coverage")
                mins = [r[mode]["extra_s"] / 60.0 for r in rs]
                whs = [r[mode]["extra_wh"] for r in rs]
                cell[mode] = {"coverage_after": C.stats(r[mode]["metrics"]["coverage"] for r in rs),
                              "coverage_gain": C.stats(gain),
                              "extra_min": C.stats(mins), "extra_wh": C.stats(whs),
                              "gain_per_min_pooled": round(sum(gain) / max(sum(mins), 1e-9), 4),
                              "gain_per_wh_pooled": round(sum(gain) / max(sum(whs), 1e-9), 4),
                              "detail_change_m": C.stats((r[mode]["metrics"]["effective_detail_m"] or 0) - (r["base"]["effective_detail_m"] or 0) for r in rs),
                              "acc_p95_after": C.stats(r[mode]["metrics"]["acc_p95"] for r in rs),
                              "soc_end_min": round(min(r[mode]["soc_end"] for r in rs), 3)}
            cell["repass"]["nofly_violations"] = sum(r["repass"]["nofly_violations"] for r in rs)
            cell["repass"]["min_clear_m"] = C.stats(r["repass"]["min_clear"] for r in rs)
            cell["repass"]["skipped"] = {k: sum(r["repass"]["skipped"][k] for r in rs) for k in ("nofly", "interior", "clearance", "budget")}
            cell["repass"]["plan_ms"] = C.stats(r["repass"]["plan_ms"] for r in rs)
            s[f"{tpl}/{gm}"] = cell
    return s


def main(which: str) -> None:
    seed = C.SEEDS["repass"][which]
    scs = MR.make_scenes(N_SCEN, np.random.default_rng(seed))
    jobs = [(sc, tpl, gm) for sc in scs for tpl in TEMPLATES for gm in GPS]
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=1)
    res = {"seed_set": which, "seed": seed, "n_scenes": N_SCEN, "max_extra_s": MAX_EXTRA_S,
           "summary": summarise(rows), "runtime_s": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(C.HERE, f"mc_repass_{which}_v3.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for k, c in res["summary"].items():
        print(k, "base", c["coverage_base"].get("median"), "| repass", c["repass"]["coverage_after"].get("median"),
              "gain/min", c["repass"]["gain_per_min_pooled"], "| repeat", c["repeat"]["coverage_after"].get("median"),
              "gain/min", c["repeat"]["gain_per_min_pooled"], "| nofly viol", c["repass"]["nofly_violations"])
    print("runtime_s", res["runtime_s"], "->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tuning")
