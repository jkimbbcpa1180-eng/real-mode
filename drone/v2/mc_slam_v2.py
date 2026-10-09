"""
Monte Carlo for the sonar SLAM layer (drone_slam_v2) on the recon missions of
mc_recon_v2: the SAME missions (same noise) give the map without SLAM ("before")
and with SLAM ("after": online map during flight, and the map re-integrated with
the smoothed poses at mission end), so before/after is a paired comparison.

Scene sets:
  tuning   seed 20261009 (the recon tuning scenes; SLAM settings were chosen here only)
  fresh    seed FRESH_SEED, never used before this run (final evaluation only)
The old held-out set (20261010) is not used: it is no longer blind for two templates.

GPS modes: gps_standard (0.5 m, tau 30 s) = the target case; gps_rtk (0.03 m) as a
no-regression check (SLAM given the RTK error model); gps_tau90 = the true GPS error
has tau 90 s while SLAM still assumes 30 s (model mismatch check).
Uplink is not simulated here (unchanged; see mc_recon_v2).

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402

import drone_slam_v2 as S  # noqa: E402
import mc_recon_v2 as M  # noqa: E402

FRESH_SEED = 20262009          # never used before the final run
TEMPLATES = ("lawnmower", "perimeter_orbit", "poi_close_pass")
MODES = {"gps_standard": dict(gps=(0.5, 30.0), slam=dict(gps_sigma_m=0.5, gps_tau_s=30.0)),
         "gps_rtk": dict(gps=(0.03, 30.0), slam=dict(gps_sigma_m=0.03, gps_tau_s=30.0)),
         "gps_tau90": dict(gps=(0.5, 90.0), slam=dict(gps_sigma_m=0.5, gps_tau_s=30.0))}


def _job(args):
    sc, tpl, mode, extra = args
    md = MODES[mode]
    kw = dict(md["slam"])
    for k_, v_ in (extra or {}).items():       # tuning runs only: module constants or constructor args
        if k_.isupper():
            setattr(S, k_, v_)
        else:
            kw[k_] = v_
    r = M.run_mission(sc, tpl, "gps_standard", links=(), slam=True, slam_kwargs=kw, gps_override=md["gps"])
    r.pop("uplink", None)
    return r


def _st(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {"median": round(float(np.median(x)), 3), "p5": round(float(np.percentile(x, 5)), 3),
            "min": round(float(np.min(x)), 3), "max": round(float(np.max(x)), 3)}


def summarize(rs: List[Dict]) -> Dict:
    sl = [r["slam"] for r in rs]
    md = lambda xs: round(float(np.median(xs)), 3)  # noqa: E731

    def block(get):
        ms = [get(r) for r in rs]
        return {"coverage": _st([m["coverage"] for m in ms]),
                "frac_missions_ge_90": round(float(np.mean([m["coverage"] >= 0.90 for m in ms])), 3),
                "point_err_median_m": md([m["acc_median"] for m in ms]),
                "point_err_p95_m": md([m["acc_p95"] for m in ms]),
                "frac_points_within_tol": md([m["frac_points_within_tol"] for m in ms]),
                "effective_detail_m": md([m["effective_detail_m"] for m in ms if m["effective_detail_m"] is not None])}
    base = block(lambda r: r)
    base["nav_err_median_m"] = md([r["pos_err_median"] for r in rs])
    base["core_nees_mean_median"] = md([s["core_nees_mean"] for s in sl])
    base["core_nees_le_7_81_median"] = md([s["core_nees_le_7_81"] for s in sl])
    after = block(lambda r: r["slam"]["smoothed"])
    after["nav_err_median_m"] = md([s["nav_err_smoothed_median"] for s in sl])
    after["nav_err_p95_median_m"] = md([s["nav_err_smoothed_p95"] for s in sl])
    after["nees_mean"] = round(float(np.mean([s["nees_smoothed_mean"] for s in sl])), 3)
    after["nees_le_7_81"] = round(float(np.mean([s["nees_smoothed_le_7_81"] for s in sl])), 3)
    online = block(lambda r: r["slam"]["online"])
    online["nav_err_median_m"] = md([s["nav_err_online_median"] for s in sl])
    online["nees_mean"] = round(float(np.mean([s["nees_online_mean"] for s in sl])), 3)
    online["nees_le_7_81"] = round(float(np.mean([s["nees_online_le_7_81"] for s in sl])), 3)
    diag = block(lambda r: r["slam"]["smoothed_datum_removed"])
    return {"missions": len(rs), "mission_s_median": md([r["mission_s"] for r in rs]),
            "before_no_slam": base, "after_slam_smoothed": after, "after_slam_online": online,
            "diagnostic_datum_removed_using_truth": diag,
            "datum": {"err_m": _st([s["datum_err_m"] for s in sl]),
                      "sigma_xyz_median_m": [md([s["datum_sigma_m"][i] for s in sl]) for i in range(3)],
                      "nees_mean": round(float(np.mean([s["datum_nees"] for s in sl])), 3),
                      "core_mission_mean_err_m": _st([s["core_mean_err_m"] for s in sl])},
            "slam_stats": {"keyframes_total": int(sum(s["keyframes"] for s in sl)),
                           "registrations_total": int(sum(s["registrations"] for s in sl)),
                           "loop_factors_total": int(sum(s["loop_factors"] for s in sl)),
                           "rejected_few_matches_total": int(sum(s["rejected_few"] for s in sl)),
                           "rejected_degenerate_total": int(sum(s["rejected_degenerate"] for s in sl)),
                           "rejected_inconsistent_total": int(sum(s["rejected_inconsistent"] for s in sl)),
                           "observable_dims_hist_0_1_2_3": [int(sum(s["n_obs_hist"][i] for s in sl)) for i in range(4)],
                           "step_ms_median": md([s["step_ms_median"] for s in sl]),
                           "step_ms_p95_max": round(float(max(s["step_ms_p95"] for s in sl)), 1),
                           "step_ms_max": round(float(max(s["step_ms_max"] for s in sl)), 1),
                           "online_map_integration_ms_max": round(float(max(s["online_map_ms_max"] for s in sl)), 1),
                           "end_smooth_and_rebuild_s_median": md([s["end_smooth_rebuild_s"] for s in sl]),
                           "end_smooth_and_rebuild_s_max": round(float(max(s["end_smooth_rebuild_s"] for s in sl)), 2)},
            "per_mission": [{"id": r["id"], "before": round(r["coverage"], 3),
                             "after": round(r["slam"]["smoothed"]["coverage"], 3),
                             "after_datum_removed": round(r["slam"]["smoothed_datum_removed"]["coverage"], 3),
                             "datum_err_xyz": r["slam"]["datum_err_xyz"]} for r in rs]}


def run(sets: Dict[str, int], modes, n: int, extra=None) -> Dict:
    jobs = []
    for tag, seed in sets.items():
        scs = M.make_scenes(n, np.random.default_rng(seed))
        for mode in modes:
            for tpl in TEMPLATES:
                for sc in scs:
                    jobs.append((tag, mode, tpl, sc))
    t0 = time.perf_counter()
    res = []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap(_job, [(sc, tpl, mode, extra) for _, mode, tpl, sc in jobs], chunksize=1)):
            res.append(r)
            print(f"[{time.perf_counter() - t0:7.1f} s] {i + 1}/{len(jobs)} done", flush=True)
    groups: Dict = {}
    for (tag, mode, tpl, _), r in zip(jobs, res):
        groups.setdefault(f"{tag}|{tpl}|{mode}", []).append(r)
    rep = {"sets": sets, "n_per_set": n, "extra": extra,
           "slam_settings": {k: getattr(S, k) for k in dir(S) if k.isupper()},
           "results": {k: summarize(v) for k, v in groups.items()},
           "runtime_s": round(time.perf_counter() - t0, 1)}
    return rep


def _print(rep):
    for k, v in rep["results"].items():
        b, a, d = v["before_no_slam"], v["after_slam_smoothed"], v["diagnostic_datum_removed_using_truth"]
        print(f"{k:40s} before cov {b['coverage']['median']} ({b['frac_missions_ge_90']}) | after {a['coverage']['median']} "
              f"({a['frac_missions_ge_90']}) min {a['coverage']['min']} | datum-removed {d['coverage']['median']} | "
              f"nav {b['nav_err_median_m']} -> {a['nav_err_median_m']} | NEES {a['nees_mean']} ({a['nees_le_7_81']}) "
              f"online {v['after_slam_online']['nees_mean']} | datum err {v['datum']['err_m']['median']} | "
              f"step ms {v['slam_stats']['step_ms_median']}/{v['slam_stats']['step_ms_max']}")
    print("runtime_s", rep["runtime_s"])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "final"
    if what == "tune":
        extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
        rep = run({"tuning": M.SEED}, ("gps_standard",), 12, extra)
        out = f"/tmp/slamdev/tune_{abs(hash(json.dumps(extra))) % 10**6}.json"
    else:
        rep = run({"tuning": M.SEED, "fresh": FRESH_SEED}, tuple(MODES), 12)
        out = "mc_slam_results_v2.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    _print(rep)
    print("written", out)
