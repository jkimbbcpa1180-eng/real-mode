# SPDX-License-Identifier: CC0-1.0
"""
Regenerates demo_output_v3.txt and the renders in out/:
  out/safe_return_paths.png   v2 vs v3 return after GPS + link loss (one demo scene)
  out/swarm_merged_map.png    3-drone perimeter task, merged + aligned cloud map
and prints the Monte Carlo summary tables from the mc_*_v3.json files.
Demo scenes use seed 7 (neither a tuning nor a held-out seed).

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout

import common_v3 as C
from common_v3 import np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mc_recon_v2 as MR  # noqa: E402
import mission_v3 as MV  # noqa: E402
import safe_return_v3 as SR  # noqa: E402
import swarm_v3 as SW  # noqa: E402

OUT = os.path.join(C.HERE, "out")


def _box(ax, lo, hi, **kw):
    ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], **kw)


def demo_safe_return():
    print("== Feature 6: offline safe return (demo scene, wind 6 m/s, loss at 60 % of the mission) ==")
    sc = SR.make_scenarios(1, np.random.default_rng(7))[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode, col in (("v2", "tab:red"), ("v3", "tab:blue")):
        r = SR.run_return(sc, mode, 0.6, 6.0, 7, record=True)
        tr = r.traj
        P = np.array(tr["p"]); E = np.array(tr["est"])
        lost = np.array(tr["t"]) >= tr["t_loss"]
        ax.plot(P[~lost, 0], P[~lost, 1], color="0.6", lw=1)
        ax.plot(P[lost, 0], P[lost, 1], color=col, lw=1.5, label=f"{mode} truth after loss")
        ax.plot(E[lost, 0], E[lost, 1], color=col, lw=0.8, ls="--", label=f"{mode} own estimate")
        print(f"{mode}: outcome={r.outcome} landing_err_m={C.fmt(r.landing_err_m)} success={r.success} "
              f"collision={r.collision} min_clear_m={C.fmt(r.min_clear_m)} pad_seen={r.pad_seen} "
              f"slam_fixes={r.slam_fixes} return_time_s={C.fmt(r.return_time_s, 1)} energy_wh={C.fmt(r.energy_used_wh)}")
    for b in (sc["building"], sc["mast"]):
        _box(ax, b[0], b[1], color="k")
    ax.plot(*sc["home"][:2], "k*", ms=12, label="take-off pad")
    ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_title("Safe return without GPS/link: v2 vs v3")
    fig.savefig(os.path.join(OUT, "safe_return_paths.png"), dpi=110); plt.close(fig)


def demo_repass():
    print("== Feature 4: re-pass (demo scene, POI close pass, RTK and standard GPS) ==")
    sc = MR.make_scenes(1, np.random.default_rng(7))[0]
    for gm in ("gps_rtk", "gps_standard"):
        r = MV.run_mission_v3(sc, "poi_close_pass", gm, repass=True)
        print(f"{gm}: coverage {C.fmt(r['first_pass']['coverage'])} -> {C.fmt(r['after_repass']['coverage'])}, "
              f"re-passes {r['repasses']}, extra {r['extra_s']} s / {C.fmt(r['extra_wh'])} Wh, skipped {r['skipped']}")


def demo_swarm():
    print("== Feature 1: swarm (demo scene, perimeter task, N = 3, standard GPS) ==")
    sc = MR.make_scenes(1, np.random.default_rng(7))[0]
    r = SW.run_swarm(sc, "perimeter_orbit", 3, record=True)
    print(f"mission_s={r['mission_s']} coverage aligned={C.fmt(r['coverage_merged_aligned'])} "
          f"unaligned={C.fmt(r['coverage_merged_raw'])} near_miss={r['near_miss_episodes']} "
          f"min_sep_m={C.fmt(r['min_sep_m'])} delivered={[C.fmt(x) for x in r['delivered_points_frac']]}")
    for a in r["alignment"]:
        print("  align", {k: (C.fmt(v) if isinstance(v, float) else v) for k, v in a.items()})
    fig, ax = plt.subplots(figsize=(7, 6))
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for k, P in enumerate(r["_per"]):
        ax.scatter(P[:, 0], P[:, 1], s=0.3, color=cols[k], label=f"drone {k} map (cloud)")
        ax.plot(r["_paths"][k][:, 0], r["_paths"][k][:, 1], color=cols[k], lw=0.6)
    _box(ax, *sc["building"], color="k"); _box(ax, *sc["poi"], color="k")
    ax.set_aspect("equal"); ax.legend(fontsize=7, markerscale=10); ax.set_title("3-drone perimeter task: merged cloud map")
    fig.savefig(os.path.join(OUT, "swarm_merged_map.png"), dpi=110); plt.close(fig)


def relay_curve(d):
    """In-flight cloud coverage (median over scenes) for n=4 with deconfliction, relay on vs off."""
    import statistics
    for t in ("perimeter_orbit", "lawnmower"):
        rr = [r for r in d["rows"] if r["template"] == t and r["n"] == 4 and r["deconflict"]]
        cell = []
        for T in (60, 90, 120):
            v = [statistics.median([max([c for tt, c in r["coverage_curve"] if tt <= T] or [0]) for r in rr if r["relay"] == rel])
                 for rel in (True, False)]
            cell.append(f"{T}s {v[0]:.3f}/{v[1]:.3f}")
        print(f"   cloud coverage in flight, {t} n4 relay on/off: " + "  ".join(cell))


def tables():
    print("== Monte Carlo summaries (from the json files) ==")
    for name in ("safe_return", "repass", "swarm", "change", "lidar", "thermal"):
        for which in ("tuning", "heldout") + (("fresh",) if name == "swarm" else ()):
            p = os.path.join(C.HERE, f"mc_{name}_{which}_v3.json")
            if not os.path.exists(p):
                print(f"{name}/{which}: not run"); continue
            d = json.load(open(p))
            print(f"-- {name} / {which} (seed {d['seed']}, runtime {d['runtime_s']} s)")
            s = d["summary"]
            if name == "safe_return":
                for m in s:
                    a = s[m]["all"]
                    print(f"   {m:10s} success {a['success_pct']}%  landing err median {a['landing_err_m'].get('median')} "
                          f"p90 {a['landing_err_m'].get('p90')} max {a['landing_err_m'].get('max')}  collisions "
                          f"{a['collision_pct']}%  land-in-place {a['land_in_place_pct']}%  depleted {a['depleted_pct']}%")
            elif name == "repass":
                for k, c in s.items():
                    print(f"   {k:28s} base {c['coverage_base'].get('median')}  repass {c['repass']['coverage_after'].get('median')} "
                          f"(+{c['repass']['gain_per_min_pooled']}/min, +{c['repass']['gain_per_wh_pooled']}/Wh)  repeat "
                          f"{c['repeat']['coverage_after'].get('median')} (+{c['repeat']['gain_per_min_pooled']}/min)  "
                          f"no-fly violations {c['repass']['nofly_violations']}")
            elif name == "change":
                for k, c in s.items():
                    if k.endswith("alignment"):
                        print(f"   {k:28s} align err median {c['err_before_m'].get('median')} -> {c['err_after_m'].get('median')} m")
                    else:
                        print(f"   {k:28s} precision {c['precision']}  recall {c['recall']}  recall by size "
                              f"{ {z: v['recall'] for z, v in c['recall_by_size_m'].items()} }  false alarms / no-change pair "
                              f"{c['false_alarms_per_nochange_pair']}")
            elif name == "lidar":
                for k, c in s.items():
                    print(f"   {k:40s} cov {c['coverage'].get('median')}  err median {c['acc_median_m'].get('median')} "
                          f"p95 {c['acc_p95_m'].get('median')}  energy {c['energy_wh'].get('median')} Wh  "
                          f"endurance {c['endurance_min'].get('median')} min")
            elif name == "thermal":
                for k, c in s.items():
                    print(f"   {k:18s} precision {c['precision']}  recall {c['recall_by_kind']}  FP/mission "
                          f"{c['fp_per_mission']} (hot-glass share {c['fp_near_hot_glass_share']})")
            else:
                relay_curve(d)
                for k, c in s.items():
                    print(f"   {k:34s} t {c['mission_s'].get('median')} s  cov {c['coverage_merged_aligned'].get('median')} "
                          f"(unaligned {c['coverage_merged_unaligned'].get('median')})  near-miss {c['near_miss_episodes_total']}  "
                          f"collision ticks {c['collision_ticks_total']}  delivered {c['delivered_points_frac'].get('mean')}  "
                          f"align obs err {c['align_err_obs_before_m'].get('median')} -> {c['align_err_obs_after_m'].get('median')}")


def main():
    os.makedirs(OUT, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo_safe_return(); demo_repass(); demo_swarm(); tables()
    txt = buf.getvalue()
    open(os.path.join(C.HERE, "demo_output_v3.txt"), "w").write(txt)
    sys.stdout.write(txt)


if __name__ == "__main__":
    main()
