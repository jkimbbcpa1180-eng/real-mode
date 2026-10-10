# SPDX-License-Identifier: CC0-1.0
"""
Monte Carlo: GNSS budget v2 -> drone v3 SLAM-only recon (NO markers, NO pad).

For each GNSS configuration x space weather, gnss_budget_v2.drone_gps_model gives a
per-axis Gauss-Markov bias (horizontal / vertical sigma, tau) plus white noise. The v3
Flight is wrapped so its GPS error is that GM bias plus white noise (the v2/v3 model was
an isotropic GM only: 0.5 m, tau 30 s). The v2 SLAM is told the matched bias model
(sigma per axis, tau). RTK is the v2 reference (0.03 m, tau 30 s, no white noise).
Runs the ground_ref_v3 mission with configs=("slam",) only on lawnmower /
perimeter_orbit / poi_close_pass.

  python3 mc_gnss_v2.py smoke | heldout

Seeds (new, never used before this run): smoke 20263008 (plumbing check, 1 scene),
held-out 20264008 (evaluated once).

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "drone", "v3"))

import common_v3 as C  # noqa: E402
from common_v3 import np  # noqa: E402
import ground_ref_v3 as G  # noqa: E402
import mc_recon_v2 as MR  # noqa: E402
import mission_v3 as MV  # noqa: E402

import gnss_budget_v2 as B  # noqa: E402

SEEDS = {"smoke": 20263008, "heldout": 20264008}
N_SCEN = {"smoke": 1, "heldout": 12}
TEMPLATES = ("lawnmower", "perimeter_orbit", "poi_close_pass")
SC = {s.name: s.cfg for s in B.SCENARIOS}
MODES = {
    "quiet/L1": (B.QUIET, SC["L1 only (bare)"]),
    "quiet/L1+SBAS": (B.QUIET, SC["L1 + SBAS (MSAS)"]),
    "quiet/L1+L5": (B.QUIET, SC["L1 + L5 dual-freq"]),
    "quiet/L1+L5+SBAS": (B.QUIET, SC["L1 + L5 + SBAS"]),
    "quiet/L1+L5+PPP_conv": (B.QUIET, SC["L1 + L5 + PPP (converged)"]),
    "quiet/L1+L5+PPP_notconv": (B.QUIET, SC["L1 + L5 + PPP (not converged)"]),
    "ref/RTK_v2": None,
    "storm/L1+SBAS": (B.STORM, SC["L1 + SBAS (MSAS)"]),
    "storm/L1+L5+SBAS": (B.STORM, SC["L1 + L5 + SBAS"]),
    "storm/L1+L5+PPP_conv": (B.STORM, SC["L1 + L5 + PPP (converged)"]),
}


def gps_model(mode: str) -> dict:
    if MODES[mode] is None:
        return {"bias_h_m": 0.03, "bias_v_m": 0.03, "tau_s": 30.0, "white_h_m": 0.0, "white_v_m": 0.0}
    sw, cfg = MODES[mode]
    return B.drone_gps_model(sw, cfg)


_CUR = {}


class GnssFlight(MV.Flight):
    """v3 Flight whose GPS error is GM bias (per-axis sigma) + white noise. The parent's
    recursion gps = phi*gps + sqrt(1-phi^2)*N(0, gsig) is kept: the property below
    removes the white part from it and rescales the innovation to the bias sigma, so the
    GM state is exact; the core is told the total sigma per axis."""

    def __init__(self, sc, scene, gps, seed, start, **kw):
        self._b = _CUR["bias"]; self._w_sig = _CUR["white"]
        self._t = np.sqrt(self._b ** 2 + self._w_sig ** 2)
        self._wrng = np.random.default_rng(seed + 424243)
        self._w = np.zeros(3); self._gm = np.zeros(3); self._ready = False
        super().__init__(sc, scene, (1.0, gps[1]), seed, start, **kw)
        self.gsig = self._t                 # pos_sigma_m to the core; innovation scale
        self._ready = True

    @property
    def gps_err(self):
        return self._gm + self._w

    @gps_err.setter
    def gps_err(self, value):
        value = np.asarray(value, float)
        if not self._ready:                                     # initial draw ~ N(0, 1)
            self._gm = value * self._b
        else:
            phi = math.exp(-MV.DT / self.gtau)
            innov = value - phi * (self._gm + self._w)          # sqrt(1-phi^2) N(0, total)
            self._gm = phi * self._gm + innov / self._t * self._b
        self._w = self._wrng.normal(0.0, 1.0, 3) * self._w_sig


class VecSLAM(G.AnchoredSLAM):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sig = _CUR["bias"].copy()
        self._P_last = np.eye(3) * self.sig ** 2


MV.Flight = GnssFlight
G.AnchoredSLAM = VecSLAM


def _job(a):
    sc, tpl, mode = a
    g = gps_model(mode)
    _CUR["bias"] = np.array([g["bias_h_m"], g["bias_h_m"], g["bias_v_m"]])
    _CUR["white"] = np.array([g["white_h_m"], g["white_h_m"], g["white_v_m"]])
    r = G.run(sc, tpl, (float(g["bias_h_m"]), g["tau_s"]), sc["seed"] + 101 * (1 + TEMPLATES.index(tpl)),
              configs=("slam",), slam_sig=float(g["bias_h_m"]), slam_tau=g["tau_s"])
    r["gps_mode"] = mode
    return r


def summarise(rows):
    out = {}
    for m in MODES:
        for t in TEMPLATES:
            v = [r["slam"] for r in rows if r["gps_mode"] == m and r["template"] == t and "slam" in r]
            if not v:
                continue
            cov = np.array([x["coverage"] for x in v])
            out[f"{m}/{t}"] = {"n": len(v), "coverage_median": round(float(np.median(cov)), 3),
                               "coverage_p5": round(float(np.percentile(cov, 5)), 3),
                               "pct_ge_90": C.pct(x >= 0.90 for x in cov),
                               "point_err_median_m": round(float(np.median([x["acc_median"] for x in v])), 3),
                               "nav_err_median_m": round(float(np.median([x["nav_err_median"] for x in v])), 3),
                               "nees_mean": round(float(np.mean([x["nees_mean"] for x in v])), 2)}
    return out


def main(which):
    seed = SEEDS[which]
    scs = MR.make_scenes(N_SCEN[which], np.random.default_rng(seed))
    jobs = [(sc, t, m) for m in MODES for t in TEMPLATES for sc in scs]
    jobs.sort(key=lambda j: j[1] != "perimeter_orbit")          # long jobs first
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        rows = p.map(_job, jobs, chunksize=1)
    res = {"seed_set": which, "seed": seed, "n_scenes": N_SCEN[which],
           "gps_models": {m: gps_model(m) for m in MODES}, "summary": summarise(rows),
           "runtime_s": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(HERE, f"mc_gnss_{which}_v2.json")
    json.dump(res, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    for k, v in res["summary"].items():
        print(k, v)
    print("runtime_s", res["runtime_s"], "->", os.path.basename(path))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke")
