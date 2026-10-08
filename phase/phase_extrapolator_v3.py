"""
REAL MODE: Physics-Fit Temporal Phase Extrapolator (v3)
Fits x(t) = a*sin(w t) + c*cos(w t) + b*exp(lam t) [+ d] to high-rate samples,
forecasts with propagated parameter uncertainty, and only trusts the physics
model when the data back it up.

How it works:
- Variable projection start: grid-search the nonlinear pair (w, lam), solve the
  linear amplitudes by least squares at every grid point, then refine all
  parameters with a numpy Levenberg-Marquardt solver (analytic Jacobian).
  An FFT/periodogram guess for w is added to the grid only when the record holds
  at least two periods; for the demo (10 ms, period 31.4 ms) it cannot resolve w.
- Standard errors come from the Jacobian: cov = s^2 (J^T J)^-1.
- Derivatives (velocity, acceleration, jerk, snap) are analytic in the fitted
  model, with delta-method standard errors.
- Model choice: physics (no offset), physics (+offset) and the v2 polynomial
  are scored by the same walk-forward backtest on common origins. The backtest
  replays the live rule (physics where its fit passes its checks, polynomial
  otherwise). Physics is used unless the polynomial is significantly better
  (paired HAC test) or physics' own backtest intervals were badly calibrated.
  The choice is made at min(horizon, validation horizon): long-horizon
  backtests can only start early in the record, where an expanding-window fit
  has seen too little to judge the full-record fit. The validation horizon is
  the longest one whose backtest still starts at >= 95% of the record.
- Reliability checks (v2 logic) still use the backtest at the actual horizon.
- When the polynomial is chosen but a physics fit passes its checks, the band
  is the envelope of both (model disagreement is real uncertainty).
  A physics fit that fails its checks (no convergence, parameter at a bound,
  R^2 < 0.99, structured residuals, unidentified omega/lambda) is reported and
  NOT used.
- Uncertainty: the forecast gradient times the parameter covariance gives a
  confidence band for the true signal; adding the residual variance gives a
  prediction band for a new noisy observation. Both are linearized (delta
  method), so they can undercover when the forecast is strongly nonlinear.
- v2's UNRELIABLE logic is kept: horizon guard, state outside backtest range,
  growing backtest error.

Limits: the model form is assumed. If the real system is not "one sinusoid plus
one exponential", the fit can look fine inside the record and still extrapolate
badly; the residual checks catch only some of these cases.

Dependencies: numpy only (plus phase_extrapolator_v2.py from this folder).
License: CC0 1.0 Universal (public domain). Copy, modify, use freely.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import phase_extrapolator_v2 as v2
from phase_extrapolator_v2 import MicroTemporalState, demo_signal, demo_truth  # noqa: F401

LAM_MAX = 15.0        # bound on |lam * span| (exp growth/decay over the record)
W_MIN = 0.05          # lowest grid value of w * span
MIN_FIT_FRACTION = 0.25
VALIDATION_FRACTION = 0.95  # assumed (picked on tuning seeds 0-49 from 0.75/0.90/0.95)
Z90 = 1.6448536269514722  # standard normal 95th percentile -> 90% two-sided
NAMES = ("velocity", "acceleration", "jerk", "snap")


# ---------- model in scaled time u = (t - t_end) / span, u in [-1, 0] ----------
def _model(p: np.ndarray, u: np.ndarray, offset: bool) -> np.ndarray:
    a, c, b, ws, ls = p[0], p[1], p[2], p[-2], p[-1]
    y = a * np.sin(ws * u) + c * np.cos(ws * u) + b * np.exp(ls * u)
    return y + p[3] if offset else y


def _jac(p: np.ndarray, u: np.ndarray, offset: bool) -> np.ndarray:
    a, c, b, ws, ls = p[0], p[1], p[2], p[-2], p[-1]
    s, co, e = np.sin(ws * u), np.cos(ws * u), np.exp(ls * u)
    cols = [s, co, e] + ([np.ones_like(u)] if offset else []) + [u * (a * co - c * s), b * u * e]
    return np.stack(cols, axis=-1)


def _basis(u, ws, ls, offset: bool) -> np.ndarray:
    cols = [np.sin(ws * u), np.cos(ws * u), np.exp(ls * u)]
    if offset:
        cols.append(np.ones_like(ws * u))
    return np.stack(cols, axis=-1)


def _grid_starts(u: np.ndarray, x: np.ndarray, offset: bool, top: int = 4) -> List[np.ndarray]:
    """Variable projection on a (w, lam) grid; returns the best `top` full parameter vectors."""
    n = len(u)
    ws_grid = np.geomspace(W_MIN, np.pi * (n - 1), 90)       # up to Nyquist
    if n >= 32:  # periodogram guess, only meaningful with >= 2 periods in the record
        xd = x - np.polyval(np.polyfit(u, x, 2), u)
        k = int(np.argmax(np.abs(np.fft.rfft(xd))[1:])) + 1
        if k >= 2:
            ws_grid = np.r_[ws_grid, 2 * np.pi * k * (n - 1) / n]
    ls_grid = np.linspace(-LAM_MAX, LAM_MAX, 61)
    W, L = (g.ravel() for g in np.meshgrid(ws_grid, ls_grid, indexing="ij"))
    B = _basis(u[None, :], W[:, None], L[:, None], offset)    # (G, n, k)
    beta = np.einsum("gkn,n->gk", np.linalg.pinv(B, rcond=1e-10), x)
    r = x[None, :] - np.einsum("gnk,gk->gn", B, beta)
    best = np.argsort(np.einsum("gn,gn->g", r, r))[:top]
    return [np.r_[beta[i], W[i], L[i]] for i in best]


def _levenberg_marquardt(p, u, x, offset, max_iter=300) -> Tuple[np.ndarray, float, bool]:
    p = np.asarray(p, dtype=float).copy()
    r = x - _model(p, u, offset)
    cost, mu, converged = float(r @ r), 1e-3, False
    for _ in range(max_iter):
        J = _jac(p, u, offset)
        A, g = J.T @ J, J.T @ r
        D = np.diag(np.diag(A)) + 1e-300
        while True:
            step = np.linalg.lstsq(A + mu * D, g, rcond=None)[0]
            pn = p + step
            pn[-1] = np.clip(pn[-1], -1.5 * LAM_MAX, 1.5 * LAM_MAX)
            if pn[-2] < 0:  # keep w >= 0: sin(-w u) = -sin(w u)
                pn[-2], pn[0] = -pn[-2], -pn[0]
            rn = x - _model(pn, u, offset)
            cn = float(rn @ rn)
            if cn <= cost:
                break
            mu *= 4
            if mu > 1e14:
                return p, cost, converged
        rel_drop = (cost - cn) / max(cost, 1e-300)
        p, r, cost, mu = pn, rn, cn, max(mu / 3, 1e-12)
        if rel_drop < 1e-12 or cost < 1e-28 or np.max(np.abs(step) / (np.abs(p) + 1e-8)) < 1e-10:
            converged = True
            break
    return p, cost, converged


@dataclass
class PhysicsFit:
    p: np.ndarray            # scaled params [a, c, b, (d), w*span, lam*span]
    offset: bool
    span: float
    t_end: float
    n: int
    rss: float
    converged: bool
    cov: Optional[np.ndarray] = None
    sigma: float = float("nan")
    diagnostics: Dict = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def omega(self) -> float:
        return float(self.p[-2] / self.span)

    @property
    def lam(self) -> float:
        return float(self.p[-1] / self.span)

    def value(self, dt_ahead: float) -> float:
        return float(_model(self.p, np.array([dt_ahead / self.span]), self.offset)[0])

    def value_se(self, dt_ahead: float) -> float:
        if self.cov is None:
            return float("nan")
        g = _jac(self.p, np.array([dt_ahead / self.span]), self.offset)[0]
        return float(np.sqrt(max(g @ self.cov @ g, 0.0)))

    def _deriv_and_grad(self, k: int) -> Tuple[float, np.ndarray]:
        a, c, b, ws, ls = self.p[0], self.p[1], self.p[2], self.p[-2], self.p[-1]
        sk, ck = np.sin(k * np.pi / 2), np.cos(k * np.pi / 2)
        val = ws ** k * (a * sk + c * ck) + b * ls ** k
        grad = [ws ** k * sk, ws ** k * ck, ls ** k] + ([0.0] if self.offset else [])
        grad += [k * ws ** (k - 1) * (a * sk + c * ck), b * k * ls ** (k - 1)]
        s = self.span ** k
        return float(val / s), np.asarray(grad) / s

    def derivative(self, k: int) -> Tuple[float, float]:
        val, g = self._deriv_and_grad(k)
        se = float(np.sqrt(max(g @ self.cov @ g, 0.0))) if self.cov is not None else float("nan")
        return val, se

    def original_time_params(self) -> Dict[str, Tuple[float, float]]:
        """Parameters for x = A sin(w t) + C cos(w t) + B exp(lam t) [+ d] on the input
        time axis, each as (value, std_err)."""
        def conv(p):
            w, lam, T = p[-2] / self.span, p[-1] / self.span, self.t_end
            out = [p[0] * np.cos(w * T) + p[1] * np.sin(w * T),
                   -p[0] * np.sin(w * T) + p[1] * np.cos(w * T),
                   p[2] * np.exp(-lam * T)] + ([p[3]] if self.offset else []) + [w, lam]
            return np.asarray(out)
        v = conv(self.p)
        G = np.empty((len(v), len(self.p)))
        for j in range(len(self.p)):
            h = 1e-6 * max(abs(self.p[j]), 1e-3)
            dp = np.zeros_like(self.p)
            dp[j] = h
            G[:, j] = (conv(self.p + dp) - conv(self.p - dp)) / (2 * h)
        se = (np.sqrt(np.maximum(np.diag(G @ self.cov @ G.T), 0)) if self.cov is not None
              else np.full(len(v), np.nan))
        names = ["A_sin", "C_cos", "B_exp"] + (["d_offset"] if self.offset else []) + ["omega", "lambda"]
        return {nm: (float(val), float(s)) for nm, val, s in zip(names, v, se)}


def fit_physics(t: np.ndarray, x: np.ndarray, offset: bool = False,
                init: Optional[Tuple[float, float]] = None) -> PhysicsFit:
    """Fit the sinusoid + exponential model. `init` = (omega, lambda) warm start in 1/s."""
    span = float(t[-1] - t[0])
    u = (t - t[-1]) / span
    if init is not None:
        ws, ls = init[0] * span, float(np.clip(init[1] * span, -LAM_MAX, LAM_MAX))
        beta = np.linalg.lstsq(_basis(u, ws, ls, offset), x, rcond=None)[0]
        starts = [np.r_[beta, ws, ls]]
    else:
        starts = _grid_starts(u, x, offset)
    best = None
    for s in starts:
        p, cost, conv = _levenberg_marquardt(s, u, x, offset)
        if best is None or cost < best[1]:
            best = (p, cost, conv)
    return PhysicsFit(best[0], offset, span, float(t[-1]), len(t), best[1], best[2])


def diagnose(fit: PhysicsFit, t: np.ndarray, x: np.ndarray) -> PhysicsFit:
    """Attach covariance and the checks that decide whether the physics model is usable."""
    u = (t - t[-1]) / fit.span
    J = _jac(fit.p, u, fit.offset)
    resid = x - _model(fit.p, u, fit.offset)
    dof = fit.n - len(fit.p)
    fit.sigma = float(np.sqrt(fit.rss / dof)) if dof > 0 else float("nan")
    if dof > 0:
        fit.cov = fit.sigma ** 2 * np.linalg.pinv(J.T @ J)
    r2 = 1.0 - fit.rss / max(float(np.sum((x - x.mean()) ** 2)), 1e-300)
    negligible = fit.sigma < 1e-9 * float(np.ptp(x))   # exact fit: residuals are rounding noise
    lag1, runs_z = 0.0, 0.0
    if not negligible:
        rc = resid - resid.mean()
        lag1 = float(rc[:-1] @ rc[1:] / max(rc @ rc, 1e-300))
        signs = resid > 0
        n1, n2 = int(signs.sum()), int((~signs).sum())
        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        if n1 and n2:
            mu = 2 * n1 * n2 / (n1 + n2) + 1
            var = (mu - 1) * (mu - 2) / (n1 + n2 - 1)
            runs_z = float((runs - mu) / np.sqrt(max(var, 1e-300)))
    params = fit.original_time_params()
    fit.diagnostics = {"r2": r2, "residual_std": fit.sigma, "resid_lag1_autocorr": lag1,
                       "runs_test_z": runs_z, "jacobian_cond": float(np.linalg.cond(J)),
                       "converged": fit.converged, "params": params}
    # Thresholds are assumed settings (user choices), not physics.
    if not fit.converged:
        fit.problems.append("solver did not converge")
    if fit.p[-2] <= W_MIN * 1.01 or fit.p[-2] > np.pi * (fit.n - 1):
        fit.problems.append("omega at a bound (no resolvable oscillation, or above Nyquist)")
    if abs(fit.p[-1]) >= LAM_MAX:
        fit.problems.append("lambda at its bound")
    if r2 < 0.99:
        fit.problems.append(f"poor fit, R^2 = {r2:.3f}")
    if abs(lag1) > 3.0 / np.sqrt(fit.n) or runs_z < -3.0:  # ~3 standard errors for white noise
        fit.problems.append(f"structured residuals (lag-1 autocorr {lag1:.2f}, runs z {runs_z:.1f})")
    for nm in ("omega", "lambda"):
        val, se = params[nm]
        if not np.isfinite(se) or se > abs(val):
            fit.problems.append(f"{nm} not identified (std err {se:.3g} > |value| {abs(val):.3g})")
    return fit


def _point(v: float) -> Dict:
    return {"level": 0.90, "low": v, "high": v}


def _envelope(bands: List[Dict]) -> Dict:
    """Smallest interval containing every band (used when models disagree)."""
    return {"level": 0.90, "low": min(b["low"] for b in bands),
            "high": max(b["high"] for b in bands), "method": "envelope"}


def _hac_z(d: np.ndarray, lag: int) -> float:
    """Mean of d over its Newey-West standard error (overlapping forecasts are correlated)."""
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    return float(d.mean() / np.sqrt(max(var, 1e-300) / n))


def _select(cands: Dict[str, Dict[int, float]], common: List[int], comparison: Dict,
            lag: int) -> str:
    """Selection rule (thresholds are assumed settings):
    1. among eligible physics variants, keep the one with the lower backtest RMSE;
    2. use it unless the polynomial is significantly better on the paired backtest
       (Diebold-Mariano-style HAC z > 1.645) or its own backtest intervals covered < 70%.
    Rationale: with noisy targets most RMSE gaps sit inside the noise, so "lowest RMSE"
    alone is close to a coin flip; the test rejects physics only on real evidence."""
    phys = [k for k in cands if k != "polynomial"]
    if not phys:
        return "polynomial"
    best = min(phys, key=lambda k: comparison[k]["backtest_rmse_common_origins"])
    d = np.array([cands[best][o] ** 2 - cands["polynomial"][o] ** 2 for o in common])
    z = _hac_z(d, lag) if np.any(d != 0) else 0.0
    c = comparison[best]
    c["paired_z_vs_polynomial"] = z
    if z > Z90:
        c["problems"].append(f"polynomial significantly better in backtest (z = {z:.2f})")
        return "polynomial"
    if c["backtest_interval_coverage"] < 0.70:
        c["problems"].append(f"backtest intervals covered only {c['backtest_interval_coverage']:.0%}")
        return "polynomial"
    return best


class PhysicsKinematicPredictor:
    """Chooses between the physics fit and v2's polynomial by walk-forward backtest."""

    def __init__(self, state: MicroTemporalState, offsets: Tuple[bool, ...] = (False, True),
                 max_rel_error: float = 0.25):
        self.poly = v2.HighSpeedKinematicPredictor(state, max_rel_error=max_rel_error)
        self.t, self.x = self.poly.t, self.poly.x
        self.sampling, self.dt = self.poly.sampling, self.poly.dt
        self.max_rel_error = max_rel_error
        n = len(self.t)
        self.min_fit = max(20, int(MIN_FIT_FRACTION * n))
        self.fits = {off: diagnose(fit_physics(self.t, self.x, off), self.t, self.x) for off in offsets}
        self._origin_fits: Dict[bool, List[PhysicsFit]] = {}

    @staticmethod
    def label(offset: bool) -> str:
        return "physics+offset" if offset else "physics"

    # ---------- backtesting ----------
    def _fits_by_origin(self, offset: bool) -> List[PhysicsFit]:
        """Expanding-window fits at each origin, warm-started from the previous origin
        (no future data used); a fresh grid search every 10 origins guards against
        tracking a bad local minimum."""
        if offset not in self._origin_fits:
            fits, prev = [], None
            for o in range(self.min_fit - 1, len(self.t) - 1):
                ts, xs = self.t[:o + 1], self.x[:o + 1]
                f = fit_physics(ts, xs, offset, init=prev)
                if prev is None or (o - self.min_fit + 1) % 10 == 0:
                    g = fit_physics(ts, xs, offset) if prev is not None else f
                    f = g if g.rss < f.rss else f
                fits.append(diagnose(f, ts, xs))
                prev = (f.omega, f.lam)
            self._origin_fits[offset] = fits
        return self._origin_fits[offset]

    def _target(self, o: int, horizon: float) -> Optional[int]:
        want = self.t[o] + horizon
        if want > self.t[-1] + 0.5 * self.dt:
            return None
        j = int(np.clip(np.searchsorted(self.t, want), o + 1, len(self.t) - 1))
        if j > o + 1 and abs(self.t[j - 1] - want) < abs(self.t[j] - want):
            j -= 1
        return j

    def backtest_physics(self, offset: bool, horizon: float,
                         poly_errs: Dict[int, float]) -> Tuple[Dict[int, float], np.ndarray]:
        """Backtest the live decision rule: at each origin use the physics fit if it passes
        its own checks there, otherwise fall back to the polynomial. Returns the errors and
        the z-scores (error / predicted std) at origins that used physics."""
        errs, zs = {}, []
        floor = 1e-9 * float(np.ptp(self.x))  # rounding-level floor for exact (noise-free) fits
        for i, f in enumerate(self._fits_by_origin(offset)):
            o = self.min_fit - 1 + i
            j = self._target(o, horizon)
            if j is None:
                break
            if not f.problems:
                lead = self.t[j] - self.t[o]
                errs[o] = float(self.x[j] - f.value(lead))
                sd = np.sqrt(f.value_se(lead) ** 2 + f.sigma ** 2 + floor ** 2)
                zs.append(errs[o] / sd)
            elif o in poly_errs:
                errs[o] = poly_errs[o]
        return errs, np.asarray(zs)

    def backtest_poly(self, horizon: float) -> Tuple[Dict[int, float], Dict]:
        sel = self.poly.select_model(horizon)
        w = sel["window"]
        return {w - 1 + i: float(e) for i, e in enumerate(sel["errors"])}, sel

    def validation_horizon(self) -> float:
        """Longest horizon whose backtest still has origins at >= VALIDATION_FRACTION of the record."""
        o_last = int(np.ceil(VALIDATION_FRACTION * (len(self.t) - 1)))
        return float(self.t[-1] - self.t[o_last])

    def physics_max_horizon(self) -> float:
        return float(self.t[-1] - self.t[self.min_fit - 1 + v2.MIN_BACKTEST_ORIGINS - 1])

    # ---------- forecast ----------
    @staticmethod
    def _physics_forecast(fit: PhysicsFit, h: float, interval: float) -> Dict:
        """Physics-model forecast with delta-method bands (reported even if not selected)."""
        x_pred, se_f = fit.value(h), fit.value_se(h)
        pse = float(np.sqrt(se_f ** 2 + fit.sigma ** 2))
        derivs = {}
        for k, nm in enumerate(NAMES, 1):
            val, se = fit.derivative(k)
            derivs[nm] = {"value": val, "std_err": se,
                          "resolved": bool(np.isfinite(se) and se < 0.5 * abs(val))}
        return {"predicted_state": x_pred, "forecast_std_err": se_f,
                "confidence_interval": {"level": interval, "low": x_pred - Z90 * se_f,
                                        "high": x_pred + Z90 * se_f},
                "prediction_interval": {"level": interval, "low": x_pred - Z90 * pse,
                                        "high": x_pred + Z90 * pse},
                "derivatives": derivs, "usable": not fit.problems}

    def project_future_horizon(self, delta_t_future: float, interval: float = 0.90) -> Dict:
        if not np.isfinite(delta_t_future) or delta_t_future <= 0:
            raise ValueError("delta_t_future must be a positive finite number of seconds")
        if abs(interval - 0.90) > 1e-12:
            raise ValueError("only 90% intervals are implemented (numpy has no normal ppf)")
        h = delta_t_future
        h_max = self.poly.max_supported_horizon()
        supported = h <= h_max
        # Compare every candidate at the same horizon: the requested one, capped at the
        # validation horizon. Backtests at longer horizons can only use origins early in the
        # record, where an expanding-window physics fit has seen too little of the motion;
        # those judge "physics with half the data", not the fit actually being used.
        h_cmp = min(h, self.validation_horizon(), self.physics_max_horizon())

        cands: Dict[str, Dict[int, float]] = {}
        comparison: Dict[str, Dict] = {"comparison_horizon_s": h_cmp}
        poly_errs, _ = self.backtest_poly(h_cmp)
        cands["polynomial"] = poly_errs
        comparison["polynomial"] = {"eligible": True, "problems": []}
        for off, fit in self.fits.items():
            name = self.label(off)
            errs, zs = self.backtest_physics(off, h_cmp, poly_errs)
            used = len(zs)
            cover = float(np.mean(np.abs(zs) <= Z90)) if used else float("nan")
            ok = not fit.problems and used >= v2.MIN_BACKTEST_ORIGINS
            comparison[name] = {"eligible": ok, "backtest_origins_using_physics": used,
                                "backtest_interval_coverage": cover,
                                "problems": list(fit.problems) or ([] if ok else [
                                    f"physics passed its checks at only {used} backtest origins"])}
            if ok:
                cands[name] = errs
        common = sorted(set.intersection(*(set(e) for e in cands.values())))
        if len(common) < v2.MIN_BACKTEST_ORIGINS:  # physics cannot be compared fairly here
            cands = {"polynomial": poly_errs}
            common = sorted(poly_errs)
        lag = int(round(h_cmp / self.dt))
        for name, errs in cands.items():
            e2 = np.array([errs[o] ** 2 for o in common])
            comparison[name]["backtest_rmse_common_origins"] = float(np.sqrt(e2.mean()))
        comparison["n_common_origins"] = len(common)
        selected = _select(cands, common, comparison, lag)

        poly_out = self.poly.project_future_horizon(h)
        phys = {self.label(o): self._physics_forecast(f, h, interval) for o, f in self.fits.items()}
        base = {"physics_forecasts": phys,"selected_model": selected, "model_comparison": comparison,
                "physics_fits": {self.label(o): {"params": f.diagnostics["params"],
                                                 "r2": f.diagnostics["r2"],
                                                 "residual_std": f.sigma,
                                                 "resid_lag1_autocorr": f.diagnostics["resid_lag1_autocorr"],
                                                 "runs_test_z": f.diagnostics["runs_test_z"],
                                                 "problems": f.problems}
                                 for o, f in self.fits.items()},
                "polynomial_prediction": poly_out["predicted_state"]}
        usable = [phys[self.label(o)] for o, f in self.fits.items() if not f.problems]
        if selected == "polynomial":
            poly_out.update(base)
            poly_out["confidence_interval"] = None
            poly_out["interval_method"] = "empirical backtest quantiles (v2)"
            if usable:  # widen by model disagreement: envelope with the usable physics band
                poly_out["polynomial_only_interval"] = poly_out["prediction_interval"]
                if poly_out["prediction_interval"] is not None:
                    poly_out["prediction_interval"] = _envelope(
                        [poly_out["prediction_interval"]] + [u["confidence_interval"] for u in usable])
                    poly_out["interval_method"] = ("envelope of the polynomial's empirical band and "
                                                   "the usable physics band (model disagreement)")
                else:
                    poly_out["model_only_interval"] = _envelope(
                        [_point(poly_out["predicted_state"])] + [u["prediction_interval"] for u in usable])
            return poly_out

        fit = self.fits[selected == "physics+offset"]
        pf = phys[selected]
        ci, pi = pf["confidence_interval"], pf["prediction_interval"]
        out = {"t_projection_s": h, "horizon_samples": h / self.dt, "sampling": self.sampling,
               "x_now_smoothed": fit.value(0.0), "x_now_raw": float(self.x[-1]),
               "predicted_state": pf["predicted_state"], "derivatives": pf["derivatives"],
               "max_supported_horizon_s": h_max,
               "interval_method": "delta method: Jacobian covariance (+ residual noise for prediction)"}
        out.update(base)
        if h > h_cmp:
            out["validation_note"] = (f"model chosen by backtest at {h_cmp * 1e3:.3g} ms; the band at "
                                      f"{h * 1e3:.3g} ms comes from the physics model's own uncertainty")
        if not supported:
            out.update(status=(f"UNRELIABLE: horizon {h:.4g} s exceeds what the data can backtest "
                               f"({h_max:.4g} s). Model-only interval shown, unvalidated."),
                       prediction_interval=None, confidence_interval=None, backtest=None,
                       model_only_interval=_envelope([pi, _point(poly_out["predicted_state"])]))
            return out

        # Reliability checks use the backtest at the ACTUAL horizon (v2 logic), not the
        # shorter validation horizon used to choose the model.
        errs_h = cands[selected]
        if h > h_cmp:
            poly_h, _ = self.backtest_poly(h)
            errs_h, _ = self.backtest_physics(selected == "physics+offset", h, poly_h)
            if len(errs_h) < v2.MIN_BACKTEST_ORIGINS:
                out.update(status=(f"UNRELIABLE: the physics forecast cannot be backtested at "
                                   f"{h * 1e3:.3g} ms (too few origins). Band is model-based only."),
                           confidence_interval=ci, prediction_interval=pi, backtest=None)
                return out
        o_sel = sorted(errs_h)
        e = np.array([errs_h[o] for o in o_sel])
        rel = float(np.sqrt(np.mean(e ** 2)) / np.ptp(self.x))
        k = max(2, len(e) // 3)
        trend = float(np.sqrt(np.mean(e[-k:] ** 2)) / max(np.sqrt(np.mean(e[:k] ** 2)), 1e-300))
        x_orig = self.x[o_sel]
        excess = max(0.0, out["x_now_smoothed"] - x_orig.max(),
                     x_orig.min() - out["x_now_smoothed"]) / max(float(np.ptp(x_orig)), 1e-300)
        out["backtest"] = {"n_origins": len(e), "rmse": float(np.sqrt(np.mean(e ** 2))),
                           "rmse_over_series_range": rel, "error_trend_last_vs_first_third": trend,
                           "state_outside_backtest_range": excess,
                           "note": "expanding-window refits; errors correlated across origins"}
        out["confidence_interval"], out["prediction_interval"] = ci, pi
        problems = []  # thresholds are assumed settings, as in v2
        if rel > self.max_rel_error:
            problems.append(f"backtest RMSE is {rel:.0%} of the series range")
        if excess > 0.10:
            problems.append(f"current state lies {excess:.0%} outside the range seen at backtest origins")
        if trend > 2.0 and np.sqrt(np.mean(e[-k:] ** 2)) > 0.01 * float(np.ptp(self.x)):
            problems.append(f"backtest errors grew {trend:.1f}x across origins")
        out["status"] = "OK" if not problems else "UNRELIABLE: " + "; ".join(problems)
        if self.sampling["uneven_sampling"]:
            out["status"] += " | WARNING: uneven sampling (fit uses actual times)"
        return out


# ---------- demo ----------
TRUE_PARAMS = {"A_sin": 1.0, "C_cos": 0.0, "B_exp": float(np.exp(-3.0)), "omega": 200.0, "lambda": 300.0}


def _band(r: Dict) -> str:
    iv = r.get("confidence_interval") or r.get("prediction_interval") or r.get("model_only_interval")
    return "-" if iv is None else f"[{iv['low']:.4f}, {iv['high']:.4f}]"


def _compare(label: str, t: np.ndarray, x: np.ndarray, truth_fn) -> None:
    try:
        import phase_extrapolator as v1  # the original v1, if present alongside
    except ImportError:
        v1 = None
    eng = PhysicsKinematicPredictor(MicroTemporalState(t, x))
    print(f"\n== {label} ==")
    print(f"{'horizon':>8} {'truth':>9} {'v1':>9} {'v2':>9} {'v3':>9} {'physics':>9}  v3 model"
          "        v3 90% band         status")
    for h in (0.0005, 0.005, 0.020):
        r = eng.project_future_horizon(h)
        p1 = (v1.HighSpeedKinematicPredictor(v1.MicroTemporalState(t, x))
              .project_future_horizon(h)["predicted_macro_state"] if v1 else float("nan"))
        print(f"{h * 1e3:>6g}ms {truth_fn(t[-1] + h):9.4f} {p1:9.4f} {r['polynomial_prediction']:9.4f} "
              f"{r['predicted_state']:9.4f} {r['physics_forecasts']['physics']['predicted_state']:9.4f}  "
              f"{r['selected_model']:<15} {_band(r):<19} {r['status'][:48]}")
    print("(physics = the physics-model forecast, shown even when it is not selected)")
    fit = eng.fits[False]
    print(f"physics fit (no offset): R^2 {fit.diagnostics['r2']:.6f}, residual std {fit.sigma:.2e}, "
          f"problems: {fit.problems or 'none'}")
    return eng


def _report_params_and_derivs(eng: "PhysicsKinematicPredictor") -> None:
    fit = eng.fits[False]
    print("  parameter      fitted (+/- std err)        truth")
    for k, (v, se) in fit.diagnostics["params"].items():
        print(f"  {k:<8} {v:14.5g} +/- {se:<10.3g} {TRUE_PARAMS.get(k, float('nan')):12.5g}")
    tr = demo_truth(eng.t[-1])
    print("  derivative     physics (+/- se)              truth        rel.err")
    for k, nm in enumerate(NAMES, 1):
        v, se = fit.derivative(k)
        print(f"  {nm:<12} {v:+.5e} +/- {se:.1e}   {tr[nm]:+.5e}  {(v - tr[nm]) / abs(tr[nm]):+.1e}")


# --- Execution Sandbox ---
if __name__ == "__main__":
    t = np.linspace(0, 0.01, 100)  # 100 samples over 10 ms -> ~9,900 fps
    truth = lambda tt: float(demo_truth(tt)["x"])  # noqa: E731
    x_clean = demo_signal(t)
    x_noisy = x_clean + np.random.default_rng(42).normal(0.0, 0.01, t.size)

    _report_params_and_derivs(_compare("clean demo signal", t, x_clean, truth))
    _report_params_and_derivs(_compare("noisy demo signal (sigma 0.01, seed 42)", t, x_noisy, truth))

    # Robustness: the true signal is NOT of the model form (a second frequency).
    other = lambda tt: float(demo_signal(np.asarray(tt)) + 0.3 * np.sin(2000 * np.asarray(tt)))  # noqa: E731
    x_other = demo_signal(t) + 0.3 * np.sin(2000 * t)
    eng = _compare("off-model signal: demo + 0.3*sin(2000t), clean", t, x_other, other)
    for name, c in eng.project_future_horizon(0.0005)["model_comparison"].items():
        if isinstance(c, dict) and c.get("problems"):
            print(f"  {name}: {'; '.join(c['problems'])}")
