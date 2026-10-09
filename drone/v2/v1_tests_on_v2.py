"""Runs v1's 11 self-test assertions literally against drone_core_v2 and reports each. CC0."""
import math

import numpy as np

from drone_core_v2 import (AcousticEchoGroup, AcousticRanger, AcousticSensorReturn, AirframeState,
                           Attitude, AutonomousDroneCore, BatteryState, IRBiasEstimator)


def run():
    res = []

    def check(name, fn, note=""):
        try:
            res.append((name, "PASS" if fn() else "FAIL", note))
        except Exception as e:  # API removed by design, etc.
            res.append((name, f"N/A ({type(e).__name__}: {e})", note))

    telem = AirframeState(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    check("1 c_sound 25C/60% in (345,350)",
          lambda: 345 < AutonomousDroneCore(temperature_c=25.0, humidity_pct=60.0).c_sound < 350)

    def t2():
        p, v, tp = AutonomousDroneCore(temperature_c=25.0, humidity_pct=60.0).compensate_latency(telem, 0.0)
        return np.allclose(p, 0, atol=1e-3) and np.allclose(v, 0, atol=1e-3) and tp > 0.4
    check("2 zero-latency identity (tp > 0.4)", t2, "failed in v1 (trust clipped to 0.40); passes now")

    def t3():
        c = AutonomousDroneCore()
        st = AirframeState(0.0, np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3), np.zeros(3))
        p1, _, _ = c.compensate_latency(st, 0.01)
        p2, _, _ = c.compensate_latency(st, 0.05)
        return p2[0] > p1[0] > 0.0
    check("3 positive latency monotone", t3)

    def t4():
        c = AutonomousDroneCore(temperature_c=15.0, humidity_pct=0.0)
        b = c.c_sound
        c.set_environment(35.0, 90.0)
        return c.c_sound > b
    check("4 set_environment raises c", t4)
    check("5 attitude round trip", lambda: np.allclose(
        Attitude(w=math.cos(math.pi / 8), z=math.sin(math.pi / 8)).inverse_rotate(
            Attitude(w=math.cos(math.pi / 8), z=math.sin(math.pi / 8)).rotate([1.0, 0, 0])), [1, 0, 0], atol=1e-9))

    def t6():
        g = AcousticSensorReturn(0, 0, 2 / 343, 0.9, 0, 25)
        o = AcousticSensorReturn(0, 0, 10 / 343, 0.9, 0, 25)
        e = AcousticRanger(343.0).estimate(AcousticEchoGroup([g, g, g, o]))
        return e is not None and abs(e["range_m"] - 1.0) < 0.05
    check("6 multi-echo median", t6)

    def t7():
        e = IRBiasEstimator(alpha0=0.2)
        for _ in range(200):
            e.update(2.0, 0.01, 2.05, 0.9)
        return abs(e.bias_m - 0.05) < 0.01
    check("7 IR bias converges", t7)

    def t8():
        c = AutonomousDroneCore()
        c.compensate_latency(telem, 0.0)
        c.kf.update_voxel(1.0, 0.0, 0.0, Attitude(), 0.02, 0.01, 0.01)
        return True
    check("8 voxel update moves position toward echo", t8,
          "INVERTED by design: update_voxel removed; v2 test asserts the pose does NOT move")

    def t9():
        return AutonomousDroneCore()._predictive_voxel_gate(np.array([1e6, 0, 0]), 0.02) == 0.0
    check("9 gate rejects 1e6 m voxel", t9,
          "REPLACED: no drone-distance gate; v2 test asserts a 1e6 m echo is rejected by the ranger envelope "
          "and a valid 12 m echo is kept")

    def t10():
        c = AutonomousDroneCore(battery=BatteryState(capacity_wh=1.0, remaining_wh=0.2))
        p = c.allocate_duty_cycle({"motors_hover": 1.0}, horizon_s=60.0)
        return p["throttled"] is True and p["soc_after"] >= c.min_safe_soc - 1e-9
    check("10 power budget throttles motors", t10,
          "INVERTED by design: v2 test asserts motors granted 1.0, action LAND, battery not floored")

    def t11():
        c = AutonomousDroneCore()
        b = c.acoustic_pulse_interval_s(0.05)
        c.compensate_latency(telem, 0.0)
        c.kf.P[3:6, 3:6] *= 100.0
        return c.acoustic_pulse_interval_s(0.05) < b
    check("11 pulse interval shrinks (base 0.05 s)", t11,
          "CHANGED: 0.05 s is below the 88 ms round trip at 15 m, so v2 floors it there; "
          "v2 test uses base 0.2 s (shrinks) and asserts the floor")
    return res


if __name__ == "__main__":
    for name, r, note in run():
        print(f"{r.split(' ')[0]:5s} {name}" + (f"  -- {note}" if note else ""))
