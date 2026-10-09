"""
Tests for drone_uplink_v2.py. Run: python3 -m unittest test_drone_uplink_v2 -v
License: CC0 1.0 Universal (public domain).
"""
import json
import math
import os
import tempfile
import time
import unittest

import numpy as np

import drone_core_v2 as v2
import drone_recon_v2 as rc
import drone_uplink_v2 as up


def tick_output(action="TRACKING_NOMINAL", mission="NOMINAL", t=0.0):
    core = v2.AutonomousDroneCore()
    tm = v2.AirframeState(t, np.array([1.0, 2.0, 3.0]), np.array([4.0, 0, 0]), np.zeros(3), np.zeros(3))
    tm.gyro_rps = np.zeros(3)
    out = core.process_flight_tick(tm, 0.05, [])
    out["flight_guidance"]["action"] = action
    out["flight_guidance"]["mission_action"] = mission
    return out


class TestStream(unittest.TestCase):
    def test_round_trip(self):
        out = tick_output()
        body = up.build_stream_body(out, 7, 12.345)
        line = up.encode_message(body)
        self.assertTrue(line.endswith(b"\n"))
        self.assertEqual(line.count(b"\n"), 1)
        back = up.decode_message(line)
        self.assertEqual(back, body)
        for k in ("v", "seq", "t", "pos", "pos_sig", "yaw", "yaw_sig", "mag_ok", "act", "near", "soc"):
            self.assertIn(k, back)
        self.assertEqual(back["seq"], 7)
        self.assertEqual(back["v"], up.SCHEMA_VERSION)
        self.assertLess(len(line), 400)                 # compact

    def test_crc_detects_corruption(self):
        line = up.encode_message(up.build_stream_body(tick_output(), 1, 1.0))
        i = line.index(b'"seq":1') + 6
        bad = line[:i] + b"2" + line[i + 1:]             # seq 1 -> 2, still valid JSON
        with self.assertRaises(ValueError) as cm:
            up.decode_message(bad)
        self.assertIn("CRC", str(cm.exception))
        with self.assertRaises(ValueError):
            up.decode_message(line[:-20])                 # truncated
        body = up.build_stream_body(tick_output(), 1, 1.0)
        body["v"] = 99
        with self.assertRaises(ValueError):
            up.decode_message(up.encode_message(body))   # unknown schema version

    def test_rate_limit_and_bandwidth(self):
        tr = up.InMemoryTransport()
        cfg = up.UplinkConfig(bandwidth_bytes_per_s=1000.0, burst_bytes=2000.0)
        m = up.UplinkManager(tr, cfg)
        out = tick_output()
        for k in range(200):                              # 10 s at 20 Hz
            t = 0.05 * k
            m.on_tick(out, t)
            m.flush(t)
        self.assertEqual(m.stats["events"], 1)            # "start"
        self.assertEqual(m.stats["states"], 49)           # 5 Hz over 10 s (+ the start event)
        self.assertLessEqual(m.stats["sent_bytes"], 1000.0 * 9.95 + 2000.0)
        seqs = [up.decode_message(b)["seq"] for b in tr.sent]
        self.assertEqual(seqs, sorted(seqs))

    def test_queue_keeps_safety_events_during_outage(self):
        tr = up.InMemoryTransport()
        cfg = up.UplinkConfig(max_state_queue=20)
        m = up.UplinkManager(tr, cfg)
        acts = ["TRACKING_NOMINAL", "CAUTION_SLOW", "COLLISION_AVOIDANCE_BRAKE", "TRACKING_NOMINAL"]
        tr.up = False
        n_changes = 0
        prev = None
        for k in range(1200):                             # 60 s outage
            a = acts[(k // 50) % 4]
            n_changes += int(a != prev)
            prev = a
            m.on_tick(tick_output(a, t=0.05 * k), 0.05 * k)
            m.flush(0.05 * k)
        self.assertEqual(tr.sent, [])
        q = m.queued()
        self.assertEqual(q["states"], 20)                 # bounded, oldest dropped
        self.assertGreater(m.stats["dropped_state"], 0)
        self.assertEqual(m.stats["dropped_event"], 0)
        self.assertEqual(q["events"], m.stats["events"])
        self.assertGreaterEqual(m.stats["events"], n_changes)
        tr.up = True
        t = 60.0
        while sum(m.queued().values()):
            t += 0.5
            m.flush(t)
        bodies = [up.decode_message(b) for b in tr.sent]
        kinds = [b["kind"] for b in bodies]
        n_ev = m.stats["events"]
        self.assertEqual(kinds[:n_ev], ["event"] * n_ev)  # events go first
        self.assertTrue(any(b["act"] == "COLLISION_AVOIDANCE_BRAKE" for b in bodies[:n_ev]))
        ev_seq = [b["seq"] for b in bodies[:n_ev]]
        self.assertEqual(ev_seq, sorted(ev_seq))

    def test_never_blocks_flight_loop(self):
        tr = up.InMemoryTransport()
        tr.delay_s = 0.1                                   # slow link
        m = up.UplinkManager(tr, threaded=True)
        out = tick_output()
        worst = 0.0
        for k in range(40):
            t0 = time.perf_counter()
            m.on_tick(out, 0.2 * k)                         # every call is due (5 Hz)
            worst = max(worst, time.perf_counter() - t0)
        m.close()
        self.assertLess(worst, 0.01)
        self.assertLess(m.stats["max_on_tick_s"], 0.01)

    def test_transport_errors_do_not_propagate(self):
        class Broken(up.Transport):
            def is_up(self):
                return True

            def send(self, data):
                raise OSError("link exploded")
        m = up.UplinkManager(Broken())
        m.on_tick(tick_output(), 0.0)
        self.assertEqual(m.flush(0.0), 0)
        self.assertEqual(m.stats["transport_errors"], 1)
        self.assertEqual(m.queued()["events"], 1)          # kept for a retry

    def test_file_transport(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stream.jsonl")
            m = up.UplinkManager(up.FileTransport(path))
            for k in range(10):
                m.on_tick(tick_output(t=0.2 * k), 0.2 * k)
                m.flush(0.2 * k)
            with open(path, "rb") as f:
                lines = f.read().splitlines(keepends=True)
            self.assertEqual(len(lines), 10)
            self.assertEqual([up.decode_message(x)["seq"] for x in lines], list(range(1, 11)))


class TestMapUpload(unittest.TestCase):
    def core_with_map(self):
        core = v2.AutonomousDroneCore(heading_init_sigma_rad=math.radians(5.0))
        for k, r in enumerate((4.0, 8.0, 12.0)):
            tm = v2.AirframeState(0.05 * k, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                                  attitude=v2.Attitude.from_yaw(0.3 * k), pos_sigma_m=np.full(3, 0.05))
            tm.gyro_rps = np.zeros(3)
            g = v2.AcousticEchoGroup([v2.AcousticSensorReturn(0, 0, 2 * r / core.c_sound, 0.9, 0.05 * k, 25)] * 3)
            core.process_flight_tick(tm, 0.0, [g])
        return core

    def test_export_round_trip_and_checksum(self):
        core = self.core_with_map()
        exp = up.export_map(core, {"flight_s": 0.15, "collided": False}, created_t=0.15)
        self.assertEqual(len(exp["points"]), len(core.map.items))
        # covariance includes the heading term: lateral sigma ~ range * heading sigma (> sensor-only)
        far = max(exp["points"], key=lambda p: np.linalg.norm(p["pos"]))
        cov = np.array(far["cov"])
        self.assertGreater(math.sqrt(np.linalg.eigvalsh(cov).max()), 12.0 * math.radians(4.0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "map.json")
            man = up.write_map_file(path, exp)
            back = up.read_map_file(path, man["sha256"])
            self.assertEqual(back, json.loads(up.map_to_bytes(exp)))
            with open(path + ".sha256") as f:
                self.assertTrue(f.read().startswith(man["sha256"]))
            with open(path, "rb") as f:
                data = bytearray(f.read())
            data[10] ^= 0x01
            with open(path, "wb") as f:
                f.write(bytes(data))
            with self.assertRaises(ValueError):
                up.read_map_file(path, man["sha256"])

    def test_chunked_resume(self):
        exp = up.export_map(self.core_with_map(), {"note": "x" * 3000})
        data = up.map_to_bytes(exp)
        man = up.make_manifest(data, 256, "flight-1")
        self.assertGreater(man["n_chunks"], 5)
        sink = up.InMemoryChunkSink()
        sink.fail_after = 3                                 # link drops after 3 chunks
        r1 = up.upload_chunked(data, man, sink)
        self.assertEqual((r1["sent_this_call"], r1["complete"], r1["interrupted"]), (3, False, True))
        with self.assertRaises(ValueError):
            sink.assemble(man)
        sink.fail_after = None
        r2 = up.upload_chunked(data, man, sink)
        self.assertEqual(r2["sent_this_call"], man["n_chunks"] - 3)   # only the missing ones
        self.assertTrue(r2["complete"])
        self.assertEqual(sink.assemble(man), data)
        self.assertEqual(up.sha256_hex(sink.assemble(man)), man["sha256"])
        r3 = up.upload_chunked(data, man, sink)
        self.assertEqual(r3["sent_this_call"], 0)

    def test_corrupted_chunk_rejected(self):
        data = b"abcdefgh" * 100
        man = up.make_manifest(data, 64, "u")
        sink = up.InMemoryChunkSink()
        self.assertFalse(sink.put_chunk("u", 0, b"X" + data[1:64], man["chunk_sha256"][0]))
        self.assertEqual(sink.received("u"), set())



def build_map(n_pings=30, seed=0):
    """A small recon map: pings at a wall x = 6 from a drone moving along y."""
    rng = np.random.default_rng(seed)
    m = rc.ReconMap()
    return m, rng


def add_pings(m, rng, k0, k1):
    for k in range(k0, k1):
        o = np.array([0.0, 0.2 * k, 2.0])
        beams = []
        for _ in range(9):
            u = np.array([1.0, rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)])
            u /= np.linalg.norm(u)
            r = 6.0 / u[0]
            cov = rc.point_covariance(r, u, 0.015, math.radians(1.0), math.radians(1.0))
            beams.append((u, r, cov))
        m.add_ping(o, beams, 0.015, 0.1 * k)


class AlwaysUp:
    """Minimal link stub for MapSync tests."""
    def __init__(self, cloud, up_=True):
        self.cloud, self.up_ = cloud, up_

    def any_up(self):
        return self.up_

    def send(self, data):
        if not self.up_:
            return False
        self.cloud.receive(data)
        return True


class TestIncrementalCloudMap(unittest.TestCase):
    def test_deltas_rebuild_identical_map(self):
        m, rng = build_map()
        sync = up.MapSync(m, max_entries_per_msg=50)
        cloud = up.CloudMapAssembler()
        sync.add_channel(AlwaysUp(cloud))
        for step in range(5):
            add_pings(m, rng, 10 * step, 10 * step + 10)
            sync.collect(float(step))
            sync.pump(float(step))
        self.assertGreater(sync.stats["delta_msgs"], 5)
        self.assertEqual(cloud.map_hash(), sync.map_hash())
        cells, pts = sync.quantized_state()
        self.assertEqual(len(cloud.cells), len(cells))
        self.assertEqual(len(cloud.points), len(m.points))

    def test_out_of_order_and_duplicate_deltas(self):
        m, rng = build_map(seed=1)
        sync = up.MapSync(m, max_entries_per_msg=40)
        msgs = []
        for step in range(6):
            add_pings(m, rng, 8 * step, 8 * step + 8)
            msgs += sync.collect(float(step))
        order = np.random.default_rng(2).permutation(len(msgs))
        shuffled = [msgs[i] for i in order] + msgs[:5]        # reversed-ish order + duplicates
        cloud = up.CloudMapAssembler()
        for d in shuffled:
            cloud.receive(d)
        self.assertEqual(cloud.stats["duplicates"], 5)
        self.assertEqual(cloud.missing_deltas(), [])
        self.assertEqual(cloud.map_hash(), sync.map_hash())

    def test_outage_then_snapshot_resync(self):
        m, rng = build_map(seed=3)
        sync = up.MapSync(m, max_entries_per_msg=40)
        cloud = up.CloudMapAssembler()
        link = AlwaysUp(cloud)
        ch = sync.add_channel(link, max_backlog_bytes=3000)
        add_pings(m, rng, 0, 5); sync.collect(0.0); sync.pump(0.0)
        r0 = ch.stats["resyncs"]
        self.assertEqual(cloud.map_hash(), sync.map_hash())
        link.up_ = False                                       # outage
        for step in range(1, 6):
            add_pings(m, rng, 5 * step, 5 * step + 5)
            sync.collect(float(step)); sync.pump(float(step))
        self.assertGreater(ch.stats["dropped_msgs"], 0)        # bounded backlog dropped deltas
        self.assertTrue(ch.need_snapshot)
        self.assertNotEqual(cloud.map_hash(), sync.map_hash())
        link.up_ = True
        add_pings(m, rng, 30, 33); sync.collect(6.0)
        sync.pump(6.0)
        self.assertEqual(ch.stats["resyncs"], r0 + 1)
        self.assertEqual(cloud.map_hash(), sync.map_hash())

    def test_corrupted_map_frame_rejected(self):
        m, rng = build_map(seed=4)
        sync = up.MapSync(m)
        add_pings(m, rng, 0, 3)
        d = bytearray(sync.collect(0.0)[0])
        d[20] ^= 0xFF
        cloud = up.CloudMapAssembler()
        self.assertFalse(cloud.receive(bytes(d)))
        self.assertEqual(cloud.stats["bad_crc"], 1)
        self.assertEqual(len(cloud.cells), 0)


class TestLinks(unittest.TestCase):
    def test_link_failover(self):
        got = []
        deliver = lambda t, d: got.append((t, d))
        rng = np.random.default_rng(0)
        wifi = up.SimLink(up.LINK_PROFILES["wifi"], rng, deliver, station_pos=np.zeros(3))
        lte = up.SimLink(up.LINK_PROFILES["lte"], rng, deliver)
        sat = up.SimLink(up.LINK_PROFILES["satellite"], rng, deliver)
        for l in (wifi, lte, sat):
            l._next_switch = 1e9                                # no random outages in this test
        lm = up.LinkManager([sat, lte, wifi])
        for l in (wifi, lte, sat):
            l.step(0.1, 0.1, np.array([10.0, 0, 0]))
        self.assertTrue(lm.send(b"x" * 100))
        self.assertEqual(lm.current, "wifi")                    # preferred when in range
        for l in (wifi, lte, sat):
            l.step(0.2, 0.1, np.array([500.0, 0, 0]))           # out of WiFi range
        self.assertFalse(wifi.is_up())
        self.assertTrue(lm.send(b"x" * 100))
        self.assertEqual(lm.current, "lte")
        lte._up = False                                         # LTE coverage gap
        self.assertTrue(lm.send(b"x" * 100))
        self.assertEqual(lm.current, "satellite")
        self.assertEqual(lm.failovers, 2)
        sat._up = False
        self.assertFalse(lm.send(b"x" * 100))                   # nothing up: refused, no exception
        self.assertEqual(len(got), 3)
        self.assertAlmostEqual(got[-1][0], 0.2 + up.LINK_PROFILES["satellite"].latency_s)

    def test_bandwidth_budget(self):
        rng = np.random.default_rng(0)
        l = up.SimLink(up.LinkProfile("x", 1000.0, 0.0, 1e9, 1.0), rng, lambda t, d: None)
        l.step(0.1, 0.1)
        self.assertTrue(l.send(b"a" * 90))
        self.assertFalse(l.send(b"a" * 90))                     # 100 B budget per 0.1 s

    def test_satellite_terminal_is_optional_power_load(self):
        core = v2.AutonomousDroneCore()
        base = core.allocate_duty_cycle({"motors_hover": 1.0, "avionics": 1.0}, horizon_s=10.0)
        with_sat = core.allocate_duty_cycle({"motors_hover": 1.0, "avionics": 1.0, "satellite_terminal": 1.0},
                                            horizon_s=10.0)
        extra_w = (with_sat["energy_needed_wh"] - base["energy_needed_wh"]) * 3600.0 / 10.0
        self.assertAlmostEqual(extra_w, v2.SAT_TERMINAL_W, delta=0.1)
        low = v2.AutonomousDroneCore(battery=v2.BatteryState(capacity_wh=100.0, remaining_wh=20.0))
        pwr = low.allocate_duty_cycle({"motors_hover": 1.0, "satellite_terminal": 1.0}, horizon_s=1.0)
        self.assertIn("satellite_terminal", pwr["shed_loads"])     # shed before flight-critical loads
        self.assertEqual(pwr["granted_fractions"]["motors_hover"], 1.0)

if __name__ == "__main__":
    unittest.main()
