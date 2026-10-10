<!-- SPDX-License-Identifier: CC0-1.0 -->
# drone

Simulation code for a small recon drone, in two versions:

- [`v2/`](v2/): flight core with a drag-consistent Kalman filter, map uplink, sonar recon
  mapping and scan-to-map SLAM, with tests, Monte Carlo scripts and results.
  See `v2/README.md`.
- [`v3/`](v3/): six features built on v2 without changing it: offline safe return,
  look-closer re-passes, a swarm with relay and reciprocal avoidance, change detection,
  lidar and thermal. Each feature has tests and Monte Carlo results on held-out seeds,
  including the failure cases. See `v3/README.md`.

Run the tests with `python3 -m unittest` inside `v2/` or `v3/` (stdlib + numpy;
matplotlib for the renders).

Everything here is simulated, and the sensor and power numbers are assumptions.

Built by Grok. License: CC0 1.0 Universal (public domain).


> **Correction:** the 0.5 m "standard GPS" assumption used in v2/v3 was optimistic; see [GPS_ASSUMPTION_NOTE.md](GPS_ASSUMPTION_NOTE.md).
