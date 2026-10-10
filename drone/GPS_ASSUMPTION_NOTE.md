<!-- SPDX-License-Identifier: CC0-1.0 -->
# Note: the "standard GPS" assumption in drone v2/v3 was optimistic

Drone v2 and v3 modelled "ordinary / standard GPS" as a slowly drifting bias of about
0.5 m (1-sigma per axis, Gauss-Markov, tau 30 s) plus small white noise.

The later GNSS error budget in [`../gnss/`](../gnss/) (v2) puts single-frequency L1 GPS,
L1+SBAS, L1+L5 and L1+L5+SBAS at roughly 1-5 m of slow, correlated error (tau minutes),
not 0.5 m. Run through the same SLAM-only recon Monte Carlo (no markers, no pad,
fresh held-out seed 20264008), those configurations reached 0-29% median coverage.

What this changes:
- The SLAM-only "standard GPS" coverage numbers in drone v2/v3 (about 66-94%) are
  optimistic for real single-frequency receivers.
- Results that do NOT depend on that assumption still hold: RTK, and the v3
  ground-reference mode (surveyed pad + markers) which removes the common GPS offset.
- Without markers, only L1+L5 with converged PPP (quiet space weather) reached
  >= 90% median coverage on all three templates in the GNSS v2 test. PPP must
  converge before take-off (typically tens of minutes for float PPP).

All GNSS numbers are order-of-magnitude assumptions; see `../gnss/README.md`. By Grok, CC0.
