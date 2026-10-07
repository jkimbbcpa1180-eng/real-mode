# Laser Charging Nightstand: Design Doc (CC0)

An open design for a fixed-spot infrared overnight phone charger. Not legal advice. All figures come from `prior_art.md`.

## 1. Goal
Charge a phone overnight with infrared light while it sits in one fixed spot on a nightstand,
and publish every design, measurement and failure openly so others can build on it.
"Fixed spot" is the point: no charging anywhere in the room, so no tracking and the beam path stays short and simple.

## 2. Honest status
- Nobody has shown IR wireless charging at real phone power (5 W or more).
- Wi-Charge, the only product on sale, delivers 0.1–0.3 W at about 15% efficiency and powers locks, not phones
  ([specs](https://wi-charge.com/technology/specifications), [The Verge](https://www.theverge.com/tech/663899/wi-charge-alfred-smart-lock-wireless-power-review)).
- The best phone demo turned more than 5 W of light into about 0.6 W of electricity at 2 m.
  Air losses were near zero, and the PV cell (about 12%) was the bottleneck ([arXiv 2105.13174](https://ar5iv.labs.arxiv.org/html/2105.13174)).
- So this project focuses on the receiver.

## 3. Architecture
```
[IR source] -> [lens/homogenizer] -> (optional: 1 fixed mirror) -> [GaAs cell on heat spreader] -> [phone]
     ^                                                                       |
     +------ 2.4 GHz BLE/Wi-Fi feedback: received power, temperature --------+
[guard beams around the path] -> [interlock MCU] -> source OFF
```
- **Wavelength matched to the cell:** GaAs cells peak at 808–850 nm. They reach 55.1% at 808 nm at room temperature
  ([AIP Advances](https://pubs.aip.org/aip/adv/article/9/10/105206/151680/Enhanced-efficiency-in-808-nm-GaAs-laser-power)),
  and a 2024 triple-junction cell reached 66.5% at 840–860 nm
  ([Cell Rep. Phys. Sci.](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(24)00568-X)).
  Cheap silicon panels are used only in Phase 0, with 940 nm light.
- **Beam shaping:** a lens or homogenizer spreads the light evenly over the cell to avoid hot spots.
- **Cooling:** start with copper or aluminum and compare against CVD diamond, which conducts more than 2200 W/mK
  ([Coherent](https://www.coherent.com/news/blog/diamond-heat-spreaders)).
- **Feedback link:** a low-power 2.4 GHz Bluetooth/Wi-Fi link where the phone or receiver reports the power it gets. It is used for
  (a) aiming and calibration and (b) a check of power sent against power received: a sudden gap means something is blocking the beam, so the source shuts off.
- **Mirrors:** at most one fixed mirror for a single redirect. Every bounce loses power and adds a stray-beam risk.
- **Magnet dock:** a MagSafe-style magnet holds the phone in exactly the same spot every night, so the beam always hits the cell.
- **Guard-beam cutoff:** IR break-beams surround the power path, and breaking one cuts the source. Shut-off time is measured and published.

## 4. Safety first
- Lasers are classed under IEC 60825-1 (1, 1M, 2, 2M, 3R, 3B, 4)
  ([IEC](https://webstore.iec.ch/en/publication/3587), [ANSI blog](https://blog.ansi.org/ansi/laser-class-safety-1-1c-1m-2-2m-3r-3b-4/)).
  Wi-Charge's product is Class 1 ([safety](https://encode.wi-charge.com/safety)).
- 808/850/976 nm light is invisible and focuses on the retina, so there is no blink reflex
  ([LBL](https://ehs.lbl.gov/resource/laser-classification-explanation/)).
- **Hobbyists: no bare 808/980 nm laser diodes.** They are Class 3B/4 eye hazards.
- Interlock requirements for any build above Phase 0:
  1. Guard beams fail safe: a lost sensor signal means the source is OFF.
  2. A sent-vs-received power check runs alongside them as a second, independent trip.
  3. Shut-off time is measured, logged and published. For reference, PowerLight's guard beams cut off in 1–8 ms ([PowerLight](https://powerlighttech.com/safety/)).
  4. The source stays off at startup until every check passes.

## 5. Phased build plan
**Phase 0: eye-safe measurement rig (LED only, no laser).** Expect microwatts to milliwatts. This phase is for measuring, not charging.
| Part | Price | Link |
|---|---|---|
| 940 nm IR LEDs, 25 pack | $7.95 | [Adafruit 388](https://www.adafruit.com/product/388) |
| IR break-beam sensor | $5.95 | [Adafruit 2168](https://www.adafruit.com/product/2168) |
| INA219 power meter | $9.95 | [Adafruit 904](https://www.adafruit.com/product/904) |
| 6 V 1 W solar panel (was out of stock) | $19.95 | [Adafruit 3809](https://www.adafruit.com/product/3809) |
| Alt: 1.2 W 6 V panel | see page | [SparkFun](https://www.sparkfun.com/small-solar-panel-1-2-watt-6-volt-etfe.html) |
| Arduino or ESP32 | not priced | - |

**Phase 1: cutoff.** Build the guard-beam interlock and the BLE sent-vs-received check, measure the shut-off time over many trials, and publish the firmware.
**Phase 2: cooling tests.** Compare bare, aluminum, copper and diamond spreaders and log cell temperature against output.
**Phase 3+: higher power.** Only with proper laser safety: a classified enclosure, IEC 60825-1 review, eyewear and a supervised lab. No home builds.

## 6. Open gaps this project targets
1. **Open safety interlock:** guard ring plus received-power check, with open firmware and measured shut-off time. Every commercial version is proprietary.
2. **Open measurement data:** efficiency against distance, wavelength, irradiance and temperature for cheap cells (Si, GaAs).
3. **Receiver cooling study:** diamond vs. copper vs. aluminum. No optical wireless power demo with a diamond-mounted cell was found
   (related: [2026 thermal study](https://citius.gal/documents/24672/mIXlcKOVfd_vKNUrwENLef/1-s2.0-s2590123026036911-main_20260910080722405.pdf)).

## 7. Patent note (not legal advice)
- Wi-Charge holds patents on the mirror-cavity/retroreflector resonator ([US8525097B2](https://patents.google.com/patent/US8525097B2/en),
  [US20160087391A1](https://patents.google.com/patent/US20160087391A1/en)), on power-loss safety monitoring ([US12176727](https://exa.ai/library/legal/patent/5lw07xgw27d))
  and on fail-safe dual sensors ([US12401428](https://exa.ai/library/legal/patent/1b8rl583sxm)).
- This design avoids resonant/retroreflector cavities and uses a simple direct beam to a fixed spot.
- The sent-vs-received check may overlap US12176727. Get a freedom-to-operate review before relying on it, or keep guard beams as the main interlock.
- "Looks more open" is unverified. Ask a patent professional before any commercial use.

## 8. Data-logging template
| Date | Source / λ (nm) | Optical in (mW) | Distance (cm) | Cell type | Spreader | Cell temp (°C) | Electrical out (mW) | Efficiency (%) | Cutoff time (ms) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | LED 940 | | | Si panel | none | | | | | |

Efficiency = electrical out ÷ optical in. Log failures too.

## 9. License
CC0 1.0 Universal: public domain. Copy, modify and sell it with no permission needed.

*Leave a 1 for the next 0.*
