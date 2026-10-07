#!/usr/bin/env python3
"""Aura Codex — Step 1: protein, energy, and modeled thermogenesis tracker.

THESIS: A higher-protein meal may be associated with higher or steadier
self-reported energy than a calorie-matched, lower-protein control meal.

CURRENT STEP: Test food input and internal state first. Thermogenesis values
are model estimates, not sensor measurements or evidence of an aura.

ESS = 0.45*energy + 0.35*mood + 0.20*(10-hunger)
P_DIT(t) = Q_heat * (t/tau^2) * exp(-t/tau)
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev

DEFAULT_LOG = Path("aura_codex_log.csv")
DEFAULT_PROTOCOL = Path("aura_codex_step1_protocol.json")
FOLLOWUP_MINUTES = (0.0, 30.0, 60.0, 120.0, 180.0)
TIME_TOLERANCE_MINUTES = 3.0
PRIMARY_OUTCOMES = (
    "mean_ess",
    "ess_change_180_minus_0",
    "ess_variance",
    "ess_auc_score_minutes",
)

# Transparent model constants, not direct readings.
PROTEIN_ENERGY_J_G = 16_700.0
DEFAULT_PROTEIN_TEF = 0.25
ABSORPTION_TAU_MINUTES = 60.0
CORE_TEMPERATURE_K = 310.15
PROTEIN_SATURATION_KM_G = 35.0

FIELDS = [
    "recorded_at", "session", "condition", "minutes_after_start",
    "protein_g", "total_calories", "carbs_g", "fat_g", "exercise_minutes",
    "energy_0_10", "mood_0_10", "hunger_0_10", "sleep_hours",
    "caffeine_mg", "notes",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ess(energy: float, mood: float, hunger: float) -> float:
    """Pre-specified 0–10 self-report index."""
    if not all(math.isfinite(v) and 0.0 <= v <= 10.0 for v in (energy, mood, hunger)):
        raise ValueError("energy, mood, and hunger must each be finite values from 0 to 10")
    return 0.45 * energy + 0.35 * mood + 0.20 * (10.0 - hunger)


def modeled_marginal_protein_coefficient(protein_g: float, k_half_g: float = PROTEIN_SATURATION_KM_G) -> float:
    """Derivative of g/(K+g): a unit-bearing illustrative saturation slope.

    This is a visualization coefficient only, not a validated biological
    measure of protein utilization.
    """
    if protein_g < 0 or k_half_g <= 0:
        raise ValueError("protein grams must be non-negative and k_half_g must be positive")
    return k_half_g / (k_half_g + protein_g) ** 2


def modeled_protein_dit_power_w(protein_g: float, minutes_after_start: float) -> float:
    """Gamma-shaped estimate of protein-related diet-induced thermogenesis."""
    if protein_g < 0 or minutes_after_start < 0:
        raise ValueError("protein grams and elapsed minutes cannot be negative")
    if protein_g == 0 or minutes_after_start == 0:
        return 0.0
    t = minutes_after_start * 60.0
    tau = ABSORPTION_TAU_MINUTES * 60.0
    q_heat = protein_g * PROTEIN_ENERGY_J_G * DEFAULT_PROTEIN_TEF
    return q_heat * (t / tau**2) * math.exp(-t / tau)


def modeled_entropy_proxy_w_k(power_w: float) -> float:
    """Simplified P/T dissipation proxy, not measured entropy production."""
    return power_w / CORE_TEMPERATURE_K


def modeled_dit_capture_fraction(minutes: float) -> float:
    x = max(0.0, minutes) / ABSORPTION_TAU_MINUTES
    return 1.0 - math.exp(-x) * (1.0 + x)


def modeled_dit_heat_analytic_j(protein_g: float, minutes: float) -> float:
    """Exact closed-form definite integral of Gamma-shaped DIT power: int_0^T P(t) dt."""
    if protein_g <= 0.0 or minutes <= 0.0:
        return 0.0
    q_total = protein_g * PROTEIN_ENERGY_J_G * DEFAULT_PROTEIN_TEF
    return q_total * modeled_dit_capture_fraction(minutes)


def ensure_log(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()


def make_protocol(sessions: int, seed: int | None, output: Path) -> None:
    if sessions < 8 or sessions % 2:
        raise ValueError("sessions must be an even number of at least 8")
    conditions = ["protein"] * (sessions // 2) + ["control"] * (sessions // 2)
    random.Random(seed).shuffle(conditions)
    protocol = {
        "title": "Aura Codex — Step 1 Protein/Energy Protocol",
        "created_at": utcnow(),
        "thesis": "A higher-protein meal may be associated with higher or steadier self-reported energy than a calorie-matched, lower-protein control meal.",
        "current_step": "Step 1: dietary input and internal-state tracking. No external-field claim is tested.",
        "pre_specified_primary_outcomes": list(PRIMARY_OUTCOMES),
        "equations": {
            "ess": "0.45*energy + 0.35*mood + 0.20*(10-hunger)",
            "modeled_dit_power": "Q_heat*(t/tau^2)*exp(-t/tau)",
            "modeled_heat_total_analytic": "protein_g*16,700 J/g*0.25*[1 - exp(-t/tau)*(1 + t/tau)]",
            "illustrative_marginal_protein_slope": "K/(K+protein_g)^2",
        },
        "randomized_session_order": [
            {"session": f"s{i + 1:02d}", "condition": condition}
            for i, condition in enumerate(conditions)
        ],
        "required_followup_minutes": list(FOLLOWUP_MINUTES),
        "time_tolerance_minutes": TIME_TOLERANCE_MINUTES,
        "controls": [
            "Match total calories as closely as practical between conditions.",
            "Protein = clearly higher-protein meal; control = lower-protein meal with similar calories.",
            "Keep timing, hydration, sleep opportunity, caffeine, and exercise similar; log deviations.",
            "Do not change score weights or outcomes after viewing results.",
        ],
        "interpretation": "DIT, entropy, and saturation fields are modeled estimates. ESS is a self-report index, not a medical metric.",
    }
    output.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def add_observation(args: argparse.Namespace) -> None:
    numeric_inputs = (args.protein_g, args.total_calories, args.carbs_g, args.fat_g, args.exercise_minutes)
    if any(value < 0 for value in numeric_inputs):
        raise ValueError("food and exercise inputs cannot be negative")
    if args.condition == "protein" and args.protein_g <= 0:
        raise ValueError("protein sessions require --protein-g greater than zero")
    ensure_log(args.log)
    row = {
        "recorded_at": utcnow(), "session": args.session, "condition": args.condition,
        "minutes_after_start": args.minutes, "protein_g": args.protein_g,
        "total_calories": args.total_calories, "carbs_g": args.carbs_g,
        "fat_g": args.fat_g, "exercise_minutes": args.exercise_minutes,
        "energy_0_10": args.energy, "mood_0_10": args.mood, "hunger_0_10": args.hunger,
        "sleep_hours": args.sleep_hours, "caffeine_mg": args.caffeine_mg, "notes": args.notes or "",
    }
    with args.log.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
    dit_w = modeled_protein_dit_power_w(args.protein_g, args.minutes)
    marginal = modeled_marginal_protein_coefficient(args.protein_g)
    print(
        f"logged ESS={ess(args.energy, args.mood, args.hunger):.2f}/10 | "
        f"modeled DIT={dit_w:.2f} W | modeled P/T={modeled_entropy_proxy_w_k(dit_w):.4f} W/K | "
        f"illustrative marginal slope={marginal:.5f} 1/g | {args.condition} | session={args.session}"
    )


def trapezoid_auc(points: list[tuple[float, float]]) -> float:
    ordered = sorted(points)
    return sum(
        (ordered[i + 1][0] - ordered[i][0]) * (ordered[i][1] + ordered[i + 1][1]) / 2.0
        for i in range(len(ordered) - 1)
    )


def required_points(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """Select one nearest reading within tolerance for every required timepoint."""
    selected: list[tuple[float, float]] = []
    used: set[int] = set()
    for target in FOLLOWUP_MINUTES:
        candidates = [
            (abs(observed - target), index, observed, score)
            for index, (observed, score) in enumerate(points)
            if index not in used and abs(observed - target) <= TIME_TOLERANCE_MINUTES
        ]
        if not candidates:
            return None
        _, index, observed, score = min(candidates)
        used.add(index)
        selected.append((observed, score))
    return selected


def group_mean(rows: list[dict], field: str, condition: str) -> float | None:
    values = [row[field] for row in rows if row["condition"] == condition]
    return round(mean(values), 3) if values else None


def analyze(path: Path, output: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("the log contains no observations")

    sessions: dict[str, dict] = {}
    for row in rows:
        try:
            score = ess(float(row["energy_0_10"]), float(row["mood_0_10"]), float(row["hunger_0_10"]))
            metadata = {
                "condition": row["condition"], "protein_g": float(row["protein_g"]),
                "total_calories": float(row["total_calories"]), "carbs_g": float(row["carbs_g"]),
                "fat_g": float(row["fat_g"]), "exercise_minutes": float(row["exercise_minutes"]),
                "sleep_hours": float(row["sleep_hours"]), "caffeine_mg": float(row["caffeine_mg"]),
            }
            if metadata["condition"] not in {"protein", "control"}:
                raise ValueError("condition must be protein or control")
            item = sessions.setdefault(row["session"], {**metadata, "points": []})
            if any(item[key] != metadata[key] for key in metadata):
                raise ValueError(f"session {row['session']} has inconsistent session inputs")
            item["points"].append((float(row["minutes_after_start"]), score))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid record in {path}: {exc}") from exc

    summary: list[dict] = []
    excluded: list[str] = []
    for session, item in sessions.items():
        selected = required_points(item["points"])
        if selected is None:
            excluded.append(session)
            continue
        scores = [score for _, score in selected]
        modeled_heat_j = modeled_dit_heat_analytic_j(item["protein_g"], 180.0)
        entropy_proxy_j_k = modeled_heat_j / CORE_TEMPERATURE_K
        variance_val = round(stdev(scores) ** 2, 4) if len(scores) > 1 else 0.0

        summary.append({
            "session": session, **{key: item[key] for key in item if key != "points"},
            "observation_times_minutes": [observed for observed, _ in selected],
            "mean_ess": round(mean(scores), 3),
            "ess_change_180_minus_0": round(scores[-1] - scores[0], 3),
            "ess_variance": variance_val,
            "ess_auc_score_minutes": round(trapezoid_auc(selected), 3),
            "modeled_protein_dit_heat_j_0_to_180": round(modeled_heat_j, 1),
            "modeled_dit_capture_fraction_0_to_180": round(modeled_dit_capture_fraction(180.0), 3),
            "modeled_entropy_proxy_j_k_0_to_180": round(entropy_proxy_j_k, 3),
            "illustrative_marginal_protein_slope_1_g": round(modeled_marginal_protein_coefficient(item["protein_g"]), 6),
        })

    conditions = ("protein", "control")
    primary = {
        field: {condition: group_mean(summary, field, condition) for condition in conditions}
        for field in PRIMARY_OUTCOMES
    }
    report = {
        "title": "Aura Codex — Step 1 Protein/Energy Report", "generated_at": utcnow(),
        "thesis": "A higher-protein meal may be associated with higher or steadier self-reported energy than a calorie-matched, lower-protein control meal.",
        "current_step": "Step 1: dietary input and internal-state tracking; no external field claim is tested.",
        "equations": {
            "ess": "0.45*energy + 0.35*mood + 0.20*(10-hunger)",
            "modeled_dit_power": "Q_heat*(t/tau^2)*exp(-t/tau)",
            "modeled_dit_heat_analytic": "Q_heat * [1 - exp(-t/tau)*(1 + t/tau)]",
        },
        "sessions_analyzed": len(summary), "sessions_excluded_for_missing_timepoints": excluded,
        "primary_outcomes_by_condition": primary,
        "mean_protein_minus_control_ess": (
            round(primary["mean_ess"]["protein"] - primary["mean_ess"]["control"], 3)
            if primary["mean_ess"]["protein"] is not None and primary["mean_ess"]["control"] is not None else None
        ),
        "mean_calorie_gap_protein_minus_control": (
            round(group_mean(summary, "total_calories", "protein") - group_mean(summary, "total_calories", "control"), 3)
            if group_mean(summary, "total_calories", "protein") is not None and group_mean(summary, "total_calories", "control") is not None else None
        ),
        "covariate_means_by_condition": {
            field: {condition: group_mean(summary, field, condition) for condition in conditions}
            for field in ("protein_g", "total_calories", "carbs_g", "fat_g", "exercise_minutes", "sleep_hours", "caffeine_mg")
        },
        "supporting_model_context": {
            "status": "descriptive only; excluded from the primary comparison",
            "fields": [
                "modeled_protein_dit_heat_j_0_to_180",
                "modeled_entropy_proxy_j_k_0_to_180",
                "illustrative_marginal_protein_slope_1_g",
            ],
        },
        "session_details": summary,
        "interpretation_limits": [
            "DIT, entropy, and marginal-slope values are modeled estimates, not measurements.",
            "This is exploratory within-person evidence; it does not establish causation.",
            "A substantial calorie gap weakens the protein-specific comparison.",
            "No result establishes an aura, external energy field, or medical effect.",
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"sessions_analyzed": len(summary), "mean_protein_minus_control_ess": report["mean_protein_minus_control_ess"], "mean_calorie_gap": report["mean_calorie_gap_protein_minus_control"]}, indent=2))
    print(f"report: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aura Codex — Step 1 protein/energy tracker.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--make-protocol", action="store_true")
    action.add_argument("--log-observation", action="store_true")
    action.add_argument("--analyze", action="store_true")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=Path("aura_codex_report.json"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--session", help="One ID per meal/control session, e.g. s01")
    parser.add_argument("--condition", choices=("protein", "control"))
    parser.add_argument("--minutes", type=float)
    parser.add_argument("--protein-g", type=float, default=0.0)
    parser.add_argument("--total-calories", type=float, default=0.0)
    parser.add_argument("--carbs-g", type=float, default=0.0)
    parser.add_argument("--fat-g", type=float, default=0.0)
    parser.add_argument("--exercise-minutes", type=float, default=0.0)
    parser.add_argument("--energy", type=float)
    parser.add_argument("--mood", type=float)
    parser.add_argument("--hunger", type=float)
    parser.add_argument("--sleep-hours", type=float, default=0.0)
    parser.add_argument("--caffeine-mg", type=float, default=0.0)
    parser.add_argument("--notes")
    args = parser.parse_args()
    if args.make_protocol:
        make_protocol(args.sessions, args.seed, args.protocol)
        print(f"protocol: {args.protocol.resolve()}")
        return
    if args.analyze:
        analyze(args.log, args.output)
        return
    required = (args.session, args.condition, args.minutes, args.energy, args.mood, args.hunger)
    if any(value is None for value in required):
        parser.error("--log-observation requires --session, --condition, --minutes, --energy, --mood, and --hunger")
    add_observation(args)


if __name__ == "__main__":
    main()
