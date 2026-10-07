#!/usr/bin/env python3
"""Quantum-inspired sequential pattern recognition with Brier scoring.

Pure Python standard library. This is a toy simulation, not a quantum-hardware
experiment and not evidence of quantum advantage.

- Three binary prototype sequences are simulated 100 times each.
- Each observed bit can be missing (None).
- Class hypotheses are represented as complex amplitudes.
- Outcomes update amplitudes by square-root likelihoods (Born-rule style).
- Euler phase factors exp(i * theta), with theta involving pi, are tracked.
- Binary forecasts are scored with the Brier loss (p - y)^2.
- Class labels occupy orthogonal states, so phase does not change their
  computational-basis posterior; phase coherence is reported as a diagnostic.
"""

from __future__ import annotations

import cmath
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

PROTOTYPES: Dict[str, Tuple[int, ...]] = {
    "A": (0, 1, 0, 1, 0, 1, 0, 1),
    "B": (1, 0, 1, 0, 1, 0, 1, 0),
    "C": (1, 1, 0, 0, 1, 1, 0, 0),
}

TRIALS_PER_SEQUENCE = 100
FLIP_RATE = 0.12
MISSING_RATE = 0.15
EMISSION_CONFIDENCE = 1.0 - FLIP_RATE
SEED = 20260925


def normalize(amplitudes: Sequence[complex]) -> List[complex]:
    """Normalize an amplitude vector; use uniform amplitudes if norm is zero."""
    norm = math.sqrt(sum(abs(a) ** 2 for a in amplitudes))
    if norm == 0.0:
        return [1.0 / math.sqrt(len(amplitudes))] * len(amplitudes)
    return [a / norm for a in amplitudes]


def born_probabilities(amplitudes: Sequence[complex]) -> List[float]:
    """Return computational-basis probabilities |amplitude|^2."""
    norm_sq = sum(abs(a) ** 2 for a in amplitudes)
    if norm_sq == 0.0:
        return [1.0 / len(amplitudes)] * len(amplitudes)
    return [abs(a) ** 2 / norm_sq for a in amplitudes]


def predicted_one_probability(
    class_probabilities: Sequence[float],
    prototypes: Sequence[Sequence[int]],
    step: int,
) -> float:
    """Bayesian mixture forecast for the next binary outcome."""
    p1_by_class = [
        EMISSION_CONFIDENCE if prototype[step] == 1
        else 1.0 - EMISSION_CONFIDENCE
        for prototype in prototypes
    ]
    return sum(weight * p1 for weight, p1 in zip(class_probabilities, p1_by_class))


def draw_outcome(
    rng: random.Random,
    true_bit: int,
) -> Optional[int]:
    """Return 0/1, or None when this step is unobserved."""
    if rng.random() < MISSING_RATE:
        return None
    p1 = EMISSION_CONFIDENCE if true_bit == 1 else 1.0 - EMISSION_CONFIDENCE
    return 1 if rng.random() < p1 else 0


def update_amplitudes(
    amplitudes: Sequence[complex],
    observation: int,
    prototypes: Sequence[Sequence[int]],
    step: int,
) -> List[complex]:
    """Apply square-root likelihood updates and deterministic Euler phases."""
    updated: List[complex] = []
    n_classes = len(prototypes)
    n_steps = len(prototypes[0])

    for class_index, (amplitude, prototype) in enumerate(zip(amplitudes, prototypes)):
        p1 = (
            EMISSION_CONFIDENCE if prototype[step] == 1
            else 1.0 - EMISSION_CONFIDENCE
        )
        likelihood = p1 if observation == 1 else 1.0 - p1

        # Amplitude likelihood update: squared magnitude receives likelihood.
        amplitude *= math.sqrt(max(likelihood, 1e-15))

        # Euler's formula: exp(i*theta) = cos(theta) + i*sin(theta).
        # Phase is retained in the state; orthogonal class readout uses |a|^2.
        theta = math.pi * (class_index + 1) * (step + 1) / (n_classes * n_steps)
        amplitude *= cmath.exp(1j * theta)
        updated.append(amplitude)

    return normalize(updated)


def simulate_trial(
    true_name: str,
    prototypes_by_name: Dict[str, Tuple[int, ...]],
    rng: random.Random,
) -> Tuple[str, List[Optional[int]], float, float]:
    names = list(prototypes_by_name)
    prototypes = [prototypes_by_name[name] for name in names]
    true_pattern = prototypes_by_name[true_name]

    # Uniform coherent prior over three orthogonal class states.
    amplitudes = normalize([
        cmath.exp(1j * math.pi * (i + 1) / len(names)) / math.sqrt(len(names))
        for i in range(len(names))
    ])

    observed: List[Optional[int]] = []
    brier_losses: List[float] = []
    coherence_trace: List[float] = []

    for step, true_bit in enumerate(true_pattern):
        class_probs = born_probabilities(amplitudes)
        p1 = predicted_one_probability(class_probs, prototypes, step)
        y = draw_outcome(rng, true_bit)
        observed.append(y)

        if y is not None:
            # Brier loss for the forecast probability of outcome 1.
            brier_losses.append((p1 - y) ** 2)
            amplitudes = update_amplitudes(amplitudes, y, prototypes, step)
        else:
            # Advance phase even when no observation arrives.
            amplitudes = normalize([
                a * cmath.exp(
                    1j * math.pi * (i + 1) / (len(names) * len(true_pattern))
                )
                for i, a in enumerate(amplitudes)
            ])

        # Interference diagnostic in the equal-superposition readout basis.
        coherence_trace.append(abs(sum(amplitudes)) ** 2 / len(names))

    posterior = born_probabilities(amplitudes)
    predicted_name = names[max(range(len(names)), key=lambda i: posterior[i])]
    mean_brier = sum(brier_losses) / len(brier_losses) if brier_losses else float("nan")
    mean_coherence = sum(coherence_trace) / len(coherence_trace)
    return predicted_name, observed, mean_brier, mean_coherence


def main() -> None:
    rng = random.Random(SEED)
    names = list(PROTOTYPES)
    confusion = {actual: Counter() for actual in names}
    brier_by_actual = defaultdict(list)
    coherence_by_actual = defaultdict(list)
    missing_counts = Counter()

    for true_name in names:
        for _ in range(TRIALS_PER_SEQUENCE):
            predicted, outcomes, brier, coherence = simulate_trial(
                true_name, PROTOTYPES, rng
            )
            confusion[true_name][predicted] += 1
            if not math.isnan(brier):
                brier_by_actual[true_name].append(brier)
            coherence_by_actual[true_name].append(coherence)
            missing_counts[true_name] += sum(y is None for y in outcomes)

    correct = sum(confusion[name][name] for name in names)
    total = TRIALS_PER_SEQUENCE * len(names)

    print("Quantum-inspired sequential pattern recognition")
    print("Euler phase: exp(i*theta), theta scaled by pi")
    print(f"Trials: {TRIALS_PER_SEQUENCE} per sequence ({total} total)")
    print(f"Flip rate: {FLIP_RATE:.0%} | Optional missing rate: {MISSING_RATE:.0%}")
    print(f"Overall prototype accuracy: {correct}/{total} = {correct / total:.3f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print("          " + "  ".join(names))
    for actual in names:
        print(f"    {actual}     " + "  ".join(
            f"{confusion[actual][predicted]:3d}" for predicted in names
        ))

    print("\nPer-sequence metrics:")
    print("Sequence | Mean Brier | Mean phase coherence | Missing outcomes")
    for name in names:
        mean_brier = sum(brier_by_actual[name]) / len(brier_by_actual[name])
        mean_coherence = sum(coherence_by_actual[name]) / len(coherence_by_actual[name])
        print(
            f"   {name}     |   {mean_brier:.4f}   |"
            f"        {mean_coherence:.4f}        |"
            f"      {missing_counts[name]}"
        )

    print("\nBrier definition: (p_hat(1) - y)^2; lower is better.")
    print("Phase coherence is diagnostic; it is not used to inflate the Brier score.")


if __name__ == "__main__":
    main()
