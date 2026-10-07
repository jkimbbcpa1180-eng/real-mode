#!/usr/bin/env python3
"""
REAL MODE v61.0
Calibration-Judge TP Runtime

Core rules
----------
1. TP is the anchor probability.
2. IP, anomaly, contagion, and pattern outputs never replace TP directly.
3. Other probabilities are evaluated conditionally against TP through Monte Carlo.
4. Duplicate claims are collapsed by provenance and source-family correlation.
5. Opposing evidence cancels claim-by-claim before pattern recognition.
6. Calibration is domain-specific, chronological, and challenger-versus-incumbent.
7. Resolved predictions update Brier score, log loss, ECE, reliability bins,
   and an internal calibration confidence score (not empirical accuracy).
8. A challenger calibrator is promoted only when held-out performance supports it.
9. Identical event/evidence state produces identical stochastic seeds and TP output.
10. Calibration models are rebuilt from the resolved ledger and persisted to disk.
11. IP/anomaly/contagion are diagnostics-only until ablation evidence proves they
    improve out-of-sample forecasting for that domain.
12. Calibration is the judge: a module earns influence only by beating the anchor
    on resolved predictions, not by sounding plausible.
13. Every Real Mode runtime boot announces itself with: "🚂 Choo choo!"

This is a production-style reference implementation, not a guarantee of perfect
real-world calibration. Reliable calibration still depends on enough clean,
resolved, domain-specific outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import hashlib
import json
import math
import random
import time
import uuid


# =============================================================================
# TYPES AND NUMERIC UTILITIES
# =============================================================================

AnomalyKind = Literal["expected", "unexpected"]
DecisionState = Literal[
    "EXECUTE_PRIMARY_DIRECTIVE",
    "STAGING_BUFFER_ACTIVE",
    "HALT_AND_GATHER_EVIDENCE",
]

EPS = 1e-9


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def clamp_probability(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("probability must be a finite number in [0,1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be a finite number in [0,1]")
    return clamp(value, 1e-6, 1.0 - 1e-6)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(probability: float) -> float:
    p = clamp_probability(probability)
    return math.log(p / (1.0 - p))


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_prediction_seed(
    *,
    event_name: str,
    domain: str,
    horizon: str,
    baseline_tp: float,
    raw_observations: Sequence[Dict[str, Any]],
    anomaly_signals: Sequence["AnomalySignal"],
) -> str:
    """
    Build a deterministic seed from the forecast state rather than prediction UUID.

    Ingestion timestamps are intentionally excluded unless the caller explicitly
    supplied them inside the observation payload/provenance. Re-running the same
    event with the same evidence therefore produces the same stochastic samples.
    """
    canonical_observations = []
    for item in raw_observations:
        canonical_observations.append(
            {
                "source": item.get("source", "UNKNOWN"),
                "source_family": item.get(
                    "source_family",
                    item.get("source", "UNKNOWN"),
                ),
                "confidence": round(float(item.get("confidence", 0.5)), 8),
                "payload": item.get("payload", {}),
                "provenance_id": item.get("provenance_id"),
                "derivative_of": item.get("derivative_of"),
            }
        )

    canonical_observations.sort(key=stable_hash)
    canonical_anomalies = [asdict(signal) for signal in anomaly_signals]
    canonical_anomalies.sort(key=stable_hash)

    return stable_hash(
        {
            "event": event_name,
            "domain": domain,
            "horizon": horizon,
            "baseline_tp": round(float(baseline_tp), 8),
            "observations": canonical_observations,
            "anomalies": canonical_anomalies,
        }
    )


def weighted_mean(
    values: Iterable[Tuple[float, float]],
    default: float = 0.0,
) -> float:
    items = list(values)
    total_weight = sum(max(0.0, weight) for _, weight in items)
    if total_weight <= EPS:
        return default
    return sum(value * max(0.0, weight) for value, weight in items) / total_weight


def beta_parameters(mean: float, concentration: float) -> Tuple[float, float]:
    mean = clamp_probability(mean)
    concentration = max(2.0, float(concentration))
    return (
        max(EPS, mean * concentration),
        max(EPS, (1.0 - mean) * concentration),
    )


def mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = sum(values) / len(values)
    if len(values) == 1:
        return mean_value, 0.0
    variance = sum((x - mean_value) ** 2 for x in values) / (len(values) - 1)
    return mean_value, math.sqrt(max(0.0, variance))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Observation:
    source: str
    source_family: str
    confidence: float
    timestamp: float
    payload: Dict[str, Any]
    provenance_id: str
    derivative_of: Optional[str] = None


@dataclass(frozen=True)
class EvidenceSignal:
    claim_id: str
    name: str
    direction: float
    strength: float
    confidence: float
    source: str
    source_family: str
    provenance_id: str
    timestamp: float
    derivative_of: Optional[str] = None
    pattern_key: Optional[str] = None

    def signed_value(self) -> float:
        return (
            max(-1.0, min(1.0, self.direction))
            * clamp(self.strength)
            * clamp(self.confidence)
        )


@dataclass(frozen=True)
class AnomalySignal:
    name: str
    kind: AnomalyKind
    probability: float
    direction: float
    severity: float
    confidence: float
    persistence: float
    contagion: float
    age: float = 0.0
    half_life: float = 1.0
    source_family: str = "unknown"
    provenance_id: str = ""
    derivative_of: Optional[str] = None

    def __post_init__(self) -> None:
        if self.half_life <= 0:
            raise ValueError("half_life must be greater than zero")
        if self.age < 0:
            raise ValueError("age cannot be negative")


@dataclass
class CalibrationMetrics:
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    ece: Optional[float] = None
    sample_count: int = 0
    reliability_bins: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class CalibrationModel:
    a: float = 1.0
    b: float = 0.0
    version: str = "identity"
    trained_count: int = 0
    metrics: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    reliability_tp: float = 0.50


@dataclass
class Prediction:
    event: str
    domain: str
    horizon: str
    baseline_tp: float
    raw_tp: float
    calibrated_tp: float
    final_tp: float
    ip: float
    anomaly_probability: float
    contagion_probability: float
    expected_loss: float
    decision: DecisionState
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prediction_timestamp: float = field(default_factory=time.time)
    resolution_timestamp: Optional[float] = None
    resolved_outcome: Optional[float] = None
    calibration_version: str = "identity"
    calibration_reliability_tp: float = 0.50
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    data_origin: str = "unlabeled"
    outcome_verified: bool = False
    outcome_source: str = ""


def verified_prediction(prediction: Prediction) -> bool:
    """Eligible for learning; a caller declaration requires an evidence reference."""
    return (prediction.data_origin == "real"
            and prediction.outcome_verified is True
            and isinstance(prediction.outcome_source, str)
            and bool(prediction.outcome_source.strip())
            and not isinstance(prediction.resolved_outcome, bool)
            and prediction.resolved_outcome in (0.0, 1.0))


# =============================================================================
# OBSERVATION INGESTION
# =============================================================================

class ObservationEngine:
    def ingest(self, raw_data: List[Dict[str, Any]]) -> List[Observation]:
        observations: List[Observation] = []

        for item in raw_data:
            payload = dict(item.get("payload", {}))
            source = str(item.get("source", "UNKNOWN"))
            source_family = str(item.get("source_family", source))
            derivative_of = item.get("derivative_of")

            provenance_id = str(
                item.get(
                    "provenance_id",
                    stable_hash(
                        {
                            "source_family": source_family,
                            "payload": payload,
                            "derivative_of": derivative_of,
                        }
                    ),
                )
            )

            observations.append(
                Observation(
                    source=source,
                    source_family=source_family,
                    confidence=clamp(item.get("confidence", 0.5)),
                    timestamp=float(item.get("timestamp", time.time())),
                    payload=payload,
                    provenance_id=provenance_id,
                    derivative_of=derivative_of,
                )
            )

        return observations


# =============================================================================
# DUPLICATE, CONTRADICTION, AND PATTERN ENGINE
# =============================================================================

class DuplicatePatternEngine:
    """
    Resolution order:
    1. Deduplicate exact provenance.
    2. Group by claim_id.
    3. Apply source-family correlation damping.
    4. Cancel support and contradiction inside each claim.
    5. Run Monte Carlo reliability sampling on residual claims.
    6. Run pattern recognition only on residuals, avoiding double counting.
    """

    def __init__(
        self,
        simulations: int = 1500,
        same_family_decay: float = 0.55,
        derivative_discount: float = 0.20,
        random_seed: int = 42,
    ) -> None:
        if simulations < 100:
            raise ValueError("simulations must be at least 100")
        self.simulations = int(simulations)
        self.same_family_decay = clamp(same_family_decay)
        self.derivative_discount = clamp(derivative_discount)
        self.random_seed = int(random_seed)

    def _deduplicate(
        self,
        signals: Sequence[EvidenceSignal],
    ) -> List[EvidenceSignal]:
        strongest: Dict[str, EvidenceSignal] = {}

        for signal in signals:
            current = strongest.get(signal.provenance_id)
            if current is None or abs(signal.signed_value()) > abs(current.signed_value()):
                strongest[signal.provenance_id] = signal

        return list(strongest.values())

    def _family_multipliers(
        self,
        signals: Sequence[EvidenceSignal],
    ) -> Dict[str, float]:
        groups: Dict[str, List[EvidenceSignal]] = defaultdict(list)
        for signal in signals:
            groups[signal.source_family].append(signal)

        multipliers: Dict[str, float] = {}

        for family_signals in groups.values():
            ordered = sorted(
                family_signals,
                key=lambda signal: abs(signal.signed_value()),
                reverse=True,
            )

            for index, signal in enumerate(ordered):
                multiplier = self.same_family_decay ** index
                if signal.derivative_of is not None:
                    multiplier *= self.derivative_discount
                multipliers[signal.provenance_id] = multiplier

        return multipliers

    def resolve(
        self,
        signals: Sequence[EvidenceSignal],
        prediction_seed: str = "",
    ) -> Dict[str, Any]:
        unique = self._deduplicate(signals)
        multipliers = self._family_multipliers(unique)

        by_claim: Dict[str, List[EvidenceSignal]] = defaultdict(list)
        for signal in unique:
            by_claim[signal.claim_id].append(signal)

        claim_residuals: List[Dict[str, Any]] = []

        for claim_id, claim_signals in by_claim.items():
            support = 0.0
            contradiction = 0.0

            for signal in claim_signals:
                weighted = signal.signed_value() * multipliers[signal.provenance_id]
                if weighted >= 0:
                    support += weighted
                else:
                    contradiction += abs(weighted)

            residual = support - contradiction
            total_mass = support + contradiction
            cancellation_ratio = (
                1.0 if total_mass <= EPS
                else 1.0 - abs(residual) / total_mass
            )

            pattern_keys = {
                signal.pattern_key
                for signal in claim_signals
                if signal.pattern_key is not None
            }

            claim_residuals.append(
                {
                    "claim_id": claim_id,
                    "residual": residual,
                    "support_mass": support,
                    "contradiction_mass": contradiction,
                    "total_mass": total_mass,
                    "cancellation_ratio": clamp(cancellation_ratio),
                    "pattern_keys": sorted(pattern_keys),
                    "signals": claim_signals,
                }
            )

        total_support = sum(item["support_mass"] for item in claim_residuals)
        total_contradiction = sum(
            item["contradiction_mass"] for item in claim_residuals
        )
        net_residual = sum(item["residual"] for item in claim_residuals)
        total_mass = total_support + total_contradiction
        global_cancellation = (
            1.0 if total_mass <= EPS
            else 1.0 - abs(net_residual) / total_mass
        )

        seed_material = stable_hash(
            {
                "base_seed": self.random_seed,
                "prediction_seed": prediction_seed,
                "claims": [
                    {
                        "claim_id": item["claim_id"],
                        "residual": item["residual"],
                    }
                    for item in claim_residuals
                ],
            }
        )
        seeded_random = random.Random(int(seed_material[:16], 16))

        simulation_totals: List[float] = []

        for _ in range(self.simulations):
            trial_total = 0.0

            for item in claim_residuals:
                claim_trial = 0.0

                for signal in item["signals"]:
                    confidence = clamp_probability(signal.confidence)
                    alpha, beta = beta_parameters(confidence, concentration=14.0)
                    sampled_reliability = seeded_random.betavariate(alpha, beta)

                    claim_trial += (
                        max(-1.0, min(1.0, signal.direction))
                        * clamp(signal.strength)
                        * sampled_reliability
                        * multipliers[signal.provenance_id]
                    )

                trial_total += claim_trial

            simulation_totals.append(trial_total)

        simulation_mean, simulation_std = mean_and_std(simulation_totals)

        # Pattern recognition only uses residual claims.
        pattern_groups: Dict[str, List[float]] = defaultdict(list)
        for item in claim_residuals:
            residual = float(item["residual"])
            if abs(residual) <= EPS:
                continue
            for pattern_key in item["pattern_keys"]:
                pattern_groups[pattern_key].append(residual)

        pattern_candidates: List[Tuple[float, float, str]] = []

        for pattern_key, values in pattern_groups.items():
            if len(values) < 2:
                continue

            positive_count = sum(1 for value in values if value > 0)
            negative_count = sum(1 for value in values if value < 0)
            consistency = abs(positive_count - negative_count) / len(values)
            avg_magnitude = sum(abs(value) for value in values) / len(values)
            score = clamp(consistency * avg_magnitude)
            direction = 0.0 if abs(sum(values)) <= EPS else math.copysign(1.0, sum(values))

            pattern_candidates.append((score, direction, pattern_key))

        if pattern_candidates:
            pattern_score, pattern_direction, pattern_key = max(
                pattern_candidates,
                key=lambda item: item[0],
            )
        else:
            pattern_score, pattern_direction, pattern_key = 0.0, 0.0, ""

        # Incremental pattern contribution only. It shrinks as the direct
        # simulation signal gets stronger, preventing double counting.
        incremental_pattern_delta = (
            pattern_direction
            * pattern_score
            * max(0.0, 1.0 - min(1.0, abs(simulation_mean)))
        )

        return {
            "unique_signal_count": len(unique),
            "claim_count": len(claim_residuals),
            "independent_source_count": len(
                {
                    signal.source_family
                    for signal in unique
                    if signal.derivative_of is None
                }
            ),
            "support_mass": round(total_support, 6),
            "contradiction_mass": round(total_contradiction, 6),
            "net_residual": round(net_residual, 6),
            "cancellation_ratio": round(clamp(global_cancellation), 6),
            "simulation_mean": round(simulation_mean, 6),
            "simulation_std": round(simulation_std, 6),
            "pattern_score": round(pattern_score, 6),
            "pattern_direction": round(pattern_direction, 6),
            "pattern_key": pattern_key,
            "incremental_pattern_delta": round(
                incremental_pattern_delta,
                6,
            ),
            "claim_residuals": [
                {
                    "claim_id": item["claim_id"],
                    "residual": round(item["residual"], 6),
                    "support_mass": round(item["support_mass"], 6),
                    "contradiction_mass": round(
                        item["contradiction_mass"],
                        6,
                    ),
                    "cancellation_ratio": round(
                        item["cancellation_ratio"],
                        6,
                    ),
                }
                for item in claim_residuals
            ],
        }


# =============================================================================
# EVIDENCE ENGINE
# =============================================================================

class EvidenceEngine:
    def __init__(self, duplicate_pattern_engine: DuplicatePatternEngine) -> None:
        self.duplicate_pattern_engine = duplicate_pattern_engine

    def verify(
        self,
        observations: Sequence[Observation],
        prediction_seed: str,
    ) -> Dict[str, Any]:
        signals: List[EvidenceSignal] = []
        merged_payload: Dict[str, Any] = {}
        list_payloads: Dict[str, List[Any]] = defaultdict(list)

        for observation in observations:
            for key, value in observation.payload.items():
                if isinstance(value, list):
                    list_payloads[key].extend(value)
                elif key not in merged_payload:
                    merged_payload[key] = value
                elif merged_payload[key] != value:
                    existing = merged_payload[key]
                    if not isinstance(existing, list):
                        existing = [existing]
                    if value not in existing:
                        existing.append(value)
                    merged_payload[key] = existing

            for index, raw_signal in enumerate(
                observation.payload.get("signals", [])
            ):
                claim_id = str(
                    raw_signal.get(
                        "claim_id",
                        stable_hash(
                            {
                                "name": raw_signal.get("name"),
                                "pattern_key": raw_signal.get("pattern_key"),
                                "subject": raw_signal.get("subject"),
                                "metric": raw_signal.get("metric"),
                                "time_window": raw_signal.get("time_window"),
                            }
                        ),
                    )
                )

                provenance_id = str(
                    raw_signal.get(
                        "provenance_id",
                        stable_hash(
                            {
                                "parent": observation.provenance_id,
                                "claim_id": claim_id,
                                "direction": round(
                                    float(raw_signal.get("direction", 0.0)),
                                    6,
                                ),
                                "strength": round(
                                    float(raw_signal.get("strength", 0.5)),
                                    6,
                                ),
                            }
                        ),
                    )
                )

                signals.append(
                    EvidenceSignal(
                        claim_id=claim_id,
                        name=str(
                            raw_signal.get("name", f"signal_{index}")
                        ),
                        direction=float(raw_signal.get("direction", 0.0)),
                        strength=clamp(raw_signal.get("strength", 0.5)),
                        confidence=clamp(
                            raw_signal.get(
                                "confidence",
                                observation.confidence,
                            )
                        ),
                        source=observation.source,
                        source_family=observation.source_family,
                        provenance_id=provenance_id,
                        timestamp=observation.timestamp,
                        derivative_of=raw_signal.get(
                            "derivative_of",
                            observation.derivative_of,
                        ),
                        pattern_key=raw_signal.get("pattern_key"),
                    )
                )

        merged_payload.update(list_payloads)

        source_confidence = weighted_mean(
            [(observation.confidence, 1.0) for observation in observations],
            default=0.5,
        )

        resolution = self.duplicate_pattern_engine.resolve(
            signals=signals,
            prediction_seed=prediction_seed,
        )

        return {
            "source_confidence": round(clamp(source_confidence), 6),
            "observation_count": len(observations),
            "verified_payload": merged_payload,
            "signals": signals,
            "resolution": resolution,
        }


# =============================================================================
# ANOMALY ENGINE
# =============================================================================

class AnomalyProbabilityEngine:
    def __init__(
        self,
        expected_weight: float = 0.65,
        unexpected_weight: float = 1.45,
        same_family_decay: float = 0.55,
        derivative_discount: float = 0.20,
    ) -> None:
        self.expected_weight = float(expected_weight)
        self.unexpected_weight = float(unexpected_weight)
        self.same_family_decay = clamp(same_family_decay)
        self.derivative_discount = clamp(derivative_discount)

    def predict(
        self,
        anomalies: Sequence[AnomalySignal],
        asset_value: float = 1.0,
    ) -> Dict[str, Any]:
        if not anomalies:
            return {
                "anomaly_probability": 0.0,
                "signed_log_odds_delta": 0.0,
                "expected_loss": 0.0,
                "black_swan_probability": 0.0,
                "persistence_probability": 0.0,
                "contagion_probability": 0.0,
                "details": [],
            }

        strongest_by_provenance: Dict[str, AnomalySignal] = {}
        for anomaly in anomalies:
            provenance = anomaly.provenance_id or stable_hash(asdict(anomaly))
            current = strongest_by_provenance.get(provenance)
            current_score = (
                -1.0
                if current is None
                else current.probability * current.severity * current.confidence
            )
            candidate_score = (
                anomaly.probability
                * anomaly.severity
                * anomaly.confidence
            )
            if current is None or candidate_score > current_score:
                strongest_by_provenance[provenance] = anomaly

        by_family: Dict[str, List[AnomalySignal]] = defaultdict(list)
        for anomaly in strongest_by_provenance.values():
            by_family[anomaly.source_family].append(anomaly)

        total_no_anomaly_probability = 1.0
        black_swan_no_event = 1.0
        signed_delta = 0.0
        expected_loss = 0.0
        persistence_values: List[Tuple[float, float]] = []
        contagion_values: List[Tuple[float, float]] = []
        details: List[Dict[str, Any]] = []

        for family_items in by_family.values():
            ordered = sorted(
                family_items,
                key=lambda item: (
                    item.probability
                    * item.severity
                    * item.confidence
                ),
                reverse=True,
            )

            for index, anomaly in enumerate(ordered):
                correlation_multiplier = self.same_family_decay ** index
                if anomaly.derivative_of is not None:
                    correlation_multiplier *= self.derivative_discount

                freshness = 0.5 ** (anomaly.age / anomaly.half_life)
                effective_probability = clamp(
                    anomaly.probability
                    * anomaly.confidence
                    * freshness
                    * correlation_multiplier
                )

                total_no_anomaly_probability *= (
                    1.0 - effective_probability
                )

                kind_weight = (
                    self.unexpected_weight
                    if anomaly.kind == "unexpected"
                    else self.expected_weight
                )

                delta = (
                    max(-1.0, min(1.0, anomaly.direction))
                    * effective_probability
                    * clamp(anomaly.severity)
                    * (0.5 + 0.5 * clamp(anomaly.persistence))
                    * kind_weight
                )
                signed_delta += delta

                expected_loss += (
                    effective_probability
                    * clamp(anomaly.severity)
                    * max(0.0, asset_value)
                )

                if anomaly.kind == "unexpected":
                    black_swan_no_event *= (
                        1.0 - effective_probability
                    )

                persistence_values.append(
                    (
                        clamp(anomaly.persistence),
                        effective_probability,
                    )
                )
                contagion_values.append(
                    (
                        clamp(anomaly.contagion),
                        effective_probability,
                    )
                )

                details.append(
                    {
                        "name": anomaly.name,
                        "kind": anomaly.kind,
                        "effective_probability": round(
                            effective_probability,
                            6,
                        ),
                        "signed_log_odds_delta": round(delta, 6),
                        "freshness": round(freshness, 6),
                        "correlation_multiplier": round(
                            correlation_multiplier,
                            6,
                        ),
                    }
                )

        return {
            "anomaly_probability": round(
                clamp(1.0 - total_no_anomaly_probability),
                6,
            ),
            "signed_log_odds_delta": round(signed_delta, 6),
            "expected_loss": round(expected_loss, 6),
            "black_swan_probability": round(
                clamp(1.0 - black_swan_no_event),
                6,
            ),
            "persistence_probability": round(
                clamp(weighted_mean(persistence_values)),
                6,
            ),
            "contagion_probability": round(
                clamp(weighted_mean(contagion_values)),
                6,
            ),
            "details": details,
        }


# =============================================================================
# RAW TP ENGINE
# =============================================================================

class TPEngine:
    def calculate(
        self,
        baseline_tp: float,
        evidence: Dict[str, Any],
        anomaly: Dict[str, Any],
        residual_sigma: float = 0.20,
        n_observed: int = 30,
        autocorrelation: float = 0.0,
    ) -> Dict[str, float]:
        baseline_tp = clamp_probability(baseline_tp)

        rho = clamp(autocorrelation, 0.0, 0.99)
        effective_n = max(
            1.0,
            n_observed * (1.0 - rho) / (1.0 + rho),
        )
        standard_error = max(0.0, residual_sigma) / math.sqrt(effective_n)

        # Shrink uncertain evidence toward neutral log odds.
        uncertainty_shrink = 1.0 / (1.0 + standard_error)
        baseline_log_odds = logit(baseline_tp) * uncertainty_shrink

        resolution = evidence["resolution"]

        evidence_delta = (
            float(resolution["simulation_mean"])
            + float(resolution["incremental_pattern_delta"])
        )
        anomaly_delta = float(anomaly["signed_log_odds_delta"])

        evidence_delta = max(-4.0, min(4.0, evidence_delta))
        anomaly_delta = max(-4.0, min(4.0, anomaly_delta))

        raw_tp = sigmoid(
            baseline_log_odds + evidence_delta + anomaly_delta
        )

        return {
            "raw_tp": round(clamp(raw_tp, 0.01, 0.99), 6),
            "effective_sample_size": round(effective_n, 6),
            "standard_error": round(standard_error, 6),
            "uncertainty_shrink": round(uncertainty_shrink, 6),
            "evidence_log_odds_delta": round(evidence_delta, 6),
            "anomaly_log_odds_delta": round(anomaly_delta, 6),
        }


# =============================================================================
# IP ENGINE
# =============================================================================

class IPEngine:
    """
    IP is a competing intuition hypothesis, not a replacement for TP.

    It uses residual pattern strength, independent sources, source quality,
    simulation stability, and contradiction penalties.
    """

    def calculate(self, evidence: Dict[str, Any]) -> float:
        resolution = evidence["resolution"]

        pattern = float(resolution["pattern_score"])
        independent_sources = int(
            resolution["independent_source_count"]
        )
        cancellation = float(resolution["cancellation_ratio"])
        simulation_std = float(resolution["simulation_std"])
        source_confidence = float(evidence["source_confidence"])

        source_depth = 1.0 - math.exp(-0.45 * independent_sources)
        stability = math.exp(-2.0 * simulation_std)

        latent_score = (
            0.34 * pattern
            + 0.24 * source_depth
            + 0.22 * source_confidence
            + 0.20 * stability
            - 0.35 * cancellation
        )

        return round(
            clamp(sigmoid(4.0 * (latent_score - 0.5))),
            6,
        )


# =============================================================================
# TP-ANCHORED CONDITIONAL MONTE CARLO
# =============================================================================

class TPAnchoredSimulationEngine:
    """
    TP is the source distribution.

    For each simulated world:
    1. Sample a TP probability around calibrated TP.
    2. Draw the baseline event from TP.
    3. Evaluate whether IP, anomaly, or contagion would change the baseline.
    4. Record hit counts and influence.

    Other probabilities are therefore interpreted relative to TP rather than
    being averaged into a separate competing final probability.
    """

    def __init__(
        self,
        base_trials: int = 1000,
        high_conflict_trials: int = 10000,
        conflict_threshold: float = 0.25,
        random_seed: int = 43,
    ) -> None:
        self.base_trials = max(100, int(base_trials))
        self.high_conflict_trials = max(
            self.base_trials,
            int(high_conflict_trials),
        )
        self.conflict_threshold = clamp(conflict_threshold)
        self.random_seed = int(random_seed)

    def simulate(
        self,
        *,
        anchor_tp: float,
        calibration_reliability_tp: float,
        ip: float,
        anomaly_probability: float,
        anomaly_direction: float,
        contagion_probability: float,
        prediction_seed: str,
    ) -> Dict[str, Any]:
        anchor_tp = clamp_probability(anchor_tp)
        ip = clamp_probability(ip)
        anomaly_probability = clamp(anomaly_probability)
        contagion_probability = clamp(contagion_probability)
        calibration_reliability_tp = clamp(
            calibration_reliability_tp
        )

        conflict_index = abs(anchor_tp - ip)
        trials = (
            self.high_conflict_trials
            if conflict_index >= self.conflict_threshold
            else self.base_trials
        )

        seed = int(
            stable_hash(
                {
                    "seed": self.random_seed,
                    "prediction_seed": prediction_seed,
                    "anchor_tp": anchor_tp,
                    "ip": ip,
                    "anomaly_probability": anomaly_probability,
                    "contagion_probability": contagion_probability,
                }
            )[:16],
            16,
        )
        rng = random.Random(seed)

        # Higher calibration reliability means a tighter beta distribution
        # around anchor TP. Lower reliability creates more spread.
        concentration = 8.0 + 192.0 * calibration_reliability_tp
        alpha_tp, beta_tp = beta_parameters(
            anchor_tp,
            concentration=concentration,
        )

        baseline_hits = 0
        baseline_misses = 0
        ip_hits = 0
        anomaly_hits = 0
        contagion_hits = 0
        ip_changed_outcome = 0
        anomaly_changed_outcome = 0
        contagion_changed_outcome = 0
        final_hits = 0

        for _ in range(trials):
            sampled_tp = rng.betavariate(alpha_tp, beta_tp)
            baseline_event = rng.random() < sampled_tp

            baseline_hits += int(baseline_event)
            baseline_misses += int(not baseline_event)

            # IP gets a chance to challenge TP proportional to disagreement.
            # If IP > TP, it mainly attempts to turn misses into hits.
            # If IP < TP, it mainly attempts to turn hits into misses.
            ip_event = rng.random() < ip
            if ip_event:
                ip_hits += 1

            event_state = baseline_event

            if ip > anchor_tp and not event_state:
                challenge_strength = clamp(
                    (ip - anchor_tp) / max(EPS, 1.0 - anchor_tp)
                )
                if ip_event and rng.random() < challenge_strength:
                    event_state = True
                    ip_changed_outcome += 1
            elif ip < anchor_tp and event_state:
                challenge_strength = clamp(
                    (anchor_tp - ip) / max(EPS, anchor_tp)
                )
                if not ip_event and rng.random() < challenge_strength:
                    event_state = False
                    ip_changed_outcome += 1

            anomaly_event = rng.random() < anomaly_probability
            if anomaly_event:
                anomaly_hits += 1
                anomaly_strength = clamp(
                    abs(anomaly_direction)
                    * anomaly_probability
                )

                if anomaly_direction < 0 and event_state:
                    if rng.random() < anomaly_strength:
                        event_state = False
                        anomaly_changed_outcome += 1
                elif anomaly_direction > 0 and not event_state:
                    if rng.random() < anomaly_strength:
                        event_state = True
                        anomaly_changed_outcome += 1

            contagion_event = (
                anomaly_event
                and rng.random() < contagion_probability
            )
            if contagion_event:
                contagion_hits += 1

                # Contagion amplifies the anomaly direction.
                contagion_strength = clamp(
                    contagion_probability
                    * max(0.25, abs(anomaly_direction))
                )

                if anomaly_direction < 0 and event_state:
                    if rng.random() < contagion_strength:
                        event_state = False
                        contagion_changed_outcome += 1
                elif anomaly_direction > 0 and not event_state:
                    if rng.random() < contagion_strength:
                        event_state = True
                        contagion_changed_outcome += 1

            final_hits += int(event_state)

        baseline_rate = baseline_hits / trials
        final_rate = final_hits / trials

        return {
            "trials": trials,
            "anchor_tp": round(anchor_tp, 6),
            "conflict_index": round(conflict_index, 6),
            "baseline_hits": baseline_hits,
            "baseline_misses": baseline_misses,
            "baseline_rate": round(baseline_rate, 6),
            "ip_hits": ip_hits,
            "ip_hit_rate": round(ip_hits / trials, 6),
            "anomaly_hits": anomaly_hits,
            "anomaly_hit_rate": round(anomaly_hits / trials, 6),
            "contagion_hits": contagion_hits,
            "contagion_hit_rate": round(contagion_hits / trials, 6),
            "ip_changed_outcome": ip_changed_outcome,
            "anomaly_changed_outcome": anomaly_changed_outcome,
            "contagion_changed_outcome": contagion_changed_outcome,
            "final_hits": final_hits,
            "final_rate": round(final_rate, 6),
            "net_influence": round(final_rate - baseline_rate, 6),
            "calibration_reliability_tp": round(
                calibration_reliability_tp,
                6,
            ),
        }


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def reliability_bins(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    bin_count: int = 10,
) -> List[Dict[str, float]]:
    bins: List[Dict[str, float]] = []

    for index in range(bin_count):
        low = index / bin_count
        high = (index + 1) / bin_count

        members = [
            i
            for i, probability in enumerate(probabilities)
            if (
                low <= probability < high
                or (
                    index == bin_count - 1
                    and probability == 1.0
                )
            )
        ]

        if not members:
            bins.append(
                {
                    "low": low,
                    "high": high,
                    "count": 0,
                    "mean_probability": 0.0,
                    "observed_frequency": 0.0,
                    "absolute_gap": 0.0,
                }
            )
            continue

        mean_probability = sum(
            probabilities[i] for i in members
        ) / len(members)
        observed_frequency = sum(
            outcomes[i] for i in members
        ) / len(members)

        bins.append(
            {
                "low": round(low, 4),
                "high": round(high, 4),
                "count": len(members),
                "mean_probability": round(mean_probability, 6),
                "observed_frequency": round(
                    observed_frequency,
                    6,
                ),
                "absolute_gap": round(
                    abs(mean_probability - observed_frequency),
                    6,
                ),
            }
        )

    return bins


def score_probabilities(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
) -> CalibrationMetrics:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal lengths")
    for probability in probabilities:
        clamp_probability(probability)
    if any(isinstance(y, bool) or y not in (0.0, 1.0) for y in outcomes):
        raise ValueError("outcomes must be binary numbers 0 or 1")
    if not probabilities:
        return CalibrationMetrics()

    count = len(probabilities)
    brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes)
    ) / count

    log_loss = -sum(
        outcome * math.log(clamp_probability(probability))
        + (1.0 - outcome)
        * math.log(clamp_probability(1.0 - probability))
        for probability, outcome in zip(probabilities, outcomes)
    ) / count

    bins = reliability_bins(probabilities, outcomes)
    ece = sum(
        (item["count"] / count) * item["absolute_gap"]
        for item in bins
        if item["count"] > 0
    )

    return CalibrationMetrics(
        brier_score=round(brier, 6),
        log_loss=round(log_loss, 6),
        ece=round(ece, 6),
        sample_count=count,
        reliability_bins=bins,
    )


# =============================================================================
# CHRONOLOGICAL PLATT CALIBRATION
# =============================================================================

class CalibrationEngine:
    def __init__(
        self,
        minimum_samples: int = 30,
        strong_samples: int = 100,
        learning_rate: float = 0.03,
        epochs: int = 1600,
        l2: float = 0.01,
        validation_fraction: float = 0.30,
        minimum_reliability_tp: float = 0.75,
        model_path: Optional[str] = None,
    ) -> None:
        self.minimum_samples = int(minimum_samples)
        self.strong_samples = int(strong_samples)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.validation_fraction = clamp(
            validation_fraction,
            0.15,
            0.50,
        )
        self.minimum_reliability_tp = clamp(
            minimum_reliability_tp
        )
        self.model_path = Path(model_path) if model_path else None
        self.models: Dict[str, CalibrationModel] = defaultdict(
            CalibrationModel
        )
        self.load()

    def save(self) -> None:
        if self.model_path is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            domain: asdict(model)
            for domain, model in self.models.items()
        }
        self.model_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            return
        raw = json.loads(self.model_path.read_text(encoding="utf-8"))
        for domain, item in raw.items():
            metrics_raw = item.get("metrics", {})
            metrics = CalibrationMetrics(**metrics_raw)
            self.models[domain] = CalibrationModel(
                a=float(item.get("a", 1.0)),
                b=float(item.get("b", 0.0)),
                version=str(item.get("version", "identity")),
                trained_count=int(item.get("trained_count", 0)),
                metrics=metrics,
                reliability_tp=float(item.get("reliability_tp", 0.50)),
            )

    def rebuild_from_predictions(
        self,
        predictions: Sequence[Prediction],
    ) -> Dict[str, Dict[str, Any]]:
        """Rebuild active domain models deterministically from resolved history."""
        domains = sorted(
            {
                p.domain
                for p in predictions
                if verified_prediction(p)
            }
        )
        self.models = defaultdict(CalibrationModel)
        results: Dict[str, Dict[str, Any]] = {}
        for domain in domains:
            results[domain] = self.train_challenger(predictions, domain)
        self.save()
        return results

    @staticmethod
    def _apply_model(
        raw_probability: float,
        model: CalibrationModel,
    ) -> float:
        return clamp(
            sigmoid(
                model.a * logit(raw_probability) + model.b
            ),
            0.01,
            0.99,
        )

    def calibrate(
        self,
        raw_tp: float,
        domain: str,
    ) -> Tuple[float, CalibrationModel]:
        model = self.models[domain]
        return (
            round(self._apply_model(raw_tp, model), 6),
            model,
        )

    def _fit_platt(
        self,
        predictions: Sequence[Prediction],
    ) -> CalibrationModel:
        xs = [logit(prediction.raw_tp) for prediction in predictions]
        ys = [
            float(prediction.resolved_outcome)
            for prediction in predictions
        ]

        a = 1.0
        b = 0.0
        count = len(predictions)

        for _ in range(self.epochs):
            gradient_a = 0.0
            gradient_b = 0.0

            for x, outcome in zip(xs, ys):
                estimate = sigmoid(a * x + b)
                error = estimate - outcome
                gradient_a += error * x
                gradient_b += error

            gradient_a = gradient_a / count + self.l2 * a
            gradient_b = gradient_b / count

            a -= self.learning_rate * gradient_a
            b -= self.learning_rate * gradient_b

        return CalibrationModel(
            a=max(0.05, min(10.0, a)),
            b=max(-10.0, min(10.0, b)),
            version=f"challenger_{count}",
            trained_count=count,
        )

    @staticmethod
    def _parameter_stability(
        incumbent: CalibrationModel,
        challenger: CalibrationModel,
    ) -> float:
        distance = math.sqrt(
            (challenger.a - incumbent.a) ** 2
            + (challenger.b - incumbent.b) ** 2
        )
        return math.exp(-0.50 * distance)

    def _calibration_reliability_tp(
        self,
        *,
        incumbent_metrics: CalibrationMetrics,
        challenger_metrics: CalibrationMetrics,
        sample_count: int,
        parameter_stability: float,
    ) -> float:
        if challenger_metrics.sample_count == 0:
            return 0.20

        sample_adequacy = 1.0 - math.exp(
            -sample_count / max(1.0, self.strong_samples)
        )

        incumbent_brier = (
            incumbent_metrics.brier_score
            if incumbent_metrics.brier_score is not None
            else 0.25
        )
        challenger_brier = (
            challenger_metrics.brier_score
            if challenger_metrics.brier_score is not None
            else 0.25
        )

        incumbent_log_loss = (
            incumbent_metrics.log_loss
            if incumbent_metrics.log_loss is not None
            else math.log(2.0)
        )
        challenger_log_loss = (
            challenger_metrics.log_loss
            if challenger_metrics.log_loss is not None
            else math.log(2.0)
        )

        brier_improvement = clamp(
            0.5
            + 2.0 * (incumbent_brier - challenger_brier)
        )
        log_loss_improvement = clamp(
            0.5
            + 0.75
            * (
                incumbent_log_loss
                - challenger_log_loss
            )
        )

        ece = (
            challenger_metrics.ece
            if challenger_metrics.ece is not None
            else 0.25
        )
        ece_quality = clamp(1.0 - ece / 0.20)

        latent = (
            1.25 * brier_improvement
            + 0.95 * log_loss_improvement
            + 1.00 * sample_adequacy
            + 0.85 * parameter_stability
            + 1.10 * ece_quality
            - 2.60
        )

        return clamp(sigmoid(latent))

    def train_challenger(
        self,
        predictions: Sequence[Prediction],
        domain: str,
    ) -> Dict[str, Any]:
        usable = sorted(
            [
                prediction
                for prediction in predictions
                if (
                    prediction.domain == domain
                    and verified_prediction(prediction)
                )
            ],
            key=lambda prediction: (
                prediction.resolution_timestamp
                or prediction.prediction_timestamp
            ),
        )

        count = len(usable)
        incumbent = self.models[domain]

        if count < self.minimum_samples:
            incumbent.version = "provisional_identity"
            incumbent.trained_count = count
            incumbent.reliability_tp = clamp(
                0.20
                + 0.50
                * count
                / max(1, self.minimum_samples)
            )

            self.save()
            return {
                "status": "PROVISIONAL",
                "domain": domain,
                "resolved_count": count,
                "minimum_required": self.minimum_samples,
                "incumbent_version": incumbent.version,
                "calibration_reliability_tp": round(
                    incumbent.reliability_tp,
                    6,
                ),
            }

        validation_count = max(
            10,
            int(round(count * self.validation_fraction)),
        )
        validation_count = min(validation_count, count - 10)

        train_set = usable[:-validation_count]
        validation_set = usable[-validation_count:]

        challenger = self._fit_platt(train_set)

        incumbent_probabilities = [
            self._apply_model(prediction.raw_tp, incumbent)
            for prediction in validation_set
        ]
        challenger_probabilities = [
            self._apply_model(prediction.raw_tp, challenger)
            for prediction in validation_set
        ]
        outcomes = [
            float(prediction.resolved_outcome)
            for prediction in validation_set
        ]

        incumbent_metrics = score_probabilities(
            incumbent_probabilities,
            outcomes,
        )
        challenger_metrics = score_probabilities(
            challenger_probabilities,
            outcomes,
        )

        parameter_stability = self._parameter_stability(
            incumbent,
            challenger,
        )

        reliability_tp = self._calibration_reliability_tp(
            incumbent_metrics=incumbent_metrics,
            challenger_metrics=challenger_metrics,
            sample_count=count,
            parameter_stability=parameter_stability,
        )

        brier_better = (
            challenger_metrics.brier_score
            < incumbent_metrics.brier_score
        )
        log_loss_safe = (
            challenger_metrics.log_loss
            <= incumbent_metrics.log_loss + 0.01
        )
        ece_safe = challenger_metrics.ece <= 0.10
        reliability_safe = (
            reliability_tp >= self.minimum_reliability_tp
        )

        promote = (
            brier_better
            and log_loss_safe
            and ece_safe
            and reliability_safe
        )

        if promote:
            challenger.version = (
                f"calibrated_{count}"
                if count >= self.strong_samples
                else f"early_calibration_{count}"
            )
            challenger.metrics = challenger_metrics
            challenger.reliability_tp = reliability_tp
            self.models[domain] = challenger
            active = challenger
            status = "CHALLENGER_PROMOTED"
        else:
            incumbent.metrics = incumbent_metrics
            incumbent.reliability_tp = reliability_tp
            incumbent.trained_count = count
            active = incumbent
            status = "INCUMBENT_RETAINED"

        self.save()
        return {
            "status": status,
            "domain": domain,
            "resolved_count": count,
            "train_count": len(train_set),
            "validation_count": len(validation_set),
            "active_version": active.version,
            "active_a": round(active.a, 6),
            "active_b": round(active.b, 6),
            "calibration_reliability_tp": round(
                reliability_tp,
                6,
            ),
            "parameter_stability": round(
                parameter_stability,
                6,
            ),
            "incumbent_metrics": asdict(incumbent_metrics),
            "challenger_metrics": asdict(challenger_metrics),
            "promotion_checks": {
                "brier_better": brier_better,
                "log_loss_safe": log_loss_safe,
                "ece_safe": ece_safe,
                "reliability_safe": reliability_safe,
            },
        }


# =============================================================================
# LEDGER WITH DISK PERSISTENCE
# =============================================================================

class PredictionLedger:
    def __init__(
        self,
        storage_path: Optional[str] = None,
    ) -> None:
        self.storage_path = (
            Path(storage_path)
            if storage_path is not None
            else None
        )
        self.records: Dict[str, Prediction] = {}

        if self.storage_path is not None:
            self.load()

    def record_prediction(self, prediction: Prediction) -> None:
        self.records[prediction.prediction_id] = prediction
        self.save()

    def resolve_prediction(
        self,
        prediction_id: str,
        outcome: float,
        resolution_timestamp: Optional[float] = None,
        *,
        outcome_verified: bool = False,
        outcome_source: str = "",
    ) -> None:
        if isinstance(outcome, bool) or outcome not in (0.0, 1.0):
            raise ValueError("outcome must be 0.0 or 1.0")
        if prediction_id not in self.records:
            raise KeyError(
                f"Unknown prediction_id: {prediction_id}"
            )

        prediction = self.records[prediction_id]
        if not isinstance(outcome_verified, bool):
            raise ValueError("outcome_verified must be boolean")
        if not isinstance(outcome_source, str):
            raise ValueError("outcome_source must be an evidence reference string")
        if outcome_verified and (prediction.data_origin != "real" or not outcome_source.strip()):
            raise ValueError("verified outcomes require real data and an evidence source")
        when = time.time() if resolution_timestamp is None else float(resolution_timestamp)
        if not math.isfinite(when) or when < prediction.prediction_timestamp:
            raise ValueError("resolution time must be finite and no earlier than prediction")
        prediction.resolved_outcome = float(outcome)
        prediction.outcome_verified = outcome_verified
        prediction.outcome_source = outcome_source.strip()
        prediction.resolution_timestamp = (
            when
        )
        self.save()

    def get_resolved(
        self,
        domain: Optional[str] = None,
        *,
        include_unverified: bool = False,
    ) -> List[Prediction]:
        records = [
            prediction
            for prediction in self.records.values()
            if prediction.resolved_outcome is not None
            and (include_unverified or verified_prediction(prediction))
        ]

        if domain is not None:
            records = [
                prediction
                for prediction in records
                if prediction.domain == domain
            ]

        return records

    def save(self) -> None:
        if self.storage_path is None:
            return

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = [
            asdict(prediction)
            for prediction in self.records.values()
        ]

        self.storage_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        if (
            self.storage_path is None
            or not self.storage_path.exists()
        ):
            return

        raw = json.loads(
            self.storage_path.read_text(encoding="utf-8")
        )

        self.records = {}
        for item in raw:
            prediction = Prediction(**item)
            if prediction.data_origin not in ("real", "demo", "synthetic", "unlabeled"):
                raise ValueError("ledger contains invalid data_origin")
            for probability in (prediction.baseline_tp, prediction.raw_tp,
                                prediction.calibrated_tp, prediction.final_tp):
                clamp_probability(probability)
            if prediction.resolved_outcome is not None and (
                    isinstance(prediction.resolved_outcome, bool)
                    or prediction.resolved_outcome not in (0.0, 1.0)):
                raise ValueError("ledger contains a nonbinary outcome")
            if not isinstance(prediction.outcome_verified, bool):
                raise ValueError("ledger outcome_verified must be boolean")
            if prediction.outcome_verified and not verified_prediction(prediction):
                raise ValueError("ledger has a verified outcome without real data and a source")
            self.records[prediction.prediction_id] = prediction


# =============================================================================
# ABLATION GATE ENGINE
# =============================================================================

class AblationGateEngine:
    """
    Secondary modules are diagnostics-only until resolved forecasts prove that
    they improve out-of-sample Brier score in the relevant domain.

    Each prediction stores counterfactual candidate probabilities. On resolution
    the gate compares each module candidate against its reference forecast:
      - IP        vs anchor TP
      - anomaly   vs anchor TP
      - contagion vs anomaly-only candidate (marginal value)

    Approval requires enough samples, a minimum Brier improvement, and no material
    log-loss regression. This makes calibration performance the authority.
    """

    MODULES = ("ip", "anomaly", "contagion")

    def __init__(
        self,
        storage_path: Optional[str] = None,
        minimum_samples: int = 40,
        minimum_brier_improvement: float = 0.005,
        log_loss_tolerance: float = 0.005,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self.minimum_samples = max(10, int(minimum_samples))
        self.minimum_brier_improvement = max(0.0, float(minimum_brier_improvement))
        self.log_loss_tolerance = max(0.0, float(log_loss_tolerance))
        self.history: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._seen: set[str] = set()
        self.load()

    @staticmethod
    def _log_loss_single(probability: float, outcome: float) -> float:
        p = clamp_probability(probability)
        return -(outcome * math.log(p) + (1.0 - outcome) * math.log(1.0 - p))

    def _pair_for_module(
        self,
        module: str,
        candidates: Dict[str, float],
    ) -> Tuple[float, float]:
        anchor = float(candidates["anchor"])
        if module == "ip":
            return float(candidates["ip"]), anchor
        if module == "anomaly":
            return float(candidates["anomaly"]), anchor
        if module == "contagion":
            return float(candidates["contagion"]), float(candidates["anomaly"])
        raise KeyError(module)

    def evaluate(self, domain: str, module: str) -> Dict[str, Any]:
        if module not in self.MODULES:
            raise KeyError(f"Unknown ablation module: {module}")
        rows = self.history.get(domain, {}).get(module, [])
        count = len(rows)
        if count == 0:
            return {
                "module": module,
                "domain": domain,
                "status": "DIAGNOSTIC_ONLY",
                "enabled": False,
                "sample_count": 0,
                "minimum_samples": self.minimum_samples,
                "candidate_brier": None,
                "reference_brier": None,
                "brier_improvement": None,
                "candidate_log_loss": None,
                "reference_log_loss": None,
            }

        candidate_brier = sum((r["candidate"] - r["outcome"]) ** 2 for r in rows) / count
        reference_brier = sum((r["reference"] - r["outcome"]) ** 2 for r in rows) / count
        candidate_log_loss = sum(
            self._log_loss_single(r["candidate"], r["outcome"]) for r in rows
        ) / count
        reference_log_loss = sum(
            self._log_loss_single(r["reference"], r["outcome"]) for r in rows
        ) / count
        improvement = reference_brier - candidate_brier

        enough_samples = count >= self.minimum_samples
        brier_pass = improvement >= self.minimum_brier_improvement
        log_loss_pass = candidate_log_loss <= reference_log_loss + self.log_loss_tolerance
        enabled = enough_samples and brier_pass and log_loss_pass

        if enabled:
            status = "APPROVED"
        elif not enough_samples:
            status = "DIAGNOSTIC_ONLY"
        else:
            status = "REJECTED"

        return {
            "module": module,
            "domain": domain,
            "status": status,
            "enabled": enabled,
            "sample_count": count,
            "minimum_samples": self.minimum_samples,
            "candidate_brier": round(candidate_brier, 6),
            "reference_brier": round(reference_brier, 6),
            "brier_improvement": round(improvement, 6),
            "candidate_log_loss": round(candidate_log_loss, 6),
            "reference_log_loss": round(reference_log_loss, 6),
            "checks": {
                "enough_samples": enough_samples,
                "brier_pass": brier_pass,
                "log_loss_pass": log_loss_pass,
            },
        }

    def is_enabled(self, domain: str, module: str) -> bool:
        return bool(self.evaluate(domain, module)["enabled"])

    def status(self, domain: str) -> Dict[str, Dict[str, Any]]:
        return {module: self.evaluate(domain, module) for module in self.MODULES}

    def observe_resolution(self, prediction: Prediction) -> Dict[str, Any]:
        if not verified_prediction(prediction):
            return self.status(prediction.domain)
        candidates = prediction.diagnostics.get("ablation_candidates")
        if not isinstance(candidates, dict):
            return self.status(prediction.domain)

        outcome = float(prediction.resolved_outcome)
        for module in self.MODULES:
            seen_key = f"{prediction.prediction_id}:{module}"
            if seen_key in self._seen:
                continue
            try:
                candidate, reference = self._pair_for_module(module, candidates)
            except (KeyError, TypeError, ValueError):
                continue
            self.history[prediction.domain][module].append(
                {
                    "prediction_id": prediction.prediction_id,
                    "candidate": clamp_probability(candidate),
                    "reference": clamp_probability(reference),
                    "outcome": outcome,
                }
            )
            self._seen.add(seen_key)
        self.save()
        return self.status(prediction.domain)

    def rebuild_from_predictions(self, predictions: Sequence[Prediction]) -> None:
        self.history = defaultdict(lambda: defaultdict(list))
        self._seen = set()
        ordered = sorted(
            [p for p in predictions if verified_prediction(p)],
            key=lambda p: p.resolution_timestamp or p.prediction_timestamp,
        )
        for prediction in ordered:
            self.observe_resolution(prediction)
        self.save()

    def save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "minimum_samples": self.minimum_samples,
            "minimum_brier_improvement": self.minimum_brier_improvement,
            "log_loss_tolerance": self.log_loss_tolerance,
            "history": {
                domain: {module: rows for module, rows in modules.items()}
                for domain, modules in self.history.items()
            },
        }
        self.storage_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        history = raw.get("history", {})
        for domain, modules in history.items():
            for module, rows in modules.items():
                if module not in self.MODULES:
                    continue
                for row in rows:
                    self.history[domain][module].append(dict(row))
                    pid = row.get("prediction_id")
                    if pid:
                        self._seen.add(f"{pid}:{module}")


# =============================================================================
# DECISION ENGINE
# =============================================================================

class DecisionEngine:
    def decide(
        self,
        tp: float,
        consequence_score: float,
        cost_of_delay: float = 0.0,
        calibration_reliability_tp: float = 0.50,
    ) -> DecisionState:
        consequence = clamp(consequence_score)
        delay = clamp(cost_of_delay)
        calibration_reliability_tp = clamp(
            calibration_reliability_tp
        )

        reliability_penalty = (
            0.10 * (1.0 - calibration_reliability_tp)
        )

        threshold = clamp(
            0.55
            + 0.30 * consequence
            - 0.15 * delay
            + reliability_penalty,
            0.50,
            0.92,
        )

        if tp >= threshold:
            return "EXECUTE_PRIMARY_DIRECTIVE"
        if tp >= threshold - 0.15:
            return "STAGING_BUFFER_ACTIVE"
        return "HALT_AND_GATHER_EVIDENCE"


# =============================================================================
# REAL MODE RUNTIME
# =============================================================================

REAL_MODE_BOOT_SIGNAL = "🚂 Choo choo!"

# EXAMPLE location only (Seoul City Hall), not anyone's real position. Replace
# with your own shared pin if you want one; it is not live GPS.
REAL_MODE_LAST_SHARED_LOCATION = {
    "latitude": 37.5665,
    "longitude": 126.9780,
    "area": "EXAMPLE: Seoul City Hall, Seoul, South Korea",
    "observed_at": None,
    "source": "example placeholder",
}


class RealModeRuntime:
    def __init__(
        self,
        ledger_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        ablation_path: Optional[str] = None,
        rebuild_learning_on_start: bool = True,
    ) -> None:
        print(REAL_MODE_BOOT_SIGNAL)

        duplicate_pattern_engine = DuplicatePatternEngine(
            simulations=1500,
            same_family_decay=0.55,
            derivative_discount=0.20,
            random_seed=42,
        )

        self.observation = ObservationEngine()
        self.evidence = EvidenceEngine(duplicate_pattern_engine)
        self.anomaly = AnomalyProbabilityEngine()
        self.tp = TPEngine()
        self.ip = IPEngine()
        self.conditional_simulation = TPAnchoredSimulationEngine(
            base_trials=1000,
            high_conflict_trials=10000,
            conflict_threshold=0.25,
            random_seed=43,
        )

        self.ledger = PredictionLedger(ledger_path)

        if ledger_path is not None:
            base = str(Path(ledger_path))
            calibration_path = calibration_path or f"{base}.calibration_models.json"
            ablation_path = ablation_path or f"{base}.ablation_gate.json"

        self.calibration = CalibrationEngine(model_path=calibration_path)
        self.ablation = AblationGateEngine(storage_path=ablation_path)
        self.decision = DecisionEngine()

        # Persisted model files without verified history are not trusted. The
        # legacy opt-out argument is retained for compatibility, never for trust.
        resolved = self.ledger.get_resolved()
        self.calibration.rebuild_from_predictions(resolved)
        self.ablation.rebuild_from_predictions(resolved)

    @staticmethod
    def _candidate_rate(simulation_result: Dict[str, Any]) -> float:
        return round(clamp(float(simulation_result["final_rate"]), 0.01, 0.99), 6)

    def run(
        self,
        *,
        event_name: str,
        domain: str,
        horizon: str,
        baseline_tp: float,
        raw_observations: List[Dict[str, Any]],
        anomaly_signals: Optional[List[AnomalySignal]] = None,
        consequence_score: float = 0.5,
        cost_of_delay: float = 0.0,
        residual_sigma: float = 0.20,
        n_observed: int = 30,
        autocorrelation: float = 0.0,
        asset_value: float = 1.0,
        data_origin: str = "unlabeled",
    ) -> Prediction:
        if data_origin not in ("real", "demo", "synthetic", "unlabeled"):
            raise ValueError("data_origin must be real, demo, synthetic, or unlabeled")
        clamp_probability(baseline_tp)
        prediction_id = str(uuid.uuid4())
        anomalies = anomaly_signals or []

        prediction_seed = deterministic_prediction_seed(
            event_name=event_name,
            domain=domain,
            horizon=horizon,
            baseline_tp=baseline_tp,
            raw_observations=raw_observations,
            anomaly_signals=anomalies,
        )

        observations = self.observation.ingest(raw_observations)

        evidence = self.evidence.verify(
            observations=observations,
            prediction_seed=prediction_seed,
        )

        anomaly_result = self.anomaly.predict(
            anomalies=anomalies,
            asset_value=asset_value,
        )

        # v61 anchor rule: secondary anomaly machinery does not enter raw TP.
        # The anchor is baseline + deduped/cancelled evidence only.
        neutral_anomaly = dict(anomaly_result)
        neutral_anomaly["signed_log_odds_delta"] = 0.0

        tp_result = self.tp.calculate(
            baseline_tp=baseline_tp,
            evidence=evidence,
            anomaly=neutral_anomaly,
            residual_sigma=residual_sigma,
            n_observed=n_observed,
            autocorrelation=autocorrelation,
        )

        ip = self.ip.calculate(evidence)

        calibrated_tp, calibration_model = self.calibration.calibrate(
            raw_tp=tp_result["raw_tp"],
            domain=domain,
        )
        calibration_reliability_tp = calibration_model.reliability_tp

        signed_anomaly_delta = float(anomaly_result["signed_log_odds_delta"])
        anomaly_direction = 0.0
        if abs(signed_anomaly_delta) > EPS:
            anomaly_direction = math.copysign(1.0, signed_anomaly_delta)

        # Counterfactual candidates are always calculated and stored. They are
        # evidence for future ablation decisions, not automatic TP adjustments.
        simulation_anchor = self.conditional_simulation.simulate(
            anchor_tp=calibrated_tp,
            calibration_reliability_tp=calibration_reliability_tp,
            ip=calibrated_tp,
            anomaly_probability=0.0,
            anomaly_direction=0.0,
            contagion_probability=0.0,
            prediction_seed=f"{prediction_seed}:anchor",
        )
        simulation_ip = self.conditional_simulation.simulate(
            anchor_tp=calibrated_tp,
            calibration_reliability_tp=calibration_reliability_tp,
            ip=ip,
            anomaly_probability=0.0,
            anomaly_direction=0.0,
            contagion_probability=0.0,
            prediction_seed=f"{prediction_seed}:ip",
        )
        simulation_anomaly = self.conditional_simulation.simulate(
            anchor_tp=calibrated_tp,
            calibration_reliability_tp=calibration_reliability_tp,
            ip=calibrated_tp,
            anomaly_probability=anomaly_result["anomaly_probability"],
            anomaly_direction=anomaly_direction,
            contagion_probability=0.0,
            prediction_seed=f"{prediction_seed}:anomaly",
        )
        simulation_contagion = self.conditional_simulation.simulate(
            anchor_tp=calibrated_tp,
            calibration_reliability_tp=calibration_reliability_tp,
            ip=calibrated_tp,
            anomaly_probability=anomaly_result["anomaly_probability"],
            anomaly_direction=anomaly_direction,
            contagion_probability=anomaly_result["contagion_probability"],
            prediction_seed=f"{prediction_seed}:contagion",
        )
        simulation_full = self.conditional_simulation.simulate(
            anchor_tp=calibrated_tp,
            calibration_reliability_tp=calibration_reliability_tp,
            ip=ip,
            anomaly_probability=anomaly_result["anomaly_probability"],
            anomaly_direction=anomaly_direction,
            contagion_probability=anomaly_result["contagion_probability"],
            prediction_seed=f"{prediction_seed}:full",
        )

        candidates = {
            "anchor": round(calibrated_tp, 6),
            "ip": self._candidate_rate(simulation_ip),
            "anomaly": self._candidate_rate(simulation_anomaly),
            "contagion": self._candidate_rate(simulation_contagion),
            "full": self._candidate_rate(simulation_full),
        }

        gate_status = self.ablation.status(domain)
        ip_enabled = bool(gate_status["ip"]["enabled"])
        anomaly_enabled = bool(gate_status["anomaly"]["enabled"])
        # Contagion is only meaningful if anomaly influence itself is approved.
        contagion_enabled = bool(
            anomaly_enabled and gate_status["contagion"]["enabled"]
        )

        if ip_enabled or anomaly_enabled or contagion_enabled:
            approved_simulation = self.conditional_simulation.simulate(
                anchor_tp=calibrated_tp,
                calibration_reliability_tp=calibration_reliability_tp,
                ip=ip if ip_enabled else calibrated_tp,
                anomaly_probability=(
                    anomaly_result["anomaly_probability"] if anomaly_enabled else 0.0
                ),
                anomaly_direction=anomaly_direction if anomaly_enabled else 0.0,
                contagion_probability=(
                    anomaly_result["contagion_probability"]
                    if contagion_enabled
                    else 0.0
                ),
                prediction_seed=f"{prediction_seed}:approved",
            )
            final_tp = self._candidate_rate(approved_simulation)
        else:
            approved_simulation = simulation_anchor
            # Until a secondary module earns its way in, the calibrated anchor
            # itself is authoritative. Avoid Monte Carlo sampling noise here.
            final_tp = round(calibrated_tp, 6)

        decision = self.decision.decide(
            tp=final_tp,
            consequence_score=consequence_score,
            cost_of_delay=cost_of_delay,
            calibration_reliability_tp=calibration_reliability_tp,
        )

        prediction = Prediction(
            prediction_id=prediction_id,
            data_origin=data_origin,
            event=event_name,
            domain=domain,
            horizon=horizon,
            baseline_tp=round(clamp_probability(baseline_tp), 6),
            raw_tp=tp_result["raw_tp"],
            calibrated_tp=calibrated_tp,
            final_tp=final_tp,
            ip=ip,
            anomaly_probability=anomaly_result["anomaly_probability"],
            contagion_probability=anomaly_result["contagion_probability"],
            expected_loss=anomaly_result["expected_loss"],
            decision=decision,
            calibration_version=calibration_model.version,
            calibration_reliability_tp=round(calibration_reliability_tp, 6),
            diagnostics={
                "deterministic_prediction_seed": prediction_seed,
                "calibration_confidence_kind": "internal_heuristic_not_accuracy",
                "verified_resolved_count": len(self.ledger.get_resolved(domain)),
                "evidence": {
                    "source_confidence": evidence["source_confidence"],
                    "observation_count": evidence["observation_count"],
                    "resolution": evidence["resolution"],
                },
                "anomaly": anomaly_result,
                "tp": tp_result,
                "ablation_candidates": candidates,
                "ablation_gate": gate_status,
                "module_activation": {
                    "ip": ip_enabled,
                    "anomaly": anomaly_enabled,
                    "contagion": contagion_enabled,
                },
                "counterfactual_simulations": {
                    "anchor": simulation_anchor,
                    "ip": simulation_ip,
                    "anomaly": simulation_anomaly,
                    "contagion": simulation_contagion,
                    "full": simulation_full,
                    "approved": approved_simulation,
                },
            },
        )

        self.ledger.record_prediction(prediction)
        return prediction

    def resolve_and_retrain(
        self,
        prediction_id: str,
        outcome: float,
        *,
        outcome_verified: bool = False,
        outcome_source: str = "",
    ) -> Dict[str, Any]:
        self.ledger.resolve_prediction(prediction_id, outcome,
            outcome_verified=outcome_verified, outcome_source=outcome_source)
        prediction = self.ledger.records[prediction_id]

        ablation_result = self.ablation.observe_resolution(prediction)
        calibration_result = self.calibration.train_challenger(
            predictions=self.ledger.get_resolved(prediction.domain),
            domain=prediction.domain,
        )

        return {
            "prediction_id": prediction_id,
            "domain": prediction.domain,
            "outcome": float(outcome),
            "outcome_verified": prediction.outcome_verified,
            "outcome_source": prediction.outcome_source,
            "data_origin": prediction.data_origin,
            "learning_eligible": verified_prediction(prediction),
            "calibration": calibration_result,
            "ablation": ablation_result,
        }


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    runtime = RealModeRuntime(
        ledger_path="real_mode_prediction_ledger.json"
    )

    telemetry = [
        {
            "source": "SENSOR_A",
            "source_family": "SORTER_CLUSTER",
            "confidence": 0.92,
            "provenance_id": "packet_001",
            "payload": {
                "signals": [
                    {
                        "claim_id": "volume_state",
                        "name": "Volume pressure",
                        "direction": +1.0,
                        "strength": 0.80,
                        "pattern_key": "volume_pattern",
                    }
                ]
            },
        },
        {
            # Exact duplicate. Same signal provenance means one copy survives.
            "source": "SENSOR_A_MIRROR",
            "source_family": "SORTER_CLUSTER",
            "confidence": 0.90,
            "provenance_id": "packet_001_mirror",
            "payload": {
                "signals": [
                    {
                        "claim_id": "volume_state",
                        "provenance_id": "same_underlying_claim_001",
                        "name": "Volume pressure mirror",
                        "direction": +1.0,
                        "strength": 0.80,
                        "pattern_key": "volume_pattern",
                    }
                ]
            },
        },
        {
            "source": "SENSOR_A_ORIGINAL",
            "source_family": "SORTER_CLUSTER",
            "confidence": 0.92,
            "provenance_id": "packet_001_original",
            "payload": {
                "signals": [
                    {
                        "claim_id": "volume_state",
                        "provenance_id": "same_underlying_claim_001",
                        "name": "Volume pressure original",
                        "direction": +1.0,
                        "strength": 0.80,
                        "pattern_key": "volume_pattern",
                    }
                ]
            },
        },
        {
            # Independent contradiction against the same claim.
            # +1 and -1 cancel claim-by-claim before pattern recognition.
            "source": "MANUAL_AUDIT",
            "source_family": "HUMAN_AUDIT",
            "confidence": 0.92,
            "provenance_id": "packet_002",
            "payload": {
                "signals": [
                    {
                        "claim_id": "volume_state",
                        "name": "Volume normal",
                        "direction": -1.0,
                        "strength": 0.80,
                        "pattern_key": "volume_pattern",
                    }
                ]
            },
        },
        {
            # Residual pattern remains after cancellation.
            "source": "HISTORY_A",
            "source_family": "LOGISTICS_HISTORY",
            "confidence": 0.88,
            "payload": {
                "signals": [
                    {
                        "claim_id": "late_dispatch_day_1",
                        "name": "Late dispatch history 1",
                        "direction": -1.0,
                        "strength": 0.55,
                        "pattern_key": "late_dispatch_pattern",
                    },
                    {
                        "claim_id": "late_dispatch_day_2",
                        "name": "Late dispatch history 2",
                        "direction": -1.0,
                        "strength": 0.50,
                        "pattern_key": "late_dispatch_pattern",
                    },
                ]
            },
        },
    ]

    anomalies = [
        AnomalySignal(
            name="Equipment fault",
            kind="unexpected",
            probability=0.35,
            direction=-1.0,
            severity=0.70,
            confidence=0.90,
            persistence=0.75,
            contagion=0.60,
            age=0.10,
            half_life=6.0,
            source_family="EQUIPMENT",
            provenance_id="fault_001",
        )
    ]

    result = runtime.run(
        event_name="STAGE_2_VOLUME_DISPATCH_SUCCEEDS",
        domain="operations.schedule_completion",
        horizon="8_HOURS",
        baseline_tp=0.30,
        raw_observations=telemetry,
        anomaly_signals=anomalies,
        consequence_score=0.70,
        cost_of_delay=0.20,
        residual_sigma=0.18,
        n_observed=60,
        autocorrelation=0.15,
        asset_value=5000.0,
    )

    simulations = result.diagnostics["counterfactual_simulations"]
    gate = result.diagnostics["ablation_gate"]

    print("=== REAL MODE v61.1 — DEMO / UNVERIFIED ESTIMATES ===")
    print(f"Prediction ID:               {result.prediction_id}")
    print(f"Baseline TP:                 {result.baseline_tp:.2%}")
    print(f"Raw TP:                      {result.raw_tp:.2%}")
    print(f"Anchor estimate:             {result.calibrated_tp:.2%}")
    print(f"IP diagnostic:               {result.ip:.2%}")
    print(f"Anomaly probability:         {result.anomaly_probability:.2%}")
    print(f"Contagion probability:       {result.contagion_probability:.2%}")
    print(f"Final TP:                    {result.final_tp:.2%}")
    print(f"Decision:                    {result.decision}")
    print(f"Calibration version:         {result.calibration_version}")
    print(
        f"Internal confidence score:   "
        f"{result.calibration_reliability_tp:.3f} (not accuracy)"
    )
    print()
    print("Ablation candidates:")
    for name, probability in result.diagnostics["ablation_candidates"].items():
        print(f"  {name:10s}: {probability:.2%}")
    print()
    print("Module gates:")
    for name, status in gate.items():
        print(
            f"  {name:10s}: {status['status']} "
            f"(n={status['sample_count']})"
        )
    print()
    approved = simulations["approved"]
    print("Approved-module simulation:")
    print(f"Trials:                       {approved['trials']}")
    print(f"Baseline rate:                {approved['baseline_rate']:.2%}")
    print(f"Final simulated rate:         {approved['final_rate']:.2%}")
