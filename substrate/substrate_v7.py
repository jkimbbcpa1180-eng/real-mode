#!/usr/bin/env python3
"""
Substrate v7 — Mammalian Autarky + Demurrage + Municipal Ledger
===============================================================

A single-file, standard-library Python reference implementation of a
post-scarcity political economy.

Layers:
  1. HouseholdPod              zero-cost life support
  2. PersonhoodRegistry        sybil-resistant one-pod-per-person
  3. MeshConsensus             neighbor attestation with rate limits
  4. AGIDataEngine             non-linear entropy minting + UN gate
  5. DemurrageEngine           per-node idle decay, reset on activity
  6. MunicipalZone             ecological-fragility-scaled tolls
  7. MunicipalLedger           trip booking with cross-border gate
  8. UNRecognition             treaty + escalation ladder
  9. TransparencyLog           public event log
 10. Watcher                   read-only observation with voice
 11. Polity + Governance       votes, disputes, treasury
 12. AuditorRegistry           bonded certificate issuers with slashing
 13. SponsorPool               multi-sponsor pods with attribution
 14. Federation                cross-polity treaties

Invariants:
  - Participation is voluntary; exit is always possible
  - Personhood is unique (one biometric, one pod)
  - Accumulation is bounded (per-node demurrage)
  - External prohibitions bind (UN escalation ladder)
  - Every event is public (transparency log)

Run:
    python substrate_v7.py --test
    python substrate_v7.py --demo
    python substrate_v7.py --municipal-demo
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ===========================================================================
# 1. HouseholdPod
# ===========================================================================
@dataclass
class HouseholdPod:
    node_id: str
    battery_capacity_kwh: float = 35.0
    battery_current_kwh: float = 28.0
    water_tank_capacity_l: float = 1200.0
    water_current_l: float = 900.0
    aeroponic_daily_calories: float = 2800.0
    transit_credit_balance: float = 0.0
    person_id: str = ""
    polity: str = ""
    neighbors: Set[str] = field(default_factory=set)
    sponsors: Dict[str, float] = field(default_factory=dict)
    watcher_only: bool = False
    months_idle: int = 0
    monthly_demurrage_rate: float = 0.04

    def credit(self, amount: float) -> None:
        self.transit_credit_balance += amount
        self.months_idle = 0

    def step_month(self, is_active: bool) -> Dict[str, float]:
        initial = self.transit_credit_balance
        if not is_active and initial > 0:
            self.months_idle += 1
            self.transit_credit_balance = round(
                initial * (1.0 - self.monthly_demurrage_rate), 2)
            decayed = round(initial - self.transit_credit_balance, 2)
        else:
            self.months_idle = 0
            decayed = 0.0
        return {
            "initial_balance": initial,
            "decayed_burned": decayed,
            "net_balance": self.transit_credit_balance,
            "months_idle": self.months_idle,
        }

    def cycle_day(self, solar_kwh: float, load_kwh: float,
                  rain_l: float, use_l: float,
                  yield_factor: float = 1.0) -> Dict:
        net_e = solar_kwh - load_kwh
        self.battery_current_kwh = min(self.battery_capacity_kwh,
                                       max(0.0, self.battery_current_kwh + net_e))
        net_w = rain_l + use_l * 0.90 - use_l
        self.water_current_l = min(self.water_tank_capacity_l,
                                   max(0.0, self.water_current_l + net_w))
        calories = self.aeroponic_daily_calories * yield_factor
        return {
            "battery_storage_kwh": round(self.battery_current_kwh, 2),
            "water_storage_l": round(self.water_current_l, 2),
            "calories_yield": round(calories, 1),
            "cost_usd": 0.00,
            "sustainable": (self.battery_current_kwh > 0 and
                            self.water_current_l > 0 and calories >= 1800),
        }

    def total_sponsor_share(self) -> float:
        return sum(self.sponsors.values())


# ===========================================================================
# 2. PersonhoodRegistry
# ===========================================================================
@dataclass
class PersonhoodAttestation:
    person_id: str
    biometric_hash: str
    sponsors: Set[str] = field(default_factory=set)
    vouched_in_person: bool = False
    registered_day: int = 0


class PersonhoodRegistry:
    def __init__(self, vouches_required: int = 2,
                 bootstrap_sponsors: Optional[Set[str]] = None):
        self.vouches_required = vouches_required
        self.bootstrap_sponsors = bootstrap_sponsors or set()
        self._biometric_index: Dict[str, str] = {}
        self.persons: Dict[str, PersonhoodAttestation] = {}
        self._pod_bindings: Dict[str, str] = {}

    @staticmethod
    def _hash(b: str, salt: str = "autarky-v7") -> str:
        return hashlib.sha256(f"{salt}|{b}".encode()).hexdigest()[:24]

    def register(self, person_id: str, biometric: str,
                 sponsors: Optional[List[str]] = None,
                 in_person: bool = False, day: int = 0) -> Dict:
        bh = self._hash(biometric)
        if bh in self._biometric_index:
            return {"status": "FAILED",
                    "reason": "biometric already registered"}
        if person_id in self.persons:
            return {"status": "FAILED", "reason": "person_id taken"}
        sponsors = sponsors or []
        if not self.persons:
            if not self.bootstrap_sponsors:
                return {"status": "FAILED",
                        "reason": "no bootstrap sponsors configured"}
            att = PersonhoodAttestation(person_id, bh,
                                        set(self.bootstrap_sponsors),
                                        True, day)
        else:
            valid = [s for s in sponsors if s in self.persons]
            if len(valid) < self.vouches_required:
                return {"status": "FAILED",
                        "reason": f"need {self.vouches_required} sponsors, "
                                  f"have {len(valid)}"}
            att = PersonhoodAttestation(person_id, bh, set(valid),
                                        in_person, day)
        self._biometric_index[bh] = person_id
        self.persons[person_id] = att
        return {"status": "REGISTERED", "person_id": person_id,
                "sponsors": len(att.sponsors)}

    def bind_pod(self, person_id: str, pod_id: str) -> Dict:
        if person_id not in self.persons:
            return {"status": "FAILED", "reason": "person not registered"}
        if person_id in self._pod_bindings:
            return {"status": "FAILED",
                    "reason": f"person already bound to "
                              f"{self._pod_bindings[person_id]}"}
        self._pod_bindings[person_id] = pod_id
        return {"status": "BOUND", "person_id": person_id, "pod_id": pod_id}

    def unbind_pod(self, person_id: str) -> None:
        self._pod_bindings.pop(person_id, None)

    def pod_count(self) -> int:
        return len(self._pod_bindings)


# ===========================================================================
# 3. MeshConsensus
# ===========================================================================
class MeshConsensus:
    def __init__(self, quorum: int = 3,
                 rate_limit_per_hour: int = 2,
                 neighbor_required: bool = True):
        self.quorum = quorum
        self.rate_limit = rate_limit_per_hour
        self.neighbor_required = neighbor_required
        self._attestation_log: Dict[str, List[Tuple[int, str]]] = {}

    @staticmethod
    def _sig(pod_id: str, claim_id: str) -> str:
        return hashlib.sha256(f"{pod_id}|{claim_id}".encode()).hexdigest()[:16]

    def register(self, claimer: HouseholdPod, domain: str,
                 hours: float, peers: List[HouseholdPod],
                 hour: int) -> Dict:
        claim_id = hashlib.sha256(
            f"{claimer.node_id}|{domain}|{hours}|{hour}".encode()
        ).hexdigest()[:12]
        accepted, rejected = [], []
        for peer in peers:
            if self.neighbor_required and peer.node_id not in claimer.neighbors:
                rejected.append((peer.node_id, "not a neighbor"))
                continue
            log = self._attestation_log.setdefault(peer.node_id, [])
            recent = [e for e in log if e[0] == hour]
            if len(recent) >= self.rate_limit:
                rejected.append((peer.node_id, "rate limit"))
                continue
            log.append((hour, claim_id))
            accepted.append(peer.node_id)
        return {
            "claim_id": claim_id,
            "accepted": accepted,
            "rejected": rejected,
            "quorum_met": len(accepted) >= self.quorum,
            "needed": self.quorum,
            "sigs": [self._sig(p, claim_id) for p in accepted],
        }


# ===========================================================================
# 4. TransparencyLog + Watcher
# ===========================================================================
class Watcher:
    def __init__(self, watcher_id: str):
        self.watcher_id = watcher_id
        self.comments: List[Dict] = []
        self.read_log: List[str] = []
        self.filed_proposals: List[str] = []

    def read(self, event_type: str, payload: Dict) -> None:
        self.read_log.append(event_type)

    def comment(self, target_id: str, text: str, day: int) -> Dict:
        c = {"watcher_id": self.watcher_id, "target": target_id,
             "text": text, "day": day}
        self.comments.append(c)
        return c


class TransparencyLog:
    def __init__(self):
        self.events: List[Dict] = []
        self.watchers: List[Watcher] = []

    def append(self, event_type: str, payload: Dict, day: int = 0) -> None:
        event = {"seq": len(self.events), "type": event_type,
                 "day": day, "payload": payload}
        self.events.append(event)
        for w in self.watchers:
            w.read(event_type, payload)

    def add_watcher(self, watcher: Watcher) -> None:
        self.watchers.append(watcher)

    def read_all(self, event_type: Optional[str] = None) -> List[Dict]:
        if event_type is None:
            return list(self.events)
        return [e for e in self.events if e["type"] == event_type]


# ===========================================================================
# 5. UNRecognition + EscalationLadder
# ===========================================================================
@dataclass
class UNProhibition:
    code: str
    description: str
    applies_to_domains: List[str]
    severity: str = "minor"


class UNEscalationLadder:
    STAGES = ["CLEAR", "WARNING", "PARTIAL_SUSP",
              "FULL_SUSPENSION", "DECERTIFICATION"]
    MINOR_CODES = {"UNP-01", "UNP-03", "UNP-04"}
    MAJOR_CODES = {"UNP-02"}

    def __init__(self):
        self.stage: Dict[str, int] = {}
        self.violation_log: Dict[str, List[Dict]] = {}

    def record_violation(self, polity: str, code: str, day: int,
                         detail: str = "") -> Dict:
        self.violation_log.setdefault(polity, []).append(
            {"code": code, "day": day, "detail": detail})
        self._recompute(polity)
        return {"polity": polity, "stage": self.stage[polity],
                "stage_name": self.STAGES[self.stage[polity]]}

    def _recompute(self, polity: str) -> None:
        v = self.violation_log.get(polity, [])
        if not v:
            self.stage[polity] = 0
            return
        last = v[-1]["day"]
        recent = [x for x in v if last - x["day"] <= 90]
        minors = [x for x in recent if x["code"] in self.MINOR_CODES]
        majors = [x for x in recent if x["code"] in self.MAJOR_CODES]
        life_majors = [x for x in v if x["code"] in self.MAJOR_CODES]
        stage = 0
        if len(minors) >= 1:
            stage = max(stage, 1)
        if len(minors) >= 3:
            stage = max(stage, 2)
        if len(majors) >= 1:
            stage = max(stage, 2)
        if len(majors) >= 2 or (len(majors) >= 1 and len(minors) >= 3):
            stage = max(stage, 3)
        if len(life_majors) >= 5:
            stage = max(stage, 4)
        if any(x["code"] == "UNP-02" and "forced_labor" in x.get("detail", "")
               for x in v):
            stage = 4
        self.stage[polity] = stage

    def current_stage(self, polity: str) -> Tuple[int, str]:
        s = self.stage.get(polity, 0)
        return s, self.STAGES[s]

    def appeal(self, polity: str, justification: str, day: int) -> Dict:
        if polity not in self.stage or self.stage[polity] == 0:
            return {"status": "FAILED", "reason": "no stage to appeal"}
        old = self.stage[polity]
        self.stage[polity] = max(0, old - 1)
        return {"status": "GRANTED", "polity": polity,
                "from_stage": old, "to_stage": self.stage[polity]}

    def can_cross_border(self, polity: str) -> bool:
        return self.stage.get(polity, 0) <= 1

    def can_register_pods(self, polity: str) -> bool:
        return self.stage.get(polity, 0) <= 2


class UNRecognition:
    STANDARD_PROHIBITIONS = [
        UNProhibition("UNP-01", "Child labor",
                      ["community_care"], "major"),
        UNProhibition("UNP-02", "Forced labor",
                      ["community_care", "permaculture", "field_study"],
                      "major"),
        UNProhibition("UNP-03", "Environmental destruction",
                      ["permaculture", "field_study"], "minor"),
        UNProhibition("UNP-04", "Cultural erasure",
                      ["ritual_observation"], "minor"),
    ]

    def __init__(self, signatories: List[str],
                 prohibitions: Optional[List[UNProhibition]] = None):
        self.signatories = set(signatories)
        self.prohibitions = prohibitions or list(self.STANDARD_PROHIBITIONS)
        self.recognized_polities: Set[str] = set()
        self.personhood_reports: List[Dict] = []
        self.ladder = UNEscalationLadder()

    def recognize(self, polity: str) -> Dict:
        stage, name = self.ladder.current_stage(polity)
        if stage >= 3:
            return {"status": "FAILED", "reason": f"polity is {name}"}
        self.recognized_polities.add(polity)
        return {"status": "RECOGNIZED", "polity": polity, "stage": name}

    def check_activity(self, domain: str, metadata: Dict,
                       polity: Optional[str] = None) -> Dict:
        for p in self.prohibitions:
            if domain in p.applies_to_domains:
                if metadata.get(f"violates_{p.code}", False):
                    if polity:
                        self.ladder.record_violation(
                            polity, p.code, metadata.get("day", 0),
                            metadata.get(f"detail_{p.code}", ""))
                    return {"status": "REJECT", "prohibition": p.code,
                            "severity": p.severity,
                            "reason": p.description}
        return {"status": "OK"}

    def report_personhood(self, person_id: str, polity: str,
                          day: int) -> None:
        self.personhood_reports.append({
            "person_hash": hashlib.sha256(person_id.encode()).hexdigest()[:16],
            "polity": polity,
            "day": day,
        })


# ===========================================================================
# 6. AGIDataEngine
# ===========================================================================
@dataclass
class ProvenanceEvidence:
    sensor_variance: float
    quorum_attestations: int
    quorum_required: int = 3

    @property
    def quorum_score(self) -> float:
        return min(1.0, self.quorum_attestations / max(1, self.quorum_required))

    @property
    def combined_entropy(self) -> float:
        return 0.6 * self.sensor_variance + 0.4 * self.quorum_score


@dataclass
class MammalianActivity:
    domain: str
    description: str
    hours_logged: float
    provenance: ProvenanceEvidence
    peer_consensus_rating: float
    pod_id: str = ""
    person_id: str = ""
    polity: str = ""
    metadata: Dict = field(default_factory=dict)


class AGIDataEngine:
    DEFAULT_DOMAIN_WEIGHTS = {
        "community_care":     1.50,
        "permaculture":       1.30,
        "ritual_observation": 1.40,
        "field_study":        1.25,
        "craft":              1.15,
        "synthetic_spam":     0.00,
    }

    def __init__(self, base_rate_per_hour: float = 12.0,
                 entropy_threshold: float = 0.75,
                 consensus_floor: float = 0.60,
                 un_recognition: Optional[UNRecognition] = None,
                 transparency_log: Optional[TransparencyLog] = None):
        self.base_rate = base_rate_per_hour
        self.entropy_threshold = entropy_threshold
        self.consensus_floor = consensus_floor
        self.domain_weights = dict(self.DEFAULT_DOMAIN_WEIGHTS)
        self.un = un_recognition
        self.log = transparency_log

    def evaluate_and_mint(self, activity: MammalianActivity,
                          day: int = 0) -> Tuple[float, Optional[str]]:
        if self.un is not None:
            check = self.un.check_activity(activity.domain,
                                           activity.metadata,
                                           polity=activity.polity or None)
            if check["status"] == "REJECT":
                if self.log:
                    self.log.append("un_reject", {
                        "pod_id": activity.pod_id,
                        "domain": activity.domain,
                        "prohibition": check["prohibition"]}, day)
                return 0.0, (f"REJECTED [UN {check['prohibition']}]: "
                             f"{check['reason']}")

        entropy = activity.provenance.combined_entropy
        if entropy < self.entropy_threshold:
            return 0.0, (f"REJECTED [{activity.domain}]: entropy "
                         f"{entropy:.2f} < {self.entropy_threshold:.2f}")

        if activity.peer_consensus_rating < self.consensus_floor:
            return 0.0, (f"REJECTED [{activity.domain}]: consensus "
                         f"{activity.peer_consensus_rating:.2f} < "
                         f"{self.consensus_floor:.2f}")

        entropy_multiplier = (
            1.0 + (entropy - self.entropy_threshold) /
            (1.0 - self.entropy_threshold)
        )
        domain_weight = self.domain_weights.get(activity.domain, 1.0)
        minted = (activity.hours_logged * self.base_rate *
                  entropy_multiplier * domain_weight *
                  activity.peer_consensus_rating)
        if self.log:
            self.log.append("mint", {
                "pod_id": activity.pod_id, "domain": activity.domain,
                "hours": activity.hours_logged,
                "entropy": round(entropy, 3),
                "minted": round(minted, 2)}, day)
        return round(minted, 2), None


# ===========================================================================
# 7. MunicipalZone + MunicipalLedger
# ===========================================================================
@dataclass
class MunicipalZone:
    zone_name: str
    base_tc_cost: float
    max_capacity: int
    current_occupancy: int
    ecological_fragility: float = 1.0

    def current_transit_toll(self) -> float:
        load_ratio = self.current_occupancy / max(1, self.max_capacity)
        congestion_penalty = (
            1.0 + max(0.0, load_ratio - 0.70) * 3.5 * self.ecological_fragility
        )
        return round(self.base_tc_cost * congestion_penalty, 2)

    def admit_traveler(self) -> bool:
        if self.current_occupancy < self.max_capacity:
            self.current_occupancy += 1
            return True
        return False


class MunicipalLedger:
    @staticmethod
    def process_trip(pod: HouseholdPod, zone: MunicipalZone,
                     polity_name: str = "",
                     un: Optional[UNRecognition] = None) -> Dict:
        if un is not None and polity_name:
            if not un.ladder.can_cross_border(polity_name):
                stage, name = un.ladder.current_stage(polity_name)
                return {"status": "DENIED",
                        "reason": f"polity {polity_name} at stage {stage} "
                                  f"({name}); cross-border travel suspended"}

        toll = zone.current_transit_toll()
        if pod.transit_credit_balance < toll:
            return {"status": "DENIED",
                    "reason": f"need {toll:.2f} TC, "
                              f"have {pod.transit_credit_balance:.2f}"}
        if zone.current_occupancy >= zone.max_capacity:
            return {"status": "DENIED",
                    "reason": f"zone {zone.zone_name} at capacity"}
        pod.transit_credit_balance -= toll
        pod.months_idle = 0
        zone.admit_traveler()
        return {
            "status": "APPROVED",
            "destination": zone.zone_name,
            "toll_paid": round(toll, 2),
            "remaining_balance": round(pod.transit_credit_balance, 2),
            "capacity_load": f"{zone.current_occupancy}/{zone.max_capacity}",
        }


# ===========================================================================
# 8. Polity + Treasury (compact)
# ===========================================================================
@dataclass
class Polity:
    name: str
    member_pods: Set[str] = field(default_factory=set)

    def admit(self, pod: HouseholdPod) -> Dict:
        self.member_pods.add(pod.node_id)
        pod.polity = self.name
        return {"status": "ADMITTED", "polity": self.name,
                "members": len(self.member_pods)}


class Treasury:
    def __init__(self):
        self.balance = 0.0

    def deposit(self, amount: float, source: str) -> None:
        self.balance += amount

    def spend(self, amount: float, purpose: str,
              approvals: Set[str]) -> Dict:
        if len(approvals) < 3 or amount > self.balance:
            return {"status": "FAILED"}
        self.balance -= amount
        return {"status": "OK", "balance": self.balance}


# ===========================================================================
# 9. AuditorRegistry + ESGCertificate
# ===========================================================================
@dataclass
class Auditor:
    auditor_id: str
    name: str
    bond_tc: float
    issued_certificates: List[str] = field(default_factory=list)
    slash_count: int = 0
    removed: bool = False


class AuditorRegistry:
    MIN_BOND = 500.0
    MAX_SLASH_RATE = 0.05

    def __init__(self):
        self.auditors: Dict[str, Auditor] = {}
        self.contested: List[Dict] = []

    def register(self, auditor_id: str, name: str, bond: float) -> Auditor:
        if bond < self.MIN_BOND:
            raise ValueError(f"bond {bond} below minimum {self.MIN_BOND}")
        a = Auditor(auditor_id, name, bond)
        self.auditors[auditor_id] = a
        return a

    def slash(self, auditor_id: str, amount: float, reason: str) -> Dict:
        a = self.auditors.get(auditor_id)
        if a is None or a.removed:
            return {"status": "FAILED", "reason": "auditor unavailable"}
        slash = min(amount, a.bond_tc)
        a.bond_tc -= slash
        a.slash_count += 1
        rec = {"auditor_id": auditor_id, "amount": slash, "reason": reason}
        self.contested.append(rec)
        if a.issued_certificates:
            rate = a.slash_count / len(a.issued_certificates)
            if rate > self.MAX_SLASH_RATE:
                a.removed = True
        return {"status": "SLASHED", **rec, "bond_after": a.bond_tc}

    def is_valid(self, auditor_id: str) -> bool:
        a = self.auditors.get(auditor_id)
        if a is None or a.removed:
            return False
        if a.bond_tc < self.MIN_BOND * 0.5:
            return False
        return True


@dataclass
class ESGCertificate:
    certificate_id: str
    sponsor_id: str
    pod_id: str
    metric: str
    amount: float
    issued_day: int
    auditor_id: str
    auditor_signature: str

    @staticmethod
    def issue(sponsor_id: str, pod_id: str, metric: str, amount: float,
              day: int, auditor: Auditor,
              registry: AuditorRegistry) -> Optional["ESGCertificate"]:
        if not registry.is_valid(auditor.auditor_id):
            return None
        cid = hashlib.sha256(
            f"{sponsor_id}|{pod_id}|{metric}|{amount:.4f}|{day}".encode()
        ).hexdigest()[:16]
        sig = hashlib.sha256(
            f"{cid}|{auditor.auditor_id}|{auditor.bond_tc}".encode()
        ).hexdigest()[:24]
        cert = ESGCertificate(cid, sponsor_id, pod_id, metric, amount,
                              day, auditor.auditor_id, sig)
        auditor.issued_certificates.append(cid)
        return cert


# ===========================================================================
# 10. SponsorPool + CorporateSponsor
# ===========================================================================
@dataclass
class SponsorAllocation:
    sponsor_id: str
    amount_tc: float
    day_funded: int
    reason: str


class SponsorPool:
    def __init__(self, notice_period_days: int = 60,
                 cancellation_fee_pct: float = 0.10):
        self.notice_period_days = notice_period_days
        self.cancellation_fee_pct = cancellation_fee_pct
        self._allocations: Dict[str, List[SponsorAllocation]] = {}

    def contribute(self, pod: HouseholdPod, sponsor_id: str,
                   amount: float, day: int, reason: str) -> Dict:
        if amount <= 0:
            return {"status": "FAILED", "reason": "amount must be positive"}
        pod.credit(amount)
        pod.sponsors[sponsor_id] = pod.sponsors.get(sponsor_id, 0.0) + amount
        self._allocations.setdefault(pod.node_id, []).append(
            SponsorAllocation(sponsor_id, amount, day, reason))
        return {"status": "CONTRIBUTED", "pod": pod.node_id,
                "sponsor": sponsor_id, "amount": amount,
                "pod_total_sponsors": len(pod.sponsors)}

    def attribution(self, pod: HouseholdPod, total_amount: float
                    ) -> Dict[str, float]:
        total = pod.total_sponsor_share()
        if total <= 0:
            return {}
        return {sid: round(total_amount * (amt / total), 4)
                for sid, amt in pod.sponsors.items()}

    def largest_share(self, pod: HouseholdPod) -> Tuple[Optional[str], float]:
        if not pod.sponsors:
            return None, 0.0
        sid = max(pod.sponsors, key=lambda k: pod.sponsors[k])
        total = pod.total_sponsor_share()
        return sid, round(pod.sponsors[sid] / max(total, 1e-9), 4)


class CorporateSponsor:
    def __init__(self, sponsor_id: str, name: str,
                 annual_budget_tc: float, max_per_pod_tc: float,
                 focus_metrics: List[str]):
        self.sponsor_id = sponsor_id
        self.name = name
        self.annual_budget_tc = annual_budget_tc
        self.max_per_pod_tc = max_per_pod_tc
        self.focus_metrics = focus_metrics
        self.committed_tc = 0.0
        self.disbursed_tc = 0.0
        self.certificates: List[ESGCertificate] = []

    def remaining(self) -> float:
        return max(0.0, self.annual_budget_tc - self.disbursed_tc)

    def fund_pod(self, pool: SponsorPool, pod: HouseholdPod,
                 amount: float, reason: str, day: int) -> Dict:
        current = pod.sponsors.get(self.sponsor_id, 0.0)
        if current + amount > self.max_per_pod_tc:
            return {"status": "FAILED",
                    "reason": f"per-pod cap {self.max_per_pod_tc} exceeded"}
        if amount > self.remaining():
            return {"status": "FAILED", "reason": "annual budget exceeded"}
        r = pool.contribute(pod, self.sponsor_id, amount, day, reason)
        self.committed_tc += amount
        self.disbursed_tc += amount
        return {**r, "remaining_budget": round(self.remaining(), 2)}

    def issue_certificate(self, pod: HouseholdPod, metric: str,
                          amount: float, day: int,
                          auditor: Auditor,
                          registry: AuditorRegistry
                          ) -> Optional[ESGCertificate]:
        cert = ESGCertificate.issue(self.sponsor_id, pod.node_id, metric,
                                    amount, day, auditor, registry)
        if cert is not None:
            self.certificates.append(cert)
        return cert


# ===========================================================================
# 11. Federation
# ===========================================================================
@dataclass
class FederationTreaty:
    polity_a: str
    polity_b: str
    recognize_attestations: bool = True
    recognize_tc: bool = True
    recognize_personhood: bool = True
    cross_polity_fee_pct: float = 0.01
    active: bool = True
    signed_day: int = 0


class Federation:
    def __init__(self):
        self.treaties: Dict[Tuple[str, str], FederationTreaty] = {}

    def sign(self, a: str, b: str, **kwargs) -> FederationTreaty:
        t = FederationTreaty(polity_a=a, polity_b=b, **kwargs)
        self.treaties[(a, b)] = t
        self.treaties[(b, a)] = t
        return t

    def get_treaty(self, a: str, b: str) -> Optional[FederationTreaty]:
        return self.treaties.get((a, b))

    def can_recognize_attestation(self, from_polity: str,
                                  to_polity: str) -> bool:
        if from_polity == to_polity:
            return True
        t = self.get_treaty(from_polity, to_polity)
        return bool(t and t.active and t.recognize_attestations)

    def cross_polity_transfer(self, src: HouseholdPod, dst: HouseholdPod,
                              amount: float, src_polity: Polity,
                              dst_polity: Polity) -> Dict:
        if src_polity.name == dst_polity.name:
            return {"status": "FAILED",
                    "reason": "use InterPodTransfer for same-polity"}
        t = self.get_treaty(src_polity.name, dst_polity.name)
        if t is None or not t.active or not t.recognize_tc:
            return {"status": "FAILED",
                    "reason": "no active treaty recognizing TC"}
        if amount <= 0 or src.transit_credit_balance < amount:
            return {"status": "FAILED", "reason": "insufficient balance"}
        base_fee = amount * 0.005
        treaty_fee = amount * t.cross_polity_fee_pct
        delivered = amount - base_fee - treaty_fee
        src.transit_credit_balance -= amount
        dst.credit(delivered)
        return {"status": "SETTLED", "amount": amount,
                "base_fee": round(base_fee, 4),
                "treaty_fee": round(treaty_fee, 4),
                "delivered": round(delivered, 4),
                "from_polity": src_polity.name,
                "to_polity": dst_polity.name}

    def recognize_personhood(self, registry_a: PersonhoodRegistry,
                             registry_b: PersonhoodRegistry,
                             person_id: str) -> Dict:
        if person_id not in registry_a.persons:
            return {"status": "FAILED", "reason": "person not in registry_a"}
        att = registry_a.persons[person_id]
        if att.biometric_hash in registry_b._biometric_index:
            return {"status": "FAILED", "reason": "already known to B"}
        registry_b._biometric_index[att.biometric_hash] = person_id
        registry_b.persons[person_id] = PersonhoodAttestation(
            person_id=person_id, biometric_hash=att.biometric_hash,
            sponsors=set(att.sponsors), vouched_in_person=True,
            registered_day=att.registered_day)
        return {"status": "RECOGNIZED", "person_id": person_id}


# ===========================================================================
# Demos
# ===========================================================================
def _setup(seed: int = 42):
    rng = random.Random(seed)
    registry = PersonhoodRegistry(bootstrap_sponsors={"bootstrap"})
    registry.register("p0", "bio0", day=0)
    persons = ["p0"]
    pods: List[HouseholdPod] = []
    for i in range(5):
        sponsors = list(set(rng.sample(persons, min(2, len(persons)))))
        if len(sponsors) < 2:
            sponsors = ["p0", "p0"]
        r = registry.register(f"p{i+1}", f"bio{i+1}",
                              sponsors=sponsors, in_person=True, day=0)
        if r["status"] != "REGISTERED":
            continue
        pod = HouseholdPod(f"POD-{i:03d}", transit_credit_balance=100.0,
                           person_id=f"p{i+1}")
        registry.bind_pod(f"p{i+1}", pod.node_id)
        pods.append(pod)
        persons.append(f"p{i+1}")
    return rng, registry, pods


def demo() -> None:
    print("=" * 88)
    print("SUBSTRATE v7 — FULL STACK DEMO")
    print("=" * 88)

    rng, registry, pods = _setup()
    log = TransparencyLog()
    mesh = MeshConsensus(quorum=3, rate_limit_per_hour=4)
    un = UNRecognition(signatories=["KR", "JP", "SG"])
    un.recognize("River Watershed")
    agi = AGIDataEngine(un_recognition=un, transparency_log=log)

    for i, p in enumerate(pods):
        p.neighbors.add(pods[(i + 1) % len(pods)].node_id)
        p.neighbors.add(pods[(i - 1) % len(pods)].node_id)
        p.polity = "River Watershed"

    # --- Minting with mesh ---
    print("\n[1] MINTING (mesh consensus + non-linear entropy)")
    for i, pod in enumerate(pods):
        peers = [q for q in pods if q.node_id in pod.neighbors]
        att = mesh.register(pod, "community_care", 6.0, peers, hour=12 + i)
        prov = ProvenanceEvidence(
            sensor_variance=0.92,
            quorum_attestations=len(att["accepted"]))
        act = MammalianActivity(
            domain="community_care", description="elder care",
            hours_logged=6.0, provenance=prov,
            peer_consensus_rating=0.90,
            pod_id=pod.node_id, polity="River Watershed")
        credits, err = agi.evaluate_and_mint(act, day=1)
        if err:
            print(f"  [-] {pod.node_id}: {err}")
        else:
            pod.credit(credits)
            print(f"  [+] {pod.node_id}: minted {credits:.2f} TC "
                  f"(entropy {prov.combined_entropy:.3f}, "
                  f"quorum {len(att['accepted'])}/{att['needed']})")

    # --- Demurrage idle vs active ---
    print("\n[2] DEMURRAGE (idle vs active, 4% monthly)")
    active = pods[0]
    idle = pods[1]
    for m in range(1, 7):
        r_a = active.step_month(is_active=True)
        r_i = idle.step_month(is_active=False)
        print(f"  month {m}: active={r_a['net_balance']:8.2f} | "
              f"idle={r_i['net_balance']:8.2f} "
              f"(burned {r_i['decayed_burned']:.2f})")

    # --- Municipal zones ---
    print("\n[3] MUNICIPAL ZONES (fragility-scaled tolls)")
    jeju = MunicipalZone("Jeju Volcanic Trail", base_tc_cost=30.0,
                         max_capacity=100, current_occupancy=85,
                         ecological_fragility=1.5)
    seorak = MunicipalZone("Seoraksan Rural Sanctuary", base_tc_cost=25.0,
                           max_capacity=200, current_occupancy=40,
                           ecological_fragility=0.8)
    print(f"  Jeju    (85% load, fragility 1.5): "
          f"toll {jeju.current_transit_toll():.2f}")
    print(f"  Seorak  (20% load, fragility 0.8): "
          f"toll {seorak.current_transit_toll():.2f}")

    # --- Trip booking ---
    print("\n[4] TRIP BOOKING")
    pod = pods[0]
    r = MunicipalLedger.process_trip(pod, jeju, pod.polity, un)
    print(f"  {pod.node_id} -> Jeju: {r['status']} | "
          f"{r.get('toll_paid', r.get('reason', ''))}")

    # --- UN escalation ---
    print("\n[5] UN ESCALATION")
    un.ladder.record_violation("River Watershed", "UNP-04", 2)
    un.ladder.record_violation("River Watershed", "UNP-04", 3)
    un.ladder.record_violation("River Watershed", "UNP-01", 4)
    un.ladder.record_violation("River Watershed", "UNP-02", 5,
                               detail="forced_labor incident")
    stage, name = un.ladder.current_stage("River Watershed")
    print(f"  polity stage: {stage} ({name})")
    r = MunicipalLedger.process_trip(pod, seorak, pod.polity, un)
    print(f"  {pod.node_id} -> Seorak: {r['status']} | {r.get('reason')}")

    # --- Auditors + sponsors ---
    print("\n[6] AUDITORS + SPONSORS")
    areg = AuditorRegistry()
    a1 = areg.register("AUD-001", "GreenVerify", bond=2000.0)
    a2 = areg.register("AUD-002", "CareAudit", bond=1500.0)
    pool = SponsorPool()
    sponsor = CorporateSponsor("SP-001", "Korea Green Energy Co.",
                               annual_budget_tc=50_000.0,
                               max_per_pod_tc=1_000.0,
                               focus_metrics=["carbon_kg", "care_hours"])
    for p in pods[:3]:
        r = sponsor.fund_pod(pool, p, 200.0, "Q1 ESG", day=1)
        print(f"  {sponsor.name} -> {p.node_id}: {r['status']}")
    cert = sponsor.issue_certificate(pods[0], "carbon_kg", 45.0, day=2,
                                     auditor=a1, registry=areg)
    print(f"  cert issued: {cert.certificate_id[:8]} by {cert.auditor_id}")
    r = areg.slash(a1.auditor_id, 300.0, "overturned")
    print(f"  slash: {r['status']} bond_after {r['bond_after']}")

    # --- Attribution ---
    print("\n[7] MULTI-SPONSOR ATTRIBUTION")
    sponsor2 = CorporateSponsor("SP-002", "Frontier Alignment Lab",
                                annual_budget_tc=200_000.0,
                                max_per_pod_tc=5_000.0,
                                focus_metrics=["care_hours"])
    sponsor2.fund_pod(pool, pods[0], 300.0, "Q1 alignment", day=1)
    attr = pool.attribution(pods[0], 1000.0)
    print(f"  1000 TC to {pods[0].node_id}: {attr}")
    sid, share = pool.largest_share(pods[0])
    print(f"  largest sponsor: {sid} at {share:.0%}")

    # --- Federation ---
    print("\n[8] FEDERATION")
    other = Polity("Jeju Watershed")
    other_pod = HouseholdPod("POD-JEJU-01")
    other.admit(other_pod)
    fed = Federation()
    fed.sign("River Watershed", "Jeju Watershed",
             recognize_tc=True, cross_polity_fee_pct=0.01)
    r = fed.cross_polity_transfer(pods[0], other_pod, 50.0, Polity("River Watershed"), other)
    print(f"  cross-polity transfer: {r['status']} | "
          f"delivered {r.get('delivered', 'N/A')}")

    # --- Watcher ---
    print("\n[9] WATCHER")
    w = Watcher("W-001")
    log.add_watcher(w)
    log.append("test_event", {"x": 1}, day=6)
    print(f"  watcher read {len(w.read_log)} events")
    print(f"  total transparency events: {len(log.events)}")

    # --- Health ---
    print("\n[10] HEALTH SNAPSHOT")
    total_tc = sum(p.transit_credit_balance for p in pods)
    print(f"  pods: {len(pods)}")
    print(f"  registered persons: {len(registry.persons)}")
    print(f"  total TC in pods: {total_tc:.2f}")
    print(f"  transparency events: {len(log.events)}")


def municipal_demo() -> None:
    print("=" * 88)
    print("MUNICIPAL DEMO — Fragility-scaled tolling")
    print("=" * 88)
    zones = [
        MunicipalZone("Robust City Park", 20.0, 500, 400,
                      ecological_fragility=0.3),
        MunicipalZone("Balanced Forest Trail", 25.0, 200, 160,
                      ecological_fragility=1.0),
        MunicipalZone("Fragile Alpine Ridge", 30.0, 50, 40,
                      ecological_fragility=2.5),
    ]
    pod = HouseholdPod("POD-TEST", transit_credit_balance=1000.0)
    for z in zones:
        print(f"  {z.zone_name:<28} fragility {z.ecological_fragility:.1f} | "
              f"load {z.current_occupancy/z.max_capacity:.0%} | "
              f"toll {z.current_transit_toll():.2f} TC")
        r = MunicipalLedger.process_trip(pod, z)
        print(f"    -> {r['status']} | {r.get('toll_paid', 'N/A')}")


# ===========================================================================
# Tests
# ===========================================================================
def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run_unit_tests() -> None:
    print("Running substrate v7 test suite...")

    # --- Non-linear entropy multiplier at max entropy ---
    log = TransparencyLog()
    agi = AGIDataEngine(transparency_log=log)
    prov_max = ProvenanceEvidence(sensor_variance=1.0, quorum_attestations=3)
    _check(abs(prov_max.combined_entropy - 1.0) < 1e-9, "max entropy = 1.0")
    act = MammalianActivity(
        domain="community_care", description="test",
        hours_logged=1.0, provenance=prov_max,
        peer_consensus_rating=1.0, pod_id="P", polity="")
    credits, err = agi.evaluate_and_mint(act)
    _check(err is None, "max entropy accepted")
    _check(abs(credits - 36.0) < 1e-6, f"expected 36.0, got {credits}")

    # --- Demurrage with reset-on-credit ---
    pod = HouseholdPod("P", transit_credit_balance=1000.0)
    r = pod.step_month(is_active=False)
    _check(r["months_idle"] == 1, "idle counted")
    _check(abs(pod.transit_credit_balance - 960.0) < 0.01, "4% decayed")
    pod.credit(50.0)
    _check(pod.months_idle == 0, "credit resets idle")

    # --- Personhood uniqueness ---
    reg = PersonhoodRegistry(bootstrap_sponsors={"boot"})
    r = reg.register("p1", "same", day=0)
    _check(r["status"] == "REGISTERED", "first")
    r = reg.register("p2", "same", day=0)
    _check(r["status"] == "FAILED", "sybil rejected")

    # --- Mesh consensus: neighbor rule ---
    pods = [HouseholdPod(f"P{i}") for i in range(5)]
    pods[0].neighbors = {"P1", "P2", "P3"}
    mesh = MeshConsensus(quorum=3, rate_limit_per_hour=2)
    r = mesh.register(pods[0], "community_care", 4.0, pods[1:5], hour=1)
    _check(r["quorum_met"], "quorum met")
    _check("P4" not in r["accepted"], "non-neighbor rejected")

    # --- Municipal fragility scaling ---
    robust = MunicipalZone("R", 100.0, 100, 85, ecological_fragility=0.5)
    fragile = MunicipalZone("F", 100.0, 100, 85, ecological_fragility=2.0)
    _check(fragile.current_transit_toll() > robust.current_transit_toll(),
           "fragile costs more")

    # --- UN escalation ladder ---
    un = UNRecognition(signatories=["KR"])
    un.ladder.record_violation("P", "UNP-02", 1, detail="forced_labor")
    stage, _ = un.ladder.current_stage("P")
    _check(stage >= 3, "forced labor -> stage >= 3")
    _check(not un.ladder.can_cross_border("P"), "cannot cross border")
    _check(not un.ladder.can_register_pods("P"),
           "decertified cannot register pods")

    # --- Municipal trip blocked by UN gate ---
    pod2 = HouseholdPod("P2", transit_credit_balance=1000.0)
    zone = MunicipalZone("Z", 30.0, 100, 0, 1.0)
    r = MunicipalLedger.process_trip(pod2, zone, "P", un)
    _check(r["status"] == "DENIED", "trip denied by UN gate")

    # --- Auditor: low bond rejected ---
    areg = AuditorRegistry()
    try:
        areg.register("A1", "LowBond", bond=100.0)
        raise AssertionError("low bond should raise")
    except ValueError:
        pass

    # --- Auditor: slash and removal ---
    a1 = areg.register("A1", "HighBond", bond=1000.0)
    _check(areg.is_valid("A1"), "valid auditor")
    for i in range(5):
        ESGCertificate.issue("SP", "POD", "carbon", 10.0, i, a1, areg)
    areg.slash("A1", 300.0, "disputed")
    areg.slash("A1", 300.0, "disputed")
    rate = a1.slash_count / max(1, len(a1.issued_certificates))
    _check(rate > areg.MAX_SLASH_RATE, f"slash rate {rate} > cap")

    # --- Sponsor pool attribution ---
    pool = SponsorPool()
    p3 = HouseholdPod("P3")
    pool.contribute(p3, "A", 100.0, day=1, reason="esg")
    pool.contribute(p3, "B", 300.0, day=1, reason="data")
    attr = pool.attribution(p3, 1000.0)
    _check(abs(attr["A"] - 250.0) < 1e-6, "A gets 25%")
    _check(abs(attr["B"] - 750.0) < 1e-6, "B gets 75%")

    # --- Federation: treaty required for cross-polity ---
    a = Polity("A"); b = Polity("B")
    pa = HouseholdPod("PA", transit_credit_balance=100)
    pb = HouseholdPod("PB")
    a.admit(pa); b.admit(pb)
    fed = Federation()
    r = fed.cross_polity_transfer(pa, pb, 50.0, a, b)
    _check(r["status"] == "FAILED", "no treaty = rejected")
    fed.sign("A", "B", cross_polity_fee_pct=0.01)
    r = fed.cross_polity_transfer(pa, pb, 50.0, a, b)
    _check(r["status"] == "SETTLED", "with treaty = settled")
    expected = 50 - 50 * 0.005 - 50 * 0.01
    _check(abs(pb.transit_credit_balance - expected) < 1e-6,
           f"delivered {pb.transit_credit_balance} vs {expected}")

    # --- Federation: personhood recognition ---
    reg_a = PersonhoodRegistry(bootstrap_sponsors={"boot"})
    reg_a.register("person_x", "bio_x")
    reg_b = PersonhoodRegistry(bootstrap_sponsors={"boot"})
    r = fed.recognize_personhood(reg_a, reg_b, "person_x")
    _check(r["status"] == "RECOGNIZED", "person recognized")
    _check("person_x" in reg_b.persons, "person in B registry")

    # --- Watcher read access ---
    w = Watcher("W-1")
    log2 = TransparencyLog()
    log2.add_watcher(w)
    log2.append("evt", {"x": 1}, day=1)
    _check(len(w.read_log) == 1, "watcher sees event")
    _check("evt" in w.read_log, "correct event type")

    print("All checks passed.")


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--municipal-demo", action="store_true")
    args = p.parse_args()

    if args.test:
        try:
            run_unit_tests()
            print("TEST SUITE: PASS")
            sys.exit(0)
        except AssertionError as e:
            print(f"TEST SUITE: FAIL — {e}")
            sys.exit(1)

    if args.municipal_demo:
        municipal_demo()
        return

    demo()


if __name__ == "__main__":
    main()
