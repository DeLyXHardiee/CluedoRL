"""Behaviour metrics computed from a Game log (torch-free, so the evaluation harness can reuse them).

    m = behaviour_metrics(log, seat)      # Metrics for one Seat (the Learner) in one finished/partial game
    total = m1 + m2                       # counts add up across games; rates are properties
    recs = analyse(log, seat)             # per-action records (flags etc.) for the replay viewer

The log is replayed through the core `Game` with one `Deduction` engine per Seat, fed exactly what
that Seat observes (`events_for`), so every number is what the Seat could actually know.

Definitions (CONTEXT.md), per Suggestion made by `seat`, judged on the Seat's engine before it acts:
- Bluff: names one of its own cards in a category whose Envelope card it does not yet know.
- Anchor: names one of its own cards in a category whose Envelope card it already knows
  (one Candidate card left in the category). A Suggestion can be both (different categories).
- Probe: names a card it already knows another Seat holds.
- Information gain: log2(Candidates before the Suggestion / Candidates once it is answered).
Per Accusation: Candidates left (exact when few remain, see the engine) and whether it was correct.
Reveal re-show rate: among the Seat's Reveals where it had a choice (>= 2 matching cards), the share
re-showing a card the asker had already seen from it.
Leakage: cells of the Seat's own hand row newly resolved to "held" in any Opponent's engine, summed
over Opponents, attributed to the event that caused it: the Seat's Suggestion, the Seat answering
(Reveal, chosen or automatic), or any other event (e.g. a clause from someone else's Suggestion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

from .cards import CAT_OF
from .deduction import Deduction
from .rules import Game


@dataclass
class Metrics:
    suggestions: int = 0
    bluffs: int = 0
    anchors: int = 0
    probes: int = 0
    accusations: int = 0
    accusations_correct: int = 0
    accusation_candidates: list = field(default_factory=list)  # Candidates left at each Accusation
    info_gains: list = field(default_factory=list)             # bits per answered Suggestion
    reveal_choices: int = 0
    reveal_reshows: int = 0
    leakage_suggest: int = 0
    leakage_reveal: int = 0
    leakage_other: int = 0

    def __add__(self, other: "Metrics") -> "Metrics":
        out = Metrics()
        for f in fields(self):
            setattr(out, f.name, getattr(self, f.name) + getattr(other, f.name))
        return out

    @staticmethod
    def _rate(num, den):
        return num / den if den else 0.0

    bluff_rate = property(lambda s: s._rate(s.bluffs, s.suggestions))
    anchor_rate = property(lambda s: s._rate(s.anchors, s.suggestions))
    probe_rate = property(lambda s: s._rate(s.probes, s.suggestions))
    accusation_accuracy = property(lambda s: s._rate(s.accusations_correct, s.accusations))
    reshow_rate = property(lambda s: s._rate(s.reveal_reshows, s.reveal_choices))
    leakage = property(lambda s: s.leakage_suggest + s.leakage_reveal + s.leakage_other)
    mean_info_gain = property(lambda s: s._rate(sum(s.info_gains), len(s.info_gains)))
    mean_accusation_candidates = property(
        lambda s: s._rate(sum(s.accusation_candidates), len(s.accusation_candidates)))

    def summary(self) -> dict:
        """Flat scalars for logging / tables."""
        return {"bluff_rate": self.bluff_rate, "anchor_rate": self.anchor_rate,
                "probe_rate": self.probe_rate, "accusation_accuracy": self.accusation_accuracy,
                "mean_accusation_candidates": self.mean_accusation_candidates,
                "mean_info_gain": self.mean_info_gain, "reshow_rate": self.reshow_rate,
                "leakage": self.leakage}


def _flags(ded: Deduction, seat: int, hand, cards) -> list[str]:
    cats = ded.category_candidates()
    own = [c for c in cards if c in hand]
    flags = []
    if any(len(cats[CAT_OF[c]]) > 1 for c in own):
        flags.append("bluff")
    if any(len(cats[CAT_OF[c]]) == 1 for c in own):
        flags.append("anchor")
    if any(ded.status(o, c) == 1 for c in cards if c not in hand for o in range(ded.n) if o != seat):
        flags.append("probe")
    return flags


def _hand_cells(engines, seat, hand) -> int:
    return sum(1 for o, e in enumerate(engines) if o != seat for c in hand if e.status(seat, c) == 1)


def _walk(log: dict, seat: int):
    """Replay `log`; returns (Metrics for `seat`, per-action records of `seat`)."""
    g = Game([frozenset(h) for h in log["hands"]], log["envelope"], log["start_seat"],
             seed=log.get("seed"), max_turns=log.get("max_turns", 400))
    engines = [Deduction.for_seat(g, s) for s in range(g.n)]
    hand = g.hands[seat]
    m, records = Metrics(), []
    shown: dict = {}      # asker -> cards `seat` has shown them
    pending = None        # (record, Candidates before) of an unanswered Suggestion by `seat`
    for idx, a in enumerate(log["actions"]):
        action = tuple(a["action"])
        d = g.decision
        if d.seat == seat:
            ded = engines[seat]
            rec = {"index": idx, "kind": d.phase, "flags": []}
            if d.phase == "suggest":
                rec["flags"] = _flags(ded, seat, hand, action[1:])
                m.suggestions += 1
                m.bluffs += "bluff" in rec["flags"]
                m.anchors += "anchor" in rec["flags"]
                m.probes += "probe" in rec["flags"]
                pending = (rec, ded.candidate_count())
            elif d.phase == "accuse" and action[0] == "accuse":
                rec["candidates"] = ded.candidate_count()
                m.accusations += 1
                m.accusation_candidates.append(rec["candidates"])
            elif d.phase == "reveal" and len(g.legal_actions()) > 1:
                asker = g.pending_suggestion()[0]
                rec["reshow"] = action[1] in shown.get(asker, ())
                m.reveal_choices += 1
                m.reveal_reshows += rec["reshow"]
            records.append(rec)
        before = _hand_cells(engines, seat, hand)
        n_events = len(g.events)
        g.apply(action)
        for e, ded in enumerate(engines):
            ded.sync(g.events_for(e))
        for ev in g.events[n_events:]:
            delta = _hand_cells(engines, seat, hand) - before
            kind = "other"
            if ev["type"] == "suggestion":
                if ev["answerer"] == seat and ev["card"] is not None:
                    shown.setdefault(ev["seat"], set()).add(ev["card"])
                if ev["seat"] == seat and pending:
                    gain = math.log2(pending[1] / engines[seat].candidate_count())
                    m.info_gains.append(gain)
                    pending[0]["info_gain"] = gain
                    pending = None
                kind = "suggest" if ev["seat"] == seat else "reveal" if ev["answerer"] == seat else "other"
            elif ev["type"] == "accusation" and ev["seat"] == seat:
                m.accusations_correct += ev["correct"]
            setattr(m, f"leakage_{kind}", getattr(m, f"leakage_{kind}") + delta)
    return m, records


def behaviour_metrics(log: dict, seat: int) -> Metrics:
    return _walk(log, seat)[0]


def analyse(log: dict, seat: int) -> list[dict]:
    """One record per action `seat` took: index into log["actions"], kind, flags (bluff/anchor/probe
    on Suggestions), info_gain, candidates (Accusations), reshow (Reveal choices)."""
    return _walk(log, seat)[1]
