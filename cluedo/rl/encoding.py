"""Seat-relative observation/mask encoder on top of the rules core (numpy only).

``SeatEncoder(game, seat, blind=False)`` owns a Deduction engine for `seat` and encodes the
frozen contract (cluedo/contract.py) from that Seat's point of view. It is used by the env for the
Learner and can be reused by policy Opponents (Checkpoints) for their own Seat.
`blind=True` zeroes all Public-history slices (PPO-SP-blind).
"""
from __future__ import annotations

import numpy as np

from cluedo import contract as C
from cluedo.core.deduction import Deduction

CLAUSE_SCALE = 10.0


class SeatEncoder:
    def __init__(self, game, seat: int, blind: bool = False):
        self.game, self.seat, self.n, self.blind = game, seat, game.n, blind
        self.ded = Deduction.for_seat(game, seat)
        self._named = np.zeros((C.MAX_SEATS, C.N_CARDS), dtype=np.float32)
        self._seen = 0  # events consumed for Public history
        self._last = None  # last Suggestion event

    def rel(self, seat: int) -> int:
        return C.relative_seat(seat, self.seat, self.n)

    def _sync(self):
        g = self.game
        self.ded.sync(g.events_for(self.seat))
        for e in g.events[self._seen:]:
            if e["type"] == "suggestion":
                self._named[self.rel(e["seat"]), e["cards"]] += 1
                self._last = e
        self._seen = len(g.events)

    def observation(self) -> np.ndarray:
        self._sync()
        g, d, n = self.game, self.ded, self.n
        obs = C.empty_obs()
        clauses = d.open_clauses()
        maybe = set()
        clause_count = np.zeros(C.MAX_SEATS, dtype=np.float32)
        for owner, cards in clauses:
            clause_count[self.rel(owner)] += 1
            maybe.update((owner, c) for c in cards)
        grid = np.zeros((C.N_OWNERS, C.N_CARDS, C.N_KNOW), dtype=np.float32)
        for owner in range(n + 1):
            row = C.ENVELOPE if owner == n else self.rel(owner)
            for c in range(C.N_CARDS):
                v = d.grid[owner][c]
                if v == 1:
                    s = C.HAS
                elif v == 0:
                    s = C.NOT
                else:
                    s = C.MAYBE if (owner, c) in maybe else C.UNKNOWN
                grid[row, c, s] = 1.0
        C.set_slice(obs, "knowledge", grid)
        C.set_slice(obs, "open_clauses", np.minimum(clause_count / CLAUSE_SCALE, 1.0))
        C.set_slice(obs, "cand_per_cat",
                    [len(c) / s for c, s in zip(d.category_candidates(), C.CAT_SIZES)])
        C.set_slice(obs, "cand_total", [d.candidate_count() / C.N_COMBOS])
        C.encode_n_players(obs, n)
        hs = np.zeros(C.MAX_SEATS, dtype=np.float32)
        el = np.zeros(C.MAX_SEATS, dtype=np.float32)
        for s in range(n):
            hs[self.rel(s)] = len(g.hands[s]) / 6
            el[self.rel(s)] = float(g.eliminated[s])
        C.set_slice(obs, "hand_sizes", hs)
        C.set_slice(obs, "eliminated", el)
        C.set_slice(obs, "turn", [g.turns_taken / C.TURN_CAP])
        C.encode_phase(obs, self.phase)
        # Public history
        C.set_slice(obs, "name_freq", self._named / max(1, g.turns_taken))
        shown = np.zeros((C.MAX_SEATS, C.N_CARDS), dtype=np.float32)
        for c in g.hands[self.seat]:
            for s in d.shown_to(c):
                shown[self.rel(s), c] = 1.0
        C.set_slice(obs, "shown_to", shown)
        last_cards = np.zeros(C.N_CARDS, dtype=np.float32)
        asker = np.zeros(C.MAX_SEATS, dtype=np.float32)
        answerer = np.zeros(C.MAX_SEATS + 1, dtype=np.float32)
        if self._last is not None:
            last_cards[self._last["cards"]] = 1.0
            asker[self.rel(self._last["seat"])] = 1.0
            a = self._last["answerer"]
            answerer[C.MAX_SEATS if a is None else self.rel(a)] = 1.0
        C.set_slice(obs, "last_cards", last_cards)
        C.set_slice(obs, "last_asker", asker)
        C.set_slice(obs, "last_answerer", answerer)
        return C.blind(obs) if self.blind else obs

    @property
    def phase(self) -> int:
        d = self.game.decision
        return C.SUGGEST if d is None else C.PHASES.index(d.phase)  # terminal: placeholder

    def mask(self) -> np.ndarray:
        """Engine mask for the pending decision of this Seat (call after observation())."""
        p = self.phase
        if p == C.SUGGEST:
            return C.suggest_mask()
        if p == C.ACCUSE:
            return C.accuse_mask(self.ded.candidates())
        return C.reveal_mask(c for (_k, c) in self.game.legal_actions())
