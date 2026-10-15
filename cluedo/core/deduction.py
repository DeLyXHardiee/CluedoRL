"""The shared Deduction engine: one instance per Seat, fed that Seat's events.

Constraint propagation over a knowledge grid (ADR 0002). Sound but not complete: it never
rules out a true fact, but may keep impossible Candidates early in a game (ticket 04 makes
the count exact once few Candidates remain, on top of the state exposed here).

Owners are Seats 0..n-1 plus owner ``n`` = the Envelope. Cards are ints (see cards.py).
"""
from __future__ import annotations

from itertools import product

from .cards import CATS, NCARDS


class Contradiction(Exception):
    """The observations cannot all be true (a bug in the engine, the feed or the rules)."""


class Deduction:
    def __init__(self, n_players, me, hand, hand_sizes):
        self.n = n_players
        self.me = me
        self.sizes = list(hand_sizes) + [3]
        # grid[p][c]: 1 holds, 0 does not hold, None unknown.
        self.grid = [[None] * NCARDS for _ in range(n_players + 1)]
        self._clauses: set = set()       # (seat, frozenset(cards)): seat holds at least one
        self._env_clauses: set = set()   # (s, w, r): NOT all three are in the Envelope (wrong Accusations)
        self._shown: dict = {}           # my card -> set of Seats I showed it to
        self._seen = 0                   # events consumed by sync()
        hand = set(hand)
        for c in range(NCARDS):
            for p in range(n_players + 1):
                if p == me:
                    self._set(p, c, 1 if c in hand else 0)
                elif c in hand:
                    self._set(p, c, 0)
        self._propagate()

    @classmethod
    def for_seat(cls, game, seat):
        """Engine for `seat` of a Game, with nothing observed yet (call sync() to catch up)."""
        return cls(game.n, seat, game.hands[seat], game.hand_sizes)

    # ------------------------------------------------------------------ feeding

    def sync(self, events):
        """Consume the not-yet-seen tail of a Seat's event list (``game.events_for(seat)``)."""
        for e in events[self._seen:]:
            self.observe(e)
        self._seen = len(events)

    def observe(self, event):
        t = event["type"]
        if t == "suggestion":
            cards = tuple(event["cards"])
            for p in event["passes"]:
                for c in cards:
                    self._set(p, c, 0)
            a, card = event["answerer"], event["card"]
            if a is not None:
                if card is not None:
                    self._set(a, card, 1)
                    if a == self.me:
                        self._shown.setdefault(card, set()).add(event["seat"])
                else:
                    self._clauses.add((a, frozenset(cards)))
        elif t == "accusation":
            cards = tuple(event["cards"])
            if event["correct"]:
                for c in cards:
                    self._set(self.n, c, 1)
            else:
                self._env_clauses.add(cards)
        self._propagate()

    # ------------------------------------------------------------------ queries

    def status(self, owner, card):
        """1 = owner holds card, 0 = does not, None = unknown. owner == n is the Envelope."""
        return self.grid[owner][card]

    def category_candidates(self):
        """Per category (suspect, weapon, room): cards not ruled out for the Envelope."""
        env = self.grid[self.n]
        return [[c for c in cat if env[c] != 0] for cat in CATS]

    def candidates(self):
        """All Candidates (s, w, r), in itertools.product order."""
        return [t for t in product(*self.category_candidates()) if t not in self._env_clauses]

    def candidate_count(self):
        return len(self.candidates())

    def is_candidate(self, triple):
        triple = tuple(triple)
        return triple not in self._env_clauses and all(
            self.grid[self.n][c] != 0 for c in triple)

    def open_clauses(self):
        """Unresolved pass/answer clauses: (seat, frozenset(cards)) meaning seat holds >= 1 of them."""
        return sorted(self._clauses, key=lambda x: (x[0], sorted(x[1])))

    def open_clause_count(self):
        return len(self._clauses)

    def envelope_clauses(self):
        """Live 'not all three of these' Envelope clauses (from wrong Accusations)."""
        return sorted(self._env_clauses)

    def shown_to(self, card):
        """Seats this engine's own Seat has shown `card` to."""
        return set(self._shown.get(card, ()))

    # ------------------------------------------------------------------ internals

    def _set(self, p, c, v):
        cur = self.grid[p][c]
        if cur is None:
            self.grid[p][c] = v
            return True
        if cur != v:
            raise Contradiction(f"owner {p} card {c}: known {cur}, now {v}")
        return False

    def _propagate(self):
        n, grid = self.n, self.grid
        changed = True
        while changed:
            changed = False
            for c in range(NCARDS):  # exactly one owner per card
                col = [grid[p][c] for p in range(n + 1)]
                if col.count(1) > 1:
                    raise Contradiction(f"card {c} has two owners")
                if 1 in col:
                    for p in range(n + 1):
                        if col[p] is None:
                            changed |= self._set(p, c, 0)
                else:
                    open_ = [p for p in range(n + 1) if col[p] is None]
                    if not open_:
                        raise Contradiction(f"card {c} has no possible owner")
                    if len(open_) == 1:
                        changed |= self._set(open_[0], c, 1)
            for p in range(n):  # hand sizes
                row = grid[p]
                ones = row.count(1)
                unk = [c for c in range(NCARDS) if row[c] is None]
                if ones > self.sizes[p] or ones + len(unk) < self.sizes[p]:
                    raise Contradiction(f"seat {p} hand size")
                if unk and ones == self.sizes[p]:
                    for c in unk:
                        changed |= self._set(p, c, 0)
                elif unk and ones + len(unk) == self.sizes[p]:
                    for c in unk:
                        changed |= self._set(p, c, 1)
            env = grid[n]  # exactly one Envelope card per category
            for cat in CATS:
                ones = [c for c in cat if env[c] == 1]
                unk = [c for c in cat if env[c] is None]
                if ones:
                    for c in unk:
                        changed |= self._set(n, c, 0)
                elif len(unk) == 1:
                    changed |= self._set(n, unk[0], 1)
                elif not unk:
                    raise Contradiction("no Envelope card left in a category")
            for cl in list(self._clauses):  # at-least-one clauses
                p, cards = cl
                if any(grid[p][c] == 1 for c in cards):
                    self._clauses.discard(cl)
                    continue
                live = [c for c in cards if grid[p][c] is None]
                if not live:
                    raise Contradiction(f"seat {p} must hold one of {sorted(cards)}")
                if len(live) == 1:
                    self._clauses.discard(cl)
                    changed |= self._set(p, live[0], 1)
            for tri in list(self._env_clauses):  # not-all-three Envelope clauses
                if any(env[c] == 0 for c in tri):
                    self._env_clauses.discard(tri)
                    continue
                live = [c for c in tri if env[c] is None]
                if not live:
                    raise Contradiction(f"Envelope is {tri}, which was accused wrongly")
                if len(live) == 1:
                    self._env_clauses.discard(tri)
                    changed |= self._set(n, live[0], 0)
