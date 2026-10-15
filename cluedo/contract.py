"""Frozen observation/action contract (torch-free; numpy only).

Single source of truth for index layouts. Env, networks, Bots and visualisations
must encode/decode through this module only.

Vocabulary follows CONTEXT.md. Cards 0-5 suspects, 6-11 weapons, 12-20 rooms.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------- cards
N_SUSPECTS, N_WEAPONS, N_ROOMS = 6, 6, 9
N_CARDS = N_SUSPECTS + N_WEAPONS + N_ROOMS  # 21
CATS = (range(0, 6), range(6, 12), range(12, 21))
CAT_SIZES = (N_SUSPECTS, N_WEAPONS, N_ROOMS)
CAT_OFFSETS = (0, N_SUSPECTS, N_SUSPECTS + N_WEAPONS)
MAX_SEATS = 6
MIN_PLAYERS, MAX_PLAYERS = 3, 6
N_OWNERS = MAX_SEATS + 1  # six Seats + Envelope (owner index 6)
ENVELOPE = MAX_SEATS
TURN_CAP = 400

# ---------------------------------------------------------------- actions
N_COMBOS = N_SUSPECTS * N_WEAPONS * N_ROOMS  # 324
SUGGEST_START = 0
ACCUSE_START = N_COMBOS  # 324
PASS_ACTION = 2 * N_COMBOS  # 648
REVEAL_START = PASS_ACTION + 1  # 649
N_ACTIONS = REVEAL_START + N_CARDS  # 670

# Phases (one-hot in the observation, in this order)
SUGGEST, ACCUSE, REVEAL = 0, 1, 2
PHASES = ("suggest", "accuse", "reveal")
N_PHASES = 3

# Knowledge-grid cell states (one-hot, last axis)
HAS, NOT, MAYBE, UNKNOWN = 0, 1, 2, 3
N_KNOW = 4


class Action(NamedTuple):
    kind: str  # "suggest" | "accuse" | "pass" | "reveal"
    cards: tuple  # (suspect, weapon, room) card ids, or (card,) for reveal, or ()


def combo_index(suspect: int, weapon: int, room: int) -> int:
    """Cards (global ids 0-5, 6-11, 12-20) -> index 0..323."""
    return (suspect * N_WEAPONS + (weapon - CAT_OFFSETS[1])) * N_ROOMS + (room - CAT_OFFSETS[2])


def combo_cards(idx: int) -> tuple[int, int, int]:
    s, rem = divmod(idx, N_WEAPONS * N_ROOMS)
    w, r = divmod(rem, N_ROOMS)
    return s, w + CAT_OFFSETS[1], r + CAT_OFFSETS[2]


def encode_suggest(suspect, weapon, room) -> int:
    return SUGGEST_START + combo_index(suspect, weapon, room)


def encode_accuse(suspect, weapon, room) -> int:
    return ACCUSE_START + combo_index(suspect, weapon, room)


def encode_reveal(card: int) -> int:
    return REVEAL_START + card


def decode_action(a: int) -> Action:
    a = int(a)
    if not 0 <= a < N_ACTIONS:
        raise ValueError(f"action {a} out of range")
    if a < ACCUSE_START:
        return Action("suggest", combo_cards(a))
    if a < PASS_ACTION:
        return Action("accuse", combo_cards(a - ACCUSE_START))
    if a == PASS_ACTION:
        return Action("pass", ())
    return Action("reveal", (a - REVEAL_START,))


def phase_of_action(a: int) -> int:
    a = int(a)
    if a < ACCUSE_START:
        return SUGGEST
    if a < REVEAL_START:
        return ACCUSE
    return REVEAL


def suggest_mask() -> np.ndarray:
    """Any Suggestion is legal at any time (Bluffs, Anchors)."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[SUGGEST_START:ACCUSE_START] = True
    return m


def accuse_mask(candidates) -> np.ndarray:
    """candidates: iterable of (suspect, weapon, room) card ids. Pass always legal."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    for s, w, r in candidates:
        m[encode_accuse(s, w, r)] = True
    m[PASS_ACTION] = True
    return m


def reveal_mask(cards) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    for c in cards:
        m[encode_reveal(c)] = True
    return m


def mask_phase(mask: np.ndarray) -> int:
    """Phase a mask belongs to; raises unless exactly one phase is active."""
    mask = np.asarray(mask)
    parts = (
        mask[SUGGEST_START:ACCUSE_START].any(),
        mask[ACCUSE_START:REVEAL_START].any(),
        mask[REVEAL_START:].any(),
    )
    # pass (648) belongs to the accuse phase
    if sum(parts) != 1:
        raise ValueError(f"mask must have exactly one active phase, got {parts}")
    return parts.index(True)


# ---------------------------------------------------------------- observation
class Slice(NamedTuple):
    start: int
    shape: tuple

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def stop(self) -> int:
        return self.start + self.size

    @property
    def sl(self) -> slice:
        return slice(self.start, self.stop)


# Seat axes are Learner-relative: index 0 = Learner, i = i-th Seat to the left; padded to 6.
_SPEC = [
    # name, shape, public_history?
    ("knowledge", (N_OWNERS, N_CARDS, N_KNOW), False),  # one-hot has/not/maybe/unknown; owner 6 = Envelope
    ("open_clauses", (MAX_SEATS,), False),  # open "has at least one of" clauses per Seat, /10 scale
    ("cand_per_cat", (3,), False),  # Envelope Candidates per category / category size
    ("cand_total", (1,), False),  # Candidate count / 324
    ("n_players", (MAX_PLAYERS - MIN_PLAYERS + 1,), False),  # one-hot 3..6
    ("hand_sizes", (MAX_SEATS,), False),  # /6, 0 for padded Seats
    ("eliminated", (MAX_SEATS,), False),  # 1 = eliminated (padded Seats 0)
    ("turn", (1,), False),  # turn number / TURN_CAP
    ("phase", (N_PHASES,), False),  # one-hot suggest/accuse/reveal
    ("name_freq", (MAX_SEATS, N_CARDS), True),  # times Seat named card / Turns so far, per Seat x card
    ("shown_to", (MAX_SEATS, N_CARDS), True),  # 1 if Learner showed this card to that Seat
    ("last_cards", (N_CARDS,), True),  # multi-hot last Suggestion
    ("last_asker", (MAX_SEATS,), True),  # one-hot
    ("last_answerer", (MAX_SEATS + 1,), True),  # one-hot, index 6 = nobody answered / none yet
]

SLICES: dict[str, Slice] = {}
PUBLIC_HISTORY: tuple[str, ...] = ()
_off = 0
_ph = []
for _name, _shape, _pub in _SPEC:
    SLICES[_name] = Slice(_off, _shape)
    _off += int(np.prod(_shape))
    if _pub:
        _ph.append(_name)
PUBLIC_HISTORY = tuple(_ph)
OBS_DIM = _off
del _off, _ph, _name, _shape, _pub


def empty_obs() -> np.ndarray:
    return np.zeros(OBS_DIM, dtype=np.float32)


def view(obs: np.ndarray, name: str) -> np.ndarray:
    """Shaped view (last axis is the flat obs axis; leading batch dims allowed)."""
    s = SLICES[name]
    out = obs[..., s.sl]
    return out.reshape(*obs.shape[:-1], *s.shape)


def set_slice(obs: np.ndarray, name: str, value) -> None:
    s = SLICES[name]
    obs[s.sl] = np.asarray(value, dtype=np.float32).reshape(-1)


def blind(obs: np.ndarray) -> np.ndarray:
    """Zero all Public-history slices (PPO-SP-blind). Returns a copy."""
    out = np.array(obs, copy=True)
    for name in PUBLIC_HISTORY:
        out[..., SLICES[name].sl] = 0.0
    return out


def encode_phase(obs: np.ndarray, phase: int) -> None:
    v = np.zeros(N_PHASES, dtype=np.float32)
    v[phase] = 1.0
    set_slice(obs, "phase", v)


def obs_phase(obs: np.ndarray) -> int:
    return int(np.argmax(view(obs, "phase")))


def encode_n_players(obs: np.ndarray, n: int) -> None:
    v = np.zeros(MAX_PLAYERS - MIN_PLAYERS + 1, dtype=np.float32)
    v[n - MIN_PLAYERS] = 1.0
    set_slice(obs, "n_players", v)


def relative_seat(seat: int, learner: int, n: int) -> int:
    """Absolute Seat -> Learner-relative index (0 = Learner, increasing to the left)."""
    return (seat - learner) % n


def absolute_seat(rel: int, learner: int, n: int) -> int:
    return (rel + learner) % n
