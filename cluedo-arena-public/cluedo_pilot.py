"""Minimal card-only Cluedo simulator for a strategy-headroom pilot.

Rules (card-only): 6 suspects, 6 weapons, 9 rooms. One card per category is the
hidden solution; the other 18 are dealt round-robin. On a turn a player may
accuse; otherwise they suggest any (suspect, weapon, room). Players to the left
are asked in order; the first who holds any of the three must show exactly one
to the suggester only. Everyone sees who showed, not what. A wrong accusation
eliminates the player, who keeps showing cards. First correct accusation wins.

Every agent does the same constraint-propagation deduction; only the strategy
(suggest / reveal / accuse) differs between variants.
"""
import random
import sys
from itertools import product

CATS = [list(range(0, 6)), list(range(6, 12)), list(range(12, 21))]
NCARDS = 21
CAT_OF = {c: k for k, cat in enumerate(CATS) for c in cat}


class Knowledge:
    """What one player knows. owner index n == the envelope."""

    def __init__(self, n, me, hand, hand_sizes):
        self.n = n
        self.me = me
        self.sizes = hand_sizes + [3]
        # m[p][c]: 1 has, 0 doesn't, None unknown
        self.m = [[None] * NCARDS for _ in range(n + 1)]
        self.clauses = []  # (player, frozenset of cards): has at least one
        for c in range(NCARDS):
            self.m[me][c] = 1 if c in hand else 0
            if c in hand:
                for p in range(n + 1):
                    if p != me:
                        self.m[p][c] = 0
        self.propagate()

    def set(self, p, c, v):
        cur = self.m[p][c]
        if cur is None:
            self.m[p][c] = v
            return True
        return False

    def propagate(self):
        n = self.n
        changed = True
        while changed:
            changed = False
            # each card has exactly one owner
            for c in range(NCARDS):
                col = [self.m[p][c] for p in range(n + 1)]
                if 1 in col:
                    for p in range(n + 1):
                        if col[p] is None:
                            changed |= self.set(p, c, 0)
                else:
                    open_ = [p for p in range(n + 1) if col[p] is None]
                    if len(open_) == 1:
                        changed |= self.set(open_[0], c, 1)
            # hand sizes
            for p in range(n):
                row = self.m[p]
                ones = sum(1 for v in row if v == 1)
                unk = [c for c in range(NCARDS) if row[c] is None]
                if unk and ones == self.sizes[p]:
                    for c in unk:
                        changed |= self.set(p, c, 0)
                elif unk and ones + len(unk) == self.sizes[p]:
                    for c in unk:
                        changed |= self.set(p, c, 1)
            # envelope: exactly one per category
            env = self.m[n]
            for cat in CATS:
                ones = [c for c in cat if env[c] == 1]
                unk = [c for c in cat if env[c] is None]
                if ones:
                    for c in unk:
                        changed |= self.set(n, c, 0)
                elif len(unk) == 1:
                    changed |= self.set(n, unk[0], 1)
            # clauses
            keep = []
            for p, cards in self.clauses:
                if any(self.m[p][c] == 1 for c in cards):
                    continue
                live = [c for c in cards if self.m[p][c] is None]
                if len(live) == 1:
                    changed |= self.set(p, live[0], 1)
                    continue
                keep.append((p, cards))
            self.clauses = keep

    def env_candidates(self):
        return [[c for c in cat if self.m[self.n][c] != 0] for cat in CATS]

    def solved(self):
        cands = self.env_candidates()
        return all(len(x) == 1 for x in cands)


# ---------------------------------------------------------------- strategies

def suggest_random(kn, hand, rng):
    return tuple(rng.choice(cat) for cat in CATS)


def suggest_candidates(kn, hand, rng):
    return tuple(rng.choice(c) for c in kn.env_candidates())


def suggest_focused(kn, hand, rng):
    """Unsolved categories: an envelope candidate. Solved ones: one of my own
    cards (nobody can show it), so every answer is about the open categories."""
    out = []
    for cat, cands in zip(CATS, kn.env_candidates()):
        if len(cands) == 1:
            mine = [c for c in cat if c in hand]
            out.append(rng.choice(mine) if mine else cands[0])
        else:
            out.append(rng.choice(cands))
    return tuple(out)


def suggest_bluff(kn, hand, rng):
    """Like focused, but in unsolved categories sometimes name an own card to
    pin down the others and confuse observers (Hansen & Hansen style)."""
    out = []
    cands_all = kn.env_candidates()
    open_cats = [k for k, c in enumerate(cands_all) if len(c) > 1]
    bluff_cat = rng.choice(open_cats) if len(open_cats) > 1 and rng.random() < 0.3 else None
    for k, (cat, cands) in enumerate(zip(CATS, cands_all)):
        mine = [c for c in cat if c in hand]
        if len(cands) == 1 or (k == bluff_cat and mine):
            out.append(rng.choice(mine) if mine else cands[0])
        else:
            out.append(rng.choice(cands))
    return tuple(out)


def reveal_random(me, matching, asker, shown_log, rng):
    return rng.choice(matching)


def reveal_smart(me, matching, asker, shown_log, rng):
    """Re-show a card the asker already saw; else the card most players saw."""
    seen = shown_log.setdefault(me, {})
    for c in matching:
        if asker in seen.get(c, set()):
            return c
    return max(matching, key=lambda c: (len(seen.get(c, ())), rng.random()))


def accuse_certain(kn, rng):
    if kn.solved():
        return tuple(c[0] for c in kn.env_candidates())
    return None


def make_accuse_gamble(max_combos):
    def accuse(kn, rng):
        cands = kn.env_candidates()
        combos = 1
        for c in cands:
            combos *= len(c)
        if combos <= max_combos:
            return tuple(rng.choice(c) for c in cands)
        return None
    return accuse


SUGGEST = {"random": suggest_random, "cand": suggest_candidates,
           "focused": suggest_focused, "bluff": suggest_bluff}
REVEAL = {"random": reveal_random, "smart": reveal_smart}
ACCUSE = {"certain": accuse_certain, "gamble2": make_accuse_gamble(2),
          "gamble3": make_accuse_gamble(3)}


class Agent:
    def __init__(self, s, r, a):
        self.name = f"{s}/{r}/{a}"
        self.suggest, self.reveal, self.accuse = SUGGEST[s], REVEAL[r], ACCUSE[a]


# ---------------------------------------------------------------- game

def play(agents, rng, max_turns=400):
    n = len(agents)
    solution = tuple(rng.choice(cat) for cat in CATS)
    rest = [c for c in range(NCARDS) if c not in solution]
    rng.shuffle(rest)
    hands = [set(rest[i::n]) for i in range(n)]
    sizes = [len(h) for h in hands]
    kns = [Knowledge(n, i, hands[i], sizes) for i in range(n)]
    alive = [True] * n
    shown_log = {}
    start = rng.randrange(n)
    for t in range(max_turns):
        q = (start + t) % n
        if not alive[q]:
            if not any(alive):
                return None, t
            continue
        ag = agents[q]
        acc = ag.accuse(kns[q], rng)
        if acc is not None:
            if acc == solution:
                return q, t
            alive[q] = False
            if not any(alive):
                return None, t
            continue
        sug = ag.suggest(kns[q], hands[q], rng)
        sset = set(sug)
        responder, card = None, None
        for d in range(1, n):
            p = (q + d) % n
            matching = [c for c in sug if c in hands[p]]
            if matching:
                responder = p
                card = agents[p].reveal(p, matching, q, shown_log, rng)
                shown_log.setdefault(p, {}).setdefault(card, set()).add(q)
                break
            for k in kns:
                for c in sug:
                    k.set(p, c, 0)
        for i, k in enumerate(kns):
            if responder is None:
                pass
            elif i == q:
                k.set(responder, card, 1)
            elif i != responder:
                k.clauses.append((responder, frozenset(sset)))
            k.propagate()
    return None, max_turns


def evaluate(test, base, n_players, games, seed):
    """One `test` agent vs (n-1) `base` agents; seat rotated each game."""
    rng = random.Random(seed)
    wins = turns = 0
    for g in range(games):
        seat = g % n_players
        agents = [base] * n_players
        agents[seat] = test
        w, t = play(agents, rng)
        wins += (w == seat)
        turns += t
    return wins / games, turns / games


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_players = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    base = Agent(*(sys.argv[3].split("/") if len(sys.argv) > 3 else ("cand", "random", "certain")))
    variants = [
        ("random", "random", "certain"),
        ("cand", "random", "certain"),
        ("focused", "random", "certain"),
        ("bluff", "random", "certain"),
        ("cand", "smart", "certain"),
        ("cand", "random", "gamble2"),
        ("cand", "random", "gamble3"),
        ("focused", "smart", "gamble2"),
    ]
    print(f"players={n_players} games={games} chance={1/n_players:.3f} baseline={base.name}")
    for i, v in enumerate(variants):
        wr, t = evaluate(Agent(*v), base, n_players, games, seed=1000 + i)
        se = (wr * (1 - wr) / games) ** 0.5
        print(f"{'/'.join(v):28s} win={wr:.3f} ±{1.96*se:.3f}  turns={t:.1f}", flush=True)