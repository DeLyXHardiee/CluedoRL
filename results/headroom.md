# Headroom re-run on the rules core (ticket 06, re-run after ticket 04 and with `reader` from ticket 07)

Regression check of the 6 Oct pilot table under the official turn order (Suggest, then Accuse or pass; Reveal is a
decision; wrong Accusations inform every Seat's Deduction engine). One test Bot vs n-1 base Bots, seat rotated, 6000
games per cell, win rate % +- 95% normal CI. Variants are suggest/reveal/accuse. Reproduce (about 200 s and 100 s, 12 cores):

    python -m cluedo.headroom --games 6000 --workers 12 --out results/headroom_vs_cand_certain.md
    python -m cluedo.headroom --games 6000 --workers 12 --base cand/random/gamble3 --seed 5000 --out results/headroom_vs_cand_gamble3.md

The numbers below are the **current engine** (ticket 04's exact Candidates, K=12). The first version of this file was
produced with the propagation-only engine (before ticket 04 was merged); it is superseded. What changed (4 players,
base field `cand/random/certain`): cand 25.6 -> 25.7, gamble2 27.2 -> 27.4, gamble3 28.0 -> 28.8, focused/smart/gamble2
28.5 -> 28.8; at 5 players gamble3 24.6 -> 25.6, 6 players 20.6 -> 21.6. Everything else moved within CI. In short, exact
Candidates make gambling fire earlier and the gambling gain grew by about +0.7 to +1.2pp, but the table's shape is
unchanged. Variant `reader/smart/reader` is new.

Pilot (6 Oct, 4 players, basic table): basic 24.3, gamble +5-6pp (30.3), focused +2pp, random suggestions -5pp,
smart reveal ~0; against a stronger table gamble3 26.3 vs 25.2.

## Base field `cand/random/certain` (the pilot's basic table)

| variant | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|
| random/random/certain | 6.2 +- 0.6 | 5.3 +- 0.6 | 5.6 +- 0.6 | 4.6 +- 0.5 |
| cand/random/certain | 33.3 +- 1.2 | 25.7 +- 1.1 | 20.2 +- 1.0 | 16.5 +- 0.9 |
| focused/random/certain | 34.4 +- 1.2 | 25.8 +- 1.1 | 20.6 +- 1.0 | 16.9 +- 0.9 |
| bluff/random/certain | 32.7 +- 1.2 | 24.0 +- 1.1 | 18.9 +- 1.0 | 15.0 +- 0.9 |
| cand/smart/certain | 33.1 +- 1.2 | 25.4 +- 1.1 | 19.5 +- 1.0 | 17.2 +- 1.0 |
| cand/random/gamble2 | 34.8 +- 1.2 | 27.4 +- 1.1 | 23.8 +- 1.1 | 19.0 +- 1.0 |
| cand/random/gamble3 | 34.2 +- 1.2 | 28.8 +- 1.1 | 25.6 +- 1.1 | 21.6 +- 1.0 |
| focused/smart/gamble2 | 36.4 +- 1.2 | 28.8 +- 1.1 | 23.0 +- 1.1 | 20.6 +- 1.0 |
| reader/smart/reader | 70.9 +- 1.1 | 54.6 +- 1.3 | 41.4 +- 1.2 | 33.5 +- 1.2 |
| chance | 33.3 | 25.0 | 20.0 | 16.7 |

## Base field `cand/random/gamble3` (stronger table)

| variant | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|
| random/random/certain | 13.1 +- 0.9 | 6.5 +- 0.6 | 4.9 +- 0.5 | 4.3 +- 0.5 |
| cand/random/certain | 33.1 +- 1.2 | 22.3 +- 1.1 | 16.8 +- 0.9 | 12.7 +- 0.8 |
| focused/random/certain | 34.4 +- 1.2 | 22.6 +- 1.1 | 16.2 +- 0.9 | 13.7 +- 0.9 |
| bluff/random/certain | 32.8 +- 1.2 | 21.2 +- 1.0 | 14.9 +- 0.9 | 12.1 +- 0.8 |
| cand/smart/certain | 33.1 +- 1.2 | 22.1 +- 1.0 | 16.1 +- 0.9 | 13.0 +- 0.8 |
| cand/random/gamble2 | 33.4 +- 1.2 | 24.1 +- 1.1 | 19.4 +- 1.0 | 15.8 +- 0.9 |
| cand/random/gamble3 | 33.0 +- 1.2 | 25.5 +- 1.1 | 19.6 +- 1.0 | 16.1 +- 0.9 |
| focused/smart/gamble2 | 34.5 +- 1.2 | 24.9 +- 1.1 | 19.1 +- 1.0 | 15.2 +- 0.9 |
| reader/smart/reader | 68.0 +- 1.2 | 49.4 +- 1.3 | 36.1 +- 1.2 | 25.7 +- 1.1 |
| chance | 33.3 | 25.0 | 20.0 | 16.7 |

## Reading (recorded, not asserted)

- The pilot's qualitative conclusions hold. Accusation gambling is the largest gain over `cand/random/certain` among the
  pilot variants: gamble3 +3.1pp at 4 players, +5.4 at 5, +5.1 at 6 (the 4-player gain is still smaller than the
  pilot's +5-6pp). Smart reveal is ~0 (within CI). Random suggestions are far worse (-20pp at 4 players vs the pilot's
  -5pp). Focused suggestions are within CI of `cand` at 4-6 players (the pilot saw +2pp).
- The gambling gain shrinks against the stronger table (the pilot saw the same).
- `bluff` is negative against Bots that ignore Suggestions (it only wastes information); against the suggestion-reading
  `reader` it matters, see `results/07-reader.md`.
- `reader/smart/reader` is by far the strongest Bot (4 players: 54.6 vs 25.7 for `cand/random/certain`). Ablation
  (3000 games, 3/4 players, base `cand/random/certain`): `reader/smart/certain` 70.4/51.5, `cand/smart/reader` 42.1/33.2,
  `reader/smart/gamble3` 70.3/51.4. So the gain is almost entirely the reader *Suggestion* rule: cards that stay Candidates
  for the other Seats get named again and again, so naming frequency is a real, public signal of Envelope membership.
- Headroom grows with the number of players; 3 players is near chance for every variant except `random` and `reader`.
- The smaller gambling gain at 4 players vs the pilot is not investigated.
