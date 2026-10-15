# Headroom re-run on the rules core (ticket 06)

Regression check of the 6 Oct pilot table under the official turn order (Suggest, then Accuse or pass; Reveal is a
decision; wrong Accusations inform every Seat's Deduction engine). One test Bot vs n-1 base Bots, seat rotated, 6000
games per cell, win rate % +- 95% normal CI. Variants are suggest/reveal/accuse. Reproduce (about 80 s each, 12 cores):

    python -m cluedo.headroom --games 6000 --workers 12 --out results/headroom_vs_cand_certain.md
    python -m cluedo.headroom --games 6000 --workers 12 --base cand/random/gamble3 --seed 5000 --out results/headroom_vs_cand_gamble3.md

Pilot (6 Oct, 4 players, basic table): basic 24.3, gamble +5-6pp (30.3), focused +2pp, random suggestions -5pp,
smart reveal ~0; against a stronger table gamble3 26.3 vs 25.2.

## Base field `cand/random/certain` (the pilot's basic table)

| variant | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|
| random/random/certain | 6.2 +- 0.6 | 5.1 +- 0.6 | 5.1 +- 0.6 | 4.3 +- 0.5 |
| cand/random/certain | 33.3 +- 1.2 | 25.6 +- 1.1 | 20.3 +- 1.0 | 16.7 +- 0.9 |
| focused/random/certain | 34.5 +- 1.2 | 25.8 +- 1.1 | 20.5 +- 1.0 | 17.0 +- 1.0 |
| bluff/random/certain | 32.8 +- 1.2 | 23.8 +- 1.1 | 18.7 +- 1.0 | 14.6 +- 0.9 |
| cand/smart/certain | 33.1 +- 1.2 | 25.5 +- 1.1 | 19.6 +- 1.0 | 16.8 +- 0.9 |
| cand/random/gamble2 | 34.7 +- 1.2 | 27.2 +- 1.1 | 23.4 +- 1.1 | 18.6 +- 1.0 |
| cand/random/gamble3 | 34.0 +- 1.2 | 28.0 +- 1.1 | 24.6 +- 1.1 | 20.6 +- 1.0 |
| focused/smart/gamble2 | 36.4 +- 1.2 | 28.5 +- 1.1 | 22.6 +- 1.1 | 20.0 +- 1.0 |
| chance | 33.3 | 25.0 | 20.0 | 16.7 |

## Base field `cand/random/gamble3` (stronger table)

| variant | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|
| random/random/certain | 11.2 +- 0.8 | 5.7 +- 0.6 | 4.7 +- 0.5 | 3.8 +- 0.5 |
| cand/random/certain | 33.2 +- 1.2 | 22.9 +- 1.1 | 17.5 +- 1.0 | 13.2 +- 0.9 |
| focused/random/certain | 34.5 +- 1.2 | 23.1 +- 1.1 | 17.0 +- 0.9 | 14.4 +- 0.9 |
| bluff/random/certain | 32.6 +- 1.2 | 22.0 +- 1.0 | 15.4 +- 0.9 | 12.8 +- 0.8 |
| cand/smart/certain | 33.2 +- 1.2 | 22.7 +- 1.1 | 16.7 +- 0.9 | 13.6 +- 0.9 |
| cand/random/gamble2 | 33.3 +- 1.2 | 24.3 +- 1.1 | 19.8 +- 1.0 | 15.8 +- 0.9 |
| cand/random/gamble3 | 33.8 +- 1.2 | 25.4 +- 1.1 | 19.7 +- 1.0 | 16.5 +- 0.9 |
| focused/smart/gamble2 | 34.8 +- 1.2 | 25.2 +- 1.1 | 19.4 +- 1.0 | 15.8 +- 0.9 |
| chance | 33.3 | 25.0 | 20.0 | 16.7 |

## Reading (recorded, not asserted)

- The pilot's qualitative conclusions hold. Accusation gambling is the largest gain over `cand/random/certain`:
  gamble3 +2.4pp at 4 players, +4.3 at 5, +3.9 at 6 (the 4-player gain is smaller than the pilot's +5-6pp). Smart
  reveal is ~0 (within CI). Random suggestions are far worse (-20pp at 4 players vs the pilot's -5pp). Focused
  suggestions are within CI of `cand` at 4-6 players (the pilot saw +2pp).
- The gambling gain shrinks against the stronger table (the pilot saw the same).
- `bluff` is negative against Bots that ignore Suggestions (expected: it only wastes information); it can only pay
  off against the suggestion-reading `reader` (ticket 07).
- Headroom grows with the number of players; 3 players is near chance for every variant except `random`.
- The smaller gambling gain at 4 players vs the pilot is not investigated; candidate causes are the new Accuse-after-Suggest
  order (every Bot accuses with the freshest knowledge) and wrong-Accusation clauses helping every Seat. Ticket 04's
  exact Candidate count will change these numbers (gamble thresholds will fire earlier), so re-run then.
