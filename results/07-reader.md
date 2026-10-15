# Ticket 07: reader Bot, bluff field vs focused field (recorded, not asserted)

`reader/smart/reader`: Suggestions name, per category, the Candidate the other Seats named most often (ties random);
gambling Accusations (at most 3 Candidates) pick the Candidate with the highest summed naming count. It reads only
`public_events()` (shown cards hidden) and never counts its own Suggestions.

Test Bot vs a field of 3-5 Bots that all play `bluff/smart/certain` or all `focused/smart/certain` (they differ only in
bluffing). 6000 games per cell, win rate % +- 95% CI, seed 9000 (same Deals in both fields). Raw tables:
`07-reader-vs-bluff-field.md`, `07-reader-vs-focused-field.md`. Reproduce:

    python -m cluedo.headroom --games 6000 --workers 12 --seed 9000 --base bluff/smart/certain \
        --variants reader/smart/reader reader/smart/certain cand/smart/gamble3

| test Bot | field | 3 players | 4 players | 5 players | 6 players |
|---|---|---|---|---|---|
| reader/smart/reader | focused | 64.1 | 49.0 | 37.5 | 31.1 |
| reader/smart/reader | bluff | 56.0 | 41.8 | 33.2 | 27.9 |
| reader/smart/certain | focused | 62.9 | 44.3 | 34.6 | 24.2 |
| reader/smart/certain | bluff | 54.1 | 38.0 | 28.0 | 21.5 |
| cand/smart/gamble3 (does not read; control) | focused | 32.9 | 26.7 | 24.9 | 21.1 |
| cand/smart/gamble3 | bluff | 34.5 | 29.1 | 25.3 | 23.0 |

- Reader loses about 8pp (3 players), 7pp (4), 4pp (5), 3pp (6) when the field bluffs: bluffing measurably fools it
  (CI about +-1.2pp). The non-reading control does not lose (it gains slightly, within noise to +2pp, because bluffers
  waste information).
- Reader remains far above chance even against a bluffing field, i.e. a field of bluffers still leaves naming frequency
  informative (bluffers bluff in only one category, 30% of the time).
- The reader's edge is mostly its Suggestion rule (see ablation in `headroom.md`).
