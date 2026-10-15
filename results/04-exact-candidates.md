# Ticket 04: exact Candidates at <= K

Measured 2026-10-15 with `scripts/bench_exact.py` (random Suggestions, a Seat accuses when its
engine has exactly one Candidate, else passes; all Seats keep an engine; single CPU process).

## Benchmark: full games per second (30 seeded games per cell)

| players | K=0 | K=4 | K=8 | K=12 | K=20 | K=40 |
|---|---|---|---|---|---|---|
| 3 | 402 | 356 | 306 | 263 | 216 | 140 |
| 4 | 274 | 250 | 223 | 199 | 159 | 111 |
| 5 | 241 | 229 | 205 | 183 | 148 | 109 |
| 6 | 168 | 167 | 151 | 138 | 118 | 90 |

K=0 is propagation only.

## Chosen K = 12 (`cluedo.core.deduction.DEFAULT_K`)

The gamble zone is 2-5 true Candidates, but the propagated count there is larger (see below, avg
extra up to 4.3 at 6 players), so K must exceed 5 to catch it. K=12 covers that with margin and costs
34% / 27% / 24% / 18% of games/s at 3/4/5/6 players versus propagation only; K=20+ costs notably
more for little extra coverage of the zone. K is a constructor argument (`Deduction(..., k=)`),
so it can be retuned against real training throughput.

## Looseness of propagation alone (150 seeded games per count, at Accuse decisions)

Measured only at points where the propagated count is <= 120 (exact check is expensive above that;
the unmeasured points are all early game, so numbers below understate the early looseness).

| players | points | impossible Candidates present | avg extra | unmeasured (>120) |
|---|---|---|---|---|
| 3 | 4423 | 47.5% | 2.13 | 1 |
| 4 | 4110 | 62.0% | 4.63 | 281 |
| 5 | 3713 | 64.4% | 7.72 | 780 |
| 6 | 3352 | 69.4% | 10.55 | 1188 |

In the 2-5 true-Candidate zone (exact count in 2..5):

| players | points | impossible Candidates present | avg extra |
|---|---|---|---|
| 3 | 1349 | 16.7% | 0.52 |
| 4 | 1088 | 26.6% | 1.37 |
| 5 | 1038 | 28.7% | 2.37 |
| 6 | 953 | 36.9% | 4.27 |

So propagation alone misstates the odds in roughly a sixth to a third of gamble-zone decisions;
the exact check removes this once the count is <= K. (Policy here differs from the pilot's, so the
numbers are not directly comparable with the ADR's pilot figures.)
