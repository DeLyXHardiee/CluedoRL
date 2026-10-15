# Ticket 09: short PPO-FB run

Command: `python -m cluedo.rl.train --condition ppo-fb --seed 0 --out runs/fb-short-s0 --total-frames 3000000 --eval-every 250000 --checkpoint-every 500000 --eval-games 400`
(default `TrainConfig`: 256x2 tanh MLPs, gamma 1, GAE 0.95, clip 0.2, lr 3e-4, entropy 0.01, 12 env workers, pass-logit prior 9.0).
Throughput about 2200 Learner steps/s on 12 cores while nothing else ran (the logged steps/s curve
in this run is distorted by a wall-clock stall mid-run; the steps/s of an undisturbed run is ~2.2k).

Evaluation: 4 players, 400 games per field, policy Seat rotated, n-1 copies of the field Bot. Chance = 25%.
Reference for the gate "beats the random-suggest Bot": a `random/smart/certain` Bot placed in the
`cand/smart/certain` field wins **3.0%** (400 games); PPO-FB in that same field:

| Learner steps | vs random/smart/certain | vs cand/smart/certain | vs focused/smart/gamble2 |
|---|---|---|---|
| 12k | 0.0 | 0.0 | 0.0 |
| 258k | 5.0 | 4.0 | 4.7 |
| 1.01M | 6.8 | 8.0 | 5.2 |
| 1.51M | 11.2 | 8.7 | 6.0 |
| 2.00M | 14.0 | 9.3 | 9.0 |
| 2.51M | 14.0 | 10.2 | 8.5 |
| 3.01M | 16.5 | 9.7 | 9.7 |

(win rate %, +-3 pp 95% CI at these sizes.) Training win rate climbed 0.1% -> 11.6%; entropy 3.0 -> 1.2; KL ~0.015.

Result: win rate rises steadily and passes the random-suggest Bot's 3% in the cand/certain field from about
250k steps on; at 3M steps it is still rising and well below chance (25%) and below cand/certain parity,
so more budget is needed for gate 1.

Finding that shaped the config: with a uniform initial policy, the first Accuse phase of a game offers 324
Accusations plus pass, so the Learner accused and died on turn one and win rate stayed at 1-2% for 1.2M steps
(lr 3e-4 and 1e-3 alike). An initial pass-logit prior (`pass_bias=9.0`, P(pass) about 0.95 at game start) fixed this.
