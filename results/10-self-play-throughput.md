# Ticket 10: self-play throughput and the ticket-11 decision

Machine: Ryzen 5 9600X, 6 physical cores / 12 threads, 12 env workers, default `TrainConfig`
(256x2 MLPs, `frames_per_batch` 12288, 4 PPO epochs, minibatch 1024). Fixed player count per run (the
env normally samples 3-6 uniformly). "Learner steps/s" = steady-state full loop (collect + PPO update +
weight sync), first 2 batches and all evaluation excluded, mean of 6-8 batches. The box was shared with other
sessions' test runs (load average 5-11 on 12 threads), so these are conservative; one 6p self-play run hit
a contention outlier (0.7-0.9k) and is excluded (re-runs: per-batch collect times 4.1-5.0 s, stable).
Scripts: scratchpad `bench1.py` (1 process), `bench2.py` (full loop), `bench3.py` (PPO minibatch).

## Result (after the fixes below)

| players | PPO-FB steps/s | PPO-SP steps/s (and SP-blind: same cost) |
|---|---|---|
| 3 | 6.7k | 4.3k |
| 4 | 5.6k | 2.7k |
| 5 | 4.3k | 2.7k |
| 6 | 3.7k | 2.4-2.5k (2.5k at fpb 4096, 2.2k at fpb 24576) |

Real PPO-SP run (random 3-6 players, 400k steps): 2.5-2.65k steps/s end to end, pool of 5 files filled,
checkpoints and eval OK (`runs/sp-short`, gitignored).

## Why ticket 09 saw ~800 steps/s with idle workers (bottlenecks found)

1. **The PPO update ran single-threaded and stalled every worker.** After building the `MultiSyncCollector`
   the trainer process is left at 1 torch thread. The update (48 minibatches of 1024, 904-dim obs, 670 actions)
   took 2.5 s per batch, equal to the collection time, and the sync collector is idle while it runs. Fix:
   `TrainConfig.update_threads` (default cpus // 2 = 6, set after the collector is created): update 2.5 s -> 0.7 s,
   PPO-FB 6p 2.4k -> 3.9k steps/s. (With 12 threads on a loaded box it is far worse: 264 ms/minibatch.)
2. **Worker torch threads oversubscribed.** Opponent policies run torch inside every worker; 12 workers x 12
   intra-op threads thrashed. Fix: `make_env` sets `torch.set_num_threads(1)` in the worker. PPO-SP 6p collection
   0.7k -> 1.9k+ steps/s at the time of the fix.
3. **Policy forward through TensorDict was ~40% of an Opponent's `act`.** `PolicyPlayer.act` now runs the MLP and a
   masked softmax sample directly (equivalence test vs `actor.get_dist`). Single-process 6p self-play: 510 -> 710
   steps/s of CPU.
4. Remaining per-step cost is the Deduction engine + observation encoding for every Seat (about 45% of a self-play
   step at 6p: each of the 5 Opponent Seats keeps its own Deduction). `frames_per_batch` does not matter (4096 to 24576
   within noise). Eval (400 games x 3 fields, ~9 s per 50k steps) also idles the workers but is a few percent.
   Workers are not truly idle at steady state: 12 workers on 6 physical cores is CPU-bound (~1k steps/s/core at 6p FB).

## Decision for ticket 11

Gate rule: numpy Opponents are needed if 6-player self-play < ~2k Learner steps/s. Measured 2.4-2.5k (noisy, conservative
due to a shared box), so **numpy Opponents are not needed**: a 10M-step self-play run takes about 70 min at 6p and
about 1 h on the 3-6 mix. The torch forward is already minor after the direct-MLP fast path; a numpy forward would save
little because the Deduction/encoding cost dominates. If more speed is wanted later, the lever is a faster
`SeatEncoder`/Deduction (or sharing one Deduction across the Opponent Seats of a Game), not numpy policies.
Caveat: margin over 2k is only ~20%; re-measure on an idle machine before the big run.
