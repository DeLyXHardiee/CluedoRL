# Reading guide: figures and TensorBoard curves

Regenerate everything with one command (reads the CSVs written by `python -m cluedo.rl.eval_grid`):

    python -m cluedo.figures --results results/eval --out figures/out

Output is one vector PDF per figure, sized for the paper (single column 3.4 in, double column 7.0 in; colour-blind safe palette, distinct markers so greyscale prints work). Conditions: blue circle PPO-SP, orange square PPO-SP-blind, green triangle PPO-FB, grey star = Bot.

General rules for every figure: error bars are 95% intervals; seeds are separate rows and never pooled in the win-rate figures (one marker per seed, a Condition's markers should agree); the behaviour figures pool the first `--metric-games` games of every cell and seed, so they are only as large as that sample (check the `n=` labels). Overlapping intervals mean "not distinguished at this sample size", not "equal". The grid has many cells, so a single separated pair among dozens is weak evidence; prefer the Holm-adjusted `p_holm` in `comparisons.csv` for any claim.

Truncation: a game that reaches the Turn cap has no winner and counts as a loss for everyone. `cells.csv` and `games.csv` carry `truncated` (and `truncation_rate` per cell), `metrics.csv` the `truncation_rate` of the measured games. The win-rate panels print an orange warning sign with the share of truncated games whenever it is nonzero. Any nonzero rate means the win rates in that panel understate every Entrant (a stalling policy shows up here, not as a draw); report it, and do not rank Conditions whose truncation rates differ much.

Holm correction: `p_holm` in `comparisons.csv` is adjusted over one family = all comparisons within one seed (every table, field, player count and pair of Entrants of that seed); seeds are separate families. The test is McNemar's exact test on the discordant games, those where exactly one of the two Entrants won; in a mixed table the same game is read for both Entrants. `bluff_anchor.csv` holds the numbers behind the `bluff_anchor` figure (rates per Condition and the PPO-SP minus PPO-SP-blind difference with its CI).

## `winrate` (double column)

Win rate with Wilson 95% CI. One panel per player count, x = Opponent (the Bot the Learner faces; the last five are held-out), one marker group per Condition, dotted line = chance (1/n).

- Good: Condition above the chance line with an interval clear of it, in the same direction for all seeds, and the ordering of Conditions stable across player counts. Held-out Opponents (bluff, reader, gamble3, random/certain) close to the training-Opponent results = generalisation.
- Bad: at or below chance everywhere (policy never learned to accuse correctly; compare with `eval/baseline/random_suggest`); large drop only on held-out Opponents (overfit to the training Bots).
- Warning: seeds of one Condition disagree (the result is noise or training-instability, do not report a mean); a win rate that rises with player count without a reason (check the chance line, not the raw height); intervals so wide that Conditions cannot be ranked (needs more games); y-axis is clipped to the data, so compare heights against the chance line, not against each other across panels.

## `winrate_mixed` (single column)

Mixed table: three Conditions plus one Bot at the same table (4 players), Seats permuted. Each Condition faces the others directly.

- Good: ordering agrees with the homogeneous panels; the Bot sits where its homogeneous strength suggests.
- Bad: a Condition below the dotted chance line (1/4) with an interval clear of it.
- Warning: a Condition that wins homogeneously but not here (it exploits the Bot field rather than playing well generally); results depend on who else is at the table, so do not read this as a stand-alone ranking.

## `bluff_anchor` (double column)

Left, middle: Bluff rate and Anchor rate per Suggestion (pooled, Wilson CI) per Condition. Right: difference PPO-SP minus PPO-SP-blind for both rates with a normal 95% CI. A Bluff names one of the Learner's own cards in a category whose Envelope card it does not yet know; an Anchor names an own card in a category it has already solved (see `cluedo/core/metrics.py`).

- Good (Bluffing is deceptive): PPO-SP Bluff rate clearly above PPO-SP-blind, difference interval excluding zero. Blind self-play cannot be rewarded for fooling a reader because its Opponents' reasoning is not modelled in its input, so a positive gap is attributable to the visibility of Opponent knowledge, i.e. to deception.
- Bad / null: equal Bluff rates in PPO-SP and PPO-SP-blind (difference interval contains zero). Then the Bluffing is a by-product of the action space (cheap safe suggestions), not a deceptive strategy; do not claim emergent deception.
- Warning: Bluff rate near 0 or near 1 for every Condition (policy collapse onto one Suggestion style); a difference whose interval is wide because few Suggestions were measured; a gap in Bluff but not in Anchor (the Conditions differ in what they hide, not in how they probe); Bluff rates are only comparable if the Conditions played the same Opponents (they do in the grid).

## `accusation_timing` (double column)

Left: cumulative distribution of Candidates left at the moment of each Accusation (log x; 1 = the solution was already uniquely determined). Right: cumulative distribution of game length (turns) of won games.

- Good: mass at or near 1 Candidate (accuse once certain) and a correspondingly high Accusation accuracy in `metrics.csv`; won games ending at similar or earlier turns than the Bot's.
- Bad: curve rising only at hundreds of Candidates (accusing at random: the early-collapse failure seen in ticket 09, where a uniform policy accuses on turn one).
- Warning: bimodal curve (a mix of safe accusations and gambles); a Condition that gambles earlier than another and wins at the same rate is trading risk for speed, which only shows up against stronger Opponents; a right panel with very long tails means stalling (too many passes).

## `gain_leakage` (single column)

One point per cell and Condition: x = Leakage (hand cells of the Learner that Opponents newly learn, per game), y = mean information gain per answered Suggestion (bits). Grey stars are Bots.

- Good: points up and to the left (learns a lot, reveals little); a Condition that dominates the Bots' cloud.
- Bad: down and to the right (reveals more than it learns).
- Warning: a positive slope across all points is normal (more informative Suggestions leak more); judge Conditions by distance from that trend, not by absolute position. Clusters by player count are expected (more players, more leakage); compare like with like.

## `reshow` (single column)

Reveal re-show rate: how often, when it could choose which card to show, the Learner shows a card the asker has already seen (only over Reveals with a choice; Wilson CI; `n=` is the number of such Reveals).

- Good: high rate (re-showing a known card gives away nothing new); PPO-SP above PPO-SP-blind indicates Opponent-aware Reveal behaviour.
- Bad: rate near the share of re-showable cards among the options (choosing uniformly at random).
- Warning: tiny `n` (intervals fill the whole axis); equal rates in the visible and blind Conditions mean Reveal behaviour is not using Opponent knowledge.

## TensorBoard curves (`tensorboard --logdir runs/NAME/tb`)

x axis is Learner steps. The three Conditions share one config, so curves should be compared at the same step.

- `train/entropy`: policy entropy. Good: decays slowly and stays clearly above zero for most of training. Bad / warning: near zero early = collapse onto one action (often "accuse now" or "pass"); never decreasing = the policy is not learning.
- `train/kl`: approximate KL per update. Good: small and steady. Warning: spikes, or consistently far above the target = the step size is too large; flat zero = nothing is being updated.
- `train/clip_fraction`: share of samples clipped by PPO. Good: roughly 0.05 to 0.3. Warning: near 1 = updates too aggressive; zero throughout = learning rate too small.
- `train/value_loss`: Critic error. Good: falls then plateaus. Warning: rising or oscillating while the win rate stalls (reward is too sparse or too noisy for the Critic; self-play makes it non-stationary, so some drift in SP is expected).
- `train/policy_loss`: no absolute meaning; only watch for NaN or huge values.
- `train/episode_win_rate`: training-time win rate against the (changing) training field. In self-play this is expected to hover around chance because the Opponents improve with the Learner; do not read it as a lack of progress, use the `eval/` curves instead.
- `eval/win_rate/<field>/<n>p`: win rate against a fixed Bot field. Good: rising and crossing `eval/chance/<n>p`. Bad: flat below chance. Warning: rise then fall (overfit or instability, pick the checkpoint by this curve); different fields diverging (specialisation to one Bot).
- `eval/chance/<n>p`: the 1/n reference; not a result.
- `eval/baseline/random_suggest/<n>p`: a random-suggest Bot in the same field. The Learner must beat it; at or below it means the policy has not learned anything beyond accusing correctly when forced.
- `eval/bluff_rate/<n>p` and `eval/anchor_rate/<n>p`: behaviour rates on a small eval sample (noisy). Good: stable trend. Warning: sudden jumps with no win-rate change; rates identical for PPO-SP and PPO-SP-blind (the same conclusion as in `bluff_anchor`, now visible over time). Pin conclusions on the final grid, not these curves.
- `train/learner_steps_per_s`: throughput only; a sudden drop means a hung or oversubscribed worker, not a learning problem.
