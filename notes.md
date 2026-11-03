### What's missing
[x] Torch needs to be seeded just like the env
[x] Code to stop training when converged (define convergence) (we should use a different eval seed for the stopping check than for the final results)
[ ] Code to plot training and other interesting data
[ ] Build a strategy that might fall for bluffing
[ ] Build a baseline "smart" strategy
[ ] Decide on and implement suggestion history in the observation (bluffing question)
[ ] Decide on an experiment list (bots only, self play, mixed pool?), how many seeds each and max frame budget vs convergence, and cross-play evaluation (all final models against each other)
[ ] Decide on card reveal limitation: choose_reveal always shows the first matching card. We should either accept that and mention it in the report, or add it to the RL actions.


### To improve training we tried
- Lowering the entropy coefficient during training since most of the valid actions the model has are accusations (324 of 648) and those aren't masked.
- Change the "how many candidates remain in total" observation to be "1 / count of accusation candidates" instead of "len(accusation_candidates) / self.combination_count" to make it more "model friendly". This makes it so if it's solved the value is 1.0, if there are two possibilities left, the value is 0.5, three is 0.33 and so on. This makes the difference between amounts of accusation candidates larger so it's more visibile to the model. Otherwise it'll have trouble learning when to accuse / gamble in order to win.
- Change the "turns left" observation to be "min(turns taken / 40, 1.0)" (40 based on how many turns a game usually lasts) instead of "turns taken / max turns" due to the same issue as above. The model doesn't see that clear of a difference between almost finished and just started with the old version. (THIS GOT WORSE?)