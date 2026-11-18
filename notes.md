### What's missing
[x] Torch needs to be seeded just like the env
[x] Code to stop training when converged (define convergence) (we should use a different eval seed for the stopping check than for the final results)
[ ] Code to plot training and other interesting data
[x] Build a strategy that might fall for bluffing
[x] Build a baseline "smart" strategy and maybe use this as a held-out opponent (only played against in evaluation)
[ ] Decide on and implement suggestion history in the observation (bluffing question)
[ ] Decide on the train, validation and test sets of combinations of bot components
[x] Decide on an experiment list (bots only, self play, mixed pool?), how many seeds each and max frame budget vs convergence, and cross-play evaluation (all final models against each other)
[ ] Decide on card reveal limitation: choose_reveal always shows the first matching card. We should either accept that and mention it in the report, or add it to the RL actions.


### To improve training we tried
- The biggest change: There are 649 possible actions. A plain network outputs 649 separate scores, so it has to learn every suggest/accuse combinaton on its own. "Suggest Scarlet + Rope + Kitchen" and "Suggest Scarlet + Rope + Hall" share nothing, so it learns slowly. To fix this we score the parts of an action, then add the parts together. So now the network outputs 45 numbers i.e. 3 kind scores, 21 suggest card scores, and 21 accuse card scores. An action then gets its score with score(SUGGEST, s, w, r) = kind[SUGGEST] + suggest_card[s] + suggest_card[w] + suggest_card[r] and so on for each kind. When including the END action we now get a vector of size 46. Each card's position is 3 + card for suggest and 3 + 21 + card for accuse (formula: 3 + 21 * k + card). From this we build four lists each 649 long (one entry per action slot) named kind_idx, s_idx, w_idx, and r_idx. For each of the 649 actions, the lists tell us where in the output to find the kind score and the three card scores. Adding those four scores up gives that action's logit. Since END has no cards, its card indices point to position 45 which is always 0 and adds nothing.
- Lowering the entropy coefficient during training since most of the valid actions the model has are accusations (324 of 648) and those aren't masked.
- Change the "how many candidates remain in total" observation to be "1 / count of accusation candidates" instead of "len(accusation_candidates) / self.combination_count" to make it more "model friendly". This makes it so if it's solved the value is 1.0, if there are two possibilities left, the value is 0.5, three is 0.33 and so on. This makes the difference between amounts of accusation candidates larger so it's more visibile to the model. Otherwise it'll have trouble learning when to accuse / gamble in order to win.
- Change the "turns left" observation to be "min(turns taken / 40, 1.0)" (40 based on how many turns a game usually lasts) instead of "turns taken / max turns" due to the same issue as above. The model doesn't see that clear of a difference between almost finished and just started with the old version. (THIS GOT WORSE?)
- At 6 players many runs ended up with very low win rates (around 1%). This is likely because of our reward model where we only reward when the actor wins. We likely need to punish accusing wrongly, or even reward getting good suggestions somehow. EDIT: a single training run after penalizing wrong accusations seems to not have fixed the problem. New hypothesis: the END action has just as small of a chance to be chosen by an untrained policy as any other combination of cards for an accusation. We need to increase the probability of choosing END in the untrained policy so it learns to use END in the accuse phase in case it's not sure yet. EDIT: the policy scored each action as kind + suspect + weapon + room, so a kind's total probability grew with the number of allowed actions of that kind. After a suggestion, END (1 action) competed against every possible accusation (about 110–195). An untrained agent therefore chose END only 0.7–1.7% of the time, and got eliminated by a near-random accusation in almost every game. At 6 players, with the most possible solutions, the agent locked into accusing on its first turn. A penalty for wrong accusations couldn't fix this, because accusing now and accusing later are penalised the same. We fixed this by turning the scores into a two-stage probability: first which kind of action, then which cards log P(action) = log P(kind) + log P(cards | kind). 
P(kind) is a softmax over the three kind scores (suggest, accuse, end).
P(cards | kind) is a softmax of the card scores over only the allowed actions of that kind. Disallowed actions are set to −1e9 before the softmax.
THIS ALSO DIDN'T WORK, the 6 player collapse is still there at some seeds.
EDIT: New idea is to simply restrict the model from accusing until there are at least K possible candidates. We need to describe this as action masking due to the exploration trap at 6 players.


# Primary metrics
1. Score against the test pool. The model alone against n − 1 test bots, at every count from 3 to 6, using best.pt.
  - We report the mean score across seeds with a 95% CI for each method, and the difference between methods with its CI.
  - The standard deviation across seeds is our robustness number.

2. Head-to-head win share. Tables with an equal number of each method's models: 2 + 2 at 4 players and 3 + 3 at 6 players, with every seating rotated, over all seed pairings.
  - We report the difference in win share (bot-trained minus self-play) with a CI across seed pairings.
  - 3 and 5 players can't be split evenly, so we use them in the secondary results.

Secondary results include 
- Results per player count.
- Each opponent type separately.
- The training pool.
- 1-vs-many lineups.
- Mixed tables with bots (1 + 1 + test bots works at every count).
- Argmax vs sampled play?
- The equal-batch comparison.
- The behaviour metrics: bluff rate, information leaked, and confidence at accusation time.

