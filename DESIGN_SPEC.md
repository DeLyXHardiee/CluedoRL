# Cluedo RL simulator: design spec

2026-10-20

## Scope and budget

Build card-only Cluedo for 4 players in three files of roughly 350 to 480 lines in total. One Learner seat is trained with PPO in TorchRL against hand-written bots. This doc is a design: it fixes rules, interfaces and tensor shapes, and you write all the code.

### Rule choices

These are deliberate simplifications. State them in the report as modelling choices.

| Topic | Choice | Why |
| --- | --- | --- |
| Cards | 21 cards as integers: suspects 0-5, weapons 6-11, rooms 12-20 | Integers make masks and tensors trivial |
| Board | No board, no movement, no dice: any room can be named | Movement adds code and no deduction |
| Players | Fixed at 4 for training; engine accepts 3-6 | A fixed count keeps the observation a fixed size |
| Deal | One envelope card per category, 18 % n left-over cards face up for everyone, the rest dealt round-robin from a shuffled deck | 4 players get hands of 4, 4, 4, 4; the other 2 lie face up |
| Turn | Up to two actions per turn: a suggestion, then an accusation or END; or an accusation straight away | Matches real Cluedo; costs one END action and a phase flag |
| Answering | Seats to the suggester's left are asked in order; the first holding a named card shows one card to the suggester only | Standard rule |
| Wrong accusation | Accuser is eliminated, takes no more turns, but still answers suggestions | Standard rule; skipping them breaks deduction |
| Game end | Correct accusation wins. If every seat is eliminated, nobody wins. 200 turns in total = truncated | Cap stops endless games of passive bots |
| Randomness | One `random.Random(seed)` per game, passed to everything | Same seed, same game: needed for tests and fair evaluation |

### Files and line budget

| File | Contents | Lines | Owner |
| --- | --- | --- | --- |
| `engine.py` | Cards, deal, game state, turn flow, answering, accusations, broadcasting events | 100-130 | Person A |
| `actors.py` | `Actor` interface, `Knowledge` notepad, 2-3 bots | 100-140 | Person A |
| `rl.py` | Action numbering, observation encoding, action mask, TorchRL env, PPO training, evaluation | 150-210 | Person B |

Tests come on top of the budget. Keep them short, but write them: section 7 lists the ones that catch the costly bugs.

How the three files connect in one env step:

```
 rl.py (Person B)                 engine.py (Person A)     actors.py (Person A)

 Policy
 (MLP + MaskedCategorical)
    | action        ^ obs, mask
    v               |
 ClueEnv._step  --action-->  Game.step  --event-->  Actors
 (runs Opponent turns)       (answer, accuse,       (choose_reveal, observe)
    ^                         advance)                  | update
    | returned by _step                                 v
 encode + mask  <-------- Learner's notepad --------  Knowledge
 (283 floats, 649 bools)                              (YES / NO / UNKNOWN per seat)
```

On Opponent turns, `ClueEnv` gets the action from the bot's `act(phase)` instead of the policy, and the same loop through `Game.step` runs.

## Game engine (`engine.py`)

The engine owns the hidden truth and the rules, and nothing else. It never decides anything itself: every choice comes from an actor. After each action it builds one event and sends each seat its own redacted copy.

### Actions

An action is a kind plus one card per category, `(kind, suspect, weapon, room)`, with kind `SUGGEST` or `ACCUSE` and each card as its global integer. END (end the turn without accusing) is `(END,)` with no cards. The engine, the actors and the bots only ever handle these tuples. Turning them into slot numbers for the network is the RL side's job (see Action numbering in section 5).

### State

| Field | Meaning |
| --- | --- |
| `n` | Number of seats |
| `hands` | One set of cards per seat |
| `envelope` | The three solution cards |
| `current` | Seat whose turn it is, plus phase: MAIN, or AFTER_SUGGEST once that seat has suggested this turn |
| `eliminated` | One bool per seat |
| `turn` | Turns taken so far |
| `winner` | Seat or `None` |
| `done`, `truncated` | Game over; over because of the turn cap |
| `rng` | The game's `random.Random` |

### Public methods

| Method | Does |
| --- | --- |
| `Game(n, seed)` | Shuffles, picks the envelope, lays out the face-up cards, deals |
| `start(actors)` | Stores the actor list and calls `actor.reset(seat, n, hand, face_up)` for each seat |
| `step(action)` | Plays the current seat's action, broadcasts the event, ends the turn unless the action was a suggestion, returns the full event |

### What `step` does

1. Check that the action comes from `current` and the game is not over, and that its kind fits the phase: SUGGEST or ACCUSE in MAIN, ACCUSE or END in AFTER_SUGGEST.
2. Suggestion: ask seats `current+1, current+2, …` (mod n), eliminated ones included. The first seat holding any of the three cards picks one with `actors[seat].choose_reveal(matching, suggester)`. Stop there.
3. END: nothing to resolve. Accusation: correct sets `winner` and `done`. Wrong marks the seat eliminated; if every seat is now eliminated, set `done`.
4. Unless the action was END, build the event and broadcast it (below).
5. After a suggestion, set phase to AFTER_SUGGEST and return: the same seat acts again. Otherwise set phase to MAIN and add 1 to `turn`; at 200 set `done` and `truncated`. Otherwise move `current` to the next non-eliminated seat.

### Events and broadcasting

One event per action. The engine calls `actors[seat].observe(view)` for every seat, eliminated ones included, because they still need their notepad to answer correctly.

| Field | Value | Who sees it |
| --- | --- | --- |
| `turn` | Turn number | All |
| `kind` | `SUGGEST` or `ACCUSE` | All |
| `seat` | Who acted | All |
| `cards` | The three named cards | All |
| `passed` | Seats asked who had none of the cards | All |
| `answerer` | Seat that showed a card, or `None` | All |
| `shown` | The card shown | Only the suggester and the answerer; `None` for everyone else |
| `correct` | Accusations only | All |

The redaction of `shown` is the one place where a bug lets the agent cheat silently. Make a copy of the event per seat and set `shown` to `None` unless the seat is the suggester or the answerer. Test it.

## Actor interface (`actors.py`)

Every seat is an `Actor` with four methods, and each actor keeps its own `Knowledge` notepad. The engine calls `choose_reveal` and `observe`. The game loop (a bot match or the RL env) calls `act`. The Learner is an actor too: the env never calls its `act`, because the policy picks its action.

### The four methods

| Method | Called by | When | Returns |
| --- | --- | --- | --- |
| `reset(seat, n, hand, face_up)` | engine `start` | Once per game | Nothing; builds a fresh `Knowledge` |
| `act(phase)` | game loop | On this seat's turn, and again after its own suggestion | An action tuple |
| `choose_reveal(matching, suggester)` | engine | This seat must show a card | One card from `matching` |
| `observe(event)` | engine | After every action, redacted | Nothing; updates `Knowledge` |

Put the shared parts in the base class: `reset`, `observe` (which only forwards to `Knowledge`) and a default random `choose_reveal`. A bot then only writes `act`, often in 5 to 15 lines.

### Knowledge: the notepad (about 60-80 lines)

Knowledge is a table of seats × 21 cards. Each cell is `YES`, `NO` or `UNKNOWN`, the paper notepad real players use. It is the base for both the bots and the observation, so it is the most valuable code in the project.

Update rules, in order:

1. **Own hand and face-up cards.** My cards are `YES` for me and `NO` for everyone else. Every other card is `NO` for me. Face-up cards are `NO` for every seat.
2. **Passed seats.** Every seat in `event.passed` is `NO` on all three named cards. This is public, so every observer applies it.
3. **Card shown to me.** The answerer is `YES` on `shown`.
4. **Seen answers ("one of" rules).** When someone else answers, I only learn they hold at least one of the three cards. Store `(answerer, {a, b, c})`. Drop cards the answerer is `NO` on; when one card is left, it becomes `YES`.
5. **One owner per card.** A `YES` in a column makes every other seat `NO` on that card.
6. **Full hands.** When a seat has as many `YES` as its hand size, the rest of its row is `NO`. Hand sizes are public: every seat holds 18 // n cards, because the 18 % n left-over cards go face up.

Repeat rules 4 to 6 until nothing changes; it settles within a few passes.

Derived queries that bots and the encoder use:

- **Envelope candidates:** a card is a candidate unless it is face up or some seat is `YES` on it. A card that is not face up is known to be in the envelope when every seat is `NO` on it.
- **Solved:** each category has exactly one candidate.
- **Accusation candidates:** all suspect × weapon × room combos built from candidates, minus combos that were accused wrongly. These drive the accusation mask.

### Bots

Two bots are enough to train against; a third gives you a held-out opponent for evaluation.

| Bot | Suggests | Accuses | Purpose |
| --- | --- | --- | --- |
| `RandomBot` | Any random combo | Only when solved | Weak baseline, crash testing |
| `CandidateBot` | A random candidate in each category | Only when solved | Main training opponent |
| `GambleBot(k)` | Like `CandidateBot` | When there are k or fewer accusation candidates, at random among them | Held-out opponent; takes risks |

After their own suggestion, bots run the same accuse rule and play END if it does not fire. All three use the default random reveal. "Show a card I already showed this suggester" is a 3-line extra if you want a smarter reveal.

## Observation: what fits the budget

Encode the Learner's `Knowledge` notepad, not the raw history. With the notepad from section 3 this costs about 20 lines and gives a fixed vector of 283 numbers at 4 players. The notepad already sums up the history, so the network does not have to learn deduction from a sequence.

### Options

| Tier | Contents | Size at 4 players | Extra lines in `rl.py` | Lets the agent learn |
| --- | --- | --- | --- | --- |
| 0 | Own hand (21), envelope candidates (21), phase (1), turn / 200 (1) | 44 | ~8 | Accusation timing only; suggestions stay blind |
| **1 (recommended)** | Notepad one-hot: 4 seats × 21 cards × {YES, NO, UNKNOWN} (252), envelope candidates (21), candidates left per category / category size (3), accusation candidates / 324 (1), eliminated per seat (4), phase (1), turn / 200 (1) | 283 | ~20 | Which cards to probe, who will answer, when to accuse or gamble |
| 2 | Tier 1 plus open "one of" rules per seat × card (63) and how often each opponent named each card (63) | 409 | ~35, plus counting in `Knowledge` | Reading opponents' suggestions; only useful against opponents whose suggestions say something |

Two rules for every tier:

- **Seats relative to me.** Row 0 is the Learner, row 1 the seat to its left, and so on. Then one network plays every seat.
- **Floats in [0, 1].** One-hots and counts divided by their maximum. No raw turn numbers.

### Too big for 300-500 lines

- **The full event history as a sequence.** It needs an RNN or transformer, padding and masking: 150+ lines and much slower training.
- **Probabilities per cell.** Sampling deals consistent with the notepad is a project of its own.
- **Mixed 3-6 player training.** Padding to 6 rows is about 10 lines, so it is a cheap later extension, not a start.

### Action mask

The mask is 649 bools in the same index order as the actions.

- In MAIN, all 324 suggestions are legal and END is not. In AFTER_SUGGEST, no suggestion is legal and END always is.
- An accusation is legal only if it is an accusation candidate in the notepad. A sound notepad never masks out the true envelope, so the mask removes only accusations that are certain to lose.

This matters a lot: without it the agent spends most of early training on losing accusations.

## TorchRL (`rl.py`)

Wrap one game as a single-agent `EnvBase`: one env step is one Learner decision, and all Opponent turns run inside it. Names below are for torch 2.14.1, torchrl 0.14.1 and tensordict 0.14.3 (CPU wheels), the versions already pinned in the repo. Older torchrl releases use other names, so do not mix versions.

### Action numbering

The network has 649 outputs, one per action, so every action needs a fixed slot number. Only `rl.py` uses these numbers.

- Build the list of all actions once: nested loops over kind (`SUGGEST`, then `ACCUSE`), suspect, weapon and room, then END at the end. That puts 2 × 6 × 6 × 9 = 648 card actions in slots 0-647 and END in slot 648.
- Slot → action: read the list at that position. Action → slot: a dictionary built from the same list.
- To check a number by hand: slot = kind × 324 + s × 54 + w × 9 + r, where s, w and r are positions within each category. The global cards are suspect s, weapon 6 + w and room 12 + r.

| Where | Direction | Does |
| --- | --- | --- |
| `ClueEnv._step` | slot → action | Decodes `tensordict["action"]` into a tuple before `game.step` |
| Mask builder | action → slot | Sets the mask to True at the slot of each legal action (rules in section 4) |

Test it: the list has 649 distinct actions, and every slot round-trips through both directions.

### The env class

`ClueEnv(EnvBase)` holds one `Game`, the Learner seat, one Learner actor and a list of opponent factories. `batch_size` is `[]` (one game); run several envs in parallel with a collector instead.

| Spec | TorchRL class | Shape | dtype |
| --- | --- | --- | --- |
| `observation_spec["observation"]` | `Unbounded` | (283,) | float32 |
| `observation_spec["action_mask"]` | `Binary` | (649,) | bool |
| `action_spec` | `Categorical(n=649)` | () | int64 |
| `reward_spec` | `Unbounded` | (1,) | float32 |
| `done_spec`: `done`, `terminated`, `truncated` | `Categorical(n=2)` each, inside a `Composite` | (1,) | bool |

The action is an index, not a one-hot, because `MaskedCategorical` samples indices.

### `_reset(tensordict)`

1. New `Game(4, seed)` from the env's own `random.Random`.
2. Pick a random Learner seat; fill the other seats with bots drawn from the opponent list.
3. Call `game.start(actors)`, then play Opponent turns until it is the Learner's turn.
4. If the game ends before the Learner moves (rare), deal again.
5. Return a `TensorDict` with `observation`, `action_mask` and `done`, `terminated`, `truncated` all False.

### `_step(tensordict)`

1. Read `tensordict["action"]`, decode it, and call `game.step`.
2. If the Learner just suggested, skip this: its next decision is accuse or END. Otherwise play Opponent turns until it is the Learner's turn again, or the game is over, or the Learner is eliminated.
3. Return `observation`, `action_mask`, `reward`, `done`, `terminated`, `truncated`. Return them flat: `EnvBase.step` moves them under `"next"` for you.

| Outcome | reward | terminated | truncated | done |
| --- | --- | --- | --- | --- |
| Game goes on | 0 | False | False | False |
| Learner accused correctly | +1 | True | False | True |
| Learner eliminated | 0 | True | False | True |
| Another seat wins | 0 | True | False | True |
| Turn cap reached | 0 | False | True | True |

End the episode as soon as the Learner is eliminated. It cannot act any more, so finishing the game only wastes time.

Also write `_set_seed(seed)`, which re-seeds the env's `random.Random`. When the episode is over, still return a valid observation and a mask with at least one True; TorchRL reads it even though nobody acts on it.

### Policy and value networks

| Part | Build | in_keys → out_keys |
| --- | --- | --- |
| Actor MLP | `MLP(in_features=283, out_features=649, num_cells=[256, 256])` in a `TensorDictModule` | `observation` → `logits` |
| Actor | `ProbabilisticActor(..., distribution_class=MaskedCategorical, return_log_prob=True)` | in_keys `{"logits": "logits", "mask": "action_mask"}` → `action`, `action_log_prob` |
| Critic | `ValueOperator(MLP(283 → [256, 256] → 1), in_keys=["observation"])` | `observation` → `state_value` |

### Training loop parts

| Part | Class | Starting values |
| --- | --- | --- |
| Data | `Collector` (one process) or `MultiSyncCollector` (one env per CPU core) | `frames_per_batch=4096` |
| Advantage | `GAE` from `torchrl.objectives.value` | `gamma=0.99`, `lmbda=0.95`, `average_gae=True` |
| Loss | `ClipPPOLoss(actor, critic, ...)` | `clip_epsilon=0.2`, `entropy_coeff=0.01` |
| Minibatches | `ReplayBuffer(storage=LazyTensorStorage(4096), sampler=SamplerWithoutReplacement())` | 4 epochs, minibatch 512 |
| Optimiser | `torch.optim.Adam` over the loss's parameters | lr 3e-4, gradient clip 0.5 |

Per batch: run `GAE` on the batch, put it in the buffer, then for each epoch and minibatch add `loss_objective + loss_critic + loss_entropy` and take one step. The official TorchRL PPO tutorial has the same structure; follow it for the loop, not for its env.

### Shapes in one collected batch (B = frames_per_batch)

| Key | Shape | dtype |
| --- | --- | --- |
| `observation` | [B, 283] | float32 |
| `action_mask` | [B, 649] | bool |
| `logits` | [B, 649] | float32 |
| `action` | [B] | int64 |
| `action_log_prob` | [B] | float32 |
| `("next", "observation")` | [B, 283] | float32 |
| `("next", "reward")` | [B, 1] | float32 |
| `("next", "done")`, `terminated`, `truncated` | [B, 1] | bool |
| `advantage`, `value_target`, `state_value` (after GAE) | [B, 1] | float32 |

A collected batch may have more than one dimension (for example [workers, steps]). Run `GAE` on the batch as it comes, so it sees episodes in time order, then flatten it with `batch.reshape(-1)` before sampling minibatches. With `MultiSyncCollector`, call `collector.update_policy_weights_()` after each update, or the workers keep playing with the old policy.

### Checks before training

- `check_env_specs(env)` must pass. It plays random actions, so wrap the env as `TransformedEnv(env, ActionMask())` to make random actions respect the mask.
- Print one `env.rollout(20)` and compare its keys and shapes with the table above.

## Splitting the work

Person A writes the game side (`engine.py`, `actors.py`). Person B writes the learning side (`rl.py`). Each part is about 230 to 250 lines. The split works because B depends on only four names from A, and those are fixed in this doc.

### The contract between the two halves

Agree on these on day one and do not change them without telling the other person.

| Name | From | Used by B for |
| --- | --- | --- |
| `Game(n, seed)`, `start`, `step`, `current`, `done`, `truncated`, `winner`, `eliminated` | `engine.py` | Running the game inside the env |
| `Actor` and its four methods | `actors.py` | The Learner actor and the opponent list |
| `Knowledge`: the YES/NO/UNKNOWN table, envelope candidates, accusation candidates | `actors.py` | Observation and action mask |
| `RandomBot`, `CandidateBot`, `GambleBot(k)` | `actors.py` | Opponents and evaluation |

### Order of work

1. **Both:** read this doc together and settle anything you disagree with. Write the contract names as empty stubs in each file.
2. **A:** engine plus `RandomBot`. 1,000 random games with no crash, every game ends. **B:** `ClueEnv` on the stub engine with a dummy observation; `check_env_specs` passes.
3. **A:** `Knowledge` and `CandidateBot`, with the soundness test from section 7. **B:** PPO loop running end to end on the dummy env; loss values print.
4. **Both:** plug the real engine and `Knowledge` into the env. Train against 3 `CandidateBot`s.
5. **B:** evaluation: the Learner's win rate over 2,000 games against each bot mix, with a 95% interval. Equal bots win 25% each at 4 players, so that is the bar to beat. **A:** `GambleBot` as the held-out opponent, plus a table of bot-vs-bot win rates as a baseline.

If one of you finishes early, the next most useful work is writing the tests in section 7 for the other person's file. A second reader catches more than the author.

## Tests and pitfalls

Six tests catch nearly every bug that would quietly ruin training. Each is 5 to 15 lines with `pytest`.

### Tests

- [ ] **Deal:** the envelope has one card per category, hands, face-up cards and envelope together hold all 21 cards once, and every hand has 18 // n cards.
- [ ] **Answering:** on a hand-built deal, the first seat to the left holding a card answers, eliminated seats still answer, and nobody answers when nobody holds a card.
- [ ] **Redaction:** in 100 random games, only the suggester and the answerer ever receive a non-`None` `shown`.
- [ ] **Notepad soundness:** in 1,000 games of bots, every `YES` and `NO` in every seat's `Knowledge` matches the true deal, and the true envelope is always an accusation candidate. This is the most important test.
- [ ] **Same seed, same game:** two games with the same seed and bots produce identical events.
- [ ] **Env:** `check_env_specs` passes, and in a rollout the chosen action is never masked out.

### Pitfalls

| Pitfall | Symptom | Fix |
| --- | --- | --- |
| `shown` sent to every seat | Learner wins far too easily | Redaction test |
| Eliminated seats skipped when answering | Notepad marks wrong cards as envelope; bots accuse wrongly | Ask every seat but the suggester |
| Turn cap reported as `terminated` | Value estimates drift; GAE treats the cap as a real loss | Set `truncated=True`, `terminated=False` |
| Game ends during Opponent turns and no reward is returned | Learner never sees losses | Check `done` after each Opponent turn inside `_step` |
| Seats not relative to the Learner | Learning is slow and depends on seat | Rotate rows so row 0 is the Learner |
| Mask with no True left | `MaskedCategorical` raises an error | A suggestion or END is always legal, so this only happens if the mask is built wrong |
| Many workers each using all CPU threads | Collection crawls | `torch.set_num_threads(1)` inside each env worker |
| Different torchrl version from the doc | Names like `Collector` or `action_log_prob` not found | Install the pinned versions from section 5 |
