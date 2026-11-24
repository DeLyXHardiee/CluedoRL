import random
import torch
import math
import os
import csv
import json
import time
import argparse
import subprocess
from torchrl.envs import EnvBase
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.data import Composite, Unbounded, Binary, Categorical, ReplayBuffer, LazyTensorStorage
from torchrl.collectors import Collector
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, MaskedCategorical
from GameEngine import CluedoGameEngine, SUGGEST, ACCUSE, END, AFTER_SUGGEST, MAIN
from Actor import CluedoActor, Knowledge, GullibleKnowledge, GoodMemoryKnowledge
from functools import partial
from Strategies import Strategy, main_first, after_suggest_certain, main_random, main_gamble, main_bluff, after_suggest_gamble

MAX_PLAYERS = 6
OBS_SIZE = MAX_PLAYERS * 21 * 3 + 21 + 4 + MAX_PLAYERS + MAX_PLAYERS + 2 + MAX_PLAYERS * 21 # 543


def bot(main=main_first, after_suggest=after_suggest_certain, knowledge=Knowledge):
      return lambda: CluedoActor(Strategy(main, after_suggest), knowledge)

bot_factory = {
    "first/certain/knowledge": bot(main_first, after_suggest_certain, Knowledge),
    "random/certain/gullible": bot(main_random, after_suggest_certain, GullibleKnowledge),
    "gamble/gamble/knowledge": bot(main_gamble, after_suggest_gamble, Knowledge),
    "bluff/certain/goodmemory": bot(main_bluff, after_suggest_certain, GoodMemoryKnowledge),
    "first/gamble/goodmemory": bot(main_first, after_suggest_gamble, GoodMemoryKnowledge),
    "gamble/gamble/goodmemory": bot(main_gamble, after_suggest_gamble, GoodMemoryKnowledge),
    "bluff/certain/knowledge": bot(main_bluff, after_suggest_certain, Knowledge),
    "first/certain/gullible": bot(main_first, after_suggest_certain, GullibleKnowledge),
    "random/gamble/goodmemory": bot(main_random, after_suggest_gamble, GoodMemoryKnowledge),
    "bluff/gamble/gullible": bot(main_bluff, after_suggest_gamble, GullibleKnowledge),
    "random/certain/knowledge": bot(main_random, after_suggest_certain, Knowledge),
}

TRAIN_POOL = [
    "first/certain/knowledge",
    "random/certain/gullible",
    "gamble/gamble/knowledge",
    "bluff/certain/goodmemory",
    "first/gamble/goodmemory",
]

VAL_POOL = [
    "gamble/gamble/goodmemory",
    "bluff/certain/knowledge",
    "first/certain/gullible",
]

TEST_POOL = [
    "random/gamble/goodmemory",
    "bluff/gamble/gullible",
    "random/certain/knowledge",
]

assert not any(bot in VAL_POOL for bot in TRAIN_POOL), "Validation pool overlaps with training pool"
assert not any(bot in TEST_POOL for bot in TRAIN_POOL), "Test pool overlaps with training pool"
assert not any(bot in TEST_POOL for bot in VAL_POOL), "Test pool overlaps with validation pool"


def build_actions():
    # lookup table
    actions = []
    for kind in (SUGGEST, ACCUSE):
        for suspect in range(0, 6):
            for weapon in range(6, 12):
                for room in range(12, 21):
                    actions.append((kind, suspect, weapon, room))
    actions.append((END,))
    return actions

ACTIONS = build_actions()
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
COMBINATION_COUNT = 6 * 6 * 9 # suspects * weapons * rooms
ACCUSE_LIMIT = 30

class ClueEnv(EnvBase):

    def __init__(self, opponent_factories, seed, player_counts=(3, 4, 5, 6)):
        super().__init__()
        self.opponent_factories = opponent_factories
        self.player_counts = player_counts

        self.actions = ACTIONS

        # dictionary of action to index
        self.action_to_index = ACTION_TO_INDEX

        self.observation_spec = Composite(
            observation=Unbounded(shape=(OBS_SIZE,), dtype=torch.float32),
            action_mask=Binary(shape=(649,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Categorical(n=649, shape=(), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Composite(
            done=Categorical(n=2, shape=(1,), dtype=torch.bool),
            terminated=Categorical(n=2, shape=(1,), dtype=torch.bool),
            truncated=Categorical(n=2, shape=(1,), dtype=torch.bool),
            shape=(),
        )
        self.penalty_scale=1.0
        self.set_seed(seed)

    def _set_seed(self, seed):
        self.rng = random.Random(seed)

    def _reset(self, tensordict=None):
        while True:
            self.player_count = self.rng.choice(self.player_counts)
            self.game = CluedoGameEngine(self.rng.getrandbits(32))
            self.seat = self.rng.randint(0, self.player_count - 1)
            self.actor = CluedoActor(Strategy())
            actors = []
            for i in range(self.player_count):
                if i != self.seat:
                    actors.append(self.rng.choice(self.opponent_factories)())
                    continue
                actors.append(self.actor)

            self.game.start(actors)
            self.game.play(stop_at=self.actor.seat)
            if not self.game.done:
                break

        self.observation = build_observation(self.actor)
        self.action_mask = build_action_mask(self.actor)

        return TensorDict({
            "observation": self.observation,
            "action_mask": self.action_mask,
            "done": torch.tensor([False], dtype=torch.bool),
            "terminated": torch.tensor([False], dtype=torch.bool),
            "truncated": torch.tensor([False], dtype=torch.bool)
        }, batch_size=())
        

    def _step(self, tensordict=None):
        action = self.actions[int(tensordict["action"])]
        self.game.step(action)

        eliminated = self.actor in self.game.eliminated_players
        if not eliminated:
            self.game.play(stop_at=self.actor.seat)

        won = self.game.winner is self.actor
        truncated = self.game.truncated
        terminated = eliminated or (self.game.done and not truncated)
        done = terminated or truncated
        reward = 0.0
        if won:
            reward = 1.0
        elif eliminated:
            reward = -self.penalty_scale / self.player_count

        observation = build_observation(self.actor)
        action_mask = build_action_mask(self.actor)

        tensordict = TensorDict({
            "observation": observation,
            "action_mask": action_mask,
            "done": torch.tensor([done], dtype=torch.bool),
            "terminated": torch.tensor([terminated], dtype=torch.bool),
            "truncated": torch.tensor([truncated], dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32)
        }, batch_size=())
        return tensordict

class FactoredPolicy(torch.nn.Module):
    
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize=normalize
        self.body = MLP(in_features=OBS_SIZE, out_features=45, num_cells=[256, 256])
        self.actions = build_actions()
        self.groups = [(0, 3), (3, 9), (9, 15), (15, 24), (24, 30), (30, 36), (36, 45)]
        kind_idx = []
        s_idx = []
        w_idx = []
        r_idx = []
        
        for action in self.actions:
            kind = action[0]
            if kind == END:
                kind_idx.append(2)
                s_idx.append(45)
                w_idx.append(45)
                r_idx.append(45) # the always zero position for the END action
                continue

            _, s, w, r = action

            k = 0 if kind == SUGGEST else 1
            kind_idx.append(k)
            s_idx.append(3 + 21 * k + s)
            w_idx.append(3 + 21 * k + w)
            r_idx.append(3 + 21 * k + r)

        kind_idx = torch.tensor(kind_idx, dtype=torch.long)
        s_idx = torch.tensor(s_idx, dtype=torch.long)
        w_idx = torch.tensor(w_idx, dtype=torch.long)
        r_idx = torch.tensor(r_idx, dtype=torch.long)
        self.register_buffer("kind_idx", kind_idx, persistent=False)
        self.register_buffer("s_idx", s_idx, persistent=False)
        self.register_buffer("w_idx", w_idx, persistent=False)
        self.register_buffer("r_idx", r_idx, persistent=False)

    def forward(self, observation, action_mask):
        output = self.body(observation)
        zeros_column = torch.zeros_like(output[..., :1])
        output = torch.cat((output, zeros_column), dim=-1)
        if self.normalize:
            kind_logp = torch.log_softmax(output[..., :3], dim=-1)
            cards = output[..., self.s_idx] + output[..., self.w_idx] + output[..., self.r_idx]
            cards = cards.masked_fill(~action_mask, -1e9)
            card_logp = torch.cat([torch.log_softmax(cards[..., :324], dim=-1),
                         torch.log_softmax(cards[..., 324:648], dim=-1),
                         torch.log_softmax(cards[..., 648:], dim=-1)], dim=-1)
            return kind_logp[..., self.kind_idx] + card_logp
        else:
            return output[..., self.kind_idx] + output[..., self.s_idx] + output[..., self.w_idx] + output[..., self.r_idx] # one logit per slot in action table's order

def build_observation(actor):
    game = actor.gameEngine
    accusation_candidates = actor.get_accusation_candidates()
    n = len(game.players)

    # Notepad grid
    knowledge = torch.tensor(actor.get_knowledge(), dtype=torch.int64)
    grid = torch.roll(knowledge, shifts=-actor.seat, dims=0)
    grid = torch.nn.functional.one_hot(grid, num_classes=3)
    grid = torch.nn.functional.pad(grid, (0, 0, 0, 0, 0, MAX_PLAYERS - n), value=0) # pad to max players
    grid = grid.flatten().float()

    # Suggestion history grid
    suggested = torch.tensor(actor.get_suggestion_counts(), dtype=torch.float32)
    suggested = torch.roll(suggested, shifts=-actor.seat, dims=0)
    suggested = suggested / (suggested + 1) # 0 stays 0, 1 becomes 0.5, 2 becomes 0.66, 3 becomes 0.75 etc.
    suggested = torch.nn.functional.pad(suggested, (0, 0, 0, MAX_PLAYERS - n), value=0)
    suggested = suggested.flatten()
    
    # Envelope candidates
    candidates = actor.get_candidates()
    cand = torch.zeros(21, dtype=torch.float32)
    for i in range(len(candidates)):
        cand[candidates[i]] = 1.0
    
    # How close learner is to solving
    suspects = cand[:6]
    weapons = cand[6:12]
    rooms = cand[12:21]
    solve = torch.zeros(4, dtype=torch.float32)
    solve[0] = suspects.sum() / 6.0 # how many candidates remain in suspects
    solve[1] = weapons.sum() / 6.0 # ... weapons
    solve[2] = rooms.sum() / 9.0 # ... rooms
    solve[3] = 1.0 / max(len(accusation_candidates), 1) # how many candidates remain in total, computed so 1.0 when solved, 0.5 at two left, 0.33 at three etc.

    # Game state
    eliminated = torch.zeros(MAX_PLAYERS, dtype=torch.float32)
    eliminated_players = torch.tensor([p in game.eliminated_players for p in game.seatToPlayer.values()], dtype=torch.float32)
    eliminated_players = torch.roll(eliminated_players, shifts=-actor.seat, dims=0)
    for i in range(len(eliminated_players)):
        if eliminated_players[i]:
            eliminated[i] = 1.0
    present_players = torch.zeros(MAX_PLAYERS, dtype=torch.float32)
    present_players[:n] = 1.0
    misc = torch.zeros(2, dtype=torch.float32)
    misc[0] = 1.0 if game.phase == AFTER_SUGGEST else 0.0
    misc[1] = game.turns_taken / game.max_turns

    encoded = torch.cat([grid, suggested, cand, solve, eliminated, present_players, misc], dim=0)
    assert encoded.shape[0] == OBS_SIZE, f"Observation size mismatch: expected {OBS_SIZE}, got {encoded.shape[0]}"
    return encoded

def build_action_mask(actor):
    game = actor.gameEngine
    accusation_candidates = actor.get_accusation_candidates()

    action_mask = torch.zeros(len(ACTIONS), dtype=torch.bool)

    if game.phase == MAIN:
        action_mask[0:COMBINATION_COUNT] = True

    if len(accusation_candidates ) <= ACCUSE_LIMIT:
        for candidate in accusation_candidates:
            index = ACTION_TO_INDEX[(ACCUSE, candidate[0], candidate[1], candidate[2])]
            action_mask[index] = True

    if game.phase == AFTER_SUGGEST:
        action_mask[-1] = True 

    assert action_mask.any(), "No valid actions available"
    return action_mask


class RLActor(CluedoActor):

    def __repr__(self):
        return f"RLActor(policy = {self.policy.__class__.__name__})"

    def __init__(self, policy):
        super().__init__(strategy=self)
        self.policy = policy

    def getStrategy(self, game_state, knowledge, seed):
        tensordict = TensorDict({
            "observation": build_observation(self),
            "action_mask": build_action_mask(self),
        }, batch_size=())

        with torch.no_grad():
            tensordict = self.policy(tensordict)
        
        return ACTIONS[int(tensordict["action"])]


def make_actor(normalize=True):
      module = TensorDictModule(FactoredPolicy(normalize), in_keys=["observation", "action_mask"], out_keys=["logits"])
      return ProbabilisticActor(
          module,
          in_keys={"logits": "logits", "mask": "action_mask"},
          out_keys=["action"],
          distribution_class=MaskedCategorical,
          return_log_prob=True,
      )

def make_critic():
    net = MLP(in_features=OBS_SIZE, out_features=1, num_cells=[256, 256])
    return ValueOperator(net, in_keys=["observation"])

def train(max_frames=None, seed=0, self_play=False, snapshot_every=50, pool_size=20, run_name="run", checkpoint_every=100, eval_games=2000, eval_seed=1000, patience=float("inf"), min_delta=0.025, player_counts=(3, 4, 5, 6), normalize=True, smooth_window=3):
    torch.manual_seed(seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_name}_{timestamp}_{seed}"
    run_dir = os.path.join("checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)

    learning_rate = 3e-4
    clip = 0.2
    entropy = 0.01
    gamma = 1.0
    lmbda = 0.95
    batch_size = 4096
    minibatch_size = 512
    epochs = 4

    train = "self-play" if self_play else [bot for bot in TRAIN_POOL]
    config = {
        "seed": seed,
        "self_play": self_play,
        "snapshot_every": snapshot_every,
        "pool_size": pool_size,
        "run_name": run_name,
        "checkpoint_every": checkpoint_every,
        "learning_rate": learning_rate,
        "clip": clip,
        "entropy": entropy,
        "gamma": gamma,
        "lambda": lmbda,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "epochs": epochs,
        "eval_games": eval_games,
        "eval_seed": eval_seed,
        "patience": patience,
        "smooth_window": smooth_window,
        "accuse_limit": ACCUSE_LIMIT,
        "min_delta": min_delta,
        "player_counts": player_counts,
        "pools": {
            "train": train,
            "val": [bot for bot in VAL_POOL],
            "test": [bot for bot in TEST_POOL]
        },
        "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(),
        "git_dirty": subprocess.run(["git", "diff", "HEAD", "--quiet"], capture_output=True, cwd=os.path.dirname(os.path.abspath(__file__))).returncode != 0,
        "normalized_policy": normalize
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    def snapshot():
        actor_inner = make_actor(normalize)
        actor_inner.load_state_dict(actor.state_dict())
        pool.append(actor_inner)

    def self_play_opponent():
        policy = pool_rng.choice(pool)
        return RLActor(policy)
    
    actor = make_actor(normalize)
    pool = []
    pool_rng = random.Random(seed)

    snapshot()

    opponent_factories = [bot_factory[bot] for bot in TRAIN_POOL]
    if self_play:
        opponent_factories = [self_play_opponent]

    env = ClueEnv(opponent_factories, seed=seed, player_counts=player_counts)
    critic = make_critic()

    # Collector is responsible for gathering experience from the environment using the actor policy.
    # GAE (Generalized Advantage Estimation) is used to compute advantage estimates for policy gradient updates.
    # ClipPPOLoss implements the PPO objective with clipping to stabilize training.
    collector = Collector(env, actor, frames_per_batch=batch_size, total_frames=max_frames or -1)
    gae = GAE(gamma=gamma, lmbda=lmbda, value_network=critic, average_gae=True)
    loss_func = ClipPPOLoss(actor, critic, clip_epsilon=clip, entropy_coeff=entropy)
    buffer = ReplayBuffer(storage=LazyTensorStorage(batch_size), sampler=SamplerWithoutReplacement())
    optim = torch.optim.Adam(loss_func.parameters(), lr=learning_rate)

    with open(os.path.join(run_dir, "train_log.csv"), "w", newline="") as f, open(os.path.join(run_dir, "eval_log.csv"), "w", newline="") as f_eval:
        writer = csv.writer(f)
        writer.writerow(["batch", "frames", "train_win_rate", "loss_objective", "loss_critic", "loss_entropy"])
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(["batch", "frames", "eval_combined_score", "eval_smoothed_score"] + [f"eval_player_count_{player_count}_win_rate" for player_count in player_counts])

        recent_scores = []
        best_smoothed_score = 0.0
        counter = 0
        stop = False

        for i, batch in enumerate(collector):
            with torch.no_grad():
                gae(batch)
            buffer.extend(batch.reshape(-1))

            for _ in range(epochs):                                    # epochs
                for _ in range(batch_size // minibatch_size):          # minibatches
                    mb = buffer.sample(minibatch_size)
                    losses = loss_func(mb)
                    loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_func.parameters(), 0.5)
                    optim.step()

            if self_play and (i + 1) % snapshot_every == 0:
                snapshot()
                if len(pool) > pool_size:
                    pool.pop(0)

            if (i + 1) % checkpoint_every == 0:
                torch.save(actor.state_dict(), os.path.join(run_dir, f"batch_{i + 1}.pt"))
                eval_dict = evaluate(actor, VAL_POOL, games=eval_games, seed=eval_seed, player_counts=player_counts)
                recent_scores.append(eval_dict["combined"])
                mean_score = sum(recent_scores[-smooth_window:]) / len(recent_scores[-smooth_window:])
                writer_eval.writerow([i, (i + 1) * batch_size, eval_dict["combined"], mean_score] + [eval_dict[player] for player in player_counts])
                f_eval.flush()
                if mean_score > best_smoothed_score + min_delta:
                    best_smoothed_score = mean_score
                    counter = 0
                    torch.save(actor.state_dict(), os.path.join(run_dir, "best.pt"))
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered after {i + 1} batches due to no improvement in score.")
                        stop = True

            wins = (batch["next", "reward"] > 0).sum()
            games = batch["next", "done"].sum()
            writer.writerow([i, (i + 1) * batch_size, (wins / games).item(), losses["loss_objective"].item(), losses["loss_critic"].item(), losses["loss_entropy"].item()])
            f.flush()
            print(f"batch {i}: win rate {(wins / games).item():.3f}, loss {loss.item():.3f}")
            if stop:
                break

    torch.save(actor.state_dict(), os.path.join(run_dir, "final.pt"))
    return actor

def evaluate(actor, opponent_factories, games=2000, seed=1, player_counts=(3, 4, 5, 6)):
    win_rates = {}
    opponent_factories = [bot_factory[bot] for bot in opponent_factories]

    for player_count in player_counts:
        env = ClueEnv(opponent_factories, seed=seed, player_counts=(player_count,))
        wins = 0
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            for _ in range(games):
                td = env.rollout(1000, policy=actor)
                wins += int(td["next", "reward"].sum().item() > 0)
        p_internal = wins / games
        # Compute the 95% confidence interval for the win rate.
        half_internal = 1.96 * math.sqrt(p_internal * (1 - p_internal) / games)
        print(f"player count {player_count}: win rate {p_internal:.3f} ± {half_internal:.3f} over {games} games")
        win_rates[player_count] = p_internal

    p = []
    for player_count in player_counts:
        p.append(win_rates[player_count] * player_count)
    combined_score = sum(p) / len(p)
    print(f"combined score across player counts {combined_score:.3f}")
    win_rates["combined"] = combined_score
    return win_rates

def load_actor(path="checkpoints/self_run_20261119-110408_0/final.pt", normalize=True):
    actor = make_actor(normalize)
    actor.load_state_dict(torch.load(path))
    return actor

if __name__ == "__main__":
    import sys
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", type=str, help="Mode to run: train, eval or plots", choices=["train", "eval", "plots"])
    argparser.add_argument("--method", type=str, default="bots", help="Method to run: bots or self play", choices=["bots", "self"])
    argparser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    argparser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to train for")
    args = argparser.parse_args()
    
    if args.mode == "train":
        train(max_frames=args.max_frames, run_name=f"{args.method}_run", seed=args.seed, self_play=(args.method == "self"))
    elif args.mode == "eval":
        evaluate(load_actor(), TEST_POOL)
    elif args.mode == "plots":
        from Analysis import plot_run
        plot_run("checkpoints/self_run_20261122-100206_5")
    else:
        argparser.print_help()