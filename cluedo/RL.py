import random
import torch
import math
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
from Actor import CluedoActor
from Strategies import RandomActor, GambleActor

class ClueEnv(EnvBase):

    def __init__(self, opponent_factories, seed, player_count):
        super().__init__()
        self.opponent_factories = opponent_factories
        self.player_count = player_count
        self.combination_count = 6 * 6 * 9 # suspects * weapons * rooms

        self.actions = build_actions()

        # dictionary of action to index
        self.action_to_index = {action: index for index, action in enumerate(self.actions)}

        self.observation_spec = Composite(
            observation=Unbounded(shape=(283,), dtype=torch.float32),
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
        self.set_seed(seed)

    def _set_seed(self, seed):
        self.rng = random.Random(seed)

    def _reset(self, tensordict=None):
        while True:
            self.game = CluedoGameEngine(self.player_count, self.rng.getrandbits(32))
            self.seat = self.rng.randint(0, self.player_count - 1)
            self.actor = CluedoActor()
            actors = []
            for i in range(self.player_count):
                if i != self.seat:
                    opponent = self.rng.choice(self.opponent_factories)()
                    actors.append(opponent)
                    continue
                actors.append(self.actor)

            self.game.start(actors)
            self.game.play(stop_at=self.seat)
            if not self.game.done:
                break

        self.observation = self.build_observation(self.actor.get_accusation_candidates())
        self.action_mask = self.build_action_mask(self.actor.get_accusation_candidates())

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
            self.game.play(stop_at=self.seat)

        won = self.game.winner is self.actor
        truncated = self.game.truncated
        terminated = eliminated or (self.game.done and not truncated)
        done = terminated or truncated
        reward = 1.0 if won else 0.0

        candidates = self.actor.get_accusation_candidates()
        observation = self.build_observation(candidates)
        action_mask = self.build_action_mask(candidates)

        tensordict = TensorDict({
            "observation": observation,
            "action_mask": action_mask,
            "done": torch.tensor([done], dtype=torch.bool),
            "terminated": torch.tensor([terminated], dtype=torch.bool),
            "truncated": torch.tensor([truncated], dtype=torch.bool),
            "reward": torch.tensor([reward], dtype=torch.float32)
        }, batch_size=())
        return tensordict

    def build_observation(self, accusation_candidates):
        # Notepad grid
        knowledge = torch.tensor(self.actor.get_knowledge(), dtype=torch.int64)
        grid = torch.roll(knowledge, shifts=-self.seat, dims=0)
        grid = torch.nn.functional.one_hot(grid, num_classes=3)
        grid = grid.flatten().float()
        
        # Envelope candidates
        candidates = self.actor.get_candidates()
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
        state = torch.zeros(6, dtype=torch.float32)
        eliminated = torch.tensor([p in self.game.eliminated_players for p in self.game.players], dtype=torch.float32)
        eliminated = torch.roll(eliminated, shifts=-self.seat, dims=0)
        for i in range(len(eliminated)):
            if eliminated[i]:
                state[i] = 1.0
        state[4] = 1.0 if self.game.phase == AFTER_SUGGEST else 0.0
        state[5] = self.game.turns_taken / self.game.max_turns

        encoded = torch.cat([grid, cand, solve, state], dim=0)
        return encoded

    def build_action_mask(self, accusation_candidates):
        action_mask = torch.zeros(len(self.actions), dtype=torch.bool)

        if self.game.phase == MAIN:
            action_mask[0:self.combination_count] = True

        for candidate in accusation_candidates:
            index = self.action_to_index[(ACCUSE, candidate[0], candidate[1], candidate[2])]
            action_mask[index] = True

        if self.game.phase == AFTER_SUGGEST:
            action_mask[-1] = True 

        assert action_mask.any(), "No valid actions available"
        return action_mask


class FactoredPolicy(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.body = MLP(in_features=283, out_features=45, num_cells=[256, 256])
        self.actions = build_actions()
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
                r_idx.append(45) # the always zero position
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

    def forward(self, observation):
        output = self.body(observation)
        zeros_column = torch.zeros_like(output[..., :1])
        output = torch.cat((output, zeros_column), dim=-1)
        return output[..., self.kind_idx] + output[..., self.s_idx] + output[..., self.w_idx] + output[..., self.r_idx] # one logit per slot in action table's order

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

def make_actor():
      module = TensorDictModule(FactoredPolicy(), in_keys=["observation"], out_keys=["logits"])
      return ProbabilisticActor(
          module,
          in_keys={"logits": "logits", "mask": "action_mask"},
          out_keys=["action"],
          distribution_class=MaskedCategorical,
          return_log_prob=True,
      )

def make_critic():
    net = MLP(in_features=283, out_features=1, num_cells=[256, 256])
    return ValueOperator(net, in_keys=["observation"])

def train(total_frames=3_000_000, seed=0):
    env = ClueEnv([CluedoActor, RandomActor, GambleActor], seed=seed, player_count=4)
    actor = make_actor() 
    critic = make_critic()

    # Collector is responsible for gathering experience from the environment using the actor policy.
    # GAE (Generalized Advantage Estimation) is used to compute advantage estimates for policy gradient updates.
    # ClipPPOLoss implements the PPO objective with clipping to stabilize training.
    collector = Collector(env, actor, frames_per_batch=4096, total_frames=total_frames)
    gae = GAE(gamma=0.99, lmbda=0.95, value_network=critic, average_gae=True)
    loss_func = ClipPPOLoss(actor, critic, clip_epsilon=0.2, entropy_coeff=0.01)
    buffer = ReplayBuffer(storage=LazyTensorStorage(4096), sampler=SamplerWithoutReplacement())
    optim = torch.optim.Adam(loss_func.parameters(), lr=3e-4)

    for i, batch in enumerate(collector):
          with torch.no_grad():
              gae(batch)
          buffer.extend(batch.reshape(-1))

          for _ in range(4):                      # epochs
              for _ in range(4096 // 512):        # minibatches
                  mb = buffer.sample(512)
                  losses = loss_func(mb)
                  loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                  optim.zero_grad()
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(loss_func.parameters(), 0.5)
                  optim.step()

          wins = batch["next", "reward"].sum()
          games = batch["next", "done"].sum()
          print(f"batch {i}: win rate {(wins / games).item():.3f}, loss {loss.item():.3f}")

    torch.save(actor.state_dict(), "learner.pt")
    return actor

def evaluate(actor, opponent_factories, games=2000, seed=1):
    env = ClueEnv(opponent_factories, seed=seed, player_count=4)
    wins = 0
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(games):
            td = env.rollout(1000, policy=actor)
            wins += int(td["next", "reward"].sum().item() > 0)
    p = wins / games
    # Compute the 95% confidence interval for the win rate.
    half = 1.96 * math.sqrt(p * (1 - p) / games)
    print(f"win rate {p:.3f} ± {half:.3f} over {games} games")
    return p

def load_actor(path="learner.pt"):
    actor = make_actor()
    actor.load_state_dict(torch.load(path))
    return actor

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(load_actor(), [CluedoActor, RandomActor, GambleActor])
    else:
        train()