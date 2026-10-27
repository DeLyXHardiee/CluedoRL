import random
import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Binary, Categorical
from tensordict import TensorDict
from GameEngine import Game, SUGGEST, ACCUSE, END, AFTER_SUGGEST, MAIN
from Actor import Actor

class ClueEnv(EnvBase):

    def __init__(self, opponent_factories, seed, player_count):
        super().__init__()
        self.opponent_factories = opponent_factories
        self.player_count = player_count
        self.combination_count = 6 * 6 * 9 # suspects * weapons * rooms

        # lookup table
        actions = []
        for kind in (SUGGEST, ACCUSE):
            for suspect in range(0, 6):
                for weapon in range(6, 12):
                    for room in range(12, 21):
                        actions.append((kind, suspect, weapon, room))
        actions.append((END,))
        self.actions = actions

        # dictionary of action to index
        self.action_to_index = {action: index for index, action in enumerate(actions)}

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
            self.game = Game(self.player_count, self.rng.getrandbits(32))
            self.seat = self.rng.randint(0, self.player_count - 1)
            self.actor = Actor(self.seat, self.rng)
            actors = []
            for i in range(self.player_count):
                if i != self.seat:
                    opponent = self.rng.choice(self.opponent_factories)(self.rng)
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
        action = int(tensordict["action"])
        self.game.step(action)

        eliminated = self.game.eliminated[self.seat]
        if not eliminated:
            self.game.play(stop_at=self.seat)

        won = self.game.winner == self.seat
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
        knowledge = self.actor.get_knowledge()
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
        solve[3] = len(accusation_candidates) / self.combination_count # how many candidates remain in total

        # Game state
        state = torch.zeros(6, dtype=torch.float32)
        eliminated = torch.tensor(self.game.eliminated, dtype=torch.float32)
        eliminated = torch.roll(eliminated, shifts=-self.seat, dims=0)
        for i in range(len(eliminated)):
            if eliminated[i]:
                state[i] = 1.0
        state[4] = 1.0 if self.game.phase == AFTER_SUGGEST else 0.0
        state[5] = self.game.turn / self.game.max_turns

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