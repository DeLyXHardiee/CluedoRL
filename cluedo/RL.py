import random
import torch
from torchrl.envs import EnvBase
from torchrl.data import TensorDict
from cluedo.game import Game, Actor

class ClueEnv(EnvBase):

    def __init__(self, opponent_factories, seed, player_count):
        super().__init__()
        self.opponent_factories = opponent_factories
        # self.observation_spec["observation"] = something
        # self.observation_spec["action_mask"] = something
        # self.action_spec = something
        # self.reward_spec = something
        # done_spec["done"] = something
        # done_spec["terminated"] = something
        # done_spec["truncated"] = something
        self.set_seed(seed)

    def _set_seed(self, seed):
        self.rng = random.Random(seed)

    def _reset(self, tensordict=None):
        self.game = Game(player_count, self.rng.getrandbits(32))
        self.seat = self.rng.randint(0, player_count - 1)
        self.actor = Actor(self.seat, self.rng)
        actors = []
        for i in range(player_count):
            if i != self.seat:
                random_index = self.rng.randint(0, len(self.opponent_factories) - 1)
                opponent = self.opponent_factories[random_index](self.rng)
                actors.append(opponent)
                continue
            actors.append(self.actor)

        self.game.start_game(actors)
        self.game.play(stop_at=self.seat)


        tensordict = TensorDict()
        return tensordict

    def _step(self, tensordict=None):
        encoded_action = int(tensordict["action"])
        decoded_action = decode_action(encoded_action)
        reward, done = self.game.step(decoded_action)
        next_state = self.game.get_state()

        tensordict = TensorDict()
        return tensordict

    def build_observation(self):
        # Notepad grid
        knowledge = self.actor.get_knowledge()
        grid = torch.roll(knowledge, shifts=-self.seat, dims=0)
        grid = torch.nn.functional.one_hot(grid, num_classes=3)
        grid = grid.flatten().float()
        
        # Envelope candidates
        candidates = knowledge.candidates
        cand = torch.zeros(21, dtype=torch.float32)
        for i in range(len(candidates)):
            cand[candidates[i]] = 1.0
        
        # How close learner is to solving
        suspects = grid[:6]
        weapons = grid[6:12]
        rooms = grid[12:21]
        solve = torch.zeros(4, dtype=torch.float32)
        solve[0] = suspects.sum() / 6.0 # how many candidates remain in suspects
        solve[1] = weapons.sum() / 6.0 # ... weapons
        solve[2] = rooms.sum() / 9.0 # ... rooms
        solve[3] = len(candidates) / 324.0 # how many candidates remain in total

        # Game state
        state = torch.zeros(6, dtype=torch.float32)
        eliminated = self.game.eliminated
        eliminated = torch.roll(eliminated, shifts=-self.seat, dims=0)
        for i in range(len(eliminated)):
            if eliminated[i]:
                state[i] = 1.0
        state[5] = 1.0 if self.game.phase == "AFTER_SUGGEST" else 0.0 # How do we get game phase?
        state[6] = self.game.turn / self.game.max_turns

        encoded = torch.cat([grid, cand, solve, state], dim=0)
        return encoded

    def build_action_mask(self):
        # Implement the logic to build the action mask based on the current game state
        pass