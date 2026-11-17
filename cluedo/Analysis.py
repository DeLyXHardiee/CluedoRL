import torch
import csv
import random
import os
from GameEngine import CluedoGameEngine, SUGGEST, ACCUSE, END
from RL import RLActor, load_actor, bot_factory, TEST_POOL
from torchrl.envs.utils import set_exploration_type, ExplorationType

class RecordingEngine(CluedoGameEngine):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.action_log = []

    def step(self, action):
        player = self.seatToPlayer[self.current_turn]
        
        own_cards = 0
        for own_card in player.hand:
            if own_card in action[1:]:
                own_cards += 1

        candidates_named = 0
        candidates = player.get_candidates()
        for card in candidates:
            if card in action[1:]:
                candidates_named += 1

        temp = {
            "turn": self.turns_taken,
            "seat": self.current_turn,
            "phase": self.phase,
            "kind": action[0],
            "cards": action[1:],
            "own_cards": own_cards,
            "candidates_named": candidates_named,
            "accusation_candidates": len(player.get_accusation_candidates()),
            "label": getattr(player, "label", "unknown"),
        }

        super().step(action)
        if action[0] == ACCUSE:
            temp["correct"] = player == self.winner
        self.action_log.append(temp)

def test_pool_lineup(label, policy, n):
    def make(rng):
        lineup = []
        model = (label, lambda: RLActor(policy))
        lineup.append(model)
        for i in range(n-1):
            name = rng.choice(TEST_POOL)
            lineup.append((name, bot_factory[name]))
        return lineup
    return make

def play_games(make_lineup, games, seed):
    action_rows = []
    game_rows = []

    rng = random.Random(seed)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for i in range(games):
            game = RecordingEngine(seed=seed + i)
            lineup = make_lineup(rng)
            actors = []
            for label, factory in lineup:
                actor = factory()
                actor.label = label
                actors.append(actor)
            game.start(actors)
            game.play()
            for row in game.action_log:
                row["game"] = i
            action_rows.extend(game.action_log)
            game_rows.append({"game": i, "player_count": len(game.players), "winner": game.winner.label if game.winner else None, "truncated": game.truncated, "turns_taken": game.turns_taken, "eliminated": [p.label for p in game.eliminated_players]})
    return action_rows, game_rows

def save_rows(rows, path):
    with open(path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        if "kind" in rows[0] and "correct" not in fieldnames:
            fieldnames.append("correct") 
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if "kind" in row:
                row["kind"] = row["kind"].name
            if "phase" in row:
                row["phase"] = row["phase"].name
            if "cards" in row:
                row["cards"] = " ".join(str(c) for c in row["cards"])
            if "eliminated" in row:
                row["eliminated"] = " ".join(row["eliminated"])
            writer.writerow(row)
        
def analyse_model(path, label, out_dir, games=2000, seed=1):
    policy = load_actor(path)
    os.makedirs(out_dir, exist_ok=True)
    for n in (3, 4, 5, 6):
        action_rows, game_rows = play_games(test_pool_lineup(label, policy, n), games, seed)
        actions_path = os.path.join(out_dir, f"test_{n}_actions.csv")
        game_path = os.path.join(out_dir, f"test_{n}_game.csv")
        save_rows(action_rows, actions_path)
        save_rows(game_rows, game_path)