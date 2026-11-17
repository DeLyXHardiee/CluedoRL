from pathlib import Path
import math
import random

from Actor import CluedoActor, Knowledge, GullibleKnowledge, GoodMemoryKnowledge
from Strategies import Strategy, main_first, main_random, main_gamble, main_bluff, after_suggest_certain, after_suggest_gamble
from GameEngine import CluedoGameEngine
from RL import RLActor, load_actor
import bots
from functools import partial
    
main_strategies = [main_first, main_random, main_gamble, main_bluff]
after_suggest_strategies = [after_suggest_certain, after_suggest_gamble]
knowledges = [Knowledge, GullibleKnowledge, GoodMemoryKnowledge]

#From RL.py
TEST_POOL = [
    bots.random_gamble_goodMemory_bot(),
    bots.bluff_gamble_gullible_bot(),
    bots.random_certain_knowledge_bot(),
]

class Experiment:
    def __init__(self, name="", players=None, seed=0, num_games=1):
        self.name = name
        self.players = players if players is not None else []
        self.player_count = len(self.players)
        self.seed = seed
        self.num_games = num_games
        self.games = []

    def run(self):
        for i in range(self.num_games):
            game = CluedoGameEngine(seed=self.seed + i)
            #if i % 1000 == 0:
            #    print(f"Starting game {i}")
            game.start(self.players)
            game.play()
            self.games.append(game.get_game_dict())
        return self.games

def evaluate(experiment):
    print(f"Experiment {experiment.name}")
    games = experiment.run()
    stats = {
        "total_games": len(games),
        "wins": {player: 0 for player in experiment.players},
        "eliminations": {player: 0 for player in experiment.players},
        "win_rates": {},
        "win_rate_95_half_widths": {},
    }
    for game in games:
        if "winner" in game and game["winner"] is not None:
            stats["wins"][game["winner"]] += 1
        for eliminated in game.get("eliminated_players", []):
            stats["eliminations"][eliminated] += 1
    for index, player in enumerate(experiment.players):
        if games:
            rate = stats["wins"][player] / len(games)
            half_width = 1.96 * math.sqrt(rate * (1 - rate) / len(games))
            stats["win_rates"][player] = rate
            stats["win_rate_95_half_widths"][player] = half_width
            print(f"Player {index} ({player}): win rate {rate:.3f} +/- {half_width:.3f} over {len(games)} games and {stats['eliminations'][player]} eliminations")
        else:
            stats["win_rates"][player] = None
            stats["win_rate_95_half_widths"][player] = None
    #print(stats)
    print()
    return stats, games

def run_experiment(name, players, seed=0, num_games=1000):
    experiment = Experiment(name=name, players=players, seed=seed, num_games=num_games)
    return evaluate(experiment)

def get_random_bots_list(n, seed=0):
    botList = []
    for i in range(n):
        botList.append(bots.randomBot(random.Random(seed + i)))
    return botList

def get_test_pool_list(numPlayers=6,seed=0):
    rng = random.Random(seed)
    return [TEST_POOL[rng.randint(0, len(TEST_POOL) - 1)] for _ in range(numPlayers)]

def run_all_experiments(seed=0, numgames=1000):
    rl_agents = []
    bot_lists = []
    for i in range(2,6):
        bot_lists.append(bots.BotList(f"{i} Random Bots", get_random_bots_list(i, seed=seed+i)))
        bot_lists.append(bots.BotList(f"{i} Test Pool", get_test_pool_list(i, seed=seed+i)))

    runs = [line for line in (Path(__file__).parent/"runs.txt").read_text().splitlines()]
    for run in runs:
        rl_agents.append(RLActor(load_actor(Path(__file__).parent / "checkpoints" / run / "best.pt")))

    for i, agent in enumerate(rl_agents):
        for j, bot_list in enumerate(bot_lists):
            run_experiment(f"{runs[i]} vs {bot_list.name}", [agent] + bot_list.get_bots(), seed=seed+i+j, num_games=numgames)

if __name__ == "__main__":
    seed = 42
    numgames = 1000
    run_all_experiments(seed=seed, numgames=numgames)
