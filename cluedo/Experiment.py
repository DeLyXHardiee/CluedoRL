from random import Random, seed, randint, random, shuffle

from Actor import CluedoActor, Knowledge, GullibleKnowledge, GoodMemoryKnowledge
from Strategies import Strategy, main_first, main_random, main_gamble, main_bluff, after_suggest_certain, after_suggest_gamble
from GameEngine import CluedoGameEngine
from RL import RLActor
import bots
from functools import partial

main_strategies = [main_first, main_random, main_gamble, main_bluff]
after_suggest_strategies = [after_suggest_certain, after_suggest_gamble]
knowledges = [Knowledge, GullibleKnowledge, GoodMemoryKnowledge]

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
            game = CluedoGameEngine(seed=i)
            if i % 1000 == 0:
                print(f"Starting game {i}")
            game.start(self.players)
            game.play()
            self.games.append(game.get_game_dict())
        return self.games

def run_game_analysis(experiment):
    games = experiment.run()
    stats = {
        "total_games": len(games),
        "wins": {player: 0 for player in experiment.players},
        "eliminations": {player: 0 for player in experiment.players}
    }
    for game in games:
        if "winner" in game and game["winner"] is not None:
            stats["wins"][game["winner"]] += 1
        for eliminated in game.get("eliminated_players", []):
            stats["eliminations"][eliminated] += 1
    print(stats)
    return stats, games

def run_rl_vs_randombots(rl_bot, num_random_bots=5, num_games=1000, seed=0):
    players = [rl_bot] + [bots.randomBot(Random(seed)) for _ in range(num_random_bots)]
    experiment = Experiment(name="RL vs Random Bot", players=players, seed=seed, num_games=num_games)
    return run_game_analysis(experiment)

def run_rl_vs_fixedbot(rl_bot, fixed_bot, num_bot=5, num_games=1000, seed=0):
    players = [rl_bot] + [fixed_bot() for _ in range(num_bot)]
    experiment = Experiment(name="RL vs Fixed Bot", players=players, seed=seed, num_games=num_games)
    return run_game_analysis(experiment)

def run_rl_vs_botList(rl_bot, bot_list, num_games=1000, seed=0):
    players = [rl_bot] + bot_list
    experiment = Experiment(name="RL vs Bot List", players=players, seed=seed, num_games=num_games)
    return run_game_analysis(experiment)

if __name__ == "__main__":
    #RLAgent = RLActor()
    smartActor = CluedoActor(Strategy(main_random, partial(after_suggest_gamble, threshold=3)), GoodMemoryKnowledge)
    run_rl_vs_randombots(smartActor)
    #run_rl_vs_fixedbot(smartActor, bots.bluff_certain_goodMemory_bot)
    #run_rl_vs_botList(smartActor, [bots.bluff_certain_goodMemory_bot(), bots.first_certain_goodMemory_bot()])
