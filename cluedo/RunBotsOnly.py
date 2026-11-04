from random import seed, randint, random, shuffle

from Actor import CluedoActor, Knowledge, GullibleKnowledge, GoodMemoryKnowledge
from Strategies import Strategy, main_first, after_suggest_certain, main_random, main_gamble, main_bluff, after_suggest_gamble
from GameEngine import CluedoGameEngine

main_strategies = [main_first, main_random, main_gamble, main_bluff]
after_suggest_strategies = [after_suggest_certain, after_suggest_gamble]
knowledges = [Knowledge, GullibleKnowledge, GoodMemoryKnowledge]

nbots = 6
wins = [0] * nbots
games = 10000

for i in range(games):
    bots = [
    CluedoActor(Strategy(main_first, after_suggest_certain), GoodMemoryKnowledge),
    CluedoActor(Strategy(main_strategies[randint(0, len(main_strategies)-1)], after_suggest_strategies[randint(0, len(after_suggest_strategies)-1)]), Knowledge),
    CluedoActor(Strategy(main_strategies[randint(0, len(main_strategies)-1)], after_suggest_strategies[randint(0, len(after_suggest_strategies)-1)]), Knowledge),
    CluedoActor(Strategy(main_strategies[randint(0, len(main_strategies)-1)], after_suggest_strategies[randint(0, len(after_suggest_strategies)-1)]), Knowledge),
    CluedoActor(Strategy(main_strategies[randint(0, len(main_strategies)-1)], after_suggest_strategies[randint(0, len(after_suggest_strategies)-1)]), Knowledge),
    CluedoActor(Strategy(main_strategies[randint(0, len(main_strategies)-1)], after_suggest_strategies[randint(0, len(after_suggest_strategies)-1)]), Knowledge)
    ]
    slot = {id(bot): idx for idx, bot in enumerate(bots)}
    shuffle(bots)
    game = CluedoGameEngine(len(bots), randint(0, 1000000))
    game.start(bots)
    game.play()
    if game.winner is not None:
        wins[slot[id(game.winner)]] += 1

print("fixed bot (slot 0) winrate:", wins[0]/games)
print("random bots (slots 1-5) avg winrate:", sum(wins[1:])/(games*(nbots-1)))
print("no winner:", (games - sum(wins))/games)
print("fair share would be:", 1/nbots)