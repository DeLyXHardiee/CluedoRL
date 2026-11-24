import random

from Actor import CluedoActor, Knowledge, GullibleKnowledge, GoodMemoryKnowledge
from Strategies import Strategy, main_first, main_random, main_gamble, main_bluff, after_suggest_certain, after_suggest_gamble

# Bots for evaluation

main_strategies = [main_first, main_random, main_gamble, main_bluff]
after_suggest_strategies = [after_suggest_certain, after_suggest_gamble]
knowledges = [Knowledge, GullibleKnowledge, GoodMemoryKnowledge]


def randomBot(seed):
    seed = random.Random(seed)
    return lambda: CluedoActor(
        Strategy(seed.choice(main_strategies), seed.choice(after_suggest_strategies)),
        seed.choice(knowledges))

def all_bots():
    bots_list = []
    for main in main_strategies:
        for after_suggest in after_suggest_strategies:
            for knowledge in knowledges:
                bots_list.append(bot(main, after_suggest, knowledge))
    return bots_list

def bot(main, after_suggest, knowledge):
    return lambda: CluedoActor(Strategy(main, after_suggest), knowledge)

first_certain_goodMemory_bot = bot(main_first, after_suggest_certain, GoodMemoryKnowledge)
first_certain_gullible_bot = bot(main_first, after_suggest_certain, GullibleKnowledge)
first_certain_knowledge_bot = bot(main_first, after_suggest_certain, Knowledge)
first_gamble_goodMemory_bot = bot(main_first, after_suggest_gamble, GoodMemoryKnowledge)
first_gamble_gullible_bot = bot(main_first, after_suggest_gamble, GullibleKnowledge)
first_gamble_knowledge_bot = bot(main_first, after_suggest_gamble, Knowledge)

random_certain_goodMemory_bot = bot(main_random, after_suggest_certain, GoodMemoryKnowledge)
random_certain_gullible_bot = bot(main_random, after_suggest_certain, GullibleKnowledge)
random_certain_knowledge_bot = bot(main_random, after_suggest_certain, Knowledge)
random_gamble_goodMemory_bot = bot(main_random, after_suggest_gamble, GoodMemoryKnowledge)
random_gamble_gullible_bot = bot(main_random, after_suggest_gamble, GullibleKnowledge)
random_gamble_knowledge_bot = bot(main_random, after_suggest_gamble, Knowledge)

gamble_certain_goodMemory_bot = bot(main_gamble, after_suggest_certain, GoodMemoryKnowledge)
gamble_certain_gullible_bot = bot(main_gamble, after_suggest_certain, GullibleKnowledge)
gamble_certain_knowledge_bot = bot(main_gamble, after_suggest_certain, Knowledge)
gamble_gamble_goodMemory_bot = bot(main_gamble, after_suggest_gamble, GoodMemoryKnowledge)
gamble_gamble_gullible_bot = bot(main_gamble, after_suggest_gamble, GullibleKnowledge)
gamble_gamble_knowledge_bot = bot(main_gamble, after_suggest_gamble, Knowledge)

bluff_certain_goodMemory_bot = bot(main_bluff, after_suggest_certain, GoodMemoryKnowledge)
bluff_certain_gullible_bot = bot(main_bluff, after_suggest_certain, GullibleKnowledge)
bluff_certain_knowledge_bot = bot(main_bluff, after_suggest_certain, Knowledge)
bluff_gamble_goodMemory_bot = bot(main_bluff, after_suggest_gamble, GoodMemoryKnowledge)
bluff_gamble_gullible_bot = bot(main_bluff, after_suggest_gamble, GullibleKnowledge)
bluff_gamble_knowledge_bot = bot(main_bluff, after_suggest_gamble, Knowledge)

class BotList:

    def __init__(self, name, bots):
        self.name = name
        self.bots = bots

    def get_bots(self):
        return self.bots


TEST_POOL = [
    random_gamble_goodMemory_bot,
    bluff_gamble_gullible_bot,
    random_certain_knowledge_bot,
]

FALLS_FOR_BLUFFS = [
    bluff_certain_gullible_bot,
    bluff_gamble_gullible_bot,
    gamble_gamble_gullible_bot,
    first_gamble_gullible_bot,
    gamble_certain_gullible_bot,
    first_certain_gullible_bot,
    random_certain_gullible_bot,
    random_gamble_gullible_bot,
]

ALWAYS_FIRST = [
    first_certain_goodMemory_bot,
    first_certain_gullible_bot,
    first_certain_knowledge_bot,
    first_gamble_goodMemory_bot,
    first_gamble_gullible_bot,
    first_gamble_knowledge_bot,
]

USES_BLUFFS = [
    bluff_certain_goodMemory_bot,
    bluff_certain_gullible_bot,
    bluff_certain_knowledge_bot,
    bluff_gamble_goodMemory_bot,
    bluff_gamble_gullible_bot,
    bluff_gamble_knowledge_bot,
]

GOOD_MEMORY = [
    random_certain_goodMemory_bot,
    random_gamble_goodMemory_bot,
    gamble_certain_goodMemory_bot,
    gamble_gamble_goodMemory_bot,
    bluff_certain_goodMemory_bot,
    bluff_gamble_goodMemory_bot,
    first_certain_goodMemory_bot,
    first_gamble_goodMemory_bot,
]