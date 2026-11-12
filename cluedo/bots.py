from random import seed, randint, random, shuffle

from Actor import CluedoActor, Knowledge, GullibleKnowledge, GoodMemoryKnowledge
from Strategies import Strategy, main_first, main_random, main_gamble, main_bluff, after_suggest_certain, after_suggest_gamble

# Bots for evaluation

main_strategies = [main_first, main_random, main_gamble, main_bluff]
after_suggest_strategies = [after_suggest_certain, after_suggest_gamble]
knowledges = [Knowledge, GullibleKnowledge, GoodMemoryKnowledge]

randombot = CluedoActor(
    Strategy(main_strategies[randint(0, len(main_strategies)-1)], after_suggest_strategies[randint(0, len(after_suggest_strategies)-1)]), 
    knowledges[randint(0, len(knowledges)-1)])

first_certain_goodMemory_bot = CluedoActor(
    Strategy(main_first, after_suggest_certain),
    GoodMemoryKnowledge)

first_certain_gullible_bot = CluedoActor(
    Strategy(main_first, after_suggest_certain),
    GullibleKnowledge)

first_certain_knowledge_bot = CluedoActor(
    Strategy(main_first, after_suggest_certain),
    Knowledge)

first_gamble_goodMemory_bot = CluedoActor(
    Strategy(main_first, after_suggest_gamble),
    GoodMemoryKnowledge)

first_gamble_gullible_bot = CluedoActor(
    Strategy(main_first, after_suggest_gamble),
    GullibleKnowledge)

first_gamble_knowledge_bot = CluedoActor(
    Strategy(main_first, after_suggest_gamble),
    Knowledge)

random_certain_goodMemory_bot = CluedoActor(
    Strategy(main_random, after_suggest_certain),
    GoodMemoryKnowledge)

random_certain_gullible_bot = CluedoActor(
    Strategy(main_random, after_suggest_certain),
    GullibleKnowledge)

random_certain_knowledge_bot = CluedoActor(
    Strategy(main_random, after_suggest_certain),
    Knowledge)

random_gamble_goodMemory_bot = CluedoActor(
    Strategy(main_random, after_suggest_gamble),
    GoodMemoryKnowledge)

random_gamble_gullible_bot = CluedoActor(
    Strategy(main_random, after_suggest_gamble),
    GullibleKnowledge)

random_gamble_knowledge_bot = CluedoActor(
    Strategy(main_random, after_suggest_gamble),
    Knowledge)

gamble_certain_goodMemory_bot = CluedoActor(
    Strategy(main_gamble, after_suggest_certain),
    GoodMemoryKnowledge)

gamble_certain_gullible_bot = CluedoActor(
    Strategy(main_gamble, after_suggest_certain),
    GullibleKnowledge)

gamble_certain_knowledge_bot = CluedoActor(
    Strategy(main_gamble, after_suggest_certain),
    Knowledge)

gamble_gamble_goodMemory_bot = CluedoActor(
    Strategy(main_gamble, after_suggest_gamble),
    GoodMemoryKnowledge)

gamble_gamble_gullible_bot = CluedoActor(
    Strategy(main_gamble, after_suggest_gamble),
    GullibleKnowledge)

gamble_gamble_knowledge_bot = CluedoActor(
    Strategy(main_gamble, after_suggest_gamble),
    Knowledge)

bluff_certain_goodMemory_bot = CluedoActor(
    Strategy(main_bluff, after_suggest_certain),
    GoodMemoryKnowledge)

bluff_certain_gullible_bot = CluedoActor(
    Strategy(main_bluff, after_suggest_certain),
    GullibleKnowledge)

bluff_certain_knowledge_bot = CluedoActor(
    Strategy(main_bluff, after_suggest_certain),
    Knowledge)

bluff_gamble_goodMemory_bot = CluedoActor(
    Strategy(main_bluff, after_suggest_gamble),
    GoodMemoryKnowledge)

bluff_gamble_gullible_bot = CluedoActor(
    Strategy(main_bluff, after_suggest_gamble),
    GullibleKnowledge)

bluff_gamble_knowledge_bot = CluedoActor(
    Strategy(main_bluff, after_suggest_gamble),
    Knowledge)