import random

from Actor import CluedoActor, Knowledge, GullibleKnowledge, NoInferenceKnowledge
from Strategies import Strategy, main_first, main_random, main_gamble, main_bluff, after_suggest_certain, after_suggest_gamble

# Bots for evaluation

class BotList:

    def __init__(self, name, bots):
        self.name = name
        self.bots = bots

    def get_bots(self):
        return self.bots

main_strategies = [main_first, main_random, main_gamble, main_bluff]
after_suggest_strategies = [after_suggest_certain, after_suggest_gamble]
knowledges = [Knowledge, GullibleKnowledge, NoInferenceKnowledge]


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
    def make():
        return CluedoActor(Strategy(main, after_suggest), knowledge)

    main_name = main.__name__.removeprefix("main_")
    after_name = after_suggest.__name__.removeprefix("after_suggest_")
    knowledge_name = knowledge.__name__.removesuffix("Knowledge").lower() or "knowledge"
    make.label = f"{main_name}/{after_name}/{knowledge_name}"
    return make

first_certain_gullible_bot = bot(main_first, after_suggest_certain, GullibleKnowledge)
first_certain_knowledge_bot = bot(main_first, after_suggest_certain, Knowledge)
first_certain_no_inference_bot = bot(main_first, after_suggest_certain, NoInferenceKnowledge)
first_gamble_gullible_bot = bot(main_first, after_suggest_gamble, GullibleKnowledge)
first_gamble_knowledge_bot = bot(main_first, after_suggest_gamble, Knowledge)
first_gamble_no_inference_bot = bot(main_first, after_suggest_gamble, NoInferenceKnowledge)

random_certain_gullible_bot = bot(main_random, after_suggest_certain, GullibleKnowledge)
random_certain_knowledge_bot = bot(main_random, after_suggest_certain, Knowledge)
random_certain_no_inference_bot = bot(main_random, after_suggest_certain, NoInferenceKnowledge)
random_gamble_gullible_bot = bot(main_random, after_suggest_gamble, GullibleKnowledge)
random_gamble_knowledge_bot = bot(main_random, after_suggest_gamble, Knowledge)
random_gamble_no_inference_bot = bot(main_random, after_suggest_gamble, NoInferenceKnowledge)

gamble_certain_gullible_bot = bot(main_gamble, after_suggest_certain, GullibleKnowledge)
gamble_certain_knowledge_bot = bot(main_gamble, after_suggest_certain, Knowledge)
gamble_certain_no_inference_bot = bot(main_gamble, after_suggest_certain, NoInferenceKnowledge)
gamble_gamble_gullible_bot = bot(main_gamble, after_suggest_gamble, GullibleKnowledge)
gamble_gamble_knowledge_bot = bot(main_gamble, after_suggest_gamble, Knowledge)
gamble_gamble_no_inference_bot = bot(main_gamble, after_suggest_gamble, NoInferenceKnowledge)

bluff_certain_no_inference_bot = bot(main_bluff, after_suggest_certain, NoInferenceKnowledge)
bluff_certain_gullible_bot = bot(main_bluff, after_suggest_certain, GullibleKnowledge)
bluff_certain_knowledge_bot = bot(main_bluff, after_suggest_certain, Knowledge)
bluff_gamble_no_inference_bot = bot(main_bluff, after_suggest_gamble, NoInferenceKnowledge)
bluff_gamble_gullible_bot = bot(main_bluff, after_suggest_gamble, GullibleKnowledge)
bluff_gamble_knowledge_bot = bot(main_bluff, after_suggest_gamble, Knowledge)

botMap = {
    "first_gamble_gullible_bot": first_gamble_gullible_bot,
    "first_gamble_knowledge_bot": first_gamble_knowledge_bot,
    "first_gamble_no_inference_bot": first_gamble_no_inference_bot,
    "first_certain_gullible_bot": first_certain_gullible_bot,
    "first_certain_knowledge_bot": first_certain_knowledge_bot,
    "first_certain_no_inference_bot": first_certain_no_inference_bot,
    "random_certain_gullible_bot": random_certain_gullible_bot,
    "random_certain_knowledge_bot": random_certain_knowledge_bot,
    "random_certain_no_inference_bot": random_certain_no_inference_bot,
    "random_gamble_gullible_bot": random_gamble_gullible_bot,
    "random_gamble_knowledge_bot": random_gamble_knowledge_bot,
    "random_gamble_no_inference_bot": random_gamble_no_inference_bot,
    "gamble_certain_gullible_bot": gamble_certain_gullible_bot,
    "gamble_certain_knowledge_bot": gamble_certain_knowledge_bot,
    "gamble_certain_no_inference_bot": gamble_certain_no_inference_bot,
    "gamble_gamble_gullible_bot": gamble_gamble_gullible_bot,
    "gamble_gamble_knowledge_bot": gamble_gamble_knowledge_bot,
    "gamble_gamble_no_inference_bot": gamble_gamble_no_inference_bot,
    "bluff_certain_gullible_bot": bluff_certain_gullible_bot,
    "bluff_certain_knowledge_bot": bluff_certain_knowledge_bot,
    "bluff_certain_no_inference_bot": bluff_certain_no_inference_bot,
    "bluff_gamble_gullible_bot": bluff_gamble_gullible_bot,
    "bluff_gamble_knowledge_bot": bluff_gamble_knowledge_bot,
    "bluff_gamble_no_inference_bot": bluff_gamble_no_inference_bot,
}

TEST_POOL = [
    bluff_gamble_gullible_bot,
    random_certain_knowledge_bot,
    bluff_gamble_no_inference_bot,
    random_certain_no_inference_bot,
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
    first_certain_gullible_bot,
    first_certain_knowledge_bot,
    first_certain_no_inference_bot,
    first_gamble_gullible_bot,
    first_gamble_knowledge_bot,
    first_gamble_no_inference_bot,


]

USES_BLUFFS = [
    bluff_certain_gullible_bot,
    bluff_certain_knowledge_bot,
    bluff_gamble_gullible_bot,
    bluff_gamble_knowledge_bot,
    bluff_certain_no_inference_bot,
    bluff_gamble_no_inference_bot,
]

NORMAL_KNOWLEDGE = [
    random_certain_knowledge_bot,
    random_gamble_knowledge_bot,
    gamble_certain_knowledge_bot,
    gamble_gamble_knowledge_bot,
    bluff_certain_knowledge_bot,
    bluff_gamble_knowledge_bot,
    first_certain_knowledge_bot,
    first_gamble_knowledge_bot,
]

NO_INFERENCE = [
    random_certain_no_inference_bot,
    random_gamble_no_inference_bot,
    gamble_certain_no_inference_bot,
    gamble_gamble_no_inference_bot,
    bluff_certain_no_inference_bot,
    bluff_gamble_no_inference_bot,
    first_certain_no_inference_bot,
    first_gamble_no_inference_bot,
]