import re

from Actor import CluedoActor
from GameEngine import ActionTypes, Phase
import random

# Class to create strategy instances for the game. 
# By running Strategy.strategy an actor should receive an action based on the current game state.
# Each strategy should be hardcoded and based on strings determine which action will be taken
# An actor should take a strategy as parameter

class Strategy:
    def __init__(
            self,
            main: callable,
            after_suggest: callable
        ):
        self.main = main
        self.after_suggest = after_suggest

    def getStrategy(self, game_state, knowledge, seed):
        phase = game_state["phase"]
        if phase == Phase.MAIN:
            return self.main(knowledge, seed)
        elif phase == Phase.AFTER_SUGGEST:
            return self.after_suggest(knowledge, seed)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
def main_first(knowledge, seed):
    suspects,weapons,rooms = _cards_setup(knowledge)
    guess = (suspects[0], weapons[0], rooms[0])
    if len(suspects) == len(weapons) == len(rooms) == 1:
        return (ActionTypes.ACCUSE, *guess)
    return (ActionTypes.SUGGEST, *guess)

def main_random(knowledge, seed):
    suspects,weapons,rooms = _cards_setup(knowledge)
    rng = random.Random(seed)
    s = rng.randrange(len(suspects))
    w = rng.randrange(len(weapons))
    r = rng.randrange(len(rooms))
    guess = (suspects[s], weapons[w], rooms[r])
    if len(suspects) == len(weapons) == len(rooms) == 1:
        return (ActionTypes.ACCUSE, *guess)
    return (ActionTypes.SUGGEST, *guess)

def main_gamble(knowledge, seed, threshold=1):
    suspects,weapons,rooms = _cards_setup(knowledge)
    rng = random.Random(seed)
    s = rng.randrange(len(suspects))
    w = rng.randrange(len(weapons))
    r = rng.randrange(len(rooms))
    guess = (suspects[s], weapons[w], rooms[r])

    accusation_candidates = knowledge.accusation_candidates()
    if len(accusation_candidates) <= threshold:
        rnd = rng.randrange(len(accusation_candidates))
        guess = accusation_candidates[rnd]
        return (ActionTypes.ACCUSE, *guess)
    return (ActionTypes.SUGGEST, *guess)

# Use one of owned cards in suggestion based on frequency
def main_bluff(knowledge, seed, frequency=0.5):
    suspects,weapons,rooms = _cards_setup(knowledge)
    rng = random.Random(seed)
    if rng.random() < frequency:
        own_cards = knowledge.hand
        card = rng.choice(own_cards)
        if card < 6:
            suspects = [card]
        elif 6 <= card < 12:
            weapons = [card]
        else:
            rooms = [card]
    s = rng.randrange(len(suspects))
    w = rng.randrange(len(weapons))
    r = rng.randrange(len(rooms))
    guess = (suspects[s], weapons[w], rooms[r])
    return (ActionTypes.SUGGEST, *guess)

def _cards_setup(knowledge):
    candidates = knowledge.candidates()
    suspects = [c for c in candidates if c < 6]
    weapons = [c for c in candidates if 6 <= c < 12]
    rooms = [c for c in candidates if c >= 12]
    return suspects, weapons, rooms

def after_suggest_certain(knowledge, seed):
    suspects,weapons,rooms = _cards_setup(knowledge)
    if len(suspects) == len(weapons) == len(rooms) == 1:
        guess = (suspects[0], weapons[0], rooms[0])
        return (ActionTypes.ACCUSE, *guess)
    return (ActionTypes.END,)

def after_suggest_gamble(knowledge, seed, threshold=1):
    suspects,weapons,rooms = _cards_setup(knowledge)
    rng = random.Random(seed)
    s = rng.randrange(len(suspects))
    w = rng.randrange(len(weapons))
    r = rng.randrange(len(rooms))
    guess = (suspects[s], weapons[w], rooms[r])

    accusation_candidates = knowledge.accusation_candidates()
    if len(accusation_candidates) <= threshold:
        rnd = rng.randrange(len(accusation_candidates))
        guess = accusation_candidates[rnd]
        return (ActionTypes.ACCUSE, *guess)
    return (ActionTypes.END,)