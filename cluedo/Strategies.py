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
            main: str = "random",
            after_suggest: str = "certain"
        ):
        self.main = main
        self.after_suggest = after_suggest

    def getStrategy(self, game_state, knowledge, seed):
        phase = game_state["phase"]
        if phase == Phase.MAIN:
            return self._main_strategy(knowledge, seed)
        elif phase == Phase.AFTER_SUGGEST:
            return self._after_suggest_strategy(knowledge, seed)
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def _main_strategy(self, knowledge, seed):
        match self.main:
            case "random":
                return self._main_random(knowledge, seed)
            case "gamble":
                return self._main_gamble(knowledge, seed)
            case "first":
                return self._main_first(knowledge)
            case _:
                return self.main

    def _after_suggest_strategy(self, knowledge, seed):
        match self.after_suggest:
            case "certain":
                return self._after_suggest_certain(knowledge)
            case _:
                return (ActionTypes.END,)
    
    def _main_first(self, knowledge):
        candidates = knowledge.candidates()
        suspects = [c for c in candidates if c < 6]
        weapons = [c for c in candidates if 6 <= c < 12]
        rooms = [c for c in candidates if c >= 12]
        guess = (suspects[0], weapons[0], rooms[0])
        if len(suspects) == len(weapons) == len(rooms) == 1:
            return (ActionTypes.ACCUSE, *guess)
        return (ActionTypes.SUGGEST, *guess)

    def _main_random(self, knowledge, seed):
        candidates = knowledge.candidates()
        suspects = [c for c in candidates if c < 6]
        weapons = [c for c in candidates if 6 <= c < 12]
        rooms = [c for c in candidates if c >= 12]
        rng = random.Random(seed)
        s = rng.randrange(len(suspects))
        w = rng.randrange(len(weapons))
        r = rng.randrange(len(rooms))
        guess = (suspects[s], weapons[w], rooms[r])
        if len(suspects) == len(weapons) == len(rooms) == 1:
            return (ActionTypes.ACCUSE, *guess)
        return (ActionTypes.SUGGEST, *guess)

    def _main_gamble(self, knowledge, seed):
        candidates = knowledge.candidates()
        suspects = [c for c in candidates if c < 6]
        weapons = [c for c in candidates if 6 <= c < 12]
        rooms = [c for c in candidates if c >= 12]
        rng = random.Random(seed)
        s = rng.randrange(len(suspects))
        w = rng.randrange(len(weapons))
        r = rng.randrange(len(rooms))
        guess = (suspects[s], weapons[w], rooms[r])

        accusation_candidates = knowledge.accusation_candidates()
        if len(accusation_candidates) <= 4:
            rnd = rng.randrange(len(accusation_candidates))
            guess = accusation_candidates[rnd]
            return (ActionTypes.ACCUSE, *guess)
        return (ActionTypes.SUGGEST, *guess)

    def _after_suggest_certain(self, knowledge):
        candidates = knowledge.candidates()
        suspects = [c for c in candidates if c < 6]
        weapons = [c for c in candidates if 6 <= c < 12]
        rooms = [c for c in candidates if c >= 12]
        if len(suspects) == len(weapons) == len(rooms) == 1:
            guess = (suspects[0], weapons[0], rooms[0])
            return (ActionTypes.ACCUSE, *guess)
        return (ActionTypes.END,)