from Actor import CluedoActor
from GameEngine import ActionTypes, AFTER_SUGGEST
import random

class RandomActor(CluedoActor):

    def strategy(self, game_state):
        candidates = self.knowledge.candidates()
        suspects = [c for c in candidates if c < 6]
        weapons = [c for c in candidates if 6 <= c < 12]
        rooms = [c for c in candidates if c >= 12]
        rng = random.Random(self.seed)
        s = rng.randrange(len(suspects))
        w = rng.randrange(len(weapons))
        r = rng.randrange(len(rooms))
        guess = (suspects[s], weapons[w], rooms[r])
        if len(suspects) == len(weapons) == len(rooms) == 1:
            return (ActionTypes.ACCUSE, *guess)
        if game_state["phase"] == AFTER_SUGGEST:
            return (ActionTypes.END,)
        return (ActionTypes.SUGGEST, *guess)


class GambleActor(CluedoActor):

    def strategy(self, game_state):
        candidates = self.knowledge.candidates()
        suspects = [c for c in candidates if c < 6]
        weapons = [c for c in candidates if 6 <= c < 12]
        rooms = [c for c in candidates if c >= 12]
        rng = random.Random(self.seed)
        s = rng.randrange(len(suspects))
        w = rng.randrange(len(weapons))
        r = rng.randrange(len(rooms))
        guess = (suspects[s], weapons[w], rooms[r])

        accusation_candidates = self.knowledge.accusation_candidates()
        if len(accusation_candidates) <= 4:
            rnd = rng.randrange(len(accusation_candidates))
            guess = accusation_candidates[rnd]
            return (ActionTypes.ACCUSE, *guess)
        if game_state["phase"] == AFTER_SUGGEST:
            return (ActionTypes.END,)
        return (ActionTypes.SUGGEST, *guess)