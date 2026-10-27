# Cluedo Actor Superclass
from GameEngine import ActionTypes

class ICluedoActor:
    def reset(self, gameEngine, seat, n, hand, public_cards):
        raise NotImplementedError

    def updateGameState(self, game_state):
        raise NotImplementedError

    def observe(self, game_state):
        raise NotImplementedError

    def act(self, game_state):
        raise NotImplementedError

    def choose_reveal(self, suggestion, suggesting_player):
        raise NotImplementedError

    def strategy(self, game_state):
        raise NotImplementedError

class CluedoActor(ICluedoActor):

    def __init__(self):
        pass

    def reset(self, gameEngine, seat, n, hand, public_cards):
        self.gameEngine = gameEngine
        self.seat = seat
        self.n = n
        self.hand = hand
        self.public_cards = public_cards
        self.noteCard = []
        self.game_state = None
        self.last_action = None

        for card in self.hand:
            if card not in self.noteCard:
                self.noteCard.append(card)
        for card in self.public_cards:
            if card not in self.noteCard:
                self.noteCard.append(card)

    def updateGameState(self, game_state):
        self.game_state = game_state
        if "revealed_info" in game_state:
            revealed_info = game_state["revealed_info"]
            if revealed_info["revealed_card"] not in self.noteCard:
                self.noteCard.append(revealed_info["revealed_card"])
    
    def observe(self, game_state):
        #print("Observing game state:", game_state)
        self.game_state = game_state
        self.updateGameState(game_state)
        if game_state["current_turn"] == self.seat:
            self.act(game_state)
    
    def act(self, game_state):
        action = self.strategy(game_state)
        #print("Acting with action:", action)
        self.last_action = action
        self.gameEngine.step(self, action)
        return

    def choose_reveal(self, suggestion, suggesting_player):
        for card in suggestion:
            if card in self.hand:
                return card

    def strategy(self, game_state):
        person = 0
        weapon = 6
        room = 12
        while person in self.noteCard and person < 5:
            person += 1
        while weapon in self.noteCard and weapon < 11:
            weapon += 1
        while room in self.noteCard and room < 20:
            room += 1
        if self.last_action == [ActionTypes.SUGGEST, person, weapon, room]:
            return [ActionTypes.ACCUSE, person, weapon, room]
        return [ActionTypes.SUGGEST, person, weapon, room]
