# Cluedo Actor Superclass
from GameEngine import ActionTypes

class CluedoActor:

    def __init__(self):
        pass

    def reset(self, gameEngine, seat, n, hand, public_cards):
        self.gameEngine = gameEngine
        self.seat = seat
        self.n = n
        self.hand = hand
        self.public_cards = public_cards
    
    def observe(self, game_state):
        print("Observing game state:", game_state)
        self.game_state = game_state
        if game_state["current_turn"] == self.seat:
            self.act(game_state)
    
    def act(self, game_state):
        self.gameEngine.step(self, self.strategy(game_state))
        return

    def choose_reveal(self, suggestion, suggesting_player):
        for card in suggestion:
            if card in self.hand:
                return card

    def strategy(self, game_state):
        return [ActionTypes.SUGGEST,0,6,12]
