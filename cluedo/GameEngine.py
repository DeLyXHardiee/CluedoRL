# Cluedo Game Engine class
import random
import enum

# Enum for action types in the game
class ActionTypes(enum.Enum):
    SUGGEST = 1
    ACCUSE = 2
    END = 3

cards = {0:"Miss Scarlet", 1:"Colonel Mustard", 2:"Mrs. White", 3:"Mr. Green", 4:"Mrs. Peacock", 5:"Professor Plum",
         6:"Candlestick", 7:"Dagger", 8:"Lead Pipe", 9:"Revolver", 10:"Rope", 11:"Wrench",
         12:"Kitchen", 13:"Ballroom", 14:"Conservatory", 15:"Dining Room", 16:"Billiard Room", 17:"Library", 18:"Lounge", 19:"Hall", 20:"Study"}

class CluedoGameEngine:
    def __init__(self):
        self.players = []
        self.envelope = None
        self.current_turn = 0
        self.playerLimit = 6
        self.hands = {}
        self.eliminated_players = []
        self.winner = None
        self.rng = None
        self.turns = []
        self.publicCards = []
        self.publicGameState = {}

    def add_player(self, player):
        if self.current_turn != 0:
            return
        if player not in self.players and len(self.players) < self.playerLimit:
            self.players.append(player)

    def start_game(self):
        print("Starting game with players:", [self.players.index(player) for player in self.players])
        # Initialize the game, deal cards, and set up the envelope
        self.current_turn = 0
        self.eliminated_players = []
        self.winner = None
        self.turns = []
        self.hands = {}
        self.rng = random.Random()

        self.setup_envelope_and_deal_cards()
    
    def setup_envelope_and_deal_cards(self):

        # Roll the cards and set up the envelope
        all_cards = list(cards.keys())
        person_cards = all_cards[:6]
        weapon_cards = all_cards[6:12]
        room_cards = all_cards[12:]
        self.rng.shuffle(person_cards)
        self.rng.shuffle(weapon_cards)
        self.rng.shuffle(room_cards)

        # Select one suspect, one weapon, and one room for the envelope
        self.envelope = {
            "suspect": person_cards.pop(0),
            "weapon": weapon_cards.pop(0),
            "room": room_cards.pop(0)
        }

        player_count = len(self.players)
        if player_count == 0:
            raise ValueError("At least one player is required to deal cards.")

        remaining_cards = person_cards + weapon_cards + room_cards
        self.rng.shuffle(remaining_cards)
        remainder = len(remaining_cards) % player_count
        dealt_cards = remaining_cards[:len(remaining_cards) - remainder]
        self.publicCards = remaining_cards[len(dealt_cards):]

        for i, card in enumerate(dealt_cards):
            player = self.players[i % player_count]
            if player not in self.hands:
                self.hands[player] = []
            self.hands[player].append(card)
        for player in self.players:
            player.reset(self, self.players.index(player), len(self.players), self.hands[player], self.publicCards)
        self.broadcastGameState()

    def next_turn(self):
        self.current_turn = (self.current_turn + 1) % len(self.players)
        if self.current_turn > 200:
            raise RuntimeError("Maximum number of turns exceeded.")

    # Currently accusing after suggesting doesnt work
    def step(self, player, action):
        if player != self.players[self.current_turn]:
            return
        if player in self.eliminated_players:
            self.next_turn()
            return
        if not self.validateAction(action):
            self.next_turn()
            return
        if action[0] == ActionTypes.END:
            self.next_turn()
            return
        elif action[0] == ActionTypes.SUGGEST:
            suggestion = action[1:]
            revealed_card, revealed_by = self.suggestLoop(player, suggestion)
            self.broadcastReveal(revealed_card, revealed_by, player)
            self.next_turn()
            return
        elif action[0] == ActionTypes.ACCUSE:
            accusation = action[1:]
            if self.accuse(accusation):
                self.winner = player
                self.broadcastGameState()
            else:
                self.eliminated_players.append(player)
                self.broadcastGameState()
            self.next_turn()
            return
        self.next_turn()
        return

    def suggestLoop(self, player, suggestion):
        for i in range(len(self.players)):
            next_player = self.players[(self.current_turn + i) % len(self.players)]
            for card in suggestion:
                if card in self.hands[next_player]:
                    revealed = next_player.choose_reveal(suggestion, player)
                    if revealed not in suggestion:
                        revealed = card
                        break
                    return revealed,next_player
        return None, None

    def accuse(self,accusation):
        combination = {
            "suspect": accusation[0],
            "weapon": accusation[1],
            "room": accusation[2]
        }
        return all(self.envelope[key] == combination.get(key) for key in ["suspect", "weapon", "room"])

    def validateAction(self, action):
        if not isinstance(action[0], ActionTypes):
            raise ValueError("Invalid action type.")
        if action[0] == ActionTypes.END:
            return True
        if len(action) != 4:
            raise ValueError("Invalid action length.")
        if action[1] < 0 or action[1] > 5:
            raise ValueError("Invalid action value.")
        if action[2] < 6 or action[2] > 11:
            raise ValueError("Invalid action value.")
        if action[3] < 12 or action[3] > 20:
            raise ValueError("Invalid action value.")
        return True

    def broadcastReveal(self, revealed_card, revealed_by, suggesting_player):
        # Broadcast the reveal to all players
        #self.notify_all()
        self.broadcastGameState()

    def broadcastGameState(self):
        # Update the public game state
        self.publicGameState = {
            "current_turn": self.current_turn % len(self.players),
            "eliminated_players": self.eliminated_players,
            "winner": self.winner,
            "turns": self.turns
        }
        # Broadcast a message to all players
        self.notify_all()

    def notify_all(self):
        for player in self.players:
            player.observe(self.publicGameState)