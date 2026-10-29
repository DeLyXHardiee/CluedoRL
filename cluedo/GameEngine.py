# Cluedo Game Engine class
import random
import enum

#set recursion limit for deep game state observations
import sys
from datetime import datetime

# Enum for action types in the game
class ActionTypes(enum.Enum):
    SUGGEST = 1
    ACCUSE = 2
    END = 3

SUGGEST, ACCUSE, END = ActionTypes.SUGGEST, ActionTypes.ACCUSE, ActionTypes.END

class Phase(enum.Enum):
    MAIN = 1
    AFTER_SUGGEST = 2

MAIN, AFTER_SUGGEST = Phase.MAIN, Phase.AFTER_SUGGEST

cards = {0:"Miss Scarlet", 1:"Colonel Mustard", 2:"Mrs. White", 3:"Mr. Green", 4:"Mrs. Peacock", 5:"Professor Plum",
         6:"Candlestick", 7:"Dagger", 8:"Lead Pipe", 9:"Revolver", 10:"Rope", 11:"Wrench",
         12:"Kitchen", 13:"Ballroom", 14:"Conservatory", 15:"Dining Room", 16:"Billiard Room", 17:"Library", 18:"Lounge", 19:"Hall", 20:"Study"}

class CluedoGameEngine:
    def __init__(self, n=4, seed=None):
        self.n = n
        self.seed = seed
        self.max_turns = 200
        self.players = []
        self.envelope = None
        self.current_turn = 0
        self.playerLimit = 6
        self.hands = {}
        self.eliminated_players = []
        self.winner = None
        self.done = False
        self.truncated = False
        self.phase = MAIN
        self.rng = None
        self.turns = []
        self.publicCards = []
        self.publicGameState = {}
        self.revealed_info = {}

    def add_player(self, player):
        if self.current_turn != 0:
            return
        if player not in self.players and len(self.players) < self.playerLimit:
            self.players.append(player)


    def start(self, actors):
        self.players = list(actors)
        self.start_game()

    #Step 1
    def start_game(self):
        #print("Starting game with players:", [self.players.index(player) for player in self.players])
        # Initialize the game, deal cards, and set up the envelope
        self.current_turn = 0
        self.turns_taken = 0
        self.eliminated_players = []
        self.winner = None
        self.done = False
        self.truncated = False
        self.phase = MAIN
        self.turns = []
        self.hands = {}
        self.rng = random.Random(self.seed)

        self.setup_envelope_and_deal_cards()

    #Step 2
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

        # Deal the remaining cards to the players evenly, leaving any remainder as public cards.
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
            player.reset(self, self.players.index(player), len(self.players), self.hands[player], self.publicCards, self.seed)

    def next_turn(self):
        self.phase = MAIN
        self.current_turn = (self.current_turn + 1) % len(self.players)
        self.turns_taken += 1
        if self.turns_taken >= self.max_turns:
            self.done = True
            self.truncated = True

    # Execute a single game step based on the player's action.
    def step(self, action):
        player = self.players[self.current_turn]
        self.turns.append((self.turns_taken, player.seat, action))
        # Skip the turn if the player is eliminated, the action is invalid, or the action is to end the turn.
        if player in self.eliminated_players or not self.validateAction(action) or action[0] == ActionTypes.END:
            self.next_turn()
            return
        elif action[0] == ActionTypes.SUGGEST:
            suggestion = action[1:]
            revealed_card, revealed_by, passed = self.suggestLoop(player, suggestion)
            #print(f"Suggestion made by {player.seat}: {suggestion}, revealed card: {revealed_card}, revealed by: {revealed_by.seat if revealed_by else revealed_by}")
            self.turns[-1] = (self.turns_taken, player.seat, action, (revealed_card, revealed_by.seat if revealed_by else revealed_by))
            self.phase = AFTER_SUGGEST
            self.broadcastEvent({
                "turn": self.turns_taken, "kind": SUGGEST, "seat": player.seat,
                "cards": tuple(suggestion), "passed": passed,
                "answerer": revealed_by.seat if revealed_by else None,
                "shown": revealed_card, "correct": None,
            })
            return
        elif action[0] == ActionTypes.ACCUSE:
            accusation = action[1:]
            correct = self.accuse(accusation)
            self.broadcastEvent({
                "turn": self.turns_taken, "kind": ACCUSE, "seat": player.seat,
                "cards": tuple(accusation), "passed": [], "answerer": None,
                "shown": None, "correct": correct,
            })
            if correct:
                self.winner = player
                self.done = True
            else:
                self.eliminated_players.append(player)
                if len(self.eliminated_players) == len(self.players):
                    self.done = True
                self.next_turn()
            return

    def play(self, stop_at=None):
        while not self.done and self.current_turn != stop_at:
            player = self.players[self.current_turn]
            action = player.act({**self.publicGameState, "phase": self.phase})
            self.step(action)

    # Determines which player should reveal and calls their own reveal strategy
    def suggestLoop(self, player, suggestion):
        passed = []
        for i in range(1, len(self.players)):
            next_player = self.players[(self.current_turn + i) % len(self.players)]
            matching = [card for card in suggestion if card in self.hands[next_player]]
            if matching:
                revealed = next_player.choose_reveal(matching, player)
                if revealed not in matching:
                    revealed = matching[0]
                return revealed, next_player, passed
            passed.append(next_player.seat)
        return None, None, passed

    def accuse(self,accusation):
        combination = {
            "suspect": accusation[0],
            "weapon": accusation[1],
            "room": accusation[2]
        }
        return all(self.envelope[key] == combination.get(key) for key in ["suspect", "weapon", "room"])

    # Validate the player's action based on the current phase and action type.
    # For suggests, ensure the suggested cards are within the valid ranges for suspects, weapons, and rooms.
    def validateAction(self, action):
        if not isinstance(action[0], ActionTypes):
            raise ValueError("Invalid action type.")
        if self.phase == MAIN and action[0] == ActionTypes.END:
            return False
        if self.phase == AFTER_SUGGEST and action[0] == ActionTypes.SUGGEST:
            return False
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

    def broadcastEvent(self, event):
        for player in self.players:
            view = dict(event)
            if player.seat not in (event["seat"], event["answerer"]):
                view["shown"] = None
            player.observe(view)

    def printToFile(self, turns):
        with open("turns_log.txt", "a") as f:
            f.write("New game: " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "\n")
            f.write("Solution: " + str(self.envelope) + "\n")
            f.write("Format: turn_number, player_seat, action, (revealed_card, revealed_by)\n")
            for turn in turns:
                f.write(str(turn) + "\n")
