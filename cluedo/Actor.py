# Cluedo Actor Superclass
from GameEngine import ActionTypes, AFTER_SUGGEST

UNKNOWN, YES, NO = 0, 1, 2

class CluedoActor():

    def __init__(self):
        pass

    def reset(self, gameEngine, seat, n, hand, public_cards, seed):
        self.gameEngine = gameEngine
        self.seat = seat
        self.n = n
        self.hand = hand
        self.public_cards = public_cards
        self.last_action = None
        self.knowledge = Knowledge(seat, n, hand, public_cards)
        self.seed = seed
    
    def observe(self, event):
        #print("Observing game state:", game_state)
        self.knowledge.observe(event)
    
    def act(self, game_state):
        action = self.strategy(game_state)
        #print("Acting with action:", action)
        self.last_action = action
        return action

    # Basic reveal strategy: reveal the first matching card in hand
    def choose_reveal(self, suggestion, suggesting_player):
        for card in suggestion:
            if card in self.hand:
                return card

    # Default strategy, always suggest the first candidate cards and accuse only when certain
    def strategy(self, game_state):
        candidates = self.knowledge.candidates()
        suspects = [c for c in candidates if c < 6]
        weapons = [c for c in candidates if 6 <= c < 12]
        rooms = [c for c in candidates if c >= 12]
        guess = (suspects[0], weapons[0], rooms[0])
        if len(suspects) == len(weapons) == len(rooms) == 1:
            return (ActionTypes.ACCUSE, *guess)
        if game_state["phase"] == AFTER_SUGGEST:
            return (ActionTypes.END,)
        return (ActionTypes.SUGGEST, *guess)

    def get_knowledge(self):
        return self.knowledge.table

    def get_candidates(self):
        return self.knowledge.candidates()

    def get_accusation_candidates(self):
        return self.knowledge.accusation_candidates()

# Knowledge representation for Cluedo game, tracking known and possible card ownerships
class Knowledge:
    def __init__(self, seat, n, hand, face_up):
        self.seat = seat
        self.n = n
        self.hand_size = 18 // n
        self.table = [[UNKNOWN] * 21 for _ in range(n)]
        self.one_of = []                # (answerer, set of cards) rules
        self.wrong_accusations = set()
        self.face_up = set(face_up)

        # Rule 1: face-up cards are NO for everyone; my row is fully known
        for card in face_up:
            for s in range(n):
                self.table[s][card] = NO
        for card in range(21):
            self.table[seat][card] = YES if card in hand else NO
        
        self.propagate()

    def apply_owner_and_full_hands(self):
      changed = False
      # Rule 5: one owner per card
      for card in range(21):
          owners = [s for s in range(self.n) if self.table[s][card] == YES]
          if owners:
              for s in range(self.n):
                  if s != owners[0] and self.table[s][card] == UNKNOWN:
                      self.table[s][card] = NO
                      changed = True
      # Rule 6: a full hand means the rest of that row is NO
      for s in range(self.n):
          if self.table[s].count(YES) == self.hand_size:
              for card in range(21):
                  if self.table[s][card] == UNKNOWN:
                      self.table[s][card] = NO
                      changed = True
      return changed

    def observe(self, event):
        if event["kind"] == ActionTypes.SUGGEST:
            # Rule 2: everyone who passed has none of the three cards
            for s in event["passed"]:
                for card in event["cards"]:
                    self.table[s][card] = NO
            # Rule 3: a card shown to me
            if event["shown"] is not None:
                self.table[event["answerer"]][event["shown"]] = YES
            # Rule 4: someone else answered, so they hold at least one of the three
            elif event["answerer"] is not None:
                self.one_of.append((event["answerer"], set(event["cards"])))
        elif event["kind"] == ActionTypes.ACCUSE and not event["correct"]:
            self.wrong_accusations.add(tuple(event["cards"]))
        self.propagate()

    # Apply the "one-of" rule: if a player must have one of a set of cards, deduce ownership when possible
    def apply_one_of(self):
      changed = False
      remaining = []
      for answerer, cards in self.one_of:
          if any(self.table[answerer][c] == YES for c in cards):
              continue    # already explained, nothing more to note
          cards = {c for c in cards if self.table[answerer][c] != NO}
          if len(cards) == 1:
              self.table[answerer][cards.pop()] = YES
              changed = True
          else:
              remaining.append((answerer, cards))
      self.one_of = remaining
      return changed

    def propagate(self):
        while self.apply_one_of() | self.apply_owner_and_full_hands(): 
            pass

    # Determine the set of candidate cards for each category (suspects, weapons, rooms) based on current knowledge.
    def candidates(self):
        result = []
        for lo, hi in ((0, 6), (6, 12), (12, 21)):
            cards = [c for c in range(lo, hi)
                    if c not in self.face_up
                    and all(self.table[s][c] != YES for s in range(self.n))]
            # Any card that is known to be held by no player is in the envelope and therefore no other cards from that category are candidates.
            known = [c for c in cards if all(self.table[s][c] == NO for s in range(self.n))]
            result += known[:1] or cards # weird syntax, but known has to be a list to append to result
        return result

    # Generate all possible accusation combinations based on current knowledge and eliminated wrong accusations
    def accusation_candidates(self):
        c = self.candidates()
        return [(s, w, r)
                for s in c if s < 6
                for w in c if 6 <= w < 12
                for r in c if r >= 12
                if (s, w, r) not in self.wrong_accusations]