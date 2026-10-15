"""Card constants. Cards are ints 0..20: suspects 0-5, weapons 6-11, rooms 12-20."""
from __future__ import annotations

SUSPECTS = ["Scarlett", "Mustard", "White", "Green", "Peacock", "Plum"]
WEAPONS = ["Candlestick", "Dagger", "Lead pipe", "Revolver", "Rope", "Wrench"]
ROOMS = ["Kitchen", "Ballroom", "Conservatory", "Dining room", "Billiard room",
         "Library", "Lounge", "Hall", "Study"]
CARD_NAMES = SUSPECTS + WEAPONS + ROOMS
NCARDS = 21
CATS = [list(range(0, 6)), list(range(6, 12)), list(range(12, 21))]
CAT_OF = {c: k for k, cat in enumerate(CATS) for c in cat}
