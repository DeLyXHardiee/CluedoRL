"""Text CLI: a human takes one Seat against random-legal Seats.

    python -m cluedo.cli --players 4 --seed 1 [--seat 0]
"""
from __future__ import annotations

import argparse
import json
import random

from .core import ACCUSE, SUGGEST, Game, random_legal
from .core.cards import CARD_NAMES, CATS
from .core.deduction import Deduction


def fmt_cards(cards):
    return " / ".join(CARD_NAMES[c] for c in cards)


def describe(e):
    s = e["seat"]
    if e["type"] == "suggestion":
        txt = f"Seat {s} suggests {fmt_cards(e['cards'])}."
        for p in e["passes"]:
            txt += f" Seat {p} passes."
        if e["answerer"] is None:
            return txt + " Nobody could answer."
        txt += f" Seat {e['answerer']} shows a card"
        if e["card"] is not None:
            txt += f": {CARD_NAMES[e['card']]}"
        return txt + "."
    if e["type"] == "accusation":
        return (f"Seat {s} ACCUSES {fmt_cards(e['cards'])}: "
                + ("CORRECT!" if e["correct"] else "wrong, eliminated."))
    return f"Seat {s} does not accuse."


def knowledge_view(ded):
    """Knowledge grid (X holds, - does not, ? unknown) per card, plus the Candidate count."""
    cols = [f"Seat {p}" for p in range(ded.n)] + ["Envelope"]
    lines = [f"{'':14s}" + " ".join(f"{c:>8s}" for c in cols)]
    sym = {1: "X", 0: "-", None: "?"}
    for c in range(len(CARD_NAMES)):
        lines.append(f"{CARD_NAMES[c]:14s}" + " ".join(
            f"{sym[ded.status(p, c)]:>8s}" for p in range(ded.n + 1)))
    lines.append(f"Candidates: {ded.candidate_count()}  Open clauses: {ded.open_clause_count()}")
    return "\n".join(lines)


def parse_cards(text):
    """Three 0-based indices within categories (suspect weapon room), e.g. '2 1 4'."""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("enter three numbers: suspect weapon room")
    return tuple(CATS[k][int(p)] for k, p in enumerate(parts))


def ask(game, inp, out):
    d = game.decision
    legal = set(game.legal_actions())
    while True:
        try:
            if d.phase == SUGGEST:
                out("Suspects: " + ", ".join(f"{i}={n}" for i, n in enumerate(CARD_NAMES[0:6])))
                out("Weapons: " + ", ".join(f"{i}={n}" for i, n in enumerate(CARD_NAMES[6:12])))
                out("Rooms: " + ", ".join(f"{i}={n}" for i, n in enumerate(CARD_NAMES[12:21])))
                a = (SUGGEST, *parse_cards(inp("suggest (suspect weapon room)> ")))
            elif d.phase == ACCUSE:
                raw = inp("accuse? Enter to pass, or 'suspect weapon room'> ").strip()
                a = ("pass",) if not raw else (ACCUSE, *parse_cards(raw))
            else:
                opts = sorted(c for (_, c) in legal)
                out("Reveal which card? " + ", ".join(f"{c}={CARD_NAMES[c]}" for c in opts))
                a = ("reveal", int(inp("reveal (card number)> ")))
        except (ValueError, IndexError) as ex:
            out(f"Invalid: {ex}")
            continue
        if a in legal:
            return a
        out("Illegal action")


def play(game, human_seat, rng, inp=input, out=print):
    out(f"You are Seat {human_seat} of {game.n}. Hand: {fmt_cards(sorted(game.hands[human_seat]))}")
    out("Hand sizes: " + ", ".join(f"S{i}={n}" for i, n in enumerate(game.hand_sizes)))
    out(f"Seat {game.decision.seat} starts.")
    shown = 0
    ded = Deduction.for_seat(game, human_seat)
    while not game.done:
        if game.decision.seat == human_seat:
            ded.sync(game.events_for(human_seat))
            if game.decision.phase == SUGGEST:
                out(knowledge_view(ded))
            a = ask(game, inp, out)
        else:
            a = random_legal(game, rng)
        game.apply(a)
        view = game.events_for(human_seat)  # public record plus what I was shown
        for e in view[shown:]:
            out(describe(e))
        shown = len(view)
    o = game.outcome
    out(f"Game over: {o['reason']}, winner: {o['winner']}, turns: {o['turns']}.")
    out("Envelope was: " + fmt_cards(game.envelope))
    return game


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--log", help="write the Game log JSON here")
    args = p.parse_args(argv)
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    game = Game.new(args.players, seed)
    play(game, args.seat, random.Random(seed + 1))
    if args.log:
        with open(args.log, "w") as f:
            json.dump(game.log(), f)


if __name__ == "__main__":
    main()
