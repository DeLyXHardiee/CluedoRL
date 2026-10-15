"""Minimal seat policies for the core (richer Bots live elsewhere)."""
import random


def random_legal(game, rng: random.Random):
    return rng.choice(game.legal_actions())
