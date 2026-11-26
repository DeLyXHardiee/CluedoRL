from collections import Counter
from random import choice, randint, shuffle

from Actor import CluedoActor, Knowledge, GullibleKnowledge, NoInferenceKnowledge
from Strategies import Strategy, main_first, after_suggest_certain, main_random, main_gamble, main_bluff, after_suggest_gamble
from GameEngine import CluedoGameEngine

main_strategies = [main_first, main_random, main_gamble, main_bluff]
after_suggest_strategies = [after_suggest_certain, after_suggest_gamble]
knowledges = [Knowledge, GullibleKnowledge, NoInferenceKnowledge]

nbots = 6
games = 1000


def main():
    wins = [0] * nbots
    categories = [
        ("Main strategy", main_strategies, lambda bot: bot.strategy.main),
        ("After-suggest strategy", after_suggest_strategies, lambda bot: bot.strategy.after_suggest),
        ("Knowledge type", knowledges, lambda bot: bot.knowledge_class),
    ]
    appearances = [Counter() for _ in categories]
    category_wins = [Counter() for _ in categories]
    eliminations = [Counter() for _ in categories]

    for game_index in range(games):
        if game_index % 1000 == 0:
            print(f"Playing game {game_index}")
        bots = [
            CluedoActor(
                Strategy(choice(main_strategies), choice(after_suggest_strategies)),
                choice(knowledges),
            )
            for _ in range(nbots)
        ]
        slot = {id(bot): idx for idx, bot in enumerate(bots)}
        for bot in bots:
            for (_, _, get_category), counts in zip(categories, appearances):
                counts[get_category(bot)] += 1
        shuffle(bots)
        game = CluedoGameEngine(randint(0, 1000000))
        game.start(bots)
        game.play()
        for bot in game.eliminated_players:
            for (_, _, get_category), counts in zip(categories, eliminations):
                counts[get_category(bot)] += 1
        if game.winner is not None:
            wins[slot[id(game.winner)]] += 1
            for (_, _, get_category), counts in zip(categories, category_wins):
                counts[get_category(game.winner)] += 1

    print("bot slot 0 winrate:", wins[0] / games)
    print("bot slots 1-5 avg winrate:", sum(wins[1:]) / (games * (nbots - 1)))
    print("no winner:", (games - sum(wins)) / games)
    print("fair share would be:", 1 / nbots)
    for (label, options, _), entered, won, eliminated in zip(
        categories, appearances, category_wins, eliminations
    ):
        print(f"\n{label} (wins / eliminations / appearances):")
        for option in options:
            print(f"  {option.__name__}: {won[option]} / {eliminated[option]} / {entered[option]}")


if __name__ == "__main__":
    main()
