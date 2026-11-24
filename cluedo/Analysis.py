import torch
import csv
import random
import os
import matplotlib.pyplot as plt
from GameEngine import CluedoGameEngine, SUGGEST, ACCUSE, END
from RL import RLActor, load_actor, bot_factory, TEST_POOL
from torchrl.envs.utils import set_exploration_type, ExplorationType

class RecordingEngine(CluedoGameEngine):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.action_log = []

    def step(self, action):
        player = self.seatToPlayer[self.current_turn]
        
        own_cards = 0
        for own_card in player.hand:
            if own_card in action[1:]:
                own_cards += 1

        candidates_named = 0
        candidates = player.get_candidates()
        for card in candidates:
            if card in action[1:]:
                candidates_named += 1

        temp = {
            "turn": self.turns_taken,
            "seat": self.current_turn,
            "phase": self.phase,
            "kind": action[0],
            "cards": action[1:],
            "own_cards": own_cards,
            "candidates_named": candidates_named,
            "accusation_candidates": len(player.get_accusation_candidates()),
            "label": getattr(player, "label", "unknown"),
        }

        super().step(action)
        if action[0] == ACCUSE:
            temp["correct"] = player == self.winner
        self.action_log.append(temp)

def create_lineup(botList, label, policy, n):
    def make(rng):
        lineup = []
        model = (label, lambda: RLActor(policy))
        lineup.append(model)
        for i in range(n-1):
            chosen = rng.choice(botList.bots)
            lineup.append((None, chosen))
        return lineup
    return make

def create_model_lineup(bot, label, policy, n):
    def make(rng):
        lineup = []
        for i in range(n-1):
            lineup.append((label, lambda: RLActor(policy)))
        lineup.append((None, bot))
        return lineup
    return make

def play_games(make_lineup, games, seed):
    action_rows = []
    game_rows = []

    rng = random.Random(seed)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for i in range(games):
            game = RecordingEngine(seed=seed + i)
            lineup = make_lineup(rng)
            actors = []
            for label, factory in lineup:
                actor = factory()
                actor.label = repr(actor) if label is None else label
                actors.append(actor)
            game.start(actors)
            game.play()
            for row in game.action_log:
                row["game"] = i
            action_rows.extend(game.action_log)
            game_rows.append({"game": i, "player_count": len(game.players), "winner": game.winner.label if game.winner else None, "truncated": game.truncated, "turns_taken": game.turns_taken, "eliminated": [p.label for p in game.eliminated_players]})
    return action_rows, game_rows

def save_rows(rows, path):
    with open(path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        if "kind" in rows[0] and "correct" not in fieldnames:
            fieldnames.append("correct") 
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if "kind" in row:
                row["kind"] = row["kind"].name
            if "phase" in row:
                row["phase"] = row["phase"].name
            if "cards" in row:
                row["cards"] = " ".join(str(c) for c in row["cards"])
            if "eliminated" in row:
                row["eliminated"] = " ".join(row["eliminated"])
            writer.writerow(row)
        
def analyse_model(path, label, botLists, out_dir, games=2000, seed=1):
    policy = load_actor(path)
    os.makedirs(out_dir, exist_ok=True)
    for n in (3, 4, 5, 6):
        for botList in botLists:
            action_rows, game_rows = play_games(create_lineup(botList, label, policy, n), games, seed)
            actions_path = os.path.join(out_dir, f"test_{n}_{botList.name}_actions.csv")
            game_path = os.path.join(out_dir, f"test_{n}_{botList.name}_game.csv")
            save_rows(action_rows, actions_path)
            save_rows(game_rows, game_path)

def analyse_bot(path, label, bot, botname, out_dir, games=2000, seed=1):
    policy = load_actor(path)
    os.makedirs(out_dir, exist_ok=True)
    for n in (3, 4, 5, 6):
        action_rows, game_rows = play_games(create_model_lineup(bot, label, policy, n), games, seed)
        actions_path = os.path.join(out_dir, f"test_{n}_{botname}_vs_{label}_actions.csv")
        game_path = os.path.join(out_dir, f"test_{n}_{botname}_vs_{label}_game.csv")
        save_rows(action_rows, actions_path)
        save_rows(game_rows, game_path)

def load_log(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def moving_average(values, window):
    averages = []
    for i in range(len(values)):
        recent = values[max(0, i + 1 - window):i + 1]
        averages.append(sum(recent) / len(recent))
    return averages

def plot_train_log(run_dir, window=50):
    log = load_log(os.path.join(run_dir, "train_log.csv"))
    if not log:
        print(f"No training data in {run_dir}")
        return

    titles = {
        "train_win_rate": "Training win rate",
        "loss_objective": "Policy loss",
        "loss_critic": "Critic loss",
        "loss_entropy": "Entropy loss",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    for ax, (key, title) in zip(axes.flat, titles.items()):
        ax.plot(log["frames"], log[key], color="tab:blue", alpha=0.25, linewidth=0.8, label="Per batch")
        smoothed = moving_average(log[key], window)
        ax.plot(log["frames"], smoothed, color="tab:blue",
                label=f"Mean of last {window} batches")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Frames")
    axes[0, 0].legend()

    fig.savefig(os.path.join(run_dir, "train_plot.png"), dpi=200)
    plt.close(fig)

def plot_eval_log(run_dir):
    log = load_log(os.path.join(run_dir, "eval_log.csv"))
    if not log:
        print(f"No evaluation data in {run_dir}")
        return

    fig, (ax_score, ax_wins) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    ax_score.plot(log["frames"], log["eval_combined_score"], marker="o", markersize=3, alpha=0.4, label="Combined score")
    if "eval_smoothed_score" in log:
        ax_score.plot(log["frames"], log["eval_smoothed_score"], label="Smoothed score")
    ax_score.axhline(1.0, color="gray", linestyle="--", label="Fair share")
    ax_score.set_title("Combined evaluation score")
    ax_score.set_ylabel("Mean of win rate x player count")

    prefix, suffix = "eval_player_count_", "_win_rate"
    for key in log:
        if key.startswith(prefix):
            n = int(key.removeprefix(prefix).removesuffix(suffix))
            line, = ax_wins.plot(log["frames"], log[key], marker="o", markersize=3, label=f"{n} players")
            ax_wins.axhline(1 / n, color=line.get_color(), linestyle="--", alpha=0.5)
    ax_wins.set_title("Win rate by player count (dashed: fair share 1/n)")
    ax_wins.set_ylabel("Win rate")

    for ax in (ax_score, ax_wins):
        ax.set_xlabel("Frames")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.savefig(os.path.join(run_dir, "eval_plot.png"), dpi=200)
    plt.close(fig)

def plot_run(run_dir):
    plot_train_log(run_dir)
    plot_eval_log(run_dir)