import torch
import csv
import random
import os
import matplotlib.pyplot as plt
import statistics
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
                actor.label = factory.label if label is None else label
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
            action_rows, game_rows = play_games(create_lineup(botList, label, policy, n), games, seed=seed)
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

def analyse_model_vs_model(path1, label1, path2, label2, out_dir, games=2000, seed=1):
    policy1 = load_actor(path1)
    policy2 = load_actor(path2)
    os.makedirs(out_dir, exist_ok=True)
    for n in (4, 6):
        model1Lineup = [(label1, lambda: RLActor(policy1)) for i in range(n//2)]
        model2Lineup = [(label2, lambda: RLActor(policy2)) for i in range(n//2)]
        lineup = lambda rng: model1Lineup + model2Lineup
        action_rows, game_rows = play_games(lineup, games, seed)
        actions_path = os.path.join(out_dir, f"test_{n}_{label1}_vs_{label2}_actions.csv")
        game_path = os.path.join(out_dir, f"test_{n}_{label1}_vs_{label2}_game.csv")
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



METHODS = {"bots_run": "Bot-trained", "self_run": "Self-play"}
COLORS = {"bots_run": "#1f77b4", "self_run": "#e8702a"}
POOLS = ["normal_knowledge", "all_bots", "test_pool", "falls_for_bluffs",
        "always_first", "uses_bluffs", "good_memory"]
T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57}

def models_of(method, root):
    return sorted(m for m in os.listdir(root) if m.startswith(method) and "_vs_" not in m)

def read_rows(root, model, n, pool, kind):
    path = os.path.join(root, model, f"test_{n}_{pool}_{kind}.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def mean_ci(values):
    half = T95[len(values)] * statistics.stdev(values) / len(values) ** 0.5
    return statistics.mean(values), half

def grouped_bars(ax, groups, get_values):
    for offset, method in zip((-0.2, 0.2), METHODS):
        stats = [mean_ci(get_values(method, group)) for group in groups]
        x = [i + offset for i in range(len(groups))]
        ax.bar(x, [s[0] for s in stats], 0.38, yerr=[s[1] for s in stats],
               capsize=3, color=COLORS[method], label=METHODS[method])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(group).replace("_", "\n") for group in groups])
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

def plot_win_rates(root="analysis"):
    def win_rate(model, n, pool):
        rows = read_rows(root, model, n, pool, "game")
        return sum(row["winner"] == model for row in rows) / len(rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, n in zip(axes.flat, (3, 4, 5, 6)):
        def values(method, pool):
            return [win_rate(m, n, pool) for m in models_of(method, root)]
        grouped_bars(ax, POOLS, values)
        ax.axhline(1 / n, color="gray", linestyle="--", label="Fair share")
        ax.set_title(f"{n} players")
        ax.set_ylabel("Win rate")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncols=3)
    fig.savefig(os.path.join(root, "win_rates.png"), dpi=200)
    plt.close(fig)

def behaviour(root, model, n, pool):
    actions = read_rows(root, model, n, pool, "actions")
    games = read_rows(root, model, n, pool, "game")
    actions = [row for row in actions if row["label"] == model]
    suggests = [row for row in actions if row["kind"] == "SUGGEST"]
    accuses = [row for row in actions if row["kind"] == "ACCUSE"]
    wins = [row for row in games if row["winner"] == model]
    return {
        "bluff": sum(int(r["own_cards"]) > 0 for r in suggests) / len(suggests),
        "correct": sum(r["correct"] == "True" for r in accuses) / len(accuses),
        "eliminated": sum(model in r["eliminated"].split() for r in games) / len(games),
        "candidates": [int(r["accusation_candidates"]) for r in accuses],
        "win_turns": [int(r["turns_taken"]) / n for r in wins],
    }

def plot_behaviour(root="analysis", pool="test_pool"):
    data = {}
    for method in METHODS:
        for n in (3, 4, 5, 6):
            data[method, n] = [behaviour(root, m, n, pool) for m in models_of(method, root)]

    titles = {
        "bluff": "Suggestions naming an own card",
        "correct": "Correct accusations",
        "eliminated": "Games where the model is eliminated",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, (key, title) in zip(axes, titles.items()):
        grouped_bars(ax, (3, 4, 5, 6), lambda method, n: [d[key] for d in data[method, n]])
        ax.set_title(title)
        ax.set_xlabel("Players")
        ax.set_ylabel("Share")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2)
    fig.savefig(os.path.join(root, f"behaviour_{pool}.png"), dpi=200)
    plt.close(fig)

    plot_histograms(data, "candidates", "Solutions left at accusation", range(1, 32),
                    os.path.join(root, f"hist_candidates_{pool}.png"))
    plot_histograms(data, "win_turns", "Rounds played in won games", range(0, 21),
                    os.path.join(root, f"hist_rounds_{pool}.png"))

def plot_histograms(data, key, xlabel, bins, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, n in zip(axes.flat, (3, 4, 5, 6)):
        for method in METHODS:
            values = [v for d in data[method, n] for v in d[key]]
            weights = [1 / len(values)] * len(values)
            ax.hist(values, bins, weights=weights, histtype="step", linewidth=2,
                    color=COLORS[method], label=METHODS[method])
        ax.set_title(f"{n} players")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Share")
        ax.grid(alpha=0.3)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncols=2)
    fig.savefig(path, dpi=200)
    plt.close(fig)

    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_head_to_head(root="analysis"):
    pairs = sorted(m for m in os.listdir(root) if "_vs_" in m)

    def values(method, n):
        rates = []
        for pair in pairs:
            rows = read_rows(root, pair, n, pair, "game")
            decided = [row for row in rows if row["winner"]]
            wins = sum(row["winner"].startswith(method) for row in decided)
            rates.append(wins / len(decided))
        return rates

    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    grouped_bars(ax, (4, 6), values)
    ax.axhline(0.5, color="gray", linestyle="--", label="Fair share")
    ax.set_title("Bot-trained vs self-play, half the seats each")
    ax.set_xlabel("Players")
    ax.set_ylabel("Share of decided games won")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(root, "head_to_head.png"), dpi=200)
    plt.close(fig)

def plot_experiments(root="analysis"):
    plot_win_rates(root)
    plot_head_to_head(root)
    for pool in POOLS:
        plot_behaviour(root, pool)

if __name__ == "__main__":
    plot_experiments()