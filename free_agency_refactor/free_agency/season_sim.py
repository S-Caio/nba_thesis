"""
Season simulation: schedule generation, game-by-game win simulation, and
the draft lottery. This is your existing utils.py content -- renamed
(not rewritten) so its role in the package is explicit at a glance.

reward_func also belongs conceptually here or in its own reward.py if it
grows; not included since it wasn't in the code you shared.
"""
import random
import numpy as np


def generate_exact_nba_schedule(n_teams=30):
    game_list = []

    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            conf_i, conf_j = i // 15, j // 15
            div_i, div_j = i // 5, j // 5

            if conf_i != conf_j:
                num_games = 2
            elif div_i == div_j:
                num_games = 4
            else:
                num_games = 4 if (i + j) % 5 < 3 else 3

            for _ in range(num_games):
                game_list.append((i, j))

    game_list = random.sample(game_list, len(game_list))
    return game_list


def play_season(team_ratings, game_list, noise_scale, pct_played=1):
    wins = np.zeros(len(team_ratings))
    games_played = np.zeros(len(team_ratings))

    n_to_play = round(len(game_list) * pct_played)

    for game_id in range(n_to_play):
        team_a, team_b = game_list[game_id]
        diff = team_ratings[team_a] - team_ratings[team_b]
        noise = np.random.logistic(loc=0.0, scale=noise_scale)

        if (diff + noise) > 0:
            wins[team_a] += 1
        else:
            wins[team_b] += 1

        games_played[team_a] += 1
        games_played[team_b] += 1

    return wins, games_played


def draft_lottery():
    """
    NBA-style weighted lottery for the 14 non-playoff teams. Seed 0 is the
    worst record (best odds); seed 13 the best of the lottery teams.
    Top 4 picks are drawn via weighted sampling without replacement,
    renormalizing after each draw. Picks 5-14 follow seed order.
    """
    weights = np.array([
        140, 140, 140, 125, 105, 90, 75,
        60, 45, 30, 20, 15, 10, 5
    ], dtype=float)

    remaining = list(range(14))
    remaining_weights = weights.copy()
    top4 = []

    for _ in range(4):
        probs = remaining_weights / remaining_weights.sum()
        winner_idx = np.random.choice(len(remaining), p=probs)
        winner = remaining.pop(winner_idx)
        top4.append(winner)
        remaining_weights = np.delete(remaining_weights, winner_idx)

    rest = sorted(remaining)
    draft_order = top4 + rest
    order_dict = {seed: pick + 1 for pick, seed in enumerate(draft_order)}

    return draft_order, order_dict


def seed_position(position):
    return 30 - position
