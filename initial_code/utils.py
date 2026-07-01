import random
import numpy as np



def generate_exact_nba_schedule(n_teams=30):
    game_list = []
    
    # Loop through every unique pair of teams exactly once
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            
            # Determine relationships (assuming teams 0-14 are Conf 1, 15-29 are Conf 2)
            # Divisions are chunks of 5 (e.g., 0-4, 5-9, etc.)
            conf_i, conf_j = i // 15, j // 15
            div_i, div_j = i // 5, j // 5
            
            if conf_i != conf_j:
                # 1. Non-Conference: 2 games
                num_games = 2
                
            elif div_i == div_j:
                # 2. Same Division: 4 games
                num_games = 4
                
            else:
                # 3. Same Conference, Different Division: 3 or 4 games
                # We need exactly 6 matchups to have 4 games, and 4 matchups to have 3 games.
                # This mathematical trick evenly and symmetrically distributes the 3s and 4s 
                # across all teams in the conference.
                if (i + j) % 5 < 3:
                    num_games = 4
                else:
                    num_games = 3
                    
            # Append this matchup the required number of times
            for _ in range(num_games):
                game_list.append((i, j))
    
    game_list = random.sample(game_list, len(game_list)) # Shuffle game sequence (sample w/o replacement)

    return game_list

def play_season(team_ratings, game_list, noise_scale, pct_played = 1):
    wins = np.zeros(len(team_ratings))
    games_played = np.zeros(len(team_ratings)) # Track total games per team

    n_to_play = round(len(game_list) * pct_played)
    
    for game_id in range(n_to_play):
        game = game_list[game_id]
        team_a = game[0]
        team_b = game[1]
        # Calculate latent difference
        diff = team_ratings[team_a] - team_ratings[team_b]
        # Generate one shared random noise for this specific game
        noise = np.random.logistic(loc=0.0, scale=noise_scale)
        
        if (diff + noise) > 0:
            wins[team_a] += 1
        else:
            wins[team_b] += 1
            
        # Log that both teams played a game
        games_played[team_a] += 1
        games_played[team_b] += 1
            
    return wins, games_played


def draft_lottery():
    weights = np.array([
        140, 140, 140, 125, 105, 90, 75,
        60, 45, 30, 20, 15, 10, 5
    ], dtype=float)

    remaining = list(range(14))
    remaining_weights = weights.copy()

    top4 = []

    # draw top 4 picks
    for _ in range(4):
        probs = remaining_weights / remaining_weights.sum()
        winner_idx = np.random.choice(len(remaining), p=probs)

        winner = remaining.pop(winner_idx)
        top4.append(winner)

        remaining_weights = np.delete(remaining_weights, winner_idx)

    # remaining picks are ordered by seed
    rest = sorted(remaining)

    draft_order = top4 + rest

    # seed -> pick position dictionary
    order_dict = {seed: pick + 1 for pick, seed in enumerate(draft_order)}

    return draft_order, order_dict

def seed_position(position):
    return 30 - position