#%%

import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
import random
from utils import generate_exact_nba_schedule, play_season, draft_lottery, seed_position
from scipy import stats


#%%

nba_teams = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards"
]

def reward_func(position, k = 0.3):
    return np.exp(-k * (position- 1))

def evolve_func(rating, age):
    z = -0.005 * (age - 27) ** 3
    # rating += np.random.normal(z, 0.1 * (rating + 0.001))
    rating += np.random.normal(z, 0.5)
    return rating



def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class FreeAgencyEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "nba_free_agency_v1"}

    def __init__(self, render_mode=None, n_teams=30, players_per_team=10, salary_cap=100.0, n_seasons = 10):
        self.n_teams = n_teams
        self.players_per_team = players_per_team
        self.salary_cap = salary_cap
        self.n_seasons = n_seasons
        self.season = 0
        self.possible_agents = ["team_" + str(r) for r in range(n_teams)]

        self.agent_name_mapping = dict(zip(self.possible_agents, nba_teams))

        # Creating players
        self.n_players = n_teams * players_per_team
        player_ratings = sorted(stats.lognorm.rvs(loc=0, s=1, size=self.n_players), reverse=True)
        player_teams = np.zeros(self.n_players)
        player_ages = np.clip(np.round(np.random.normal(27, 4, size=self.n_players)), 19, 40)
        player_contract_lens = np.zeros(self.n_players)

        self.players = np.vstack([player_ratings, player_teams, player_ages, player_contract_lens]).T

        self.teams = {agent: np.zeros(players_per_team) for agent in self.possible_agents}
        self.team_salaries = {agent: 0.0 for agent in self.possible_agents}  # Track spent salary

        self.salary_ranges = np.arange(0, 35, 5)
        self.contract_lengths = np.arange(1, 6)
        
        total_actions = self.n_players * len(self.salary_ranges) * len(self.contract_lengths)
        self._action_spaces = {agent: Discrete(total_actions) for agent in self.possible_agents}


        self.team_win_pct = {agent: 0.5 for agent in self.possible_agents}
        self.team_has_history = {agent: 0.0 for agent in self.possible_agents}

        #Flattened
        # flat_obs_size = (self.n_players * 4) + self.players_per_team

        # self._observation_spaces = {
        #     agent: Box(low=0, high=np.inf, shape=(flat_obs_size,), dtype=np.float32) 
        #     for agent in self.possible_agents
        # }
        ### Original obs space
        self._observation_spaces = {
            agent: Dict({
                "player_market": Box(low=0, high=np.inf, shape=self.players.shape, dtype=np.float32),
                "my_team": Box(low=0, high=np.inf, shape=(players_per_team,), dtype=np.float32),
                "win_pct": Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "has_history": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }) for agent in self.possible_agents
        }

        self.render_mode = render_mode
        self.g_list = generate_exact_nba_schedule(n_teams)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
        
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return
        print(f"Current Market Free Agents: {np.sum(self.players[:, 1] == 0)}")

    def observe(self, agent):
        observation = {
            "player_market": self.players.astype(np.float32),
            "my_team": self.teams[agent].astype(np.float32)
        }
        return observation

    # def observe(self, agent):
    #     # Flatten the player market matrix and your team's vector
    #     flat_market = self.players.astype(np.float32).ravel()
    #     flat_team = self.teams[agent].astype(np.float32)
        
    #     # Combine them into a single 1D array
    #     return np.concatenate([flat_market, flat_team])
    
    def close(self):
        pass

    def print_team_rosters(self):
        """
        Prints a detailed layout of all teams, showing their individual signed 
        player ratings, their filled roster spots, and their overall team strength.
        """
        print("\n" + "═"*75)
        print(f"{'TEAM NAME':<25} | {'ROSTER RATINGS (Top 10 Slots)':<35} | {'TOTAL STRENGTH':<12}")
        print("═"*75)
        
        for agent in self.possible_agents:
            real_name = self.agent_name_mapping[agent]
            team_vector = self.teams[agent]
            
            # Filter out the 0.0s to see how many active players are on the roster
            active_players = team_vector[team_vector > 0.0]
            total_strength = np.sum(team_vector)
            
            # Format the player ratings nicely to 2 decimal places
            # e.g., "[1.45, 0.92, 0.41]" or "Empty" if they haven't drafted yet
            if len(active_players) > 0:
                ratings_str = ", ".join([f"{r:.2f}" for r in active_players])
                # Truncate string representation if it gets too wide for the console column
                if len(ratings_str) > 33:
                    ratings_str = ratings_str[:30] + "..."
                roster_display = f"[{ratings_str}] ({len(active_players)}/{self.players_per_team})"
            else:
                roster_display = "Empty (0/10)"
                
            print(f"{real_name:<25} | {roster_display:<35} | {total_strength:<12.2f}")
            
        print("═"*75 + "\n")

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Re-generate player metrics
        player_ratings = sorted(stats.lognorm.rvs(loc=0, s=1, size=self.n_players), reverse=True)
        player_teams = np.zeros(self.n_players)
        player_ages = np.clip(np.round(np.random.normal(27, 4, size=self.n_players)), 19, 40)
        player_contract_lens = np.zeros(self.n_players)

        self.players = np.vstack([player_ratings, player_teams, player_ages, player_contract_lens]).T

        # Reset roster structure & salary pools
        self.teams = {agent: np.zeros(self.players_per_team) for agent in self.agents}
        self.team_salaries = {agent: 0.0 for agent in self.agents}

        self.team_win_pct = {agent: 0.5 for agent in self.agents}
        self.team_has_history = {agent: 0.0 for agent in self.agents}
        
        self.num_moves = 0
        self.season = 0

        self.g_list = generate_exact_nba_schedule(self.n_teams)

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    
    # --- Decomposed Helper Methods ---
    def _handle_agent_signing(self, agent, action):
        """
        Decodes the discrete action integer into Player, Salary, and Contract Length.
        """
        n_salaries = len(self.salary_ranges)
        n_lengths = len(self.contract_lengths)
        
        # Decode step-by-step
        contract_len_idx = action % n_lengths
        action //= n_lengths
        
        salary_idx = action % n_salaries
        player_id = action // n_salaries
        
        offered_salary = self.salary_ranges[salary_idx]
        chosen_length = self.contract_lengths[contract_len_idx]
        
        # --- GUARD CLAUSES ---
        # 1. Is the player already under contract with someone?
        if self.players[player_id, 1] != 0:
            return
            
        # 2. Does the annual salary break this team's cap space?
        if self.team_salaries[agent] + offered_salary > self.salary_cap:
            return
            
        # 3. Is the team roster already completely full?
        team_vector = self.teams[agent]
        empty_slots = np.where(team_vector == 0.0)[0]
        if len(empty_slots) == 0:
            return
            
        # --- EXECUTE SIGNING ---
        first_empty_slot = empty_slots[0]
        self.teams[agent][first_empty_slot] = self.players[player_id, 0]
        self.team_salaries[agent] += offered_salary
        
        # Save structural details to Market Matrix
        agent_numeric_idx = int(agent.split("_")[1])
        self.players[player_id, 1] = agent_numeric_idx + 1 # Team marker
        self.players[player_id, 3] = chosen_length          # Years marker

    def _is_draft_complete(self):
        """
        Evaluates whether all teams have finished their 10 rounds of draft moves.
        """
        max_moves = self.n_teams * self.players_per_team
        return self.num_moves >= max_moves

    def _simulate_and_reward_season(self):
        """
        Runs the mathematical season simulation and distributes playoff point payouts.
        Prints full league standings at completion.
        """
        # Extract aggregate team power metrics
        team_ratings = [np.sum(self.teams[f"team_{i}"]) for i in range(self.n_teams)]
        
        # Execute schedule calculation engine
        wins, games_played = play_season(team_ratings, self.g_list, noise_scale=10)

        for team_idx, total_wins in enumerate(wins):
            agent_name = f"team_{team_idx}"
            team_games = games_played[team_idx]
            self.team_win_pct[agent_name] = total_wins / team_games if team_games > 0 else 0.5
            self.team_has_history[agent_name] = 1.0
        
        # Rank teams 1 through 30 based on performance
        ranked_teams = sorted(enumerate(wins), key=lambda item: item[1], reverse=True)
        
        # --- PRINT COMPLETE STANDINGS DASHBOARD ---
        print("\n" + "="*55)
        print(f"{'RANK':<6} | {'TEAM NAME':<25} | {'WINS':<5} | {'REWARD':<8}")
        print("="*55)
        
        draft_order, order_dict = draft_lottery()
        # Distribute scores and print row items
        for position, (team_idx, total_wins) in enumerate(ranked_teams, start=1):
            agent_name = f"team_{team_idx}"
            real_name = self.agent_name_mapping[agent_name]
            
            # Calculate reward values
            if position <= 16:
                reward_val = reward_func(position)
                self.rewards[agent_name] = reward_val
                playoff_marker = "⭐"  # Mark playoff teams
            else:
                draft_seed = seed_position(position)
                draft_position = draft_order[draft_seed]
                reward_val = reward_func(draft_position, k = 0.4)
                self.rewards[agent_name] = reward_val
                playoff_marker = "  "

            # Print aligned standings row
            print(f"{position:<2} {playoff_marker} | {real_name:<25} | {int(total_wins):<5} | {reward_val:.4f}")
            
            if position == 16:
                print("-" * 55) # Draw a line under the playoff cutoff
                
        print("="*55 + "\n")

    def _player_update(self):
        ratings = self.players[:, 0]
        age = self.players[:, 2]
        print(self.players[:5])
        self.players[:, 2] = age + 1
        self.players[:, 0] = [evolve_func(r, a) for r, a in zip(ratings, age)]
        self.players[:, 0] = np.clip(self.players[:, 0], 0.1, np.inf)
        print(self.players[:5])

    def _new_entrants(self, n_new_players = 60):
        player_ratings = sorted(stats.lognorm.rvs(loc=0, s=0.8, size=n_new_players), reverse=True)
        player_teams = np.zeros(n_new_players)
        player_ages = np.clip(np.round(np.random.normal(21, 1, size = n_new_players)), 19, 40)
        player_contract_lens = np.zeros(n_new_players)

        new_players = np.vstack([player_ratings, player_teams, player_ages, player_contract_lens]).T
        return new_players

    # def retire_old_folk(self, n_to_retire = 60):

    def _prepare_next_season(self):
        """
        Updates multi-year contracts, returns expired players to the open market,
        and retains signed cores while resetting team salary calculations.
        """
        # 1. Find all currently signed players (where contract years > 0)
        signed_mask = self.players[:, 3] > 0
        
        # Dwindle contract lengths by 1 year
        self.players[signed_mask, 3] -= 1
        
        # 2. Find players whose contracts have officially expired this offseason
        expired_mask = (self.players[:, 3] == 0) & (self.players[:, 1] != 0)
        
        # Release expired players back to free agency
        self.players[expired_mask, 1] = 0.0 # Clear team ownership
        
        # 3. Rebuild rosters and calculate modern cap spendings for the next draft loop
        self.teams = {agent: np.zeros(self.players_per_team) for agent in self.possible_agents}
        self.team_salaries = {agent: 0.0 for agent in self.possible_agents}
        
        # Populate rosters with players who are still under contract
        for player_idx, player in enumerate(self.players):
            team_marker = int(player[1])
            if team_marker > 0: # If currently owned
                agent = f"team_{team_marker - 1}"
                rating = player[0]
                
                # Re-add to team vector
                team_vector = self.teams[agent]
                empty_slots = np.where(team_vector == 0.0)[0]
                if len(empty_slots) > 0:
                    self.teams[agent][empty_slots[0]] = rating
                    
                # Note: This basic system assumes fixed salaries per year. 
                # For high fidelity, you could add a 5th column to explicitly track self.players contract salary!
                # For now, we will assume an arbitrary default cap hold or recalculate dynamically.
                
    def step(self, action):
        """
        Executes exactly ONE environment step for the currently selected agent.
        Processes multiple seasons sequentially across separate steps.
        """
        current_agent = self.agent_selection
        
        # 1. Handle termination safety checks
        if self.terminations[current_agent] or self.truncations[current_agent]:
            self._was_dead_step(action)
            return
        
        # 2. Process free agency signing or no-op if market tracking phase is over
        max_draft_moves = self.n_teams * self.players_per_team  # 30 * 10 = 300 moves
        
        if self.num_moves < max_draft_moves:
            if self.num_moves == 0:
                print(f"\n--- Welcome to Season {self.season} Free Agency! ---")
            self._handle_agent_signing(current_agent, action)
            
        self.num_moves += 1

        # 3. Check if the draft is over and it's time to run the season simulation
        # We trigger this on the final agent's turn of the draft sequence
        if self.num_moves >= max_draft_moves and self._agent_selector.is_last():
            print(f"\nSimulating Season {self.season}!")
            self.print_team_rosters()
            self._simulate_and_reward_season()  # This populates self.rewards
            self._player_update()
            self._prepare_next_season()
            
            # Advance to the next season
            self.season += 1
            self.num_moves = 0  # Reset draft move counter for the next season
            
            # Clean up market contracts & rosters for the upcoming season's draft
            if self.season == self.n_seasons:
                # ALL seasons are finished -> Terminate the entire multi-season episode
                for agent in self.possible_agents:
                    self.terminations[agent] = True
            else:
                self._clear_rewards()

            # At the end of your multi-season loop
            print("--- Final Cumulative Rewards (Direct Inspection) ---")
            for agent in self.possible_agents:
                real_name = self.agent_name_mapping[agent]
                rolling_total = self._cumulative_rewards[agent]
                print(f"{real_name:<25}: {rolling_total:.4f}")
        else:
            # Multi-agent intermediate turns require reward clearing
            self._clear_rewards()
            
        # 4. Cycle turn pointer and accumulate values for PettingZoo wrappers
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()


#%%
env = FreeAgencyEnv(render_mode="human")
env.reset()

#%%
sorted_players = env.players[env.players[:, 0].argsort(descending = True)]
# sorted_players[:60]
sorted_players

#%%

def retirement_risk(age, k = 2.5):
    return np.exp((age - 35) / k)

print(env.players[:20])
scores = retirement_risk(env.players[:, 2])
sum_scores = np.sum(scores)
probs = scores / sum_scores
print(probs[:20])
print(np.max(probs))

p_to_retire = np.random.choice(len(env.players), size=60, p=probs, replace=False)
env.players[p_to_retire]

#%%

def retire_players(self, n_to_retire):
    scores = retirement_risk(self.players[:, 2])
    sum_scores = np.sum(scores)
    probs = scores / sum_scores
    players_to_retire = np.random.choice(len(self.players), size=60, p=probs, replace=False)
    env.players[p_to_retire]
    self.players = np.delete(self.players, (players_to_retire), axis = 0)
    return self.players[self.players[:, 0].argsort(descending = True)]


#%%
if __name__ == "__main__":
    # Initialize the environment
    env = FreeAgencyEnv(render_mode="human")
    env.reset(seed=42)

    print("Starting Random Free Agency Draft Loop...\n")
    # Standard PettingZoo AEC loop
    for agent in env.agent_iter():
        # 1. Fetch current observation, reward, termination, truncation, and info
        observation, reward, termination, truncation, info = env.last()
        # print(f"Observation = {reward}")
        # 2. Check if the episode is finished for this agent
        if termination or truncation:
            # We pass None as the action to dead/terminated agents
            env.step(None)
            continue
        # print(env.rewards)
        # 3. Sample a random action from this specific agent's action space
        # (Since it's a Discrete space, it returns a random integer)
        random_action = env.action_space(agent).sample()
        
        # 4. Step the environment forward with the random move
        env.step(random_action)
