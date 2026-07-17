"""
FreeAgencyEnv: the PettingZoo AEC surface. Every method here is now
either (a) pure API plumbing (reset/step/observe/action_space) or
(b) a thin call into contracts.py / rosters.py / player_lifecycle.py /
season_runner.py. The env no longer *contains* the business logic --
it *sequences* it. That sequencing (what happens each step, what
happens at season boundary) is genuinely the env's job and stays here.
"""
import functools
import numpy as np
import gymnasium
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from gymnasium.utils import seeding

from .constants import LeagueConfig, N_PLAYER_COLS
from .state import LeagueState, new_league_state
from .contracts import handle_signing, contract_update, make_action_mask
from .rosters import rebuild_rosters, print_team_rosters
from .player_lifecycle import player_update, run_rookie_draft
from .season_sim import generate_exact_nba_schedule
from .season_runner import simulate_and_reward_season

nba_teams = [f"Placeholder Team {i}" for i in range(30)]  # swap for your real nba_teams list


class FreeAgencyEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "nba_free_agency_v1"}

    def __init__(self, render_mode=None, config: LeagueConfig | None = None):
        self.config = config or LeagueConfig()
        self.possible_agents = [f"team_{i}" for i in range(self.config.n_teams)]
        self.agent_name_mapping = dict(zip(self.possible_agents, nba_teams))
        self.render_mode = render_mode

        self.league = new_league_state(self.config, self.possible_agents)
        self.g_list = generate_exact_nba_schedule(self.config.n_teams)
        self.season = 0
        self.team_standing = {agent : self.config.n_teams // 2 for agent in self.possible_agents}

        self.n_contract_actions = self.config.n_players * len(self.config.salary_ranges) * len(self.config.contract_lengths)
        self.n_actions = self.n_contract_actions + 1

        self._action_spaces = {
            agent: Discrete(self.n_actions)
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: Dict({
                "action_mask" : Box(low = 0, high = 1, shape = (self.n_actions, ), dtype = np.int8),
                "player_market": Box(low=0, high=np.inf,
                                      shape=(self.config.n_players, N_PLAYER_COLS), dtype=np.float32),
                "my_team": Box(low=0, high=np.inf, shape=(self.config.players_per_team,), dtype=np.float32),
                "win_pct": Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "team_salary" : Box(low = 0, high = self.config.salary_cap, shape = (1,), dtype = np.int32),
                "standing" : Box(low=0, high=1, shape=(1,), dtype=np.int32),
                "has_history": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }) for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return
        print(f"Current Market Free Agents: {np.sum(self.league.players[:, 1] == 0)}")

    def observe(self, agent):
        return {
            "action_mask" : make_action_mask(self.league, self.config, agent),
            "player_market": self.league.players.astype(np.float32),
            "my_team": self.league.teams[agent].astype(np.float32),
            "win_pct": np.array([self.league.team_win_pct[agent]], dtype=np.float32),
            "team_salary" : np.array([self.league.team_salaries[agent] / self.config.salary_cap], dtype=np.int32),
            "standing" : np.array([self.team_standing[agent] / self.config.n_teams], dtype=np.int32),
            "has_history": np.array([self.league.team_has_history[agent]], dtype=np.float32),
        }

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.team_standing = {agent : self.config.n_teams // 2 for agent in self.possible_agents}


        self.league = new_league_state(self.config, self.agents)
        self.g_list = generate_exact_nba_schedule(self.config.n_teams)

        self.num_moves = 0
        self.season = 0
        self.full_draft_order = self.possible_agents[:]

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def _league_ready(self):
        min_players = self.config.players_per_team - 1

        for team in self.possible_agents:
            roster_size = np.count_nonzero(self.league.teams[team])
            if roster_size < min_players:
                return False

        return True

    def step(self, action):
        current_agent = self.agent_selection

        if self.terminations[current_agent] or self.truncations[current_agent]:
            self._was_dead_step(action)
            return

        # max_draft_moves = self.config.n_teams * self.config.players_per_team

        # if self.num_moves < max_draft_moves:
        if self.num_moves == 0:
            print(f"\n--- Welcome to Season {self.season} Free Agency! ---")
        handle_signing(self.league, self.config, current_agent, action)

        self.num_moves += 1

        if self._agent_selector.is_last():
            if self._league_ready():
                self._run_season_boundary()
            else:
                self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
        
        print("--- Final Cumulative Rewards (Direct Inspection) ---")
        for agent in self.possible_agents:
            real_name = self.agent_name_mapping[agent]
            print(f"{real_name:<25}: {self._cumulative_rewards[agent]:.4f}")

    def _run_season_boundary(self) -> None:
        """Everything that happens once a season's draft is complete:
        simulate the season, age/retire/replace players, run the next
        rookie draft, and either advance or terminate."""
        self._clear_rewards() 

        print(f"\nSimulating Season {self.season}!")
        print_team_rosters(self.league, self.config, self.agent_name_mapping)

        self.full_draft_order, self.team_standings = simulate_and_reward_season(
            self.league, self.config, self.g_list, self.agent_name_mapping, self.rewards
        )

        player_update(self.league)
        contract_update(self.league)
        # Rebuild twice: once so the rookie draft's roster/cap checks see
        # accurate occupancy, once more to fold newly drafted rookies in.
        rebuild_rosters(self.league, self.config)
        run_rookie_draft(self.league, self.config, self.full_draft_order,
                          n_to_retire=self.config.n_new_entrants_per_season)
        rebuild_rosters(self.league, self.config)

        self.season += 1
        self.num_moves = 0

        if self.season == self.config.n_seasons:
            for agent in self.possible_agents:
                self.terminations[agent] = True
        # else:
        #     self._clear_rewards()
