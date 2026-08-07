"""
FreeAgencyEnvParallel: the PettingZoo Parallel surface. Every method here is now
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
from pettingzoo import ParallelEnv
from gymnasium.utils import seeding
import pprint

from .constants import LeagueConfig, N_PLAYER_COLS, HISTORY_WINDOW, CAP_HORIZON, TEAM, AGE
from .state import LeagueState, new_league_state, record_season_history, agent_to_team_id, initial_endowment
# from .contracts import handle_signing, contract_update, make_action_mask
from .contracts import submit_offer, resolve_offers, contract_update, make_action_mask, compute_cap_projection, make_free_agent_market_and_mapping, FREE_AGENT_MARKER
from .rosters import rebuild_rosters, print_team_rosters
from .player_lifecycle import player_update, run_rookie_draft
from .season_sim import generate_exact_nba_schedule
from .season_runner import simulate_and_reward_season_parallel

nba_teams = [f"Placeholder Team {i}" for i in range(30)]  # swap for your real nba_teams list


class FreeAgencyEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "nba_free_agency_v1"}

    def __init__(self, render_mode=None, config: LeagueConfig | None = None):
        self.config = config or LeagueConfig()
        self.possible_agents = [f"team_{i}" for i in range(self.config.n_teams)]
        self.agents = [f"team_{i}" for i in range(self.config.n_teams)]
        self.agent_name_mapping = dict(zip(self.possible_agents, nba_teams))
        self.render_mode = render_mode

        self.league = new_league_state(self.config, self.possible_agents)
        initial_endowment(self.league, self.config)
        rebuild_rosters(self.league, self.config)
        self.ages = self.calculate_ages()
        self.n_players_z = self.players_relative_z_score()
        self.rel_strength = self.strength_relative_z_score()
        self.free_agents, self.free_agent_mapping = make_free_agent_market_and_mapping(self.league, self.config)
        self.g_list = generate_exact_nba_schedule(self.config.n_teams)
        self.season = 0
        self.team_standing = {agent : self.config.n_teams // 2 for agent in self.possible_agents}

        self.n_contract_actions = self.config.n_proper_actions
        self.n_actions = self.n_contract_actions + 1

        self.max_moves = 150

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
                "action_mask" : Box(low = 0, high = 1, shape = (self.n_actions, ), dtype = np.int8),
                # "player_market": Box(low=0, high=np.inf,
                #                       shape=(self.config.n_players, N_PLAYER_COLS), dtype=np.float32),
                "free_agents" : Box(low=-1, high=np.inf,
                                      shape=(self.config.n_free_agents, N_PLAYER_COLS), dtype=np.float32),
                "my_team": Box(low=0, high=np.inf, shape=(self.config.players_per_team,), dtype=np.float32),
                "my_team_rating" : Box(low = 0, high = np.inf, shape = (1, ), dtype = np.float32),
                "relative_team_strength" : Box(low = -np.inf, high = np.inf, shape = (1, ), dtype = np.float32),
                "my_team_avg_age" : Box(low = 18, high = 40, shape = (1,), dtype = np.float32),
                "n_players_team" : Box(low = 0, high = 10, shape = (1,), dtype = np.int8),
                "n_players_team_relative" : Box(low = -np.inf, high = np.inf, shape = (1,), dtype = np.float32),
                "win_pct": Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "season" : Box(low = 0, high = 1, shape=(1,), dtype = np.float32),
                "team_salary" : Box(low = 0, high = self.config.salary_cap, shape = (1,), dtype = np.float32),
                "standing" : Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "has_history": Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "win_pct_history": Box(low=0, high=1, shape=(HISTORY_WINDOW,), dtype=np.float32),
                "rating_history": Box(low=0, high=np.inf, shape=(HISTORY_WINDOW,), dtype=np.float32),
                "history_mask": Box(low=0, high=1, shape=(HISTORY_WINDOW,), dtype=np.int8),
                "cap_projection" : Box(low = 0, high = 1, shape = (CAP_HORIZON, ), dtype = np.float32),
            })

    @functools.lru_cache(maxsize = None)
    def action_space(self, agent):
        return Discrete(self.n_actions)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return
        print(f"Current Market Free Agents: {np.sum(self.league.players[:, 1] == 0)}")

    def observe(self, agent):
        return {
            "action_mask": make_action_mask(self.league, self.config, agent, self.free_agents),
            # "player_market": self.league.players.astype(np.float32),
            "free_agents" : self.free_agents.astype(np.float32),
            "my_team": self.league.teams[agent].astype(np.float32),
            "my_team_rating": np.array([np.sum(self.league.teams[agent])], dtype=np.float32),
            "relative_team_strength" : np.array([self.rel_strength[agent]], dtype=np.float32),
            "my_team_avg_age" : np.array([self.ages[agent]]),
            "n_players_team" : np.array([np.count_nonzero(self.league.teams[agent]).astype(np.int8)]),
            "n_players_team_relative" : np.array([self.n_players_z[agent] / 10], dtype=np.float32),
            "win_pct": np.array([self.league.team_win_pct[agent]], dtype=np.float32),
            "team_salary": np.array([self.league.team_salaries[agent] / self.config.salary_cap], dtype=np.float32),
            "standing": np.array([self.team_standing[agent] / self.config.n_teams], dtype=np.float32),
            "has_history": np.array([self.league.team_has_history[agent]], dtype=np.float32),
            "win_pct_history": self.league.team_win_pct_history[agent],
            "rating_history": self.league.team_rating_history[agent],
            "history_mask": self._history_mask,
            "cap_projection" : (compute_cap_projection(self.league, agent, CAP_HORIZON) / self.config.salary_cap).astype(np.float32)
        }

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        # print("RESET")
        self.np_random, self.np_random_seed = seeding.np_random(seed)


        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.team_standing = {agent : self.config.n_teams // 2 for agent in self.possible_agents}


        self.league = new_league_state(self.config, self.agents)
        initial_endowment(self.league, self.config)
        rebuild_rosters(self.league, self.config)
        self.free_agents, self.free_agent_mapping = make_free_agent_market_and_mapping(self.league, self.config)
        self.ages = self.calculate_ages()
        self.n_players_z = self.players_relative_z_score()
        self.rel_strength = self.strength_relative_z_score()
        self.g_list = generate_exact_nba_schedule(self.config.n_teams)

        self.num_moves = 0
        self.season = 0
        self._history_mask = self._compute_history_mask()
        self.full_draft_order = self.possible_agents[:]

        observations = {agent : self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # self._agent_selector = AgentSelector(self.agents)
        # self.agent_selection = self._agent_selector.next()

        return observations, infos

    def _league_ready(self):
        min_players = self.config.players_per_team - 1

        for team in self.possible_agents:
            roster_size = np.count_nonzero(self.league.teams[team])
            if roster_size < min_players:
                return False

        return True

    def step(self, actions):
        # if self.num_moves % 5 == 0:
        #     print(f"step {self.num_moves}")
        # If environment has no active agents, return empty dictionaries
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Lock in the list of agents participating in this step
        current_agents = self.agents[:]

        self.rewards = {agent: 0.0 for agent in current_agents}
        self.terminations = {agent: False for agent in current_agents}
        self.truncations = {agent: False for agent in current_agents}

        # Submit actions for active agents
        for agent, action in actions.items():
            if agent in current_agents:
                submit_offer(self.league, self.config, agent, action, self.free_agent_mapping)

        self.num_moves += 1
        resolve_offers(self.league, self.config, self.np_random)
        self.free_agents, self.free_agent_mapping = make_free_agent_market_and_mapping(self.league, self.config)
        self.ages = self.calculate_ages()
        self.n_players_z = self.players_relative_z_score()
        self.rel_strength = self.strength_relative_z_score()


        # Truncation safety: prevent infinite free-agency loops
        stalled = self.num_moves >= self.max_moves and not self._league_ready()

        if self._league_ready():
            self._run_season_boundary()  # may overwrite self.terminations, keyed over possible_agents


        if stalled:
            print("Stalled the simulation!")
            for agent in current_agents:
                self.truncations[agent] = True

        # 2. Build output dicts scoped to current_agents, no matter what internal
        #    state looks like. This is the ONLY place PettingZoo-facing dicts get built.
        observations = {agent: self.observe(agent) for agent in current_agents}
        rewards      = {agent: self.rewards.get(agent, 0.0) for agent in current_agents}
        terminations = {agent: self.terminations.get(agent, False) for agent in current_agents}
        truncations  = {agent: self.truncations.get(agent, False) for agent in current_agents}
        infos        = {agent: {} for agent in current_agents}

        # 3. Update self.agents AFTER building output for step t
        self.agents = [
            agent for agent in current_agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos
        # if self._agent_selector.is_last():
        #     resolve_offers(self.league, self.config, self.np_random)
        #     if self._league_ready():
        #         self._run_season_boundary()
        #     else:
        #         self._clear_rewards()

        # self.agent_selection = self._agent_selector.next()
        # self._accumulate_rewards()


    def calculate_ages(self):
        ages_dict = {}
        all_ages = self.league.players[self.league.players[:, TEAM] != FREE_AGENT_MARKER][:, AGE]
        league_avg_age = np.mean(all_ages).astype(np.float32) if all_ages.size > 0 else np.float32(25.0)

        for agent in self.agents:
            team_ages = self.league.players[self.league.players[:, TEAM] == agent_to_team_id(agent)][:, AGE]
            if team_ages.size > 0:
                ages_dict[agent] = np.mean(team_ages).astype(np.float32)
            else:
                ages_dict[agent] = league_avg_age

        return ages_dict

    @staticmethod
    def _relative_z(values_dict, eps=1e-8, use_std = True):
        """
        values_dict: {agent: scalar}
        Returns {agent: z_score}, computed across all agents in values_dict.
        """
        agents = list(values_dict.keys())
        values = np.array([values_dict[a] for a in agents], dtype=np.float32)

        mean, std = values.mean(), values.std()
        if use_std:
            z = (values - mean) / (std + eps)
        else:
            z = values - mean

        return {agent: np.float32(z_i) for agent, z_i in zip(agents, z)}

    def players_relative_z_score(self):
        n_players = {
            agent: np.count_nonzero(roster)
            for agent, roster in self.league.teams.items()
        }
        return self._relative_z(n_players, use_std=False)

    def strength_relative_z_score(self):
        strength = {
            agent: np.sum(roster)
            for agent, roster in self.league.teams.items()
        }
        return self._relative_z(strength)
        
    def _run_season_boundary(self) -> None:
        # print("Season boundary")
        # self._clear_rewards()

        self.full_draft_order, self.rewards, self.team_standing = simulate_and_reward_season_parallel(
            self.league, self.config, self.g_list, self.agent_name_mapping, self.rewards
        )

        for agent in self.possible_agents:                      
            record_season_history(
                self.league, agent,                               
                self.league.team_win_pct[agent],                  
                float(np.sum(self.league.teams[agent])),          
            )                                                      

        player_update(self.league)
        contract_update(self.league)
        rebuild_rosters(self.league, self.config)
        run_rookie_draft(self.league, self.config, self.full_draft_order,
                        n_to_retire=self.config.n_new_entrants_per_season)
        rebuild_rosters(self.league, self.config)

        self.season += 1
        self._history_mask = self._compute_history_mask()
        self.num_moves = 0

        if self.season == self.config.n_seasons:
            for agent in self.possible_agents:
                self.terminations[agent] = True
        else:
            for agent in self.possible_agents:
                self.terminations[agent] = False



    def _compute_history_mask(self):
        filled = min(self.season, self.config.history_window)
        mask = np.zeros(self.config.history_window, dtype=np.int8)
        if filled > 0:
            mask[-filled:] = 1
        return mask