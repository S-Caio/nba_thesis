import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN


class FreeAgencyMaskedModel(TorchRLModule, ValueFunctionAPI):
    """
    Expects the Dict observation space defined in FreeAgencyEnv:
        action_mask               : (n_actions,)
        free_agents                : (n_free_agents, N_PLAYER_COLS)
        my_team                   : (players_per_team,)
        my_team_rating             : (1,)
        my_team_avg_age             : (1,)
        n_players_team               : (1,)
        win_pct                     : (1,)
        season                       : (1,)
        team_salary                   : (1,)
        standing                       : (1,)
        has_history                     : (1,)
        relative_team_strength           : (1,)   # z-score of team strength vs league
        n_players_team_relative            : (1,)   # z-score of roster headcount vs league
    """

    def setup(self):
        super().setup()
        print("=== MODEL CUDA DIAGNOSTIC ===")
        print("torch:", torch.__version__)
        print("torch CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())

        try:
            print("CUDA device:", torch.cuda.current_device())
            print("CUDA name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("CUDA INIT ERROR:", repr(e))
        # self.observation_space / self.action_space / self.model_config are
        # already set by the base class before setup() runs.
        self.orig_space = getattr(
            self.observation_space, "original_space", self.observation_space
        )

        self.n_free_agents, self.n_player_cols = self.orig_space["free_agents"].shape
        self.players_per_team = self.orig_space["my_team"].shape[0]
        self.history_window = self.orig_space["win_pct_history"].shape[0]
        self.cap_horizon = self.orig_space["cap_projection"].shape[0]

        player_embed_dim = 32
        conv_hidden = self.model_config.get("conv_hidden", 64)

        # Positional encoder over the ratings-sorted free-agent list.
        self.player_conv = nn.Sequential(
            nn.Conv1d(self.n_player_cols, conv_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(conv_hidden, player_embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        player_feat_dim = player_embed_dim * self.n_free_agents

        self.market_encoder = nn.Sequential(
            nn.Linear(player_feat_dim, 128),
            nn.LeakyReLU(),
        )

        # my_team_rating, win_pct, team_salary, standing, has_history,
        # relative_team_strength, n_players_team_relative,
        # my_team_avg_age, n_players_team
        scalar_dim = 9
        history_input_dim = self.history_window * 3
        team_raw_dim = (
            self.players_per_team + scalar_dim + history_input_dim + self.cap_horizon
        )
        self.team_norm = nn.LayerNorm(team_raw_dim)
        self.team_encoder = nn.Sequential(
            nn.Linear(team_raw_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )

        combined_dim = 128 + 128 + 32
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )

        num_outputs = self.action_space.n
        self.logits_layer = nn.Linear(128, num_outputs)
        self.value_layer = nn.Linear(128, 1)

    def _compute_trunk(self, batch):
        """Shared feature extraction used by both the policy and value heads."""
        obs = batch[Columns.OBS]

        action_mask = obs["action_mask"].float()
        free_agents = obs["free_agents"].float()  # (B, n_free_agents, n_cols)
        my_team = obs["my_team"].float()
        my_team_rating = obs["my_team_rating"].float()
        my_team_avg_age = obs["my_team_avg_age"].float()
        n_players_team = obs["n_players_team"].float()
        win_pct = obs["win_pct"].float()
        team_salary = obs["team_salary"].float()
        standing = obs["standing"].float()
        has_history = obs["has_history"].float()
        relative_team_strength = obs["relative_team_strength"].float()
        n_players_team_relative = obs["n_players_team_relative"].float()
        history_mask = obs["history_mask"].float()
        win_pct_history = obs["win_pct_history"].float() * history_mask
        rating_history = obs["rating_history"].float() * history_mask
        cap_projection = obs["cap_projection"].float()

        # Conv1d wants channel-first: (B, C=n_cols, L=n_free_agents)
        x = free_agents.permute(0, 2, 1)
        x = self.player_conv(x)
        market_summary = torch.mean(x, dim=2)
        x_flat = x.reshape(x.shape[0], -1)
        market_feat = self.market_encoder(x_flat)

        history_vars = torch.cat([win_pct_history, rating_history, history_mask], dim=-1)
        scalars = torch.cat(
            [
                my_team_rating,
                win_pct,
                team_salary,
                standing,
                has_history,
                relative_team_strength,
                n_players_team_relative,
                my_team_avg_age,
                n_players_team,
            ],
            dim=-1,
        )

        raw_team_context = torch.cat([my_team, scalars, history_vars, cap_projection], dim=-1)
        normed_team_context = self.team_norm(raw_team_context)
        team_feat = self.team_encoder(normed_team_context)

        combined = torch.cat([market_feat, market_summary, team_feat], dim=-1)
        trunk_out = self.trunk(combined)

        return trunk_out, action_mask

    def _masked_logits(self, trunk_out, action_mask):
        logits = self.logits_layer(trunk_out)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        return logits + inf_mask

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Covers forward_exploration() and forward_inference() as long as you
        # don't need separate exploration/inference behavior.
        trunk_out, action_mask = self._compute_trunk(batch)
        masked_logits = self._masked_logits(trunk_out, action_mask)
        return {Columns.ACTION_DIST_INPUTS: masked_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        # Called separately by the PPO Learner to get value estimates.
        # We recompute the trunk here since we didn't cache embeddings from
        # _forward(); this costs one extra forward pass through the shared
        # trunk during training but keeps the module simple and correct.
        trunk_out, _ = self._compute_trunk(batch)
        return self.value_layer(trunk_out).squeeze(-1)