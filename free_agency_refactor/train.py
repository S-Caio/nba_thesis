"""
Custom RLlib TorchModelV2 for FreeAgencyEnv.

Key idea (action masking):
    RLlib gives us `obs["action_mask"]`, a {0,1} vector over the full
    Discrete(n_actions) space. We compute raw logits over all actions as
    normal, then add `log(mask)` to them before returning. For mask==1
    that's `+log(1) == 0` (no change). For mask==0 that's `log(0) == -inf`,
    which we clamp to `FLOAT_MIN` (a very large negative float, not
    literal -inf) so the softmax/log-softmax stays numerically stable
    and never produces NaNs, while still driving the probability of
    invalid actions to ~0.

Observation handling:
    player_market is (n_players, N_PLAYER_COLS) and is sorted by rating
    every step, so position in the array is meaningful. We do NOT use a
    permutation-invariant (DeepSets / attention-pooling) encoder here --
    a positional Conv1d + flatten is enough, and is cheaper. If you ever
    stop sorting player_market (e.g. add/remove/shuffle entries such
    that order becomes arbitrary), swap the player encoder for a
    DeepSets-style mean/max pool over per-player embeddings instead --
    the rest of the model doesn't need to change.
"""

import torch
import torch.nn as nn

from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN


class FreeAgencyMaskedModel(TorchModelV2, nn.Module):
    """
    Expects the Dict observation space defined in FreeAgencyEnv:
        action_mask  : (n_actions,)
        player_market: (n_players, N_PLAYER_COLS)
        my_team      : (players_per_team,)
        win_pct      : (1,)
        team_salary  : (1,)
        standing     : (1,)
        has_history  : (1,)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # obs_space may already be a Dict (new-stack, no preprocessor) or a
        # flattened Box with the real Dict stashed on `.original_space`
        # (old-stack default preprocessor). Handle both.
        self.orig_space = getattr(obs_space, "original_space", obs_space)
        print("ORIG SPACE KEYS:", list(self.orig_space.spaces.keys()))


        self.n_players, self.n_player_cols = self.orig_space["player_market"].shape
        self.players_per_team = self.orig_space["my_team"].shape[0]

        player_embed_dim = 32
        conv_hidden = model_config.get("custom_model_config", {}).get("conv_hidden", 64)

        # Positional encoder over the ratings-sorted player list.
        # Conv1d over the player axis lets each player's embedding see its
        # rank-neighbors (players just above/below it in rating).
        self.player_conv = nn.Sequential(
            nn.Conv1d(self.n_player_cols, conv_hidden, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_hidden, player_embed_dim, kernel_size=1, padding=1),
            nn.ReLU(),
        )
        player_feat_dim = player_embed_dim * self.n_players  # flatten preserves order

        self.team_mlp = nn.Sequential(
            nn.Linear(self.players_per_team, 64),
            nn.ReLU(),
        )

        scalar_dim = 4  # win_pct, team_salary, standing, has_history

        combined_dim = player_feat_dim + 64 + scalar_dim

        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.logits_layer = nn.Linear(256, num_outputs)
        self.value_layer = nn.Linear(256, 1)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # If RLlib handed us a flattened tensor instead of the Dict
        # (depends on stack/version), unflatten it back into the pieces.
        if not isinstance(obs, dict):
            obs = restore_original_dimensions(obs, self.obs_space, "torch")

        action_mask = obs["action_mask"].float()
        player_market = obs["player_market"].float()   # (B, n_players, n_cols)
        my_team = obs["my_team"].float()                # (B, players_per_team)
        win_pct = obs["win_pct"].float()
        team_salary = obs["team_salary"].float()
        standing = obs["standing"].float()
        has_history = obs["has_history"].float()

        # Conv1d wants channel-first: (B, C=n_cols, L=n_players)
        x = player_market.permute(0, 2, 1)
        x = self.player_conv(x)                          # (B, embed_dim, n_players)
        x = x.reshape(x.shape[0], -1)                     # flatten, order preserved

        team_feat = self.team_mlp(my_team)

        scalars = torch.cat([win_pct, team_salary, standing, has_history], dim=-1)

        combined = torch.cat([x, team_feat, scalars], dim=-1)
        trunk_out = self.trunk(combined)

        logits = self.logits_layer(trunk_out)
        self._value_out = self.value_layer(trunk_out).squeeze(-1)

        # --- action masking ---
        # log(0) = -inf, log(1) = 0. Clamp so invalid-action logits become
        # a very large negative finite number (FLOAT_MIN) instead of -inf,
        # which keeps softmax / log-prob computations finite.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return self._value_out


ModelCatalog.register_custom_model("free_agency_masked_model", FreeAgencyMaskedModel)


# ---------------------------------------------------------------------------
# Usage sketch: wiring FreeAgencyEnv (a PettingZoo AECEnv) + this model into
# RLlib's PPO. Adjust import paths for your project layout.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from free_agency.env import FreeAgencyEnv
    from ray.rllib.env import PettingZooEnv
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    # from free_agency_env import FreeAgencyEnv  # your module

    def env_creator(config):
        # PettingZooEnv wraps an AECEnv for RLlib's multi-agent API.
        return PettingZooEnv(FreeAgencyEnv())  # noqa: F821 (import above)

    register_env("free_agency_v1", env_creator)

    sample_env = PettingZooEnv(FreeAgencyEnv())  # noqa: F821
    obs_space = sample_env.observation_space["team_0"]
    act_space = sample_env.action_space["team_0"]

    # print(f"OBS SPACE: {obs_space}")

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("free_agency_v1")
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .training(
            model={
                "custom_model": "free_agency_masked_model",
                # example: "custom_model_config": {"conv_hidden": 64},
            }
        )
    )

    algo = config.build()
    for i in range(5):
        result = algo.train()
        print(f"iter {i}: episode_reward_mean={result.get('episode_reward_mean')}")