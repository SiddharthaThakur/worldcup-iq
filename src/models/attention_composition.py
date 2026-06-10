"""
Attention-based team composition model (Challenger 2).

What this does in simple English:
    Challenger 1 averages player stats by position — it treats a team as
    the sum of its parts. But football intuition says players INTERACT:
    a ball-playing center-back is worth more next to a fast partner who
    covers space; a creative midfielder needs runners ahead of him.

    This model uses a small transformer that "attends" over all 11 player
    vectors at once, letting it learn interaction patterns from ~100K club
    matches with known lineups. The isolated research question:

        Does modeling player interactions beat simple averaging?

    HIGFormer (KDD 2025) found interactions matter for club football.
    Nobody has tested whether that transfers to international football,
    where teammates train together a few weeks a year. Plausible answer:
    club chemistry signal does NOT transfer. That's why this is Challenger 2
    behind a kill criterion, not the headline.

Kill criteria (pre-registered, DECISIONS.md D009):
    - Must beat the aggregated model on held-out club seasons (Poisson NLL)
    - Must not degrade Brier vs Challenger 1 on the 2018+2022 WC backtests
    If either fails: report the negative result, do not deploy to the
    live scorecard.

Requires: pip install torch (the [torch] extra in pyproject.toml).
"""

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None


@dataclass
class AttentionConfig:
    n_features: int = 17        # player embedding dim (matches player_embeddings.py)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_position_groups: int = 4  # GK / DEF / MID / FWD


if TORCH_AVAILABLE:

    class TeamCompositionTransformer(nn.Module):
        """Transformer over 11 player embeddings → team strength scalar.

        Architecture:
            player features → linear projection to d_model
            + learned position-group embedding (GK/DEF/MID/FWD)
            → TransformerEncoder (n_layers, n_heads)
            → mean-pool over players → MLP head → scalar strength

        The output strength feeds the SAME Dixon-Coles likelihood as the
        other models — only the strength source differs, keeping the
        champion-challenger comparison clean.
        """

        def __init__(self, config: AttentionConfig | None = None):
            super().__init__()
            self.config = config or AttentionConfig()
            c = self.config

            self.input_proj = nn.Linear(c.n_features, c.d_model)
            self.position_embedding = nn.Embedding(c.n_position_groups, c.d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=c.d_model, nhead=c.n_heads,
                dim_feedforward=c.d_model * 4, dropout=c.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.n_layers)
            self.strength_head = nn.Sequential(
                nn.Linear(c.d_model, c.d_model // 2),
                nn.GELU(),
                nn.Linear(c.d_model // 2, 1),
            )

        def forward(self, player_features: "torch.Tensor",
                    position_groups: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                player_features: (batch, 11, n_features)
                position_groups: (batch, 11) ints in {0:GK, 1:DEF, 2:MID, 3:FWD}
            Returns:
                (batch,) team strength scores
            """
            x = self.input_proj(player_features)
            x = x + self.position_embedding(position_groups)
            x = self.encoder(x)
            pooled = x.mean(dim=1)
            return self.strength_head(pooled).squeeze(-1)


    class MatchOutcomeModel(nn.Module):
        """Wraps two team-strength computations into a match goal model.

        log(lambda_home) = intercept + scale * (s_home - s_away) + home_adv * is_home
        log(lambda_away) = intercept - scale * (s_home - s_away)

        Trained end-to-end with Poisson NLL on observed goals — the same
        functional form fitted for the Elo baseline, so the only varying
        factor is where strength comes from.
        """

        def __init__(self, config: AttentionConfig | None = None):
            super().__init__()
            self.composer = TeamCompositionTransformer(config)
            self.intercept = nn.Parameter(torch.tensor(0.3))
            self.scale = nn.Parameter(torch.tensor(0.1))
            self.home_adv = nn.Parameter(torch.tensor(0.2))

        def forward(self, home_players, home_positions, away_players,
                    away_positions, is_home_advantage):
            s_home = self.composer(home_players, home_positions)
            s_away = self.composer(away_players, away_positions)
            diff = s_home - s_away
            log_lam_h = self.intercept + self.scale * diff + self.home_adv * is_home_advantage
            log_lam_a = self.intercept - self.scale * diff
            return log_lam_h.clamp(-3, 2.5), log_lam_a.clamp(-3, 2.5)


    def poisson_nll(log_lam: "torch.Tensor", goals: "torch.Tensor") -> "torch.Tensor":
        """Poisson negative log-likelihood (up to a constant)."""
        return (torch.exp(log_lam) - goals * log_lam).mean()


def check_torch() -> None:
    """Loud failure if torch isn't installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch required for the attention model: pip install -e '.[torch]'"
        )
