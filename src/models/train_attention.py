"""
Phase 4 head-to-head: attention vs aggregation (H3).

What this does in simple English:
    Two models read the EXACT same starting XIs. The aggregation model
    averages the 11 players' features (Challenger 1's idea, as a neural
    net). The attention model lets the 11 players "look at" each other
    before being combined, so it can learn that some players are worth
    more alongside certain teammates. Both then predict the match's goals
    the same way. We train them on past Big-5 matches and compare their
    error on a held-out later period it never saw.

    The verdict (H3 / D009): does attention beat aggregation on held-out
    Poisson error? If yes by a clear margin, interactions carry signal. If
    not, "averaging is enough" — a publishable negative result, and the
    attention model does NOT deploy.

Run: python -m src.models.train_attention
"""

import json
from pathlib import Path

import numpy as np

from src.models.attention_composition import AttentionConfig, check_torch

DATA = Path("data/processed/lineup_dataset.npz")
REPORT = Path("data/predictions/phase4_h3_report.json")


def _load_split(holdout_frac: float = 0.2):
    d = np.load(DATA)
    order = np.argsort(d["dates"])  # chronological
    n = len(order)
    cut = int(n * (1 - holdout_frac))
    tr, te = order[:cut], order[cut:]
    return d, tr, te


def run(epochs: int = 40, seed: int = 0) -> dict:
    check_torch()
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    d, tr, te = _load_split()

    # Standardize features on train stats
    Xall = np.concatenate([d["Xh"], d["Xa"]], axis=0)
    mu = Xall[np.concatenate([tr, tr + len(d["Xh"])])].reshape(-1, Xall.shape[-1]).mean(0)
    sd = Xall[np.concatenate([tr, tr + len(d["Xh"])])].reshape(-1, Xall.shape[-1]).std(0) + 1e-6

    def t(name, idx):
        return torch.tensor((d[name][idx] - mu) / sd, dtype=torch.float32)

    def tp(name, idx):
        return torch.tensor(d[name][idx], dtype=torch.long)

    Xh_tr, Ph_tr = t("Xh", tr), tp("Ph", tr)
    Xa_tr, Pa_tr = t("Xa", tr), tp("Pa", tr)
    gh_tr = torch.tensor(d["gh"][tr], dtype=torch.float32)
    ga_tr = torch.tensor(d["ga"][tr], dtype=torch.float32)
    Xh_te, Ph_te = t("Xh", te), tp("Ph", te)
    Xa_te, Pa_te = t("Xa", te), tp("Pa", te)
    gh_te = torch.tensor(d["gh"][te], dtype=torch.float32)
    ga_te = torch.tensor(d["ga"][te], dtype=torch.float32)

    cfg = AttentionConfig(n_features=Xall.shape[-1], d_model=48, n_heads=4, n_layers=2)

    class Composer(nn.Module):
        """Shared backbone; `attention` toggles the transformer on/off."""

        def __init__(self, attention: bool):
            super().__init__()
            self.attention = attention
            self.proj = nn.Linear(cfg.n_features, cfg.d_model)
            self.pos = nn.Embedding(4, cfg.d_model)
            if attention:
                layer = nn.TransformerEncoderLayer(
                    cfg.d_model, cfg.n_heads, cfg.d_model * 2,
                    dropout=0.1, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, cfg.n_layers)
            self.head = nn.Sequential(nn.Linear(cfg.d_model, 24), nn.GELU(), nn.Linear(24, 1))

        def forward(self, X, P):
            h = self.proj(X) + self.pos(P)
            if self.attention:
                h = self.enc(h)
            return self.head(h.mean(1)).squeeze(-1)

    class Match(nn.Module):
        def __init__(self, attention: bool):
            super().__init__()
            self.composer = Composer(attention)
            self.intercept = nn.Parameter(torch.tensor(0.2))
            self.scale = nn.Parameter(torch.tensor(0.3))
            self.home_adv = nn.Parameter(torch.tensor(0.2))

        def loglams(self, Xh, Ph, Xa, Pa):
            diff = self.composer(Xh, Ph) - self.composer(Xa, Pa)
            lh = (self.intercept + self.scale * diff + self.home_adv).clamp(-3, 2.5)
            la = (self.intercept - self.scale * diff).clamp(-3, 2.5)
            return lh, la

    def nll(lh, la, gh, ga):
        return ((torch.exp(lh) - gh * lh) + (torch.exp(la) - ga * la)).mean()

    def train_one(attention: bool) -> float:
        torch.manual_seed(seed)
        m = Match(attention)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            lh, la = m.loglams(Xh_tr, Ph_tr, Xa_tr, Pa_tr)
            loss = nll(lh, la, gh_tr, ga_tr)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lh, la = m.loglams(Xh_te, Ph_te, Xa_te, Pa_te)
            return float(nll(lh, la, gh_te, ga_te))

    agg_nll = train_one(attention=False)
    att_nll = train_one(attention=True)

    improvement = agg_nll - att_nll
    verdict = {
        "n_train": len(tr), "n_holdout": len(te),
        "aggregation_holdout_nll": round(agg_nll, 4),
        "attention_holdout_nll": round(att_nll, 4),
        "attention_improvement": round(improvement, 4),
        "attention_wins": bool(improvement > 0.001),
        "h3_verdict": ("attention beats aggregation"
                       if improvement > 0.001 else
                       "no, averaging is enough — interactions do not transfer"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(verdict, indent=2))
    return verdict


if __name__ == "__main__":
    v = run()
    print(json.dumps(v, indent=2))
    print("\nH3:", v["h3_verdict"])
