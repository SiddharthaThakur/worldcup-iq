"""
Offline Bayesian hierarchical model — champion probabilities WITH error bars.

What this does in simple English:
    A separate, more statistically honest take on "who wins the World Cup".
    Instead of a single point estimate, it gives a RANGE (credible interval)
    for every team's chance, reflecting how unsure we are.

    The model (the standard Bayesian football model, Baio & Blangiardo 2010):
        goals_home ~ Poisson(λ_home),  goals_away ~ Poisson(λ_away)
        log(λ_home) = μ + home_adv·(host) + attack[home] − defence[away]
        log(λ_away) = μ + home_adv·(host) + attack[away] − defence[home]
    Each team's attack/defence is a partially-pooled random effect — teams
    with little data are automatically shrunk toward average with wide
    uncertainty (no manual fallback rule needed). Fitted with PyMC (NUTS).

    Then we simulate the 2026 tournament across many POSTERIOR DRAWS, so the
    champion probabilities carry the model's parameter uncertainty, not just
    match randomness. The spread across draws = the credible interval.

    Offline by design (MCMC is too slow for the daily free pipeline). Run:
        python -m src.models.bayesian
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.results_loader import load_processed_results
from src.data.wc2026 import load_wc2026

OUT = Path("data/predictions/bayesian_champions.csv")
IDATA_CACHE = Path("data/processed/bayesian_idata.nc")
IDX_CACHE = Path("data/processed/bayesian_team_idx.json")
HOST = {"USA", "CAN", "MEX"}


def fit(window_from: int = 2019, draws: int = 1000, tune: int = 1000, seed: int = 0):
    """Fit the hierarchical Poisson model on recent internationals."""
    import pymc as pm

    res = load_processed_results()
    df = res[res["date"].dt.year >= window_from].dropna(
        subset=["home_score", "away_score"]).copy()
    teams = sorted(set(df["home_code"]) | set(df["away_code"]))
    idx = {t: i for i, t in enumerate(teams)}
    hi = df["home_code"].map(idx).to_numpy()
    ai = df["away_code"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy(int)
    ag = df["away_score"].to_numpy(int)
    not_neutral = (~df["neutral"].astype(bool)).to_numpy(float)
    n = len(teams)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.2, 0.5)
        home_adv = pm.Normal("home_adv", 0.25, 0.1)
        sd_att = pm.HalfNormal("sd_att", 0.5)
        sd_def = pm.HalfNormal("sd_def", 0.5)
        # native sum-to-zero (identifiable vs mu, samples far better than
        # manual mean-subtraction)
        att = pm.ZeroSumNormal("att", sigma=sd_att, shape=(n,))
        deff = pm.ZeroSumNormal("deff", sigma=sd_def, shape=(n,))

        log_lh = mu + home_adv * not_neutral + att[hi] - deff[ai]
        log_la = mu + att[ai] - deff[hi]
        pm.Poisson("hg", mu=pm.math.exp(log_lh), observed=hg)
        pm.Poisson("ag", mu=pm.math.exp(log_la), observed=ag)

        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=4,
                          target_accept=0.95, random_seed=seed, progressbar=False)

    import arviz as az
    post = idata.posterior
    samples = {
        "att": post["att"].stack(s=("chain", "draw")).values,   # (n_teams, S)
        "deff": post["deff"].stack(s=("chain", "draw")).values,
        "mu": post["mu"].stack(s=("chain", "draw")).values,
        "ha": post["home_adv"].stack(s=("chain", "draw")).values,
        "rhat": float(az.summary(idata)["r_hat"].max()),
        "idx": idx,
    }
    return samples


def fit_cached(refit: bool = False, **kw) -> dict:
    """Fit, caching the posterior arrays (npz) so re-runs skip sampling."""
    import json
    if not refit and IDATA_CACHE.exists() and IDX_CACHE.exists():
        z = np.load(IDATA_CACHE)
        meta = json.loads(IDX_CACHE.read_text())
        return {"att": z["att"], "deff": z["deff"], "mu": z["mu"], "ha": z["ha"],
                "rhat": meta["rhat"], "idx": meta["idx"]}
    s = fit(**kw)
    IDATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(IDATA_CACHE, att=s["att"], deff=s["deff"], mu=s["mu"], ha=s["ha"])
    IDX_CACHE.write_text(json.dumps({"idx": s["idx"], "rhat": s["rhat"]}))
    return s


def _match_lambdas(h, a, att, deff, mu, home_adv, hosts):
    h_adv = home_adv if h in hosts else 0.0
    a_adv = home_adv if a in hosts else 0.0
    return (np.exp(mu + h_adv + att[h] - deff[a]),
            np.exp(mu + a_adv + att[a] - deff[h]))


def simulate_champion(att, deff, mu, home_adv, groups, rng, hosts=HOST):
    """One tournament from one posterior draw. Returns the champion team code."""
    standings = {}
    for g, teams in groups.items():
        tab = {t: [0, 0, 0] for t in teams}  # pts, gd, gf
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                h, a = teams[i], teams[j]
                lh, la = _match_lambdas(h, a, att, deff, mu, home_adv, hosts)
                gh, ga = rng.poisson(lh), rng.poisson(la)
                tab[h][1] += gh - ga; tab[h][2] += gh
                tab[a][1] += ga - gh; tab[a][2] += ga
                if gh > ga:
                    tab[h][0] += 3
                elif ga > gh:
                    tab[a][0] += 3
                else:
                    tab[h][0] += 1; tab[a][0] += 1
        ranked = sorted(teams, key=lambda t: (tab[t][0], tab[t][1], tab[t][2], rng.random()),
                        reverse=True)
        standings[g] = (ranked, tab)

    winners = [standings[g][0][0] for g in groups]
    runners = [standings[g][0][1] for g in groups]
    thirds = [(standings[g][0][2], standings[g][1][standings[g][0][2]]) for g in groups]
    best = [t for t, _ in sorted(thirds, key=lambda x: (x[1][0], x[1][1], x[1][2], rng.random()),
                                 reverse=True)[:8]]
    bracket = winners + runners + best
    rng.shuffle(bracket)  # structural approximation (D010); fine for champion odds
    while len(bracket) > 1:
        nxt = []
        for k in range(0, len(bracket), 2):
            h, a = bracket[k], bracket[k + 1]
            lh, la = _match_lambdas(h, a, att, deff, mu, home_adv, hosts)
            gh, ga = rng.poisson(lh), rng.poisson(la)
            if gh > ga:
                nxt.append(h)
            elif ga > gh:
                nxt.append(a)
            else:  # extra time / penalties: stronger side edges it
                nxt.append(h if rng.random() < lh / (lh + la) else a)
        bracket = nxt
    return bracket[0]


def champion_probabilities(samples: dict, n_draws: int = 200, sims_per_draw: int = 200,
                           seed: int = 1) -> pd.DataFrame:
    """Champion probability per team with a credible interval, propagating
    posterior uncertainty (across draws) and match randomness (sims)."""
    wc = load_wc2026(save=False)
    teams = [t for ts in wc.groups.values() for t in ts]
    idx = samples["idx"]
    att_s, def_s, mu_s, ha_s = samples["att"], samples["deff"], samples["mu"], samples["ha"]
    n_samples = att_s.shape[1]

    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(n_samples, size=min(n_draws, n_samples), replace=False)

    per_draw = {t: [] for t in teams}  # champion prob for each posterior draw
    for sid in sample_ids:
        att = {t: att_s[idx[t], sid] for t in teams}
        deff = {t: def_s[idx[t], sid] for t in teams}
        mu, ha = float(mu_s[sid]), float(ha_s[sid])
        counts = {t: 0 for t in teams}
        for _ in range(sims_per_draw):
            counts[simulate_champion(att, deff, mu, ha, wc.groups, rng)] += 1
        for t in teams:
            per_draw[t].append(counts[t] / sims_per_draw)

    rows = []
    for t in teams:
        arr = np.array(per_draw[t])
        rows.append({"team": t, "champ_mean": arr.mean(),
                     "champ_lo": np.percentile(arr, 5),
                     "champ_hi": np.percentile(arr, 95)})
    return pd.DataFrame(rows).sort_values("champ_mean", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Fitting Bayesian hierarchical model (cached after first run)...")
    samples = fit_cached()
    print(f"max R-hat over all params (want <1.01): {samples['rhat']:.3f}")
    print(f"home advantage (log): {samples['ha'].mean():.3f}  baseline mu: {samples['mu'].mean():.3f}")
    df = champion_probabilities(samples)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\nChampion probabilities with 90% credible intervals:")
    for _, r in df.head(12).iterrows():
        print(f"  {r['team']}: {r['champ_mean']*100:4.1f}%  [{r['champ_lo']*100:4.1f}% – {r['champ_hi']*100:4.1f}%]")
