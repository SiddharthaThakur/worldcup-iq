"""
Fit Dixon-Coles parameters on historical international results.

What this does in simple English:
    Earlier versions of this project used made-up constants to convert
    Elo ratings into expected goals. That's a calibration landmine: if the
    mapping is wrong, every probability is systematically wrong, and the
    whole "honest scorecard" measures the error of an arbitrary constant.

    This module fits everything from data:
    1. Run the Elo system over all post-2010 international matches,
       recording each team's PRE-MATCH rating (no lookahead).
    2. Poisson regression: log(goals) ~ intercept + elo_diff + home_advantage.
    3. Profile-likelihood fit of rho (the low-scoring-draw correction).
    4. Save fitted parameters with provenance to models/dixon_coles_params.json.

Run: python -m src.models.fit_params
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from src.models.dixon_coles import DixonColesParams, dixon_coles_correction
from src.models.elo import EloSystem


def build_training_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Replay matches chronologically, recording PRE-match Elo for both teams.

    Critical: ratings are captured BEFORE each match is fed to the Elo
    update, so there is no information leakage from the match outcome
    into its own features.

    Expects columns: date, home_code, away_code, home_score, away_score,
                     tournament, neutral
    """
    elo = EloSystem()
    rows = []
    for _, row in results.sort_values("date").iterrows():
        home_elo = elo.get_rating(row["home_code"])
        away_elo = elo.get_rating(row["away_code"])
        rows.append({
            "date": row["date"],
            "home_code": row["home_code"],
            "away_code": row["away_code"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "neutral": bool(row.get("neutral", True)),
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
        })
        elo.update(
            home_team=row["home_code"], away_team=row["away_code"],
            home_score=row["home_score"], away_score=row["away_score"],
            tournament=row.get("tournament", "Friendly"),
            neutral=row.get("neutral", True),
            date=str(row["date"])[:10],
        )
    return pd.DataFrame(rows)


def fit_goal_model(train: pd.DataFrame) -> tuple[float, float, float]:
    """Poisson regression for expected goals as a function of Elo difference.

    Model (stacked: each match contributes a home row and an away row):
        log(lambda) = intercept + elo_coef * (signed_diff / 100) + home_adv * is_home_nonneutral

    Returns (intercept, elo_coef, home_adv).
    """
    diff = (train["home_elo_pre"] - train["away_elo_pre"]).values / 100.0
    not_neutral = (~train["neutral"]).astype(float).values

    # Stack home and away observations
    y = np.concatenate([train["home_score"].values, train["away_score"].values])
    x_diff = np.concatenate([diff, -diff])
    x_home = np.concatenate([not_neutral, np.zeros(len(train))])

    def neg_log_lik(theta: np.ndarray) -> float:
        intercept, elo_coef, home_adv = theta
        log_lam = intercept + elo_coef * x_diff + home_adv * x_home
        log_lam = np.clip(log_lam, -3.0, 2.5)
        lam = np.exp(log_lam)
        return float(-np.sum(y * log_lam - lam))  # Poisson NLL up to a constant

    res = minimize(neg_log_lik, x0=np.array([0.3, 0.15, 0.2]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"Goal model fit failed: {res.message}")
    return tuple(float(v) for v in res.x)


def fit_rho(train: pd.DataFrame, intercept: float, elo_coef: float,
            home_adv: float) -> float:
    """Profile-likelihood fit of the Dixon-Coles rho parameter.

    With the goal model fixed, find rho maximizing the likelihood of
    observed scorelines under the corrected bivariate Poisson.
    """
    diff = (train["home_elo_pre"] - train["away_elo_pre"]).values / 100.0
    not_neutral = (~train["neutral"]).astype(float).values
    lam_h = np.exp(np.clip(intercept + elo_coef * diff + home_adv * not_neutral, -3, 2.5))
    lam_a = np.exp(np.clip(intercept - elo_coef * diff, -3, 2.5))
    hg = train["home_score"].values
    ag = train["away_score"].values

    base_log_pmf = poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
    low_mask = (hg <= 1) & (ag <= 1)

    def neg_log_lik(rho: float) -> float:
        tau = np.ones(len(train))
        idx = np.where(low_mask)[0]
        for i in idx:
            tau[i] = dixon_coles_correction(int(hg[i]), int(ag[i]),
                                            float(lam_h[i]), float(lam_a[i]), rho)
        tau = np.clip(tau, 1e-10, None)
        return float(-np.sum(base_log_pmf + np.log(tau)))

    res = minimize_scalar(neg_log_lik, bounds=(-0.5, 0.5), method="bounded")
    return float(res.x)


def fit_all(results: pd.DataFrame, description: str = "", save: bool = True) -> DixonColesParams:
    """Full fitting pipeline: Elo replay → goal model → rho → save.

    save=False is for backtests, which must never overwrite the live
    params file with parameters fitted on a historical cutoff.
    """
    train = build_training_rows(results)
    intercept, elo_coef, home_adv = fit_goal_model(train)
    rho = fit_rho(train, intercept, elo_coef, home_adv)

    params = DixonColesParams(
        intercept=intercept,
        elo_coef=elo_coef,
        home_adv=home_adv,
        rho=rho,
        fitted_on=description or f"international results {train['date'].min()} to {train['date'].max()}",
        n_matches=len(train),
    )
    if save:
        params.save()
    return params


if __name__ == "__main__":
    from src.data.results_loader import load_processed_results

    results = load_processed_results()
    params = fit_all(results, description="post-2010 internationals (Kaggle martj42)")
    print(f"Fitted on {params.n_matches} matches:")
    print(f"  intercept = {params.intercept:.4f}  (avg goals = {np.exp(params.intercept):.3f})")
    print(f"  elo_coef  = {params.elo_coef:.4f}  per 100 Elo")
    print(f"  home_adv  = {params.home_adv:.4f}  (log scale)")
    print(f"  rho       = {params.rho:.4f}")
