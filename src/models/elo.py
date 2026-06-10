"""
Elo rating system for international football teams.

What this does in simple English:
    Every team gets a number (their "Elo rating") that represents how
    strong they are. When a strong team beats a weak team, ratings barely
    change. When a weak team upsets a strong team, ratings shift a lot.
    We update ratings after every match going back to 2010, so by the
    time the World Cup starts, each team has a current strength estimate.

    The key tuning parameters:
    - K-factor: how much ratings change per match (higher for important tournaments)
    - Home advantage: bonus Elo points for the home team
    - Mean reversion: pull ratings toward average at the start of each year
      (prevents dead teams from keeping outdated high ratings)
"""

from dataclasses import dataclass, field

import pandas as pd

# Tournament importance weights for K-factor.
# EMPTY by design: the conventional importance ladder (WC=60 ... Friendly=15)
# was tested against flat K on the 2018+2022 backtest and LOST decisively
# (pooled Brier 0.2054 vs 0.2009; steeper weighting was monotonically worse).
# High-K tournaments made ratings overreact to small-sample, high-variance
# results. Selected before any 2026 match was scored — see D017.
TOURNAMENT_K: dict[str, float] = {}

DEFAULT_K = 25.0
DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0
# Annual pull toward 1500. Set to 0 after a backtest experiment on 2018+2022
# (selected BEFORE any 2026 match was scored — see D015): reversion=1/3 gave
# pooled Brier 0.2160; reversion=0 gave 0.2054. The 1/3 reversion also
# produced an artifact: teams with competitive matches early in a year
# re-earned rating while everyone else sat compressed at the January reset.
MEAN_REVERSION_FACTOR = 0.0


@dataclass
class EloSystem:
    """Elo rating tracker for international football.

    Attributes:
        ratings: current Elo rating per team (FIFA code → float)
        history: list of (date, team_code, rating) snapshots after each match
    """

    ratings: dict[str, float] = field(default_factory=dict)
    history: list[tuple[str, str, float]] = field(default_factory=list)
    _last_year: int = 0

    def get_rating(self, team: str) -> float:
        """Get current rating for a team, defaulting to 1500."""
        return self.ratings.get(team, DEFAULT_RATING)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A (1=win, 0.5=draw, 0=loss)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def get_k_factor(self, tournament: str) -> float:
        """Get K-factor based on tournament importance."""
        return TOURNAMENT_K.get(tournament, DEFAULT_K)

    def _apply_mean_reversion(self, year: int) -> None:
        """At the start of a new year, regress all ratings toward 1500."""
        if year > self._last_year and self._last_year > 0:
            for team in self.ratings:
                self.ratings[team] = (
                    self.ratings[team]
                    + MEAN_REVERSION_FACTOR * (DEFAULT_RATING - self.ratings[team])
                )
        self._last_year = year

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str = "Friendly",
        neutral: bool = True,
        date: str = "",
    ) -> tuple[float, float]:
        """Update ratings after a match.

        Args:
            home_team: FIFA code of home team
            away_team: FIFA code of away team
            home_score: goals scored by home team
            away_score: goals scored by away team
            tournament: tournament name (for K-factor)
            neutral: whether the match is on neutral ground
            date: match date string (for year tracking)

        Returns:
            (new_home_rating, new_away_rating)
        """
        if date:
            year = int(date[:4])
            self._apply_mean_reversion(year)

        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        # Home advantage (only if not neutral)
        effective_home = home_rating + (0 if neutral else HOME_ADVANTAGE)

        # Actual result (1=win, 0.5=draw, 0=loss)
        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        expected_home = self.expected_score(effective_home, away_rating)
        k = self.get_k_factor(tournament)

        # Goal difference multiplier (rewards dominant wins)
        goal_diff = abs(home_score - away_score)
        if goal_diff <= 1:
            gd_mult = 1.0
        elif goal_diff == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11.0 + goal_diff) / 8.0

        delta = k * gd_mult * (actual_home - expected_home)

        self.ratings[home_team] = home_rating + delta
        self.ratings[away_team] = away_rating - delta

        if date:
            self.history.append((date, home_team, self.ratings[home_team]))
            self.history.append((date, away_team, self.ratings[away_team]))

        return self.ratings[home_team], self.ratings[away_team]

    def fit_from_results(self, results: pd.DataFrame) -> "EloSystem":
        """Fit Elo ratings from a DataFrame of match results.

        Expects columns: date, home_code, away_code, home_score, away_score,
                         tournament, neutral
        """
        results_sorted = results.sort_values("date").reset_index(drop=True)

        for _, row in results_sorted.iterrows():
            self.update(
                home_team=row["home_code"],
                away_team=row["away_code"],
                home_score=row["home_score"],
                away_score=row["away_score"],
                tournament=row.get("tournament", "Friendly"),
                neutral=row.get("neutral", True),
                date=str(row["date"])[:10],
            )
        return self

    def get_ratings_df(self, teams: list[str] | None = None) -> pd.DataFrame:
        """Get current ratings as a sorted DataFrame."""
        if teams:
            data = {t: self.get_rating(t) for t in teams}
        else:
            data = dict(self.ratings)
        df = pd.DataFrame(list(data.items()), columns=["team", "elo"])
        return df.sort_values("elo", ascending=False).reset_index(drop=True)
