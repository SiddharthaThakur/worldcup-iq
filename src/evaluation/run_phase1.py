"""
Phase 1 report: fitted baseline vs the bookmaker market, 2018 + 2022 World Cups.

Run: python -m src.evaluation.run_phase1

Both the model and the market are scored on the same matches with the
same outcome definition: the 90-MINUTE result (1X2 odds settle at 90
minutes, so extra-time winners count as draws). The model is fitted with
a strict pre-tournament cutoff for each World Cup — no leakage.
"""

import json
from dataclasses import replace
from pathlib import Path

from src.data.odds_loader import load_wc_odds_workbook
from src.data.results_loader import load_processed_results
from src.evaluation.backtest import backtest_world_cup
from src.evaluation.calibration import PredictionRecord, scorecard
from src.evaluation.market_compare import align_market_to_results

WORKBOOK = Path("data/raw/WorldCup2026.xlsx")
OUT_PATH = Path("data/predictions/phase1_backtest_report.json")


def main() -> dict:
    results = load_processed_results()
    report = {"note": "All scores on 90-minute results; model fitted pre-tournament only."}
    pooled_model, pooled_market = [], []

    for year, sheet in [(2018, "WorldCup2018"), (2022, "WorldCup2022")]:
        bt = backtest_world_cup(results, wc_year=year)
        wc = results[(results["tournament"] == "FIFA World Cup")
                     & (results["date"].dt.year == year)]
        odds = load_wc_odds_workbook(WORKBOOK, sheet)
        market = align_market_to_results(wc, odds)
        if market.attrs["unmatched_match_ids"]:
            raise RuntimeError(f"{year}: no odds for {market.attrs['unmatched_match_ids']}")

        result_90 = dict(zip(market["match_id"], market["result_90"]))
        model_recs = [replace(r, actual_result=result_90[r.match_id]) for r in bt.records]
        market_recs = [
            PredictionRecord(
                match_id=m["match_id"], model_name=f"market_{year}",
                prob_home=m["prob_home"], prob_draw=m["prob_draw"],
                prob_away=m["prob_away"], actual_result=m["result_90"],
            )
            for _, m in market.iterrows()
        ]
        pooled_model += model_recs
        pooled_market += market_recs
        report[str(year)] = {
            "n_train_matches": bt.n_train_matches,
            "model": scorecard(model_recs, f"elo_dixon_coles_{year}"),
            "market": scorecard(market_recs, f"market_avg_devig_{year}"),
        }

    report["pooled"] = {
        "model": scorecard(pooled_model, "elo_dixon_coles_pooled"),
        "market": scorecard(pooled_market, "market_pooled"),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=float)
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps(rep, indent=2, default=float))
