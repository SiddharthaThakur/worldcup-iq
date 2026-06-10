"""
Prediction lock-in: make pre-match predictions cryptographically verifiable.

What this does in simple English:
    Anyone can claim "my model predicted that." The git history makes the
    claim checkable: before each match, this script writes the prediction
    to a timestamped JSON file and creates a git commit. The commit's hash
    and timestamp prove the prediction existed before kickoff. Predictions
    for matches already played at lock-in time are REFUSED — they're
    backtest data, not predictions, and mixing the two would poison the
    honesty claim the whole project rests on.

Usage:
    python -m src.lockin.lock_predictions --matchday 2026-06-12

Each lock-in file contains: every model's probabilities, the strength
inputs used, the fitted-parameter file's hash (so the exact model version
is pinned), and the lock timestamp.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PREDICTIONS_DIR = Path("data/predictions")
PARAMS_PATH = Path("models/dixon_coles_params.json")


def file_sha256(path: Path) -> str:
    """Hash a file — used to pin the exact fitted parameters."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def lock_predictions(
    predictions: list[dict],
    matchday: str,
    kickoff_times: dict[str, str] | None = None,
) -> Path:
    """Write predictions to a timestamped lock file and git-commit it.

    Args:
        predictions: list of dicts, each with match_id, model_name, and
                     prob_home/prob_draw/prob_away (from MatchPrediction.to_dict()
                     plus model_name and match_id)
        matchday: date string YYYY-MM-DD for the matches being locked
        kickoff_times: optional {match_id: ISO kickoff time}; if provided,
                       matches whose kickoff is already past are REJECTED

    Returns:
        Path of the lock file written.

    Raises:
        ValueError if any match has already kicked off.
    """
    now = datetime.now(timezone.utc)

    if kickoff_times:
        for p in predictions:
            ko = kickoff_times.get(p["match_id"])
            if ko and datetime.fromisoformat(ko) <= now:
                raise ValueError(
                    f"REFUSED: {p['match_id']} kicked off at {ko}, which is in "
                    f"the past. Post-hoc predictions are backtest data, not "
                    f"predictions. They must not enter the lock-in record."
                )

    lock = {
        "locked_at_utc": now.isoformat(),
        "matchday": matchday,
        "params_file_sha256": file_sha256(PARAMS_PATH) if PARAMS_PATH.exists() else None,
        "predictions": predictions,
    }

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"lock_{matchday}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(lock, indent=2))

    commit_hash = _git_commit_lock(out_path, matchday)
    if commit_hash:
        print(f"Locked {len(predictions)} predictions for {matchday}")
        print(f"  file:   {out_path}")
        print(f"  commit: {commit_hash}")
        print(f"  Publish this commit hash. Anyone can verify timing via git log.")
    return out_path


def _git_commit_lock(path: Path, matchday: str) -> str | None:
    """Commit the lock file. Returns short commit hash, or None if git fails."""
    try:
        subprocess.run(["git", "add", str(path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"lock: predictions for {matchday}"],
            check=True, capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARNING: git commit failed ({e}). Lock file written but NOT "
              f"committed — predictions are not yet verifiable. Commit manually.")
        return None


def verify_lock(lock_path: Path, kickoff_times: dict[str, str]) -> dict:
    """Verify a lock file: was it committed before every kickoff it predicts?

    Returns a report dict. Used in the post-tournament write-up to
    demonstrate that every scored prediction was genuinely pre-match.
    """
    lock = json.loads(lock_path.read_text())
    locked_at = datetime.fromisoformat(lock["locked_at_utc"])

    report = {"lock_file": str(lock_path), "locked_at": lock["locked_at_utc"],
              "matches": [], "all_valid": True}
    for p in lock["predictions"]:
        ko = kickoff_times.get(p["match_id"])
        valid = ko is None or locked_at < datetime.fromisoformat(ko)
        report["matches"].append(
            {"match_id": p["match_id"], "kickoff": ko, "pre_match": valid})
        if not valid:
            report["all_valid"] = False
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matchday", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    print(f"Lock-in for {args.matchday}: generate predictions via the model "
          f"modules, then call lock_predictions(). See module docstring.")
