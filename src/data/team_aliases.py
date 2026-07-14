"""
Resolve team names to FIFA 3-letter codes.

What this does in simple English:
    Different data sources call the same team different things —
    "Korea Republic", "South Korea", "Korea Rep." are all the same team.
    This module maps everything to the standard FIFA 3-letter code (KOR).
"""

import json
from pathlib import Path

ALIASES_PATH = Path("data/processed/team_aliases.json")

# Default aliases — extend as needed during entity resolution
_DEFAULT_ALIASES: dict[str, list[str]] = {
    "ARG": ["Argentina"],
    "AUS": ["Australia"],
    "AUT": ["Austria"],
    "BEL": ["Belgium"],
    "BOL": ["Bolivia"],
    "BRA": ["Brazil"],
    "CMR": ["Cameroon"],
    "CAN": ["Canada"],
    "CHI": ["Chile"],
    "COL": ["Colombia"],
    "CRC": ["Costa Rica"],
    "CRO": ["Croatia"],
    "CZE": ["Czech Republic", "Czechia"],
    "DEN": ["Denmark"],
    "ECU": ["Ecuador"],
    "EGY": ["Egypt"],
    "ENG": ["England"],
    "FRA": ["France"],
    "GER": ["Germany"],
    "GHA": ["Ghana"],
    "GRE": ["Greece"],
    "HTI": ["Haiti"],
    "HUN": ["Hungary"],
    "ISL": ["Iceland"],
    "IDN": ["Indonesia"],
    "IRN": ["Iran", "IR Iran"],
    "IRQ": ["Iraq"],
    "ISR": ["Israel"],
    "ITA": ["Italy"],
    "CIV": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"],
    "JAM": ["Jamaica"],
    "JPN": ["Japan"],
    "JOR": ["Jordan"],
    "KEN": ["Kenya"],
    "KOR": ["Korea Republic", "South Korea", "Korea Rep."],
    "PRK": ["Korea DPR", "North Korea", "DPR Korea"],
    "MEX": ["Mexico"],
    "MAR": ["Morocco"],
    "NED": ["Netherlands", "Holland"],
    "NZL": ["New Zealand"],
    "NGA": ["Nigeria"],
    "NOR": ["Norway"],
    "PAN": ["Panama"],
    "PAR": ["Paraguay"],
    "PER": ["Peru"],
    "POL": ["Poland"],
    "POR": ["Portugal"],
    "QAT": ["Qatar"],
    "ROU": ["Romania"],
    "RUS": ["Russia"],
    "KSA": ["Saudi Arabia"],
    "SCO": ["Scotland"],
    "SEN": ["Senegal"],
    "SRB": ["Serbia"],
    "SVK": ["Slovakia"],
    "SVN": ["Slovenia"],
    "RSA": ["South Africa"],
    "ESP": ["Spain"],
    "SWE": ["Sweden"],
    "SUI": ["Switzerland"],
    "TUN": ["Tunisia"],
    "TUR": ["Turkey", "Türkiye"],
    "UKR": ["Ukraine"],
    "UAE": ["United Arab Emirates"],
    "URU": ["Uruguay"],
    "USA": ["United States", "United States of America", "US"],
    "UZB": ["Uzbekistan"],
    "VEN": ["Venezuela"],
    "WAL": ["Wales"],
    "ALG": ["Algeria"],
    "BIH": ["Bosnia and Herzegovina", "Bosnia-Herzegovina", "Bosnia"],
    "CPV": ["Cape Verde", "Cabo Verde", "Cape Verde Islands"],
    "CUW": ["Curaçao", "Curacao"],
    "COD": ["DR Congo", "Congo DR", "Democratic Republic of the Congo"],
}

# Build reverse lookup: name → code
_name_to_code: dict[str, str] = {}


def _build_lookup() -> None:
    """Build the reverse name→code lookup from aliases."""
    global _name_to_code
    aliases = _DEFAULT_ALIASES
    if ALIASES_PATH.exists():
        with open(ALIASES_PATH) as f:
            aliases = json.load(f)
    _name_to_code = {}
    for code, names in aliases.items():
        _name_to_code[code.lower()] = code
        for name in names:
            _name_to_code[name.lower()] = code


def resolve_team_code(name: str) -> str:
    """Resolve a team name to its FIFA 3-letter code.

    Returns the input unchanged if no match is found (logged for review).
    """
    if not _name_to_code:
        _build_lookup()
    if not isinstance(name, str):
        return str(name)
    return _name_to_code.get(name.strip().lower(), name)


def save_aliases(path: Path = ALIASES_PATH) -> None:
    """Save current aliases to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_DEFAULT_ALIASES, f, indent=2, ensure_ascii=False)
