"""NFL team names and the abbreviation variants fantasy platforms use.

nflverse abbreviations are canonical. Platforms differ for a few teams (ESPN uses ``WSH`` and
``JAX``, Yahoo ``JAC``, and so on), and defenses usually appear by nickname ("Vikings D/ST").
"""

# Canonical nflverse abbreviation -> (location, nickname).
TEAMS: dict[str, tuple[str, str]] = {
    "ARI": ("Arizona", "Cardinals"),
    "ATL": ("Atlanta", "Falcons"),
    "BAL": ("Baltimore", "Ravens"),
    "BUF": ("Buffalo", "Bills"),
    "CAR": ("Carolina", "Panthers"),
    "CHI": ("Chicago", "Bears"),
    "CIN": ("Cincinnati", "Bengals"),
    "CLE": ("Cleveland", "Browns"),
    "DAL": ("Dallas", "Cowboys"),
    "DEN": ("Denver", "Broncos"),
    "DET": ("Detroit", "Lions"),
    "GB": ("Green Bay", "Packers"),
    "HOU": ("Houston", "Texans"),
    "IND": ("Indianapolis", "Colts"),
    "JAX": ("Jacksonville", "Jaguars"),
    "KC": ("Kansas City", "Chiefs"),
    "LA": ("Los Angeles", "Rams"),
    "LAC": ("Los Angeles", "Chargers"),
    "LV": ("Las Vegas", "Raiders"),
    "MIA": ("Miami", "Dolphins"),
    "MIN": ("Minnesota", "Vikings"),
    "NE": ("New England", "Patriots"),
    "NO": ("New Orleans", "Saints"),
    "NYG": ("New York", "Giants"),
    "NYJ": ("New York", "Jets"),
    "PHI": ("Philadelphia", "Eagles"),
    "PIT": ("Pittsburgh", "Steelers"),
    "SEA": ("Seattle", "Seahawks"),
    "SF": ("San Francisco", "49ers"),
    "TB": ("Tampa Bay", "Buccaneers"),
    "TEN": ("Tennessee", "Titans"),
    "WAS": ("Washington", "Commanders"),
}

_ALIASES: dict[str, str] = {
    "ARZ": "ARI",
    "GNB": "GB",
    "GBP": "GB",
    "JAC": "JAX",
    "KAN": "KC",
    "KCC": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "OAK": "LV",
    "NWE": "NE",
    "NEP": "NE",
    "NOR": "NO",
    "NOS": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "TBB": "TB",
    "WSH": "WAS",
}


def canonical_team(abbr: str | None) -> str | None:
    """Map a platform's team abbreviation to the nflverse one; None if unknown or missing."""
    if abbr is None:
        return None
    key = abbr.strip().upper()
    key = _ALIASES.get(key, key)
    return key if key in TEAMS else None
