"""Match extracted roster names to the player index.

Scoring is deliberately simple and explainable: normalized first and last names are compared
separately with ``difflib.SequenceMatcher`` and averaged, so a shared surname alone ("Mike
Williams" vs "Kyle Williams") can't look like a near-match; an abbreviated first name ("J.
Jefferson") scores high when the last name matches exactly and the initial agrees; a team that
agrees with the screenshot adds a small bonus that only breaks ties between similar names (it
never lifts a weak name over a threshold, and a mismatch is not penalized because screenshots
can lag trades). Anything short of a clear,
unique winner is returned as ``ambiguous`` or ``unmatched`` for the user to resolve.
"""

from difflib import SequenceMatcher

from backend.app.parsing.players import IndexedPlayer, PlayerIndex, normalize_name
from backend.app.parsing.schemas import Candidate, ExtractedPlayer, PlayerMatch
from backend.app.parsing.teams import canonical_team

MATCH_THRESHOLD = 0.90  # best score needed to accept a match
MATCH_MARGIN = 0.05  # and its lead over the runner-up
AMBIGUOUS_THRESHOLD = 0.80  # below this, report unmatched
CANDIDATE_THRESHOLD = 0.60  # lowest score worth showing the user as an option
MAX_CANDIDATES = 3
TEAM_BONUS = 0.05
INITIAL_SCORE = 0.95

_DEFENSE_WORDS = frozenset({"d", "st", "dst", "def", "defense", "def/st"})


def query_key(extracted: ExtractedPlayer) -> str:
    """Normalized lookup key; for defenses, drop words like "D/ST" and "DEF"."""
    key = normalize_name(extracted.name)
    if extracted.position == "DST":
        key = " ".join(t for t in key.split() if t not in _DEFENSE_WORDS) or key
    return key


def name_score(query: str, player: IndexedPlayer) -> float:
    """Similarity in [0, 1] between a normalized query and a player's names."""
    keys = (player.key, *player.aliases)
    if query in keys:
        return 1.0
    q_first, _, q_rest = query.partition(" ")
    p_first, _, p_rest = player.key.partition(" ")
    if len(q_first) == 1 and q_rest and q_rest == p_rest and p_first.startswith(q_first):
        return INITIAL_SCORE
    return max(split_ratio(query, k) for k in keys)


def ratio(a: str, b: str) -> float:
    """``SequenceMatcher`` similarity of two strings."""
    return SequenceMatcher(None, a, b).ratio()


def split_ratio(a: str, b: str) -> float:
    """Mean of first-name and rest-of-name similarity; whole-string ratio for one-word names."""
    a_first, _, a_rest = a.partition(" ")
    b_first, _, b_rest = b.partition(" ")
    if not a_rest or not b_rest:
        return ratio(a, b)
    return (ratio(a_first, b_first) + ratio(a_rest, b_rest)) / 2


def match_player(extracted: ExtractedPlayer, index: PlayerIndex) -> PlayerMatch:
    """Return the best index match for one extracted player, with alternatives."""
    query = query_key(extracted)
    team = canonical_team(extracted.nfl_team)
    # (name score, ranking score with team bonus, player); thresholds use the name score only.
    scored = sorted(
        (
            (s, s + (TEAM_BONUS if team is not None and p.team == team else 0.0), p)
            for p in index.by_position(extracted.position)
            for s in [name_score(query, p)]
        ),
        key=lambda t: (-t[1], -t[0], t[2].player_id),
    )
    candidates = [
        Candidate(
            player_id=p.player_id, name=p.name, position=p.position, team=p.team, score=round(s, 3)
        )
        for s, _, p in scored[:MAX_CANDIDATES]
        if s >= CANDIDATE_THRESHOLD
    ]
    if not scored or scored[0][0] < AMBIGUOUS_THRESHOLD:
        return PlayerMatch(status="unmatched", candidates=candidates)

    best_score, best_rank, best = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else 0.0
    # Small epsilon so an exact name plus the team bonus clears the margin despite float error.
    if best_score >= MATCH_THRESHOLD and best_rank - runner_up >= MATCH_MARGIN - 1e-9:
        return PlayerMatch(
            status="matched",
            player_id=best.player_id,
            score=round(best_score, 3),
            candidates=candidates,
        )
    return PlayerMatch(status="ambiguous", candidates=candidates)
