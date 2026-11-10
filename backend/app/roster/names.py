"""Name similarity between a typed query and indexed players, used by player search.

Scoring is deliberately simple and explainable: normalized first and last names are compared
separately with ``difflib.SequenceMatcher`` and averaged, so a shared surname alone ("Mike
Williams" vs "Kyle Williams") can't look like a near-match, and an abbreviated first name ("J.
Jefferson") scores high when the last name matches exactly and the initial agrees.
"""

from difflib import SequenceMatcher

from backend.app.roster.players import IndexedPlayer

INITIAL_SCORE = 0.95


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
