"""Free, local screenshot extraction: OCR text lines turned into roster rows.

RapidOCR (PaddleOCR PP-OCRv6 models on ONNX Runtime) reads every text line with a bounding box
and a confidence. OCR doesn't know which line is a player, so the layout is reconstructed with
simple, explainable rules:

1. **Player lines** are name-shaped lines whose best fuzzy score against this season's player
   index (the same ``name_score`` the matcher uses) clears ``NAME_THRESHOLD``. Headers, slot
   labels, records, and point totals don't, so they drop out. Lines that score between
   ``WEAK_NAME_THRESHOLD`` and ``NAME_THRESHOLD`` are kept as ``guessed`` rows, because they are
   usually misread names, and the user decides in review.
2. **Context lines** (a "BUF - QB" subline, an injury tag, a slot label) attach to the nearest
   player line at or above them in the same row. Slot labels sit left of the name; anything to
   the right is a tag.
3. **Legibility** comes from the OCR confidence of the name line.

Every row then goes through the unchanged matcher and review flags, so OCR mistakes surface as
``needs_review`` rows rather than silent errors. The image is decoded in memory only.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from backend.app.parsing.extract import ExtractionError
from backend.app.parsing.image import ValidatedImage
from backend.app.parsing.match import AMBIGUOUS_THRESHOLD, name_score, query_key
from backend.app.parsing.players import IndexedPlayer, PlayerIndex, normalize_name
from backend.app.parsing.schemas import (
    ExtractedPlayer,
    ExtractedRoster,
    Legibility,
    Platform,
    Position,
)
from backend.app.parsing.teams import TEAMS, canonical_team

logger = logging.getLogger(__name__)

OCR_MODEL_NAME = "local-ocr (RapidOCR PP-OCRv6)"
NAME_THRESHOLD = AMBIGUOUS_THRESHOLD  # a confident name line
WEAK_NAME_THRESHOLD = 0.65  # probably a misread name: keep it, flagged
CLEAR_CONFIDENCE = 0.95
PARTIAL_CONFIDENCE = 0.80
ALWAYS_WARNING = "Read with local OCR, which can misread names: check every row."

_POS = r"(?:QB|RB|WR|TE|K|DEF|DST|D/ST)"
# Hyphen, en/em dash, middle dot, bullet, bar, or comma between team and position.
_SEP = r"\s*[-\u2013\u2014\u00b7\u2022|,]?\s*"
TEAM_POS = re.compile(rf"^(?P<team>[A-Z]{{2,3}}){_SEP}(?P<pos>{_POS})$")
POS_TEAM = re.compile(rf"^(?P<pos>{_POS}){_SEP}(?P<team>[A-Z]{{2,3}})$")
# The same, trailing a name on one line: "Josh Allen BUF - QB", "Josh Allen QB - BUF".
TRAILING_TEAM_POS = re.compile(
    rf"\s+(?:(?P<team>[A-Z]{{2,3}}){_SEP}(?P<pos>{_POS})"
    rf"|(?P<pos2>{_POS}){_SEP}(?P<team2>[A-Z]{{2,3}}))\s*$"
)
INJURY_TAGS = {"Q", "D", "O", "OUT", "IR", "PUP", "SUS", "NA", "DTD", "COV"}
SLOT_LABELS = {
    "QB", "RB", "WR", "TE", "FLEX", "W/R/T", "W/R", "WR/TE", "R/W/T", "SUPERFLEX", "SFLX",
    "OP", "BN", "BE", "BENCH", "IR", "K", "DEF", "D/ST", "DST",
}  # fmt: skip
# Sleeper's bye week after team/position: "C Stroud QB - HOU (8)".
BYE_WEEK = re.compile(r"\s*\(\d{1,2}\)\s*$")
TRAILING_TAG = re.compile(r"\s+(?P<tag>Q|D|O|OUT|IR|PUP|SUS|DTD)$")
DEFENSE_WORDS = re.compile(r"\b(?:D/ST|DST|DEF|Defense)\b", re.IGNORECASE)
# Letters (accented too), periods, straight or curly apostrophes, slashes, hyphens, spaces.
NAME_SHAPED = re.compile(r"^[A-Za-z\u00c0-\u00ff][A-Za-z\u00c0-\u00ff.'\u2019/\- ]{2,40}$")
NICKNAMES = {nickname.lower() for _, nickname in TEAMS.values()}
PLATFORMS: dict[str, Platform] = {"espn": "espn", "yahoo": "yahoo", "sleeper": "sleeper"}


@dataclass(frozen=True)
class OcrLine:
    """One line of text OCR found, with its box (pixels) and confidence."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def height(self) -> float:
        """Box height in pixels (at least 1)."""
        return max(self.y1 - self.y0, 1.0)

    @property
    def center_y(self) -> float:
        """Vertical center of the box."""
        return (self.y0 + self.y1) / 2


@dataclass
class _Row:
    """A player line and the context attached to it."""

    line: OcrLine
    name: str
    position: Position | None
    team: str | None
    tag: str | None = None
    slot: str | None = None
    weak: bool = False


@lru_cache(maxsize=1)
def engine() -> Any:
    """The OCR engine, created once per process (loading models takes a moment)."""
    from rapidocr import RapidOCR

    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    return RapidOCR()


def read_lines(image: ValidatedImage) -> list[OcrLine]:
    """Run OCR on the image bytes, in memory.

    Raises:
        ExtractionError: if the engine fails on this image.
    """
    try:
        result = engine()(image.data)
    except Exception as e:  # the engine raises assorted cv2/onnx errors on odd inputs
        logger.warning("Local OCR failed: %s", type(e).__name__)
        raise ExtractionError("Local OCR could not read this image.", retryable=False) from e
    if result.boxes is None:
        return []
    lines = []
    for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
        xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
        lines.append(OcrLine(str(text).strip(), min(xs), min(ys), max(xs), max(ys), float(score)))
    return lines


def position_of(token: str) -> Position | None:
    """``QB``/``RB``/... as a schema position; defense spellings become ``DST``."""
    positions: dict[str, Position] = {
        "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K",
        "DEF": "DST", "DST": "DST", "D/ST": "DST",
    }  # fmt: skip
    return positions.get(token.upper())


def team_and_position(text: str) -> tuple[str | None, Position | None] | None:
    """Parse a "BUF - QB" / "QB - BUF" context line; None if it isn't one."""
    compact = text.strip().upper()
    m = TEAM_POS.match(compact) or POS_TEAM.match(compact)
    if not m or canonical_team(m["team"]) is None:
        return None
    return canonical_team(m["team"]), position_of(m["pos"])


def is_defense(text: str) -> bool:
    """A team defense: "Ravens D/ST", "Baltimore Defense", or a bare nickname."""
    return bool(DEFENSE_WORDS.search(text)) or normalize_name(text) in NICKNAMES


def split_name(text: str) -> tuple[str, str | None, Position | None, str | None]:
    """Strip team/position and an injury tag trailing a name: (name, team, position, tag)."""
    team: str | None = None
    position: Position | None = None
    m = TRAILING_TEAM_POS.search(text)
    if m and canonical_team(m["team"] or m["team2"]) is not None:
        team = canonical_team(m["team"] or m["team2"])
        position = position_of(m["pos"] or m["pos2"])
        text = text[: m.start()]
    tag = None
    t = TRAILING_TAG.search(text)
    if t:
        tag = t["tag"]
        text = text[: t.start()]
    return text.strip(), team, position, tag


def best_score(name: str, position: Position | None, index: PlayerIndex) -> float:
    """Best name similarity of a candidate line to any player at that position."""
    probe = ExtractedPlayer(
        name=name,
        position=position,
        nfl_team=None,
        slot=None,
        injury_tag=None,
        legibility="clear",
        note=None,
    )
    query = query_key(probe)
    if not query:
        return 0.0
    pool: tuple[IndexedPlayer, ...] = index.by_position(position)
    # A cheap length filter skips most of the index for long queries.
    return max(
        (name_score(query, p) for p in pool if abs(len(p.key) - len(query)) <= 12), default=0.0
    )


def player_row(line: OcrLine, index: PlayerIndex) -> _Row | None:
    """A ``_Row`` if this line is (probably) a player name, else None."""
    text = BYE_WEEK.sub("", line.text)
    if not NAME_SHAPED.match(text) or text.upper() in SLOT_LABELS | INJURY_TAGS:
        return None
    if team_and_position(text) is not None:  # a "BAL - DEF" subline, not a name
        return None
    name, team, position, tag = split_name(text)
    if is_defense(name):
        position = "DST"
    if len(normalize_name(name)) < 3:
        return None
    score = best_score(name, position, index)
    if score < WEAK_NAME_THRESHOLD:
        return None
    return _Row(line, name, position, team, tag=tag, weak=score < NAME_THRESHOLD)


def owner(rows: list[_Row], ctx: OcrLine) -> _Row | None:
    """The player row a context line belongs to: the nearest name starting at or above it.

    OCR boxes are padded and overlap the next row, so "at or above" compares the name's top edge
    with the context line's center. A line more than about one and a half rows below every
    name belongs to nobody.
    """
    near = [r for r in rows if r.line.y0 <= ctx.center_y <= r.line.y1 + 1.6 * r.line.height]
    return max(near, key=lambda r: r.line.y0, default=None)


def attach(row: _Row, ctx: OcrLine) -> None:
    """Fill team/position, injury tag, or slot from one context line."""
    parsed = team_and_position(ctx.text)
    if parsed is not None:
        row.team = row.team or parsed[0]
        if row.position != "DST":
            row.position = row.position or parsed[1]
        return
    token = ctx.text.strip().upper()
    left_of_name = ctx.x1 <= row.line.x0 + 2
    if left_of_name and token in SLOT_LABELS:
        row.slot = row.slot or token
    elif not left_of_name and (token in INJURY_TAGS or token == "0"):
        row.tag = row.tag or ("O" if token == "0" else token)  # OCR reads the "O" tag as zero


def legibility(row: _Row) -> tuple[Legibility, str | None]:
    """Legibility from OCR confidence; weak name matches are always ``guessed``."""
    confidence = row.line.score
    if row.weak:
        return "guessed", f"low-confidence name read (OCR {confidence:.2f}); check the player"
    if confidence >= CLEAR_CONFIDENCE:
        return "clear", None
    if confidence >= PARTIAL_CONFIDENCE:
        return "partial", f"OCR confidence {confidence:.2f}"
    return "guessed", f"OCR confidence {confidence:.2f}"


def detect_platform(lines: list[OcrLine]) -> Platform:
    """Platform from brand words anywhere on screen, else ``unknown``."""
    text = " ".join(line.text.lower() for line in lines)
    return next((p for word, p in PLATFORMS.items() if word in text), "unknown")


def lines_to_roster(lines: list[OcrLine], index: PlayerIndex) -> ExtractedRoster:
    """Reconstruct roster rows from OCR lines, in screen order."""
    ordered = sorted(lines, key=lambda line: (round(line.center_y), line.x0))
    rows: list[_Row] = []
    context: list[OcrLine] = []
    for line in ordered:
        row = player_row(line, index)
        if row is None:
            context.append(line)
        else:
            rows.append(row)
    for ctx in context:
        target = owner(rows, ctx)
        if target is not None:
            attach(target, ctx)

    players = []
    for row in rows:
        leg, note = legibility(row)
        players.append(
            ExtractedPlayer(
                name=row.name,
                position=row.position,
                nfl_team=row.team,
                slot=row.slot,
                injury_tag=row.tag,
                legibility=leg,
                note=note,
            )
        )
    warnings = [ALWAYS_WARNING]
    if not players:
        warnings.append(
            "No player names were found. Try a sharper, uncropped screenshot, "
            "or add players by name."
        )
    return ExtractedRoster(platform=detect_platform(lines), players=players, warnings=warnings)
