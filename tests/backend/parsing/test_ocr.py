import pytest

from backend.app.parsing.extract import ExtractionError
from backend.app.parsing.image import validate_image
from backend.app.parsing.ocr import (
    ALWAYS_WARNING,
    OcrLine,
    lines_to_roster,
    read_lines,
    split_name,
    team_and_position,
)
from backend.app.parsing.players import PlayerIndex, build_player_index
from tests.backend.parsing.frames import rosters

ROW = 56  # pixels between roster rows


@pytest.fixture(scope="module")
def index() -> PlayerIndex:
    return build_player_index(rosters(), season=2026)


def line(text: str, row: int, x: float = 80, *, dy: float = 0, score: float = 0.99) -> OcrLine:
    """A padded OCR box like RapidOCR's: about 33 px tall, overlapping the next row."""
    y0 = 60 + row * ROW + dy
    return OcrLine(text, x, y0, x + 10 * len(text), y0 + 33, score)


def roster_lines(*names: str) -> list[OcrLine]:
    """Slot label, name, "TEAM - POS" subline, and points, per row (shuffled like OCR)."""
    meta = {"Justin Jefferson": "MIN - WR", "Josh Allen": "BUF - QB", "Sam LaPorta": "DET - TE"}
    out = [line("My Team  7-1  Week 8", -1, 13)]
    for i, name in enumerate(names):
        out += [
            line(meta.get(name, "MIN - WR"), i, dy=22),
            line(name, i),
            line("WR" if i else "QB", i, 12, dy=6),
            line(f"{10 + i:.1f}", i, 590, dy=7),
        ]
    return out


def test_rows_get_name_team_position_and_slot(index: PlayerIndex) -> None:
    roster = lines_to_roster(roster_lines("Josh Allen", "Justin Jefferson"), index)
    allen, jj = roster.players
    assert (allen.name, allen.nfl_team, allen.position, allen.slot) == (
        "Josh Allen",
        "BUF",
        "QB",
        "QB",
    )
    assert (jj.name, jj.nfl_team, jj.position, jj.slot) == ("Justin Jefferson", "MIN", "WR", "WR")
    assert all(p.legibility == "clear" for p in roster.players)


def test_headers_points_and_slots_are_not_players(index: PlayerIndex) -> None:
    roster = lines_to_roster(roster_lines("Sam LaPorta"), index)
    assert [p.name for p in roster.players] == ["Sam LaPorta"]


def test_screen_order_is_kept(index: PlayerIndex) -> None:
    names = ["Sam LaPorta", "Josh Allen", "Justin Jefferson"]
    lines = roster_lines(*names)
    roster = lines_to_roster(list(reversed(lines)), index)
    assert [p.name for p in roster.players] == names


def test_injury_tag_right_of_name_and_zero_read_as_out(index: PlayerIndex) -> None:
    lines = [
        line("Josh Allen", 0),
        line("Q", 0, 420, dy=12),
        line("Justin Jefferson", 1),
        line("0", 1, 420, dy=12),
    ]
    allen, jj = lines_to_roster(lines, index).players
    assert allen.injury_tag == "Q"
    assert jj.injury_tag == "O"


def test_ir_left_of_name_is_a_slot_and_right_of_name_a_tag(index: PlayerIndex) -> None:
    lines = [line("IR", 0, 12, dy=6), line("Josh Allen", 0), line("IR", 0, 420, dy=12)]
    (allen,) = lines_to_roster(lines, index).players
    assert (allen.slot, allen.injury_tag) == ("IR", "IR")


def test_team_and_position_on_the_name_line(index: PlayerIndex) -> None:
    (allen,) = lines_to_roster([line("Josh Allen BUF - QB", 0)], index).players
    assert (allen.name, allen.nfl_team, allen.position) == ("Josh Allen", "BUF", "QB")


def test_abbreviated_first_name_is_kept_for_the_matcher(index: PlayerIndex) -> None:
    (jj,) = lines_to_roster([line("J. Jefferson", 0)], index).players
    assert jj.name == "J. Jefferson"


def test_defense_rows(index: PlayerIndex) -> None:
    lines = [line("Ravens D/ST", 0), line("BAL - DEF", 0, dy=22), line("Vikings", 1)]
    ravens, vikings = lines_to_roster(lines, index).players
    assert (ravens.position, ravens.nfl_team) == ("DST", "BAL")
    assert vikings.position == "DST"


def test_confidence_sets_legibility(index: PlayerIndex) -> None:
    lines = [
        line("Josh Allen", 0, score=0.99),
        line("Justin Jefferson", 1, score=0.88),
        line("Sam LaPorta", 2, score=0.6),
    ]
    legibility = [p.legibility for p in lines_to_roster(lines, index).players]
    assert legibility == ["clear", "partial", "guessed"]


def test_misread_name_is_kept_as_guessed(index: PlayerIndex) -> None:
    (row,) = lines_to_roster([line("Jusin Jaffarsan", 0)], index).players
    assert row.legibility == "guessed"
    assert row.note is not None


def test_always_warns_and_explains_an_empty_read(index: PlayerIndex) -> None:
    roster = lines_to_roster([line("Standings", 0), line("12.4", 1)], index)
    assert roster.players == []
    assert roster.warnings[0] == ALWAYS_WARNING
    assert "No player names" in roster.warnings[1]


def test_platform_from_brand_words(index: PlayerIndex) -> None:
    roster = lines_to_roster([line("ESPN Fantasy", -1), line("Josh Allen", 0)], index)
    assert roster.platform == "espn"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BUF - QB", ("BUF", "QB")),
        ("WR · LAR", ("LA", "WR")),
        ("BAL - DEF", ("BAL", "DST")),
        ("ZZZ - QB", None),
        ("Josh Allen", None),
    ],
)
def test_team_and_position_lines(text: str, expected: tuple[str, str] | None) -> None:
    assert team_and_position(text) == expected


def test_split_name_strips_trailing_tag() -> None:
    assert split_name("Josh Allen Q") == ("Josh Allen", None, None, "Q")


def test_engine_failure_is_an_extraction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> object:
        def run(_: bytes) -> object:
            raise ValueError("cv2 error")

        return run

    monkeypatch.setattr("backend.app.parsing.ocr.engine", broken)
    with pytest.raises(ExtractionError) as info:
        read_lines(validate_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32))
    assert not info.value.retryable


def test_sleeper_line_with_initial_and_bye_week(index: PlayerIndex) -> None:
    lines = [line("J Allen QB - BUF (7)", 0), line("BAL DEF- BAL (7)", 1)]
    allen, ravens = lines_to_roster(lines, index).players
    assert (allen.name, allen.nfl_team, allen.position) == ("J Allen", "BUF", "QB")
    assert (ravens.name, ravens.nfl_team, ravens.position) == ("BAL", "BAL", "DST")
