"""Real OCR on a rendered roster screenshot (needs the ~31 MB models; run with ``-m ocr``)."""

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.app.parsing.image import validate_image
from backend.app.parsing.match import match_player
from backend.app.parsing.ocr import lines_to_roster, read_lines
from backend.app.parsing.players import build_player_index
from tests.backend.parsing.frames import rosters

pytestmark = pytest.mark.ocr

# slot, name, "TEAM - POS", injury tag, expected player ID
ROWS = [
    ("QB", "Josh Allen", "BUF - QB", "Q", "00-004"),
    ("RB", "Kenneth Walker III", "SEA - RB", "", "00-010"),
    ("WR", "Justin Jefferson", "MIN - WR", "", "00-001"),
    ("WR", "Amon-Ra St. Brown", "DET - WR", "", "00-008"),
    ("TE", "Sam LaPorta", "DET - TE", "", "00-012"),
    ("FLEX", "A.J. Brown", "PHI - WR", "", "00-003"),
    ("BN", "Marvin Harrison Jr.", "ARI - WR", "IR", "00-002"),
    ("BN", "Davante Adams", "LAR - WR", "", "00-011"),
    ("BN", "Justin Herbert", "LAC - QB", "", "00-014"),
    ("K", "Brandon Aubrey", "DAL - K", "", "00-009"),
    ("DEF", "Ravens D/ST", "BAL - DEF", "", "DST:BAL"),
]


def render(dark: bool) -> bytes:
    """A roster screenshot in the usual app layout: slot | name over team-pos | tag | points."""
    bg, card, fg, dim = (
        ((24, 26, 32), (36, 40, 48), (250, 250, 250), (150, 160, 170))
        if dark
        else ((245, 245, 245), (255, 255, 255), (20, 20, 20), (110, 110, 110))
    )
    big, small = ImageFont.load_default(size=22), ImageFont.load_default(size=15)
    img = Image.new("RGB", (720, 60 + 56 * len(ROWS)), bg)
    draw = ImageDraw.Draw(img)
    draw.text((20, 15), "My Team   7-1   Week 8", font=small, fill=fg)
    for i, (slot, name, meta, tag, _) in enumerate(ROWS):
        y = 60 + i * 56
        draw.rectangle((10, y, 710, y + 50), fill=card)
        draw.text((22, y + 14), slot, font=small, fill=dim)
        draw.text((90, y + 5), name, font=big, fill=fg)
        draw.text((90, y + 31), meta, font=small, fill=dim)
        if tag:
            draw.text((420, y + 14), tag, font=small, fill=(220, 60, 60))
        draw.text((600, y + 14), f"{10 + i * 1.3:.1f}", font=big, fill=(60, 160, 90))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
def test_ocr_reads_a_rendered_roster(dark: bool) -> None:
    index = build_player_index(rosters(), season=2026)
    roster = lines_to_roster(read_lines(validate_image(render(dark))), index)
    matched = [match_player(p, index).player_id for p in roster.players]
    expected = [r[4] for r in ROWS]
    hits = sum(m == e for m, e in zip(matched, expected, strict=False))
    assert len(roster.players) == len(ROWS), [p.name for p in roster.players]
    assert hits >= len(ROWS) - 1, list(zip(matched, expected, strict=False))
    allen = roster.players[0]
    assert (allen.nfl_team, allen.position, allen.injury_tag, allen.slot) == (
        "BUF",
        "QB",
        "Q",
        "QB",
    )
