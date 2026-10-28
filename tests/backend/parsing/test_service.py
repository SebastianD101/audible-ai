import logging
from pathlib import Path

import pytest
from pydantic import SecretStr

from backend.app.config import Settings
from backend.app.parsing import __main__ as cli
from backend.app.parsing.extract import ConfigurationError
from backend.app.parsing.image import InvalidImageError
from backend.app.parsing.ocr import ALWAYS_WARNING, OCR_MODEL_NAME, OcrLine
from backend.app.parsing.players import PlayerIndex, build_player_index
from backend.app.parsing.schemas import ExtractedPlayer, ExtractedRoster
from backend.app.parsing.service import make_client, parse_roster_screenshot
from tests.backend.parsing.frames import extracted, rosters
from tests.backend.parsing.test_extract import fake_client, response

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
SECRET = "sk-ant-test-secret"


@pytest.fixture(scope="module")
def index() -> PlayerIndex:
    return build_player_index(rosters(), season=2026)


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "anthropic_api_key": SecretStr(SECRET),
        "anthropic_model": "configured-model",
    }
    return Settings.model_validate(values | overrides)


def roster_of(*players: ExtractedPlayer, warnings: list[str] | None = None) -> ExtractedRoster:
    return ExtractedRoster(platform="sleeper", players=list(players), warnings=warnings or [])


def test_clear_unique_matches_need_no_review(index: PlayerIndex) -> None:
    client, _ = fake_client(
        response(
            roster_of(extracted("Justin Jefferson", "WR", "MIN"), extracted("Josh Allen", "QB"))
        )
    )
    parsed = parse_roster_screenshot(PNG, client, settings(), index)
    assert [p.match.player_id for p in parsed.players] == ["00-001", "00-004"]
    assert not any(p.needs_review for p in parsed.players)
    assert parsed.model == "configured-model"
    assert parsed.platform == "sleeper"


@pytest.mark.parametrize(
    "player",
    [
        extracted("Justin Jefferson", "WR", legibility="partial"),  # matched but blurry
        extracted("Mike Williams", "WR"),  # ambiguous
        extracted("Zxqv Plorbin", "WR"),  # unmatched
    ],
)
def test_uncertain_rows_are_flagged(index: PlayerIndex, player: ExtractedPlayer) -> None:
    client, _ = fake_client(response(roster_of(player)))
    (parsed,) = parse_roster_screenshot(PNG, client, settings(), index).players
    assert parsed.needs_review


def test_two_rows_matching_one_player_are_both_flagged(index: PlayerIndex) -> None:
    client, _ = fake_client(
        response(
            roster_of(
                extracted("Justin Jefferson", "WR"),
                extracted("J. Jefferson", "WR"),
                extracted("Josh Allen", "QB"),
            )
        )
    )
    parsed = parse_roster_screenshot(PNG, client, settings(), index)
    assert [p.needs_review for p in parsed.players] == [True, True, False]
    assert any("same player" in w for w in parsed.warnings)


def test_model_warnings_are_passed_through(index: PlayerIndex) -> None:
    client, _ = fake_client(response(roster_of(warnings=["roster cut off at bottom"])))
    assert parse_roster_screenshot(PNG, client, settings(), index).warnings == [
        "roster cut off at bottom"
    ]


def test_invalid_image_never_reaches_the_api(index: PlayerIndex) -> None:
    client, messages = fake_client(response())
    with pytest.raises(InvalidImageError):
        parse_roster_screenshot(b"not an image", client, settings(), index)
    with pytest.raises(InvalidImageError):
        parse_roster_screenshot(PNG, client, settings(max_upload_bytes=8), index)
    assert messages.calls == []


def test_settings_reject_upload_limit_above_api_limit() -> None:
    with pytest.raises(ValueError, match="max_upload_bytes"):
        settings(max_upload_bytes=50 * 1024 * 1024)


def test_api_key_never_appears_in_output_or_logs(
    index: PlayerIndex, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    s = settings()
    client, _ = fake_client(response())
    parsed = parse_roster_screenshot(PNG, client, s, index)
    assert SECRET not in parsed.model_dump_json()
    assert SECRET not in caplog.text
    assert SECRET not in repr(s)


def test_make_client_uses_configured_key() -> None:
    client = make_client(settings())
    assert client is not None
    assert client.api_key == SECRET


def test_cli_reports_missing_model_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "roster.png"
    image.write_bytes(PNG)
    monkeypatch.setattr(
        cli, "get_settings", lambda: settings(parser="claude", anthropic_model=None)
    )
    monkeypatch.setattr(
        cli, "load_player_index", lambda season: build_player_index(rosters(), season)
    )
    assert cli.main([str(image)]) == 1
    assert "ANTHROPIC_MODEL" in capsys.readouterr().err


def test_cli_success_prints_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "roster.png"
    image.write_bytes(PNG)
    client, _ = fake_client(response(roster_of(extracted("Josh Allen", "QB"))))
    monkeypatch.setattr(cli, "get_settings", settings)
    monkeypatch.setattr(cli, "make_client", lambda s: client)
    monkeypatch.setattr(
        cli, "load_player_index", lambda season: build_player_index(rosters(), season)
    )
    assert cli.main([str(image)]) == 0
    out = capsys.readouterr()
    assert '"player_id": "00-004"' in out.out
    assert "0 flagged" in out.err


def fake_ocr(monkeypatch: pytest.MonkeyPatch, *texts: str) -> None:
    """Make local OCR "read" one line per text, one row apart."""
    lines = [OcrLine(t, 80, 60 + 56 * i, 300, 93 + 56 * i, 0.99) for i, t in enumerate(texts)]
    monkeypatch.setattr("backend.app.parsing.service.read_lines", lambda image: lines)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "claude"),
        ({"anthropic_model": None}, "ocr"),
        ({"anthropic_api_key": None}, "ocr"),
        ({"parser": "ocr"}, "ocr"),
        ({"parser": "claude", "anthropic_model": None}, "claude"),
    ],
)
def test_auto_parser_uses_claude_only_when_configured(
    overrides: dict[str, object], expected: str
) -> None:
    assert settings(**overrides).resolved_parser() == expected


def test_ocr_path_needs_no_client_and_shares_review_flags(
    index: PlayerIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ocr(monkeypatch, "Josh Allen BUF - QB", "Jusin Jaffarsan")
    s = settings(parser="ocr")
    assert make_client(s) is None
    parsed = parse_roster_screenshot(PNG, None, s, index)
    assert parsed.model == OCR_MODEL_NAME
    assert parsed.players[0].match.player_id == "00-004"
    assert [p.needs_review for p in parsed.players] == [False, True]
    assert parsed.warnings[0] == ALWAYS_WARNING


def test_claude_parser_without_client_is_a_configuration_error(index: PlayerIndex) -> None:
    with pytest.raises(ConfigurationError):
        parse_roster_screenshot(PNG, None, settings(parser="claude"), index)


def test_cli_parser_flag_overrides_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "roster.png"
    image.write_bytes(PNG)
    fake_ocr(monkeypatch, "Josh Allen")
    monkeypatch.setattr(cli, "get_settings", settings)
    monkeypatch.setattr(
        cli, "load_player_index", lambda season: build_player_index(rosters(), season)
    )
    assert cli.main([str(image), "--parser", "ocr"]) == 0
    assert OCR_MODEL_NAME in capsys.readouterr().out
