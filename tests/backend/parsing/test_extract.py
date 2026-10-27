import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
import pydantic
import pytest

from backend.app.parsing.extract import (
    ConfigurationError,
    ExtractionError,
    extract_roster,
)
from backend.app.parsing.image import validate_image
from backend.app.parsing.schemas import ExtractedRoster
from tests.backend.parsing.frames import extracted

IMAGE = validate_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
ROSTER = ExtractedRoster(
    platform="espn", players=[extracted("Justin Jefferson", "WR")], warnings=[]
)


@dataclass
class FakeMessages:
    """Stands in for ``client.messages``: records calls, returns or raises a canned result."""

    result: object
    calls: list[dict[str, Any]] = field(default_factory=list)

    def parse(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def fake_client(result: object) -> tuple[anthropic.Anthropic, FakeMessages]:
    messages = FakeMessages(result)
    return cast(anthropic.Anthropic, SimpleNamespace(messages=messages)), messages


def response(
    parsed: ExtractedRoster | None = ROSTER, stop_reason: str = "end_turn"
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        _request_id="req_123",
        usage=SimpleNamespace(input_tokens=1500, output_tokens=300),
    )


def api_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    http_response = httpx2.Response(status, request=request, headers={"request-id": "req_err"})
    return cls("boom", response=http_response, body=None)


def test_returns_parsed_roster_and_sends_configured_model() -> None:
    client, messages = fake_client(response())
    assert extract_roster(IMAGE, client, model="configured-model", effort="low") == ROSTER
    (call,) = messages.calls
    assert call["model"] == "configured-model"
    assert call["output_format"] is ExtractedRoster
    assert call["output_config"] == {"effort": "low"}
    image_block, text_block = call["messages"][0]["content"]
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["data"] == IMAGE.to_base64()
    assert text_block["type"] == "text"


@pytest.mark.parametrize("model", [None, ""])
def test_missing_model_fails_before_any_call(model: str | None) -> None:
    client, messages = fake_client(response())
    with pytest.raises(ConfigurationError, match="ANTHROPIC_MODEL"):
        extract_roster(IMAGE, client, model=model)
    assert messages.calls == []


@pytest.mark.parametrize(
    ("stop_reason", "message"), [("refusal", "declined"), ("max_tokens", "too long")]
)
def test_unusable_stop_reasons_raise(stop_reason: str, message: str) -> None:
    client, _ = fake_client(response(stop_reason=stop_reason))
    with pytest.raises(ExtractionError, match=message) as exc:
        extract_roster(IMAGE, client, model="m")
    assert exc.value.retryable is False


def test_missing_parsed_output_raises() -> None:
    client, _ = fake_client(response(parsed=None))
    with pytest.raises(ExtractionError, match="no roster"):
        extract_roster(IMAGE, client, model="m")


def test_schema_mismatch_inside_parse_is_an_extraction_error() -> None:
    try:
        ExtractedRoster.model_validate_json('{"platform": "espn", "players": [')
    except pydantic.ValidationError as e:
        truncated = e
    client, _ = fake_client(truncated)
    with pytest.raises(ExtractionError, match="could not be read"):
        extract_roster(IMAGE, client, model="m")


@pytest.mark.parametrize(
    ("error", "expected", "retryable"),
    [
        (api_error(anthropic.AuthenticationError, 401), ConfigurationError, None),
        (api_error(anthropic.PermissionDeniedError, 403), ConfigurationError, None),
        (api_error(anthropic.NotFoundError, 404), ConfigurationError, None),
        (api_error(anthropic.BadRequestError, 400), ExtractionError, False),
        (api_error(anthropic.RateLimitError, 429), ExtractionError, True),
        (api_error(anthropic.InternalServerError, 500), ExtractionError, True),
        (api_error(anthropic.UnprocessableEntityError, 422), ExtractionError, False),
    ],
)
def test_api_errors_are_mapped(
    error: Exception, expected: type[Exception], retryable: bool | None
) -> None:
    client, _ = fake_client(error)
    with pytest.raises(expected) as exc:
        extract_roster(IMAGE, client, model="m")
    if retryable is not None:
        assert isinstance(exc.value, ExtractionError)
        assert exc.value.retryable is retryable


def test_connection_error_is_retryable() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, _ = fake_client(anthropic.APIConnectionError(request=request))
    with pytest.raises(ExtractionError) as exc:
        extract_roster(IMAGE, client, model="m")
    assert exc.value.retryable is True


def test_api_errors_log_the_request_id(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client, _ = fake_client(api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(ConfigurationError):
        extract_roster(IMAGE, client, model="m")
    assert "req_err" in caplog.text  # the request id is logged for support


def test_success_logs_request_id_and_usage(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    client, _ = fake_client(response())
    extract_roster(IMAGE, client, model="m")
    assert "req_123" in caplog.text
    assert "input_tokens=1500" in caplog.text
