"""Read a roster off a screenshot with a Claude vision model and structured outputs.

The model is asked to transcribe, not to reason about fantasy value: it reports only what is
visible, leaves unknown fields null, and marks every row's legibility so uncertain reads reach
the user for review. The response is constrained to the ``ExtractedRoster`` JSON schema, so a
successful call always yields a validated object.

The client is passed in, so tests use a fake and nothing here touches the network on import.
The model name comes from settings (``ANTHROPIC_MODEL``) and is never hardcoded.
"""

import logging
from typing import Literal

import anthropic
import pydantic

from backend.app.parsing.image import ValidatedImage
from backend.app.parsing.schemas import ExtractedRoster

logger = logging.getLogger(__name__)

MAX_TOKENS = 16_000

SYSTEM_PROMPT = """\
You transcribe fantasy football roster screenshots into structured data.

Transcribe only what is visible in the image. Do not add players who are not shown, and do not \
fill in a team, position, or injury tag from your own knowledge when the screenshot doesn't \
show it; leave that field null instead. Include every rostered player in screen order, starters \
and bench and IR. Skip empty slots, opponents, projections, and point totals.

For each player, set legibility to "clear" only if you read every field without doubt. Use \
"partial" when some text is cut off, truncated with an ellipsis, or blurry, and "guessed" when \
the name itself is uncertain; say why in the note. For a team defense, put the team name as \
shown in name and use position DST.

If the image is not a fantasy football roster, return an empty players list and explain in \
warnings. Also use warnings for whole-image problems such as a roster cut off at the bottom.\
"""

USER_PROMPT = "Transcribe the roster in this screenshot."


class ConfigurationError(RuntimeError):
    """The parser can't run because the API key or model is missing or rejected."""


class ExtractionError(RuntimeError):
    """The model call failed or returned something unusable.

    ``retryable`` tells the caller whether trying again later might succeed.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        """Store the message and whether a retry could help."""
        super().__init__(message)
        self.retryable = retryable


def build_messages(image: ValidatedImage) -> list[anthropic.types.MessageParam]:
    """The single user turn: the screenshot followed by the instruction."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": image.to_base64(),
                    },
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]


def extract_roster(
    image: ValidatedImage,
    client: anthropic.Anthropic,
    model: str | None,
    effort: Literal["low", "medium", "high"] = "medium",
) -> ExtractedRoster:
    """Send ``image`` to ``model`` and return the transcribed roster.

    Raises:
        ConfigurationError: no model is configured, or the API rejected the credentials.
        ExtractionError: the request failed, was declined, or the output was cut off.
    """
    if not model:
        raise ConfigurationError("ANTHROPIC_MODEL is not set; add it to .env.")
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=build_messages(image),
            output_config={"effort": effort},
            output_format=ExtractedRoster,
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        logger.error("Anthropic credentials rejected (request_id=%s)", e.request_id)
        raise ConfigurationError("The Anthropic API rejected the configured credentials.") from e
    except anthropic.NotFoundError as e:
        logger.error("Model %r not found (request_id=%s)", model, e.request_id)
        raise ConfigurationError(f"Model {model!r} was not found; check ANTHROPIC_MODEL.") from e
    except anthropic.BadRequestError as e:
        logger.warning("Extraction request rejected (request_id=%s)", e.request_id)
        raise ExtractionError("The image could not be processed.", retryable=False) from e
    except anthropic.RateLimitError as e:
        logger.warning("Rate limited (request_id=%s)", e.request_id)
        raise ExtractionError("Too many requests; try again shortly.", retryable=True) from e
    except anthropic.APIStatusError as e:
        logger.warning("API error %d (request_id=%s)", e.status_code, e.request_id)
        retryable = e.status_code >= 500
        raise ExtractionError("The vision service returned an error.", retryable=retryable) from e
    except anthropic.APIConnectionError as e:
        logger.warning("Could not reach the Anthropic API: %s", type(e).__name__)
        raise ExtractionError("Could not reach the vision service.", retryable=True) from e
    except pydantic.ValidationError as e:
        # parse() validates inside the call; a truncated or declined reply lands here.
        logger.warning("Model output did not match the roster schema: %d errors", e.error_count())
        raise ExtractionError(
            "The model's reply could not be read as a roster.", retryable=True
        ) from e

    logger.info(
        "Extracted roster (request_id=%s, stop_reason=%s, input_tokens=%d, output_tokens=%d)",
        response._request_id,
        response.stop_reason,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    if response.stop_reason == "refusal":
        raise ExtractionError("The model declined to read this image.", retryable=False)
    if response.stop_reason == "max_tokens":
        raise ExtractionError("The roster was too long to read in one pass.", retryable=False)
    roster = response.parsed_output
    if roster is None:
        raise ExtractionError("The model returned no roster.", retryable=True)
    return roster
