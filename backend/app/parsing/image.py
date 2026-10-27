"""Validate uploaded screenshot bytes before they are sent anywhere.

The media type comes from the file's magic bytes, never from a filename or a client-supplied
content type. Images are held in memory only; nothing here writes to disk.
"""

import base64
from dataclasses import dataclass
from typing import Literal

MediaType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]

DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # the Messages API's per-image limit


class InvalidImageError(ValueError):
    """The upload is empty, too large, or not a supported image format."""


@dataclass(frozen=True)
class ValidatedImage:
    """An in-memory image whose type and size have been checked."""

    data: bytes
    media_type: MediaType

    def to_base64(self) -> str:
        """Return the image as a base64 string with no line breaks."""
        return base64.standard_b64encode(self.data).decode("ascii")


def sniff_media_type(data: bytes) -> MediaType | None:
    """Return the image type implied by the leading magic bytes, or None if unrecognized."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def validate_image(data: bytes, max_bytes: int = DEFAULT_MAX_BYTES) -> ValidatedImage:
    """Check that ``data`` is a supported image no larger than ``max_bytes``.

    Raises:
        InvalidImageError: if the upload is empty, oversized, or not PNG/JPEG/WebP/GIF.
    """
    if not data:
        raise InvalidImageError("Upload is empty.")
    if len(data) > max_bytes:
        raise InvalidImageError(
            f"Image is {len(data) / 1_048_576:.1f} MB; the limit is {max_bytes / 1_048_576:.0f} MB."
        )
    media_type = sniff_media_type(data)
    if media_type is None:
        raise InvalidImageError("Unsupported file type; upload a PNG, JPEG, WebP, or GIF image.")
    return ValidatedImage(data=data, media_type=media_type)
