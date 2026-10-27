import base64

import pytest

from backend.app.parsing.image import InvalidImageError, validate_image

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 8
GIF = b"GIF89a" + b"\x00" * 16


@pytest.mark.parametrize(
    ("data", "media_type"),
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp"), (GIF, "image/gif")],
)
def test_detects_media_type_from_magic_bytes(data: bytes, media_type: str) -> None:
    assert validate_image(data).media_type == media_type


def test_rejects_empty_upload() -> None:
    with pytest.raises(InvalidImageError, match="empty"):
        validate_image(b"")


def test_rejects_oversized_upload() -> None:
    with pytest.raises(InvalidImageError, match="limit"):
        validate_image(PNG + b"\x00" * 100, max_bytes=50)


def test_accepts_upload_at_exact_limit() -> None:
    assert validate_image(PNG, max_bytes=len(PNG)).media_type == "image/png"


@pytest.mark.parametrize(
    "data",
    [
        b"name,team\nJustin Jefferson,MIN\n",  # a CSV renamed to .png
        b"%PDF-1.7\n",
        b"<svg xmlns='http://www.w3.org/2000/svg'/>",
        b"RIFF\x10\x00\x00\x00WAVEfmt ",  # RIFF but not WebP
    ],
)
def test_rejects_non_images_regardless_of_name(data: bytes) -> None:
    with pytest.raises(InvalidImageError, match="Unsupported"):
        validate_image(data)


def test_base64_round_trips_without_newlines() -> None:
    encoded = validate_image(PNG * 100).to_base64()
    assert "\n" not in encoded
    assert base64.standard_b64decode(encoded) == PNG * 100
