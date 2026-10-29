# Screenshot parsing

Turns a roster screenshot into an ID-matched roster that the user reviews before anything
downstream uses it. Code: `backend/app/parsing/`. Design choices: `docs/decisions.md`
(2026-10-27).

Sleeper leagues don't need a screenshot: the Sleeper import (`backend/app/sleeper/`, see
`docs/api.md`) reads the roster from Sleeper's API with exact IDs. Screenshots are for ESPN,
Yahoo, and everything else.

## Pipeline

```
bytes -> validate_image -> extract_roster (Claude vision, structured output)   PARSER=claude
                        \-> read_lines + lines_to_roster (local OCR)          PARSER=ocr
      -> match_player (per row, against nflverse rosters) -> ParsedRoster
```

`PARSER=auto` (the default) uses Claude when `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` are set
and local OCR otherwise, so the app works with no API key. `ParsedRoster.model` records which
reader produced each parse.

| Stage | Module | Fails with |
| --- | --- | --- |
| Validate type (magic bytes) and size (<= 5 MB) | `image.py` | `InvalidImageError` |
| Transcribe rows into `ExtractedRoster` | `extract.py` | `ConfigurationError`, `ExtractionError` (`retryable` flag) |
| ...or OCR text lines and rebuild rows | `ocr.py` | `ExtractionError` (not retryable) |
| Index current-season players + 32 defenses | `players.py` | `FileNotFoundError` with the ingest command |
| Match each row to a `gsis_id` (or `DST:<TEAM>`) | `match.py` | never; returns `matched` / `ambiguous` / `unmatched` |
| Combine and flag | `service.py` | n/a |

## What the user sees per player

- `extracted`: the row as read, including `legibility` (`clear` / `partial` / `guessed`) and the
  model's note on any uncertainty.
- `match.status` and up to three `candidates` with scores, so a wrong or missing match can be
  fixed by picking from a short list.
- `needs_review`: set unless the row was read clearly, matched uniquely, and no other row
  matched the same player.

## Try it

```bash
uv run python -m ml.ingest.rosters_weekly --start-season 2026 --end-season 2026
uv run python -m backend.app.parsing path/to/screenshot.png   # real, billed API call
uv run python -m backend.app.parsing path/to/screenshot.png --parser ocr   # local, free
```

## Local OCR

RapidOCR (PaddleOCR PP-OCRv6 detection and recognition models on ONNX Runtime, CPU only)
returns text lines with boxes and confidences. `ocr.py` rebuilds rows with explainable rules:

1. A line is a **player** if it is name-shaped and its best fuzzy score against the player
   index (the matcher's own `name_score`) is >= 0.80. Lines scoring 0.65-0.80 are kept as
   `guessed`, since they are usually misread names. Headers, records, and point totals drop out.
2. **Context lines** ("BUF - QB", injury tags, slot labels) attach to the nearest player line
   at or above them. A slot label sits left of the name; a tag sits right of it. A lone "0"
   right of a name is read as the "O" (out) tag.
3. **Legibility** comes from OCR confidence: `clear` >= 0.95, `partial` >= 0.80, else `guessed`.

Every OCR parse carries a warning to check every row. The engine's models (~31 MB) download on
first use; the Docker image fetches them at build time. `uv run pytest -m ocr` runs the real
engine on rendered rosters (light and dark).

OCR limits compared with Claude: it doesn't understand layout, so an unusual arrangement
(several players per line, a name split across lines) can drop or merge rows. It can't detect
the platform unless a brand word is on screen. It also needs the player index to recognize
names, so a player missing from current rosters is dropped instead of being returned
`unmatched`.

## Known limitations

- **Players missing from current NFL rosters** (released free agents, for example) come back
  `unmatched` because the index holds only the current season's rostered players.
- **Nicknames** like "Hollywood Brown" don't match the legal name in nflverse. There's no
  nickname table yet, so these come back `unmatched` and need a manual pick.
- **Team-name-only defenses** that are ambiguous ("New York") come back `ambiguous`.
- **Thresholds were set by hand**, not fit to labeled data.

## Measuring extraction accuracy (not done yet)

Unit tests use a fake client or fake OCR lines, so they check the plumbing, not how well either
reader handles real screenshots. The OCR reader got 12 of 12 rows right (names, slots, tags,
defense) on one synthetic screenshot drawn in a generic app layout. That is a smoke check, not
an accuracy estimate. To measure that honestly:

1. Collect 20+ real screenshots across ESPN, Yahoo, and Sleeper (own leagues; light and dark
   mode; full and cut-off rosters), and hand-label the expected `gsis_id` per slot.
2. Report per platform and per reader (Claude vs. OCR): row recall and precision (players
   found vs. on screen), match accuracy on found rows, the share flagged `needs_review`, and how often a flagged row was actually
   wrong. Show counts alongside rates; with about 20 images, intervals are wide.
3. Keep labeled screenshots out of git unless they contain no personal league data.

No accuracy number is claimed until this is done.
