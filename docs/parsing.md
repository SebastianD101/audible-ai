# Screenshot parsing

Turns a roster screenshot into an ID-matched roster that the user reviews before anything
downstream uses it. Code: `backend/app/parsing/`. Design choices: `docs/decisions.md`
(2026-10-27).

## Pipeline

```
bytes -> validate_image -> extract_roster (Claude vision, structured output)
      -> match_player (per row, against nflverse rosters) -> ParsedRoster
```

| Stage | Module | Fails with |
| --- | --- | --- |
| Validate type (magic bytes) and size (<= 5 MB) | `image.py` | `InvalidImageError` |
| Transcribe rows into `ExtractedRoster` | `extract.py` | `ConfigurationError`, `ExtractionError` (`retryable` flag) |
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
```

## Known limitations

- **Players missing from current NFL rosters** (released free agents, for example) come back
  `unmatched` because the index holds only the current season's rostered players.
- **Nicknames** like "Hollywood Brown" don't match the legal name in nflverse. There's no
  nickname table yet, so these come back `unmatched` and need a manual pick.
- **Team-name-only defenses** that are ambiguous ("New York") come back `ambiguous`.
- **Thresholds were set by hand**, not fit to labeled data.

## Measuring extraction accuracy (not done yet)

Unit tests use a fake client, so they check the plumbing, not how well the model reads
screenshots. To measure that honestly:

1. Collect 20+ real screenshots across ESPN, Yahoo, and Sleeper (own leagues; light and dark
   mode; full and cut-off rosters), and hand-label the expected `gsis_id` per slot.
2. Report per platform: row recall and precision (players found vs. on screen), match accuracy
   on found rows, the share flagged `needs_review`, and how often a flagged row was actually
   wrong. Show counts alongside rates; with about 20 images, intervals are wide.
3. Keep labeled screenshots out of git unless they contain no personal league data.

No accuracy number is claimed until this is done.
