# Deployment

One Render web service hosts everything: the FastAPI API and the static Next.js UI on the
same origin, from a prebuilt image with the week's data baked in. It costs $0.

| Piece | Host | Cost |
| --- | --- | --- |
| API + UI (one Docker image) | Render, Free web service (512 MB, sleeps when idle) | $0 |
| Image registry | GitHub Container Registry, public package | $0 |
| Weekly data refresh, builds, and deploys | GitHub Actions | $0 on a public repo |

Prices and free-tier limits are the providers' and change; check their pricing pages.

**What the free tier costs in behavior.** The service sleeps after 15 idle minutes, and the
next visit waits about a minute while it wakes. The UI is served by the same service, so the
first page load waits too. Memory is capped at 512 MB. Every deploy smoke-tests the image under
that cap, so an out-of-memory optimizer fails before it ships. Bandwidth is limited (about 5 GB
a month), which is plenty at demo traffic. There's no persistent database: the image defaults
`DATABASE_URL` to a SQLite file in `/tmp`, so the audit log (`parse_records`,
`recommendation_records`) resets on every restart. Nothing reads those tables, so the app behaves the same.
Postgres stays supported: set `DATABASE_URL` and it's used, as in docker-compose and CI.

## One-time setup

1. **Publish the first image.** Merge to `main`, or run Actions → Deploy on `main`. The
   workflow pushes `ghcr.io/sebastiand101/audible-ai:latest`. On GitHub, open the package
   (your profile → Packages → audible-ai → Package settings) and set its visibility to
   **Public**, so Render can pull it without credentials.
2. **Create the Render service.** In Render: New → Web Service → **Existing image**, image URL
   `ghcr.io/sebastiand101/audible-ai:latest`, instance type **Free**. Under Advanced, set the
   health check path to `/health`. Leave the environment variables empty: the image sets
   `STATIC_DIR` and `DATABASE_URL`, and Render sets `PORT`.
3. **Deploy hook.** In the service's Settings, copy the **Deploy Hook** URL. Save it as a repo
   secret without printing it: `gh secret set RENDER_DEPLOY_HOOK_URL` (paste at the prompt).
4. **First deploy.** Run `gh workflow run deploy.yml --ref main`, then check
   `curl https://<service>.onrender.com/health`, which should return `{"status":"ok"}`. Then
   open the site and run Sleeper import → review → recommendations.

## What runs when

- **Push to `main`:** CI runs, and then `deploy.yml` does these steps:
  - ingests the 2026 rosters
  - builds the `serve` image (code, committed projections, rosters, and the static UI)
  - smoke-tests it under 512 MB with `scripts/smoke_serve.py`, which checks the UI, data,
    search, and a full-size recommendation, and writes peak memory to the job summary
  - pushes `:<sha12>` and `:latest` to GHCR
  - calls the deploy hook with `imgURL` set to the `:<sha12>` tag, so Render runs the exact
    image that was tested

  Render's image services don't redeploy on their own when a tag changes; the hook is what
  deploys. Migrations run at container start (the free tier has no pre-deploy step).
- **Wednesday 12:00 UTC (scheduled):** `refresh.yml` runs these steps:
  - ingests the live season
  - runs `ml.serve.project --week auto`
  - runs `ml.serve.evaluate`
  - commits any new projection file and `docs/live-2026.md`
  - redeploys

  GitHub can start scheduled runs hours late. Recent runs began around 19:15 UTC, still well
  before Thursday kickoff.
- **Off `main`:** the deploy workflow builds and smoke-tests only.
- **No `RENDER_DEPLOY_HOOK_URL`:** it builds, smoke-tests, and pushes the image, then stops
  with a notice. Forks and a not-yet-configured repo stay green.

## When the refresh does nothing

`--week auto` logs why it didn't project and exits successfully. The log says one of these:

- **already projected:** the week's file exists. Published projections are never overwritten.
- **games have already kicked off:** the run came after Thursday night. Projecting now could
  see results, so the week is skipped. Run it earlier next week, or project by hand before
  kickoff with `--week W`.
- **no final score:** this comes from `--week W`. Week W-1's last game isn't in nflverse yet.
  Re-run later; nflverse usually updates overnight.
- **season is over.**

## Rollback

Every deployed image stays in GHCR under the first 12 characters of its git SHA. To go back,
call the deploy hook with an older tag, URL-encoded:
`curl "$RENDER_DEPLOY_HOOK_URL&imgURL=ghcr.io%2Fsebastiand101%2Faudible-ai%3A<sha12>"`, or use
Render's dashboard (Deploys → Rollback). Each image carries its own week's data, so a
rollback restores the matching projections too.
