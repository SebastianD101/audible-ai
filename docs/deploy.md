# Deployment

| Piece | Host | Cost |
| --- | --- | --- |
| API (FastAPI, Docker) | Fly.io, `shared-cpu-1x` with 1 GB, scales to zero | a few dollars a month at demo traffic |
| Database | Neon Postgres, free tier | $0 |
| Frontend (Next.js) | Vercel, Hobby tier | $0 |
| Weekly data refresh and deploys | GitHub Actions | $0 on a public repo |

Prices are the providers' and change; check their pricing pages.

## One-time setup

1. **Fly app.** Install `flyctl` and run `fly auth login`. Then, from the repo root, run
   `fly launch --no-deploy --copy-config`. If `audible-ai` is taken, pick another name and
   keep the `app =` line it writes into `fly.toml`.
2. **Database.** Create a Neon project in AWS us-east-1, near Fly's `iad` region. Copy its
   connection string, then run:
   `fly secrets set DATABASE_URL='postgresql://...neon.tech/neondb?sslmode=require'`.
   The app converts `postgres://` and `postgresql://` URLs to its psycopg 3 driver.
3. **Frontend.** In Vercel, import the GitHub repo and set the root directory to `frontend`.
   Add the env var `NEXT_PUBLIC_API_URL=https://<app>.fly.dev`, then deploy. Vercel redeploys
   the frontend on every push to `main`.
4. **CORS.** Run `fly secrets set CORS_ORIGINS='["https://<project>.vercel.app"]'`.
5. **Deploy token.** Run `fly tokens create deploy`, and save the output as the repository
   secret `FLY_API_TOKEN` (Settings → Secrets and variables → Actions).
6. **First deploy.** Actions → Deploy → Run workflow. Then check it with
   `curl https://<app>.fly.dev/health`, which should return `{"status":"ok"}`.

Optional: to use Claude for screenshots instead of OCR, set `PARSER = "auto"` in `fly.toml`.
Then run `fly secrets set ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=...`. Every parse is then a
billed API call, so leave it off for a public demo.

## What runs when

- **Push to `main`:** CI runs, and then `deploy.yml` builds the `serve` image and deploys it.
  The image is the code plus the committed projections plus freshly ingested 2026 rosters.
  Migrations run as Fly's `release_command` before new machines take traffic.
- **Wednesday 12:00 UTC:** `refresh.yml` runs these steps:
  - ingests the live season
  - runs `ml.serve.project --week auto`
  - runs `ml.serve.evaluate`
  - commits any new projection file and `docs/live-2026.md`
  - redeploys
- **No `FLY_API_TOKEN`:** the deploy workflow builds and smoke-tests the image, then stops
  with a notice. Forks stay green.

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

Run `fly releases` to list the deployed images. Then run
`fly deploy --image registry.fly.io/<app>:<label>` to go back. Each image carries its own
week's data, so a rollback restores the matching projections too. Labels are the first 12
characters of the git SHA.
