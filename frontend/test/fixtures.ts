/** Shared API-shaped fixtures for component and unit tests. */

import type {
  ParseResponse,
  PlayerHit,
  Recommendation,
  SleeperImportResponse,
  SleeperLeaguesResponse,
} from "@/lib/api";

export const allen: PlayerHit = {
  player_id: "qb1",
  name: "Josh Allen",
  position: "QB",
  team: "BUF",
  projection: 24.1,
  hold_value: 26.9,
};

export const williams: PlayerHit = {
  player_id: "wr9",
  name: "Mike Williams",
  position: "WR",
  team: "NYJ",
  projection: 6.2,
  hold_value: 7,
};

export function parseResponse(): ParseResponse {
  const extracted = (
    name: string,
    legibility: "clear" | "partial" = "clear",
  ) => ({
    name,
    position: null,
    nfl_team: null,
    slot: null,
    injury_tag: null,
    legibility,
    note: null,
  });
  return {
    id: "p1",
    kind: "roster",
    roster: {
      platform: "sleeper",
      model: "m",
      warnings: [],
      players: [
        {
          extracted: extracted("Josh Allen"),
          match: {
            status: "matched",
            player_id: "qb1",
            score: 1,
            candidates: [],
          },
          needs_review: false,
        },
        {
          extracted: extracted("M. Williams", "partial"),
          match: {
            status: "ambiguous",
            player_id: null,
            score: 0.85,
            candidates: [
              {
                player_id: "wr9",
                name: "Mike Williams",
                position: "WR",
                team: "NYJ",
                score: 0.85,
              },
              {
                player_id: "wr8",
                name: "Mike Williams",
                position: "WR",
                team: "PIT",
                score: 0.85,
              },
            ],
          },
          needs_review: true,
        },
      ],
    },
    players: { qb1: allen, wr9: williams },
  };
}

export const waiverWideout: PlayerHit = {
  player_id: "wr5",
  name: "Waiver Wideout",
  position: "WR",
  team: "MIA",
  projection: 14,
  hold_value: 9,
};

export function recommendation(
  overrides: Partial<Recommendation> = {},
): Recommendation {
  return {
    id: "r1",
    season: 2026,
    week: 8,
    projections_version: "abc123def456",
    adds: [waiverWideout],
    drops: [williams],
    projected_gain: 7.8,
    before: {
      starters: [
        { slot: "QB", player: allen },
        { slot: "WR", player: williams },
      ],
      points: 30.3,
    },
    after: {
      starters: [
        { slot: "QB", player: allen },
        { slot: "WR", player: waiverWideout },
      ],
      points: 38.1,
    },
    alternatives: [{ add: waiverWideout, drop: williams, projected_gain: 7.8 }],
    protected: [{ player: allen, reason: "season_starter" }],
    caveats: ["Projections are for week 8 only."],
    evidence: {
      realized_gain: 1.98,
      realized_gain_lo: 1.27,
      realized_gain_hi: 2.75,
      hit_rate: 0.61,
      roster_weeks: 5050,
      weeks: 101,
      summary: "In a 2020-2025 backtest, moves gained 1.98 points per week.",
    },
    ...overrides,
  };
}

export const sleeperLeagues: SleeperLeaguesResponse = {
  user_id: "111",
  username: "someone",
  display_name: "Someone",
  season: 2026,
  leagues: [
    {
      league_id: "900",
      name: "Test League",
      status: "in_season",
      total_rosters: 12,
      scoring: "PPR",
    },
    {
      league_id: "901",
      name: "Half League",
      status: "in_season",
      total_rosters: 10,
      scoring: "Half PPR",
    },
  ],
};

/** A Sleeper import: one exact roster player and one free agent. */
export function sleeperImport(): SleeperImportResponse {
  const row = (player: PlayerHit, slot: string | null) => ({
    extracted: {
      name: player.name,
      position: player.position,
      nfl_team: player.team,
      slot,
      injury_tag: null,
      legibility: "clear" as const,
      note: null,
    },
    match: {
      status: "matched" as const,
      player_id: player.player_id,
      score: 1,
      candidates: [],
    },
    needs_review: false,
  });
  const parse = (
    id: string,
    kind: "roster" | "free_agents",
    player: PlayerHit,
    slot: string | null,
  ): ParseResponse => ({
    id,
    kind,
    roster: {
      platform: "sleeper",
      model: "sleeper-api",
      warnings: [],
      players: [row(player, slot)],
    },
    players: { [player.player_id]: player },
  });
  return {
    league_name: "Test League",
    roster: parse("s1", "roster", allen, "QB"),
    free_agents: parse("s2", "free_agents", waiverWideout, null),
    settings: {
      scoring: "ppr",
      slots: { QB: 1, RB: 2, WR: 3, TE: 1, WRRB_FLEX: 1, FLEX: 1 },
      flex_positions: ["RB", "WR", "TE"],
    },
    warnings: ["Your league scores 0.5 per catch."],
  };
}
