/** Shared API-shaped fixtures for component and unit tests. */

import type { ParseResponse, PlayerHit } from "@/lib/api";

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
