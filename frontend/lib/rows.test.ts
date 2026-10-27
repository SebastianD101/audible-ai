import { describe, expect, it } from "vitest";

import type { ParseResponse, PlayerHit } from "./api";
import {
  duplicateIds,
  isResolved,
  manualRow,
  resolvedIds,
  rowsFromParse,
  unresolvedCount,
} from "./rows";

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

describe("rowsFromParse", () => {
  it("resolves clear matches and flags the rest", () => {
    const [clear, ambiguous] = rowsFromParse(parseResponse());
    expect(clear.player?.player_id).toBe("qb1");
    expect(isResolved(clear)).toBe(true);
    expect(ambiguous.player).toBeNull();
    expect(ambiguous.needsReview).toBe(true);
    expect(isResolved(ambiguous)).toBe(false);
  });

  it("keeps candidates without a projection as plain hits", () => {
    const [, ambiguous] = rowsFromParse(parseResponse());
    expect(ambiguous.candidates.map((c) => c.player_id)).toEqual([
      "wr9",
      "wr8",
    ]);
    expect(ambiguous.candidates[1].projection).toBeNull();
  });
});

describe("resolution helpers", () => {
  it("count unresolved rows and skip them in the ID list", () => {
    const rows = rowsFromParse(parseResponse());
    expect(unresolvedCount(rows)).toBe(1);
    expect(resolvedIds(rows)).toEqual(["qb1"]);
  });

  it("treat a hand-picked player as confirmed", () => {
    expect(isResolved(manualRow(williams))).toBe(true);
  });

  it("find players picked on more than one row", () => {
    const rows = [...rowsFromParse(parseResponse()), manualRow(allen)];
    expect(duplicateIds(rows)).toEqual(new Set(["qb1"]));
    expect(resolvedIds(rows)).toEqual(["qb1"]);
  });
});
