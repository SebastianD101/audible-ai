import { describe, expect, it } from "vitest";

import { allen, parseResponse, williams } from "@/test/fixtures";

import {
  duplicateIds,
  isResolved,
  manualRow,
  resolvedIds,
  rowsFromParse,
  unresolvedCount,
} from "./rows";

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
