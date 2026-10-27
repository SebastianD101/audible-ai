import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { allen, parseResponse, williams } from "@/test/fixtures";
import { rowsFromParse, unresolvedCount, type ReviewRow } from "@/lib/rows";
import { jsonResponse, renderWithClient, stubApi } from "@/test/render";

import { ReviewTable } from "./review-table";

afterEach(() => vi.unstubAllGlobals());

function Harness({ onRows }: { onRows?: (rows: ReviewRow[]) => void }) {
  const [rows, setRows] = useState(() => rowsFromParse(parseResponse()));
  return (
    <ReviewTable
      rows={rows}
      onChange={(next) => {
        setRows(next);
        onRows?.(next);
      }}
      allowLock
      emptyText="Nothing yet"
      addLabel="Add a player"
    />
  );
}

describe("ReviewTable", () => {
  it("flags rows that need review and shows read-as details", () => {
    renderWithClient(<Harness />);
    expect(screen.getByText(/1 row needs your review/)).toBeInTheDocument();
    const [clear, flagged] = screen.getAllByTestId("review-row");
    expect(within(clear).getByText("Matched")).toBeInTheDocument();
    expect(within(clear).getByText("24.1")).toBeInTheDocument();
    expect(within(flagged).getByText("partly legible")).toBeInTheDocument();
    expect(within(flagged).getByText("No match")).toBeInTheDocument();
  });

  it("resolves an ambiguous row by picking a candidate", async () => {
    const onRows = vi.fn();
    renderWithClient(<Harness onRows={onRows} />);
    await userEvent.click(
      screen.getByRole("button", { name: "Change M. Williams" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Mike Williams · NYJ" }),
    );
    const rows = onRows.mock.lastCall![0] as ReviewRow[];
    expect(rows[1].player?.player_id).toBe(williams.player_id);
    expect(unresolvedCount(rows)).toBe(0);
    expect(screen.queryByText(/needs your review/)).not.toBeInTheDocument();
  });

  it("removes a row and locks a player", async () => {
    const onRows = vi.fn();
    renderWithClient(<Harness onRows={onRows} />);
    await userEvent.click(
      screen.getByRole("button", { name: "Keep Josh Allen" }),
    );
    expect((onRows.mock.lastCall![0] as ReviewRow[])[0].locked).toBe(true);
    await userEvent.click(
      screen.getByRole("button", { name: "Remove M. Williams" }),
    );
    expect(onRows.mock.lastCall![0]).toHaveLength(1);
  });

  it("adds a player found by search", async () => {
    stubApi({
      "/api/players/search": () => jsonResponse(200, [allen]),
    });
    const onRows = vi.fn();
    renderWithClient(<Harness onRows={onRows} />);
    await userEvent.type(screen.getByPlaceholderText("Add a player"), "jos");
    const results = await screen.findByRole("list", { name: "Search results" });
    await userEvent.click(
      within(results).getByRole("button", { name: /Josh Allen/ }),
    );
    const rows = onRows.mock.lastCall![0] as ReviewRow[];
    expect(rows).toHaveLength(3);
    expect(rows[2].extracted).toBeNull();
    expect(screen.getAllByText("Duplicate")).toHaveLength(2); // both rows for qb1
  });

  it("shows search errors", async () => {
    stubApi({
      "/api/players/search": () =>
        jsonResponse(503, { detail: "No projections yet" }),
    });
    renderWithClient(<Harness />);
    await userEvent.type(screen.getByPlaceholderText("Add a player"), "jos");
    expect(await screen.findByText("No projections yet")).toBeInTheDocument();
  });
});
