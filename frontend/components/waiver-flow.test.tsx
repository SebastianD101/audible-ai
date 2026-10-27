import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { manualRow, rowsFromParse } from "@/lib/rows";
import {
  allen,
  parseResponse,
  recommendation,
  waiverWideout,
} from "@/test/fixtures";
import { jsonResponse, renderWithClient, stubApi } from "@/test/render";

import { blocker, WaiverFlow } from "./waiver-flow";

afterEach(() => vi.unstubAllGlobals());

const META = {
  season: 2026,
  week: 8,
  version: "abc123def456",
  generated_at_utc: "2026-10-27T23:00:00+00:00",
  rows: 546,
  model: "lgbm",
};

describe("blocker", () => {
  it("explains what is missing, in order", () => {
    const parsed = rowsFromParse(parseResponse());
    expect(blocker([], [])).toBe("Add your roster first.");
    expect(blocker(parsed, [])).toBe("Resolve 1 flagged row first.");
    expect(blocker([parsed[0]], [])).toBe("Add at least one free agent.");
    expect(blocker([parsed[0]], [manualRow(waiverWideout)])).toBeNull();
  });
});

describe("WaiverFlow", () => {
  it("goes from screenshot to recommendation once flagged rows are resolved", async () => {
    const fetchMock = stubApi({
      "/api/projections/meta": () => jsonResponse(200, META),
      "/api/parse": () => jsonResponse(200, parseResponse()),
      "/api/players/search": () => jsonResponse(200, [waiverWideout]),
      "/api/recommendations": () => jsonResponse(200, recommendation()),
    });
    renderWithClient(<WaiverFlow />);
    expect(await screen.findByText(/2026 week 8/)).toBeInTheDocument();

    await userEvent.upload(
      screen.getByLabelText("Roster screenshot screenshot"),
      new File(["png"], "roster.png", { type: "image/png" }),
    );
    expect(
      await screen.findByText("Resolve 1 flagged row first."),
    ).toBeVisible();
    const button = screen.getByRole("button", { name: "Get recommendation" });
    expect(button).toBeDisabled();

    await userEvent.click(
      screen.getByRole("button", { name: "Remove M. Williams" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Keep Josh Allen" }),
    );
    await userEvent.type(
      screen.getByPlaceholderText("Add a free agent by name"),
      "waiv",
    );
    const results = await screen.findByRole("list", { name: "Search results" });
    await userEvent.click(within(results).getByRole("button"));
    expect(button).toBeEnabled();

    await userEvent.click(button);
    expect(
      await screen.findByText("Add Waiver Wideout, drop Mike Williams"),
    ).toBeVisible();

    const call = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/api/recommendations"),
    )!;
    const body = JSON.parse(call[1]!.body as string);
    expect(body).toMatchObject({
      roster: [allen.player_id],
      free_agents: [waiverWideout.player_id],
      locked: [allen.player_id],
      max_moves: 1,
    });
  });

  it("shows when projections are unavailable", async () => {
    stubApi({
      "/api/projections/meta": () =>
        jsonResponse(503, { detail: "no projections in ml/data/projections" }),
    });
    renderWithClient(<WaiverFlow />);
    expect(await screen.findByText("Projections unavailable")).toBeVisible();
    expect(screen.getByText(/no projections in/)).toBeVisible();
  });
});
