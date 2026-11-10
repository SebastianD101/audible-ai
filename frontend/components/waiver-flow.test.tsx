import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { manualRow, rowsFromParse } from "@/lib/rows";
import {
  allen,
  parseResponse,
  recommendation,
  sleeperImport,
  sleeperLeagues,
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
  it("goes from players added by name to a recommendation", async () => {
    const fetchMock = stubApi({
      "/api/projections/meta": () => jsonResponse(200, META),
      "/api/players/search": () => jsonResponse(200, [allen, waiverWideout]),
      "/api/recommendations": () => jsonResponse(200, recommendation()),
    });
    renderWithClient(<WaiverFlow />);
    expect(await screen.findByText(/2026 week 8/)).toBeInTheDocument();
    const button = screen.getByRole("button", { name: "Get recommendation" });
    expect(button).toBeDisabled();

    await userEvent.type(
      screen.getByPlaceholderText("Add a roster player by name"),
      "allen",
    );
    const rosterSection = screen.getByRole("region", {
      name: "1. Your roster",
    });
    let results = await within(rosterSection).findByRole("list", {
      name: "Search results",
    });
    await userEvent.click(
      within(results).getByRole("button", { name: /Josh Allen/ }),
    );
    expect(
      await screen.findByText("Add at least one free agent."),
    ).toBeVisible();

    await userEvent.type(
      screen.getByPlaceholderText("Add a free agent by name"),
      "waiv",
    );
    const faSection = screen.getByRole("region", {
      name: "2. Available free agents",
    });
    results = await within(faSection).findByRole("list", {
      name: "Search results",
    });
    await userEvent.click(
      within(results).getByRole("button", { name: /Waiver Wideout/ }),
    );
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
      max_moves: 1,
    });
  });

  it("fills roster, free agents, and settings from a Sleeper import", async () => {
    stubApi({
      "/api/projections/meta": () => jsonResponse(200, META),
      "/api/sleeper/leagues": () => jsonResponse(200, sleeperLeagues),
      "/api/sleeper/import": () => jsonResponse(200, sleeperImport()),
    });
    renderWithClient(<WaiverFlow />);
    await userEvent.type(screen.getByLabelText("Sleeper username"), "someone");
    await userEvent.click(screen.getByRole("button", { name: "Find leagues" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Import league" }),
    );

    expect(await screen.findByText(/Imported 1 player /)).toBeVisible();
    expect(screen.getByLabelText("W/R flex (RB/WR) starters")).toHaveValue(1);
    expect(screen.getByLabelText("WR starters")).toHaveValue(3);
    expect(
      screen.getByRole("button", { name: "Get recommendation" }),
    ).toBeEnabled();
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
