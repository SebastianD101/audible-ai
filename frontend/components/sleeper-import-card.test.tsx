import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { sleeperImport, sleeperLeagues } from "@/test/fixtures";
import { jsonResponse, renderWithClient, stubApi } from "@/test/render";

import { SleeperImportCard } from "./sleeper-import-card";

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

function setup() {
  const onImported = vi.fn();
  renderWithClient(<SleeperImportCard onImported={onImported} />);
  return { onImported, username: screen.getByLabelText("Sleeper username") };
}

describe("SleeperImportCard", () => {
  it("finds leagues, imports the chosen one, and remembers the username", async () => {
    const fetchMock = stubApi({
      "/api/sleeper/leagues": () => jsonResponse(200, sleeperLeagues),
      "/api/sleeper/import": () => jsonResponse(200, sleeperImport()),
    });
    const { onImported, username } = setup();
    await userEvent.type(username, "someone");
    await userEvent.click(screen.getByRole("button", { name: "Find leagues" }));

    const select = await screen.findByLabelText("Someone's 2026 leagues");
    await userEvent.selectOptions(select, "901");
    await userEvent.click(
      screen.getByRole("button", { name: "Import league" }),
    );

    expect(
      await screen.findByText(
        /Imported 1 player from Test League, the best 1 free agent,/,
      ),
    ).toBeVisible();
    expect(screen.getByText("Your league scores 0.5 per catch.")).toBeVisible();
    expect(onImported).toHaveBeenCalledWith(sleeperImport());
    const call = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/api/sleeper/import"),
    )!;
    expect(JSON.parse(call[1]!.body as string)).toEqual({
      league_id: "901",
      user_id: "111",
    });
    expect(window.localStorage.getItem("audible.sleeperUsername")).toBe(
      "someone",
    );
  });

  it("prefills the saved username", () => {
    window.localStorage.setItem("audible.sleeperUsername", "saved");
    stubApi({});
    const { username } = setup();
    expect(username).toHaveValue("saved");
  });

  it("shows an unknown user's error", async () => {
    stubApi({
      "/api/sleeper/leagues": () =>
        jsonResponse(404, { detail: "Sleeper has no user named 'nobody'." }),
    });
    const { onImported, username } = setup();
    await userEvent.type(username, "nobody");
    await userEvent.click(screen.getByRole("button", { name: "Find leagues" }));
    expect(await screen.findByText("Couldn't find leagues")).toBeVisible();
    expect(screen.getByText(/no user named 'nobody'/)).toBeVisible();
    expect(onImported).not.toHaveBeenCalled();
  });

  it("says when to retry if Sleeper is down", async () => {
    stubApi({
      "/api/sleeper/leagues": () => jsonResponse(200, sleeperLeagues),
      "/api/sleeper/import": () =>
        jsonResponse(
          503,
          { detail: "Couldn't reach Sleeper; try again shortly." },
          { "Retry-After": "30" },
        ),
    });
    const { username } = setup();
    await userEvent.type(username, "someone");
    await userEvent.click(screen.getByRole("button", { name: "Find leagues" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Import league" }),
    );
    expect(
      await screen.findByText("Couldn't import that league"),
    ).toBeVisible();
    expect(screen.getByText(/about 30 seconds/)).toBeVisible();
  });

  it("explains a user with no leagues this season", async () => {
    stubApi({
      "/api/sleeper/leagues": () =>
        jsonResponse(200, { ...sleeperLeagues, leagues: [] }),
    });
    const { username } = setup();
    await userEvent.type(username, "someone");
    await userEvent.click(screen.getByRole("button", { name: "Find leagues" }));
    expect(
      await screen.findByText("Someone has no 2026 leagues on Sleeper."),
    ).toBeVisible();
  });
});
