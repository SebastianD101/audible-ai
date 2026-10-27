import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { jsonResponse, renderWithClient, stubApi } from "@/test/render";

import Home from "./page";

afterEach(() => vi.unstubAllGlobals());

describe("Home", () => {
  it("renders the product name and the four steps", () => {
    stubApi({ "/api/projections/meta": () => jsonResponse(503, {}) });
    renderWithClient(<Home />);
    expect(
      screen.getByRole("heading", { level: 1, name: "Audible AI" }),
    ).toBeInTheDocument();
    for (const step of [
      "1. Your roster",
      "2. Available free agents",
      "3. League settings",
      "4. Recommendation",
    ]) {
      expect(screen.getByRole("heading", { name: step })).toBeInTheDocument();
    }
  });
});
