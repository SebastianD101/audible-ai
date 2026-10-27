import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { recommendation } from "@/test/fixtures";

import { RecommendationView } from "./recommendation-view";

describe("RecommendationView", () => {
  it("leads with the move, the gain, and how much to trust it", () => {
    render(<RecommendationView rec={recommendation()} />);
    expect(
      screen.getByText("Add Waiver Wideout, drop Mike Williams"),
    ).toBeInTheDocument();
    expect(screen.getByText(/\+7\.8 projected/)).toBeInTheDocument();
    expect(screen.getByText(/moves gained 1\.98 points/)).toBeInTheDocument();
    expect(
      screen.getByText("Lineup after: 38.1 projected"),
    ).toBeInTheDocument();
  });

  it("explains why players were protected", () => {
    render(<RecommendationView rec={recommendation()} />);
    expect(
      screen.getByText(/starter on season-long value/),
    ).toBeInTheDocument();
  });

  it("says plainly when there is no move, and when there is no evidence", () => {
    render(
      <RecommendationView
        rec={recommendation({
          adds: [],
          drops: [],
          projected_gain: 0,
          alternatives: [],
          evidence: null,
        })}
      />,
    );
    expect(screen.getByText("No move recommended")).toBeInTheDocument();
    expect(screen.getByText(/were not backtested/)).toBeInTheDocument();
    expect(screen.queryByText(/Lineup after/)).not.toBeInTheDocument();
  });
});
