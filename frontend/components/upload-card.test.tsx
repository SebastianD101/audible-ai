import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseResponse } from "@/test/fixtures";
import { jsonResponse, renderWithClient, stubApi } from "@/test/render";

import { UploadCard } from "./upload-card";

afterEach(() => vi.unstubAllGlobals());

const png = () => new File(["png"], "roster.png", { type: "image/png" });

function setup() {
  const onParsed = vi.fn();
  renderWithClient(
    <UploadCard
      kind="roster"
      title="Your roster"
      description="Upload your team"
      onParsed={onParsed}
    />,
  );
  return { onParsed, input: screen.getByLabelText("Your roster screenshot") };
}

describe("UploadCard", () => {
  it("uploads, reports what was read, and hands back the result", async () => {
    const fetchMock = stubApi({
      "/api/parse": () => jsonResponse(200, parseResponse()),
    });
    const { onParsed, input } = setup();
    await userEvent.upload(input, png());
    expect(
      await screen.findByText(/Read 2 players from sleeper/),
    ).toBeVisible();
    expect(onParsed).toHaveBeenCalledWith(parseResponse());
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("kind")).toBe("roster");
  });

  it("rejects non-images before uploading", async () => {
    const fetchMock = stubApi({});
    const { input } = setup();
    await userEvent.upload(
      input,
      new File(["%PDF"], "roster.pdf", { type: "application/pdf" }),
      { applyAccept: false },
    );
    expect(await screen.findByText(/Upload a PNG/)).toBeVisible();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("shows the server's error and when to retry", async () => {
    stubApi({
      "/api/parse": () =>
        jsonResponse(
          503,
          { detail: "Too many requests; try again shortly." },
          { "Retry-After": "30" },
        ),
    });
    const { onParsed, input } = setup();
    await userEvent.upload(input, png());
    expect(
      await screen.findByText("Too many requests; try again shortly."),
    ).toBeVisible();
    expect(screen.getByText(/about 30 seconds/)).toBeVisible();
    expect(onParsed).not.toHaveBeenCalled();
  });
});
