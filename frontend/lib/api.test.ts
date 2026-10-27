import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  checkImage,
  createRecommendation,
  DEFAULT_SETTINGS,
  detailMessage,
  getProjectionsMeta,
  parseScreenshot,
} from "./api";

function jsonResponse(
  status: number,
  body: unknown,
  headers: Record<string, string> = {},
) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("detailMessage", () => {
  it("passes through a string detail", () => {
    expect(detailMessage({ detail: "No projections" }, 503)).toBe(
      "No projections",
    );
  });

  it("joins validation issues", () => {
    const body = { detail: [{ msg: "too short" }, { msg: "not projected" }] };
    expect(detailMessage(body, 422)).toBe("too short; not projected");
  });

  it("falls back to the status", () => {
    expect(detailMessage(null, 500)).toBe("Request failed (500)");
  });
});

describe("checkImage", () => {
  it("accepts a small PNG", () => {
    expect(
      checkImage(new File(["x"], "a.png", { type: "image/png" })),
    ).toBeNull();
  });

  it("rejects other types and oversized files", () => {
    expect(
      checkImage(new File(["x"], "a.pdf", { type: "application/pdf" })),
    ).toMatch(/PNG/);
    const big = new File([new Uint8Array(5 * 1024 * 1024 + 1)], "big.png", {
      type: "image/png",
    });
    expect(checkImage(big)).toMatch(/5 MB/);
  });
});

describe("request", () => {
  it("returns parsed JSON on success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(200, { week: 8 })),
    );
    await expect(getProjectionsMeta()).resolves.toEqual({ week: 8 });
  });

  it("raises ApiError with the detail and Retry-After", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse(
            503,
            { detail: "Too many requests" },
            { "Retry-After": "30" },
          ),
        ),
    );
    const error = await parseScreenshot(
      new File(["x"], "a.png", { type: "image/png" }),
      "roster",
    ).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 503,
      message: "Too many requests",
      retryAfterSeconds: 30,
    });
  });

  it("explains a network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("fetch failed")),
    );
    await expect(getProjectionsMeta()).rejects.toThrow(
      /Is the backend running/,
    );
  });

  it("posts recommendation requests as JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, {}));
    vi.stubGlobal("fetch", fetchMock);
    await createRecommendation({
      roster: ["a"],
      free_agents: [],
      locked: [],
      max_moves: 1,
      settings: DEFAULT_SETTINGS,
    });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/api\/recommendations$/);
    expect(JSON.parse(init.body as string).roster).toEqual(["a"]);
  });
});
