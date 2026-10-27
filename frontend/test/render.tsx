import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import { vi } from "vitest";

/** Render with a fresh, non-retrying query client. */
export function renderWithClient(ui: ReactElement): RenderResult {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
  );
}

export function jsonResponse(
  status: number,
  body: unknown,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

/** Stub `fetch` with a router: the first matching path prefix answers. */
export function stubApi(
  routes: Record<string, (init?: RequestInit) => Response>,
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url).pathname;
    const match = Object.keys(routes).find((p) => path.startsWith(p));
    if (!match) throw new Error(`unexpected request ${path}`);
    return routes[match](init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
