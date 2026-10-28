/**
 * Typed client for the FastAPI backend.
 *
 * Types mirror the backend's Pydantic models. The backend snapshots its OpenAPI schema
 * (tests/backend/api/openapi.json), so a contract change shows up in review next to this file.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const ACCEPTED_IMAGE_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
] as const;
export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

export type Position = "QB" | "RB" | "WR" | "TE" | "K" | "DST";
export type Legibility = "clear" | "partial" | "guessed";
export type MatchStatus = "matched" | "ambiguous" | "unmatched";
export type ParseKind = "roster" | "free_agents";
export type SlotName =
  "QB" | "RB" | "WR" | "TE" | "WRRB_FLEX" | "FLEX" | "SUPERFLEX";

export interface PlayerHit {
  player_id: string;
  name: string;
  position: Position;
  team: string | null;
  projection: number | null;
  hold_value: number | null;
}

export interface ExtractedPlayer {
  name: string;
  position: Position | null;
  nfl_team: string | null;
  slot: string | null;
  injury_tag: string | null;
  legibility: Legibility;
  note: string | null;
}

export interface Candidate {
  player_id: string;
  name: string;
  position: Position;
  team: string | null;
  score: number;
}

export interface PlayerMatch {
  status: MatchStatus;
  player_id: string | null;
  score: number;
  candidates: Candidate[];
}

export interface ParsedPlayer {
  extracted: ExtractedPlayer;
  match: PlayerMatch;
  needs_review: boolean;
}

export interface ParsedRoster {
  platform: string;
  players: ParsedPlayer[];
  warnings: string[];
  model: string;
}

export interface ParseResponse {
  id: string;
  kind: ParseKind;
  roster: ParsedRoster;
  players: Record<string, PlayerHit>;
}

export interface ProjectionsMeta {
  season: number;
  week: number;
  version: string;
  generated_at_utc: string;
  rows: number;
  model: string;
}

export interface LeagueSettings {
  scoring: "ppr";
  slots: Partial<Record<SlotName, number>>;
  flex_positions: ("RB" | "WR" | "TE")[];
}

export interface RecommendationRequest {
  roster: string[];
  free_agents: string[];
  locked: string[];
  max_moves: number;
  settings: LeagueSettings;
}

export interface LineupSlot {
  slot: SlotName;
  player: PlayerHit;
}

export interface LineupOut {
  starters: LineupSlot[];
  points: number;
}

export interface SwapOut {
  add: PlayerHit;
  drop: PlayerHit;
  projected_gain: number;
}

export type ProtectionReason = "locked" | "no_projection" | "season_starter";

export interface Protection {
  player: PlayerHit;
  reason: ProtectionReason;
}

export interface BacktestEvidence {
  realized_gain: number;
  realized_gain_lo: number;
  realized_gain_hi: number;
  hit_rate: number;
  roster_weeks: number;
  weeks: number;
  summary: string;
}

export interface Recommendation {
  id: string;
  season: number;
  week: number;
  projections_version: string;
  adds: PlayerHit[];
  drops: PlayerHit[];
  projected_gain: number;
  before: LineupOut;
  after: LineupOut;
  alternatives: SwapOut[];
  protected: Protection[];
  caveats: string[];
  evidence: BacktestEvidence | null;
}

export const DEFAULT_SETTINGS: LeagueSettings = {
  scoring: "ppr",
  slots: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1 },
  flex_positions: ["RB", "WR", "TE"],
};

/** A failed request, with the backend's message when it sent one. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly retryAfterSeconds: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** FastAPI sends `detail` as a string, or as a list of validation issues. */
export function detailMessage(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return (detail as ValidationIssue[])
      .map((d) => d.msg ?? "invalid value")
      .join("; ");
  }
  return `Request failed (${status})`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(
      0,
      `Can't reach the API at ${API_URL}. Is the backend running?`,
    );
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const retry = Number(response.headers.get("Retry-After"));
    throw new ApiError(
      response.status,
      detailMessage(body, response.status),
      Number.isFinite(retry) && retry > 0 ? retry : null,
    );
  }
  return body as T;
}

export function getProjectionsMeta(): Promise<ProjectionsMeta> {
  return request("/api/projections/meta");
}

export function searchPlayers(
  q: string,
  position?: Position,
  limit = 8,
): Promise<PlayerHit[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (position) params.set("position", position);
  return request(`/api/players/search?${params}`);
}

export function parseScreenshot(
  file: File,
  kind: ParseKind,
): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("kind", kind);
  return request("/api/parse", { method: "POST", body: form });
}

export function createRecommendation(
  body: RecommendationRequest,
): Promise<Recommendation> {
  return request("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Client-side check before uploading; the server re-validates by file contents. */
export function checkImage(file: File): string | null {
  if (!(ACCEPTED_IMAGE_TYPES as readonly string[]).includes(file.type)) {
    return "Upload a PNG, JPEG, WebP, or GIF screenshot.";
  }
  if (file.size > MAX_UPLOAD_BYTES)
    return "Screenshots must be 5 MB or smaller.";
  if (file.size === 0) return "That file is empty.";
  return null;
}
