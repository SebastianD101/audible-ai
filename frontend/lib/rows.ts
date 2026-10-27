/**
 * Review rows: the user-editable version of a parsed screenshot.
 *
 * A row the parser flagged (unclear text, ambiguous or missing match, duplicate) must be
 * confirmed, changed, or removed before recommendations run. A row the user picked by hand
 * is confirmed by definition.
 */

import type {
  Candidate,
  ExtractedPlayer,
  ParseResponse,
  PlayerHit,
} from "./api";

export interface ReviewRow {
  key: string;
  extracted: ExtractedPlayer | null;
  player: PlayerHit | null;
  candidates: PlayerHit[];
  needsReview: boolean;
  confirmed: boolean;
  locked: boolean;
}

function candidateHit(
  c: Candidate,
  players: Record<string, PlayerHit>,
): PlayerHit {
  return (
    players[c.player_id] ?? {
      player_id: c.player_id,
      name: c.name,
      position: c.position,
      team: c.team,
      projection: null,
      hold_value: null,
    }
  );
}

export function rowsFromParse(parse: ParseResponse): ReviewRow[] {
  return parse.roster.players.map((p, i) => {
    const id = p.match.player_id;
    return {
      key: `${parse.id}:${i}`,
      extracted: p.extracted,
      player: id ? (parse.players[id] ?? null) : null,
      candidates: p.match.candidates
        .filter((c) => c.player_id !== id)
        .map((c) => candidateHit(c, parse.players)),
      needsReview: p.needs_review || id === null,
      confirmed: false,
      locked: false,
    };
  });
}

export function manualRow(player: PlayerHit): ReviewRow {
  return {
    key: `manual:${player.player_id}:${Date.now()}`,
    extracted: null,
    player,
    candidates: [],
    needsReview: false,
    confirmed: true,
    locked: false,
  };
}

export function isResolved(row: ReviewRow): boolean {
  return row.player !== null && (!row.needsReview || row.confirmed);
}

export function unresolvedCount(rows: ReviewRow[]): number {
  return rows.filter((r) => !isResolved(r)).length;
}

/** IDs of resolved rows, without duplicates, in row order. */
export function resolvedIds(rows: ReviewRow[]): string[] {
  return [...new Set(rows.filter(isResolved).map((r) => r.player!.player_id))];
}

/** IDs that appear on more than one resolved row. */
export function duplicateIds(rows: ReviewRow[]): Set<string> {
  const seen = new Set<string>();
  const dupes = new Set<string>();
  for (const r of rows) {
    if (!r.player) continue;
    if (seen.has(r.player.player_id)) dupes.add(r.player.player_id);
    seen.add(r.player.player_id);
  }
  return dupes;
}
