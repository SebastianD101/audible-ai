"use client";

import { Check, Lock, Pencil, Trash2, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { PlayerHit } from "@/lib/api";
import {
  duplicateIds,
  isResolved,
  manualRow,
  unresolvedCount,
  type ReviewRow,
} from "@/lib/rows";
import { cn } from "@/lib/utils";

import { formatPoints, PlayerLabel } from "./player-label";
import { PlayerSearch } from "./player-search";

interface ReviewTableProps {
  rows: ReviewRow[];
  onChange: (rows: ReviewRow[]) => void;
  /** Roster tables let the user lock players the optimizer must keep. */
  allowLock?: boolean;
  emptyText: string;
  addLabel: string;
}

function StatusBadge({
  row,
  duplicate,
}: {
  row: ReviewRow;
  duplicate: boolean;
}) {
  if (duplicate) return <Badge variant="destructive">Duplicate</Badge>;
  if (!row.player) return <Badge variant="warning">No match</Badge>;
  if (!isResolved(row)) return <Badge variant="warning">Check</Badge>;
  if (row.extracted === null) return <Badge variant="secondary">Added</Badge>;
  if (row.needsReview) return <Badge variant="success">Confirmed</Badge>;
  return <Badge variant="success">Matched</Badge>;
}

function ReadAs({ row }: { row: ReviewRow }) {
  if (!row.extracted) {
    return <span className="text-muted-foreground">Added by you</span>;
  }
  const { name, legibility, injury_tag, note } = row.extracted;
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span>{name}</span>
      {legibility !== "clear" && (
        <Badge variant="warning" title={note ?? undefined}>
          {legibility === "partial" ? "partly legible" : "name guessed"}
        </Badge>
      )}
      {injury_tag && <Badge variant="outline">{injury_tag}</Badge>}
    </span>
  );
}

export function ReviewTable({
  rows,
  onChange,
  allowLock,
  emptyText,
  addLabel,
}: ReviewTableProps) {
  const [editing, setEditing] = useState<string | null>(null);
  const unresolved = unresolvedCount(rows);
  const duplicates = duplicateIds(rows);

  function update(key: string, patch: Partial<ReviewRow>) {
    onChange(rows.map((r) => (r.key === key ? { ...r, ...patch } : r)));
  }

  function choose(key: string, player: PlayerHit) {
    update(key, { player, confirmed: true });
    setEditing(null);
  }

  return (
    <div className="grid gap-3">
      {unresolved > 0 && (
        <Alert variant="warning">
          <TriangleAlert aria-hidden />
          <AlertDescription>
            {unresolved} row{unresolved === 1 ? "" : "s"} need
            {unresolved === 1 ? "s" : ""} your review: confirm the match, pick
            the right player, or remove the row.
          </AlertDescription>
        </Alert>
      )}

      {rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">{emptyText}</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Screenshot</TableHead>
              <TableHead>Player</TableHead>
              <TableHead className="text-right">Proj</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => {
              const resolved = isResolved(row);
              const label = row.player?.name ?? row.extracted?.name ?? "row";
              return (
                <TableRow
                  key={row.key}
                  className={cn(!resolved && "bg-warning/5")}
                  data-testid="review-row"
                >
                  <TableCell>
                    <ReadAs row={row} />
                  </TableCell>
                  <TableCell className="whitespace-normal">
                    {row.player ? (
                      <PlayerLabel player={row.player} />
                    ) : (
                      <span className="text-muted-foreground">
                        Pick a player
                      </span>
                    )}
                    {editing === row.key && (
                      <div className="mt-2 grid max-w-sm gap-2">
                        {row.candidates.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {row.candidates.map((c) => (
                              <Button
                                key={c.player_id}
                                size="sm"
                                variant="outline"
                                onClick={() => choose(row.key, c)}
                              >
                                {c.name} · {c.team ?? c.position}
                              </Button>
                            ))}
                          </div>
                        )}
                        <PlayerSearch
                          label="Search for the right player"
                          autoFocus
                          onSelect={(p) => choose(row.key, p)}
                        />
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatPoints(row.player?.projection ?? null)}
                  </TableCell>
                  <TableCell>
                    <StatusBadge
                      row={row}
                      duplicate={
                        row.player !== null &&
                        duplicates.has(row.player.player_id)
                      }
                    />
                  </TableCell>
                  <TableCell>
                    <div className="flex justify-end gap-1">
                      {row.player && !resolved && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => update(row.key, { confirmed: true })}
                        >
                          <Check aria-hidden />
                          Confirm
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        aria-label={`Change ${label}`}
                        aria-expanded={editing === row.key}
                        onClick={() =>
                          setEditing(editing === row.key ? null : row.key)
                        }
                      >
                        <Pencil aria-hidden />
                      </Button>
                      {allowLock && (
                        <Button
                          size="sm"
                          variant={row.locked ? "secondary" : "ghost"}
                          aria-label={`Keep ${label}`}
                          aria-pressed={row.locked}
                          title="Never drop this player"
                          disabled={!row.player}
                          onClick={() =>
                            update(row.key, { locked: !row.locked })
                          }
                        >
                          <Lock aria-hidden />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        aria-label={`Remove ${label}`}
                        onClick={() =>
                          onChange(rows.filter((r) => r.key !== row.key))
                        }
                      >
                        <Trash2 aria-hidden />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}

      <PlayerSearch
        label={addLabel}
        onSelect={(p) => onChange([...rows, manualRow(p)])}
      />
    </div>
  );
}
