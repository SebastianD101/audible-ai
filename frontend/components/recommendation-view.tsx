import { ArrowDown, ArrowUp, Info, ShieldCheck } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { LineupOut, ProtectionReason, Recommendation } from "@/lib/api";

import { formatPoints, PlayerLabel } from "./player-label";

const REASONS: Record<ProtectionReason, string> = {
  locked: "you locked this player",
  no_projection: "no projection this week (bye, or not active)",
  season_starter:
    "a starter on season-long value, so a bad week isn't a reason to cut",
};

function names(players: { name: string }[]): string {
  return players.map((p) => p.name).join(" and ");
}

function LineupTable({ title, lineup }: { title: string; lineup: LineupOut }) {
  return (
    <div className="grid gap-2">
      <h3 className="text-sm font-medium">
        {title}: {formatPoints(lineup.points)} projected
      </h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Slot</TableHead>
            <TableHead>Player</TableHead>
            <TableHead className="text-right">Proj</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {lineup.starters.map(({ slot, player }) => (
            <TableRow key={`${slot}-${player.player_id}`}>
              <TableCell className="text-muted-foreground">{slot}</TableCell>
              <TableCell>
                <PlayerLabel player={player} />
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {formatPoints(player.projection)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function RecommendationView({ rec }: { rec: Recommendation }) {
  const moved = rec.adds.length > 0;
  return (
    <Card aria-labelledby="recommendation-title">
      <CardHeader>
        <CardTitle id="recommendation-title" className="text-xl">
          {moved
            ? `Add ${names(rec.adds)}, drop ${names(rec.drops)}`
            : "No move recommended"}
        </CardTitle>
        <CardDescription>
          {moved
            ? `+${rec.projected_gain.toFixed(1)} projected starting-lineup points for week ${rec.week}.`
            : `No listed free agent raises your week ${rec.week} lineup by at least half a point.`}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6">
        {moved && (
          <ul className="grid gap-1 text-sm">
            {rec.adds.map((p) => (
              <li key={p.player_id} className="flex items-center gap-2">
                <ArrowUp className="size-4 text-success" aria-label="Add" />
                <PlayerLabel player={p} showProjection />
              </li>
            ))}
            {rec.drops.map((p) => (
              <li key={p.player_id} className="flex items-center gap-2">
                <ArrowDown
                  className="size-4 text-destructive"
                  aria-label="Drop"
                />
                <PlayerLabel player={p} showProjection />
              </li>
            ))}
          </ul>
        )}

        <Alert>
          <Info aria-hidden />
          <AlertTitle>How much to trust this</AlertTitle>
          <AlertDescription>
            <p>
              {rec.evidence?.summary ??
                "Three-move recommendations were not backtested, so there is no historical estimate for them."}
            </p>
            <p>
              The projected gain is the model&apos;s expectation for one week,
              not a promise.
            </p>
          </AlertDescription>
        </Alert>

        {moved && (
          <div className="grid gap-6 md:grid-cols-2">
            <LineupTable title="Lineup now" lineup={rec.before} />
            <LineupTable title="Lineup after" lineup={rec.after} />
          </div>
        )}

        {rec.alternatives.length > 0 && (
          <section className="grid gap-2">
            <h3 className="text-sm font-medium">Other single moves</h3>
            <ul className="grid gap-1 text-sm">
              {rec.alternatives.map((s) => (
                <li
                  key={`${s.add.player_id}-${s.drop.player_id}`}
                  className="flex flex-wrap items-baseline gap-x-2"
                >
                  <span>
                    Add <strong>{s.add.name}</strong>, drop {s.drop.name}
                  </span>
                  <span className="text-muted-foreground tabular-nums">
                    +{s.projected_gain.toFixed(1)}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {rec.protected.length > 0 && (
          <section className="grid gap-2">
            <h3 className="flex items-center gap-2 text-sm font-medium">
              <ShieldCheck className="size-4" aria-hidden />
              Not considered for dropping
            </h3>
            <ul className="grid gap-1 text-sm text-muted-foreground">
              {rec.protected.map(({ player, reason }) => (
                <li key={player.player_id}>
                  <span className="text-foreground">{player.name}</span>:{" "}
                  {REASONS[reason]}
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="grid gap-1 text-xs text-muted-foreground">
          {rec.caveats.map((c) => (
            <p key={c}>{c}</p>
          ))}
          <p>
            Projections {rec.season} week {rec.week}, version{" "}
            {rec.projections_version}.
          </p>
        </section>
      </CardContent>
    </Card>
  );
}
