import type { PlayerHit } from "@/lib/api";

export function formatPoints(points: number | null): string {
  return points === null ? "—" : points.toFixed(1);
}

interface PlayerLabelProps {
  player: PlayerHit;
  showProjection?: boolean;
}

/** "Josh Allen  QB · BUF  24.1 proj" */
export function PlayerLabel({ player, showProjection }: PlayerLabelProps) {
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-2">
      <span className="font-medium">{player.name}</span>
      <span className="text-xs text-muted-foreground">
        {player.position}
        {player.team && ` · ${player.team}`}
      </span>
      {showProjection && (
        <span className="text-xs text-muted-foreground tabular-nums">
          {player.projection === null
            ? "no projection this week"
            : `${formatPoints(player.projection)} proj`}
        </span>
      )}
    </span>
  );
}
