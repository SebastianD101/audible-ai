"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertCircle, Loader2 } from "lucide-react";
import { useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  createRecommendation,
  DEFAULT_SETTINGS,
  getProjectionsMeta,
  type LeagueSettings,
  type RecommendationRequest,
} from "@/lib/api";
import {
  resolvedIds,
  rowsFromParse,
  unresolvedCount,
  type ReviewRow,
} from "@/lib/rows";

import { LeagueSettingsForm } from "./league-settings-form";
import { RecommendationView } from "./recommendation-view";
import { ReviewTable } from "./review-table";
import { UploadCard } from "./upload-card";

function ProjectionsBanner() {
  const meta = useQuery({
    queryKey: ["projections-meta"],
    queryFn: getProjectionsMeta,
  });
  if (meta.isPending) {
    return (
      <p className="text-sm text-muted-foreground">Loading projections…</p>
    );
  }
  if (meta.isError) {
    return (
      <Alert variant="destructive">
        <AlertCircle aria-hidden />
        <AlertTitle>Projections unavailable</AlertTitle>
        <AlertDescription>
          <p>{meta.error.message}</p>
        </AlertDescription>
      </Alert>
    );
  }
  return (
    <p className="text-sm text-muted-foreground">
      Projections for {meta.data.season} week {meta.data.week} ({meta.data.rows}{" "}
      players, version {meta.data.version}).
    </p>
  );
}

/** Why the recommend button is disabled, or null when it's ready. */
export function blocker(
  roster: ReviewRow[],
  freeAgents: ReviewRow[],
): string | null {
  if (resolvedIds(roster).length === 0) return "Add your roster first.";
  const open = unresolvedCount(roster) + unresolvedCount(freeAgents);
  if (open > 0) {
    return `Resolve ${open} flagged row${open === 1 ? "" : "s"} first.`;
  }
  if (resolvedIds(freeAgents).length === 0) {
    return "Add at least one free agent.";
  }
  return null;
}

export function WaiverFlow() {
  const [roster, setRoster] = useState<ReviewRow[]>([]);
  const [freeAgents, setFreeAgents] = useState<ReviewRow[]>([]);
  const [settings, setSettings] = useState<LeagueSettings>(DEFAULT_SETTINGS);
  const [maxMoves, setMaxMoves] = useState(1);
  const recommendation = useMutation({ mutationFn: createRecommendation });

  // Any edit makes a shown recommendation stale, so clear it.
  function edit<T>(set: (value: T) => void) {
    return (value: T) => {
      recommendation.reset();
      set(value);
    };
  }

  const blocked = blocker(roster, freeAgents);

  function submit() {
    const request: RecommendationRequest = {
      roster: resolvedIds(roster),
      free_agents: resolvedIds(freeAgents),
      locked: roster
        .filter((r) => r.locked && r.player)
        .map((r) => r.player!.player_id),
      max_moves: maxMoves,
      settings,
    };
    recommendation.mutate(request);
  }

  return (
    <div className="grid gap-8">
      <ProjectionsBanner />

      <section className="grid gap-4" aria-labelledby="roster-heading">
        <h2 id="roster-heading" className="text-lg font-semibold">
          1. Your roster
        </h2>
        <UploadCard
          kind="roster"
          title="Roster screenshot"
          description="A screenshot of your team page from ESPN, Yahoo, Sleeper, or similar."
          onParsed={(r) => edit(setRoster)(rowsFromParse(r))}
        />
        <ReviewTable
          rows={roster}
          onChange={edit(setRoster)}
          allowLock
          emptyText="Upload a screenshot or add players by name."
          addLabel="Add a roster player by name"
        />
      </section>

      <section className="grid gap-4" aria-labelledby="fa-heading">
        <h2 id="fa-heading" className="text-lg font-semibold">
          2. Available free agents
        </h2>
        <UploadCard
          kind="free_agents"
          title="Waiver-wire screenshot"
          description="A screenshot of your league's available players. Only players you list are considered."
          onParsed={(r) =>
            edit(setFreeAgents)([...freeAgents, ...rowsFromParse(r)])
          }
        />
        <ReviewTable
          rows={freeAgents}
          onChange={edit(setFreeAgents)}
          emptyText="Upload a waiver screenshot or add free agents by name."
          addLabel="Add a free agent by name"
        />
      </section>

      <section className="grid gap-4" aria-labelledby="settings-heading">
        <h2 id="settings-heading" className="text-lg font-semibold">
          3. League settings
        </h2>
        <Card>
          <CardHeader>
            <CardTitle>Starting lineup</CardTitle>
            <CardDescription>
              Used to decide who starts before and after a move.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LeagueSettingsForm
              settings={settings}
              maxMoves={maxMoves}
              onSettingsChange={edit(setSettings)}
              onMaxMovesChange={edit(setMaxMoves)}
            />
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4" aria-labelledby="rec-heading">
        <h2 id="rec-heading" className="text-lg font-semibold">
          4. Recommendation
        </h2>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="lg"
            disabled={blocked !== null || recommendation.isPending}
            onClick={submit}
          >
            {recommendation.isPending && (
              <Loader2 className="animate-spin" aria-hidden />
            )}
            Get recommendation
          </Button>
          {blocked && (
            <span className="text-sm text-muted-foreground">{blocked}</span>
          )}
        </div>
        {recommendation.isError && (
          <Alert variant="destructive">
            <AlertCircle aria-hidden />
            <AlertTitle>Couldn&apos;t get a recommendation</AlertTitle>
            <AlertDescription>
              <p>{recommendation.error.message}</p>
            </AlertDescription>
          </Alert>
        )}
        {recommendation.isSuccess && (
          <RecommendationView rec={recommendation.data} />
        )}
      </section>
    </div>
  );
}
