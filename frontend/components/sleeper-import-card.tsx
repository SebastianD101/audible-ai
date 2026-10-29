"use client";

import { useMutation } from "@tanstack/react-query";
import { AlertCircle, Loader2 } from "lucide-react";
import { useState, useSyncExternalStore, type FormEvent } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  ApiError,
  getSleeperLeagues,
  importSleeperLeague,
  type SleeperImportResponse,
} from "@/lib/api";

const USERNAME_KEY = "audible.sleeperUsername";

/** localStorage can be missing or throw (private mode, blocked storage); it's only a convenience. */
function readUsername(): string {
  try {
    return window.localStorage.getItem(USERNAME_KEY) ?? "";
  } catch {
    return "";
  }
}

// The saved username never changes behind our back, so there is nothing to subscribe to.
const noSubscription = () => () => {};

function saveUsername(username: string) {
  try {
    window.localStorage.setItem(USERNAME_KEY, username);
  } catch {
    // Not remembering the username is fine.
  }
}

function count(items: unknown[], noun: string): string {
  return `${items.length} ${noun}${items.length === 1 ? "" : "s"}`;
}

function ErrorAlert({ title, error }: { title: string; error: Error }) {
  const retryAfter = error instanceof ApiError ? error.retryAfterSeconds : null;
  return (
    <Alert variant="destructive">
      <AlertCircle aria-hidden />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        <p>{error.message}</p>
        {retryAfter && <p>Try again in about {retryAfter} seconds.</p>}
      </AlertDescription>
    </Alert>
  );
}

interface SleeperImportCardProps {
  onImported: (result: SleeperImportResponse) => void;
}

export function SleeperImportCard({ onImported }: SleeperImportCardProps) {
  // Server render and hydration see "", then the client fills in the saved name.
  const saved = useSyncExternalStore(noSubscription, readUsername, () => "");
  const [draft, setUsername] = useState<string | null>(null);
  const username = draft ?? saved;
  const [leagueId, setLeagueId] = useState("");
  const leagues = useMutation({
    mutationFn: getSleeperLeagues,
    onSuccess: (data) => {
      saveUsername(username.trim());
      setLeagueId(data.leagues[0]?.league_id ?? "");
    },
  });
  const imported = useMutation({
    mutationFn: ({ league, user }: { league: string; user: string }) =>
      importSleeperLeague(league, user),
    onSuccess: (result) => onImported(result),
  });

  function findLeagues(event: FormEvent) {
    event.preventDefault();
    const name = username.trim();
    if (!name) return;
    imported.reset();
    leagues.mutate(name);
  }

  const found = leagues.data;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Import from Sleeper</CardTitle>
        <CardDescription>
          Exact players, your league&apos;s real free agents, and its lineup
          settings, from Sleeper&apos;s public API. No login needed.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <form className="flex flex-wrap items-end gap-3" onSubmit={findLeagues}>
          <div className="grid min-w-48 flex-1 gap-1.5">
            <Label htmlFor="sleeper-username">Sleeper username</Label>
            <Input
              id="sleeper-username"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <Button
            type="submit"
            variant="outline"
            disabled={!username.trim() || leagues.isPending}
          >
            {leagues.isPending && (
              <Loader2 className="animate-spin" aria-hidden />
            )}
            Find leagues
          </Button>
        </form>

        {leagues.isError && (
          <ErrorAlert title="Couldn't find leagues" error={leagues.error} />
        )}

        {found && found.leagues.length === 0 && (
          <p className="text-sm text-muted-foreground" role="status">
            {found.display_name} has no {found.season} leagues on Sleeper.
          </p>
        )}

        {found && found.leagues.length > 0 && (
          <div className="flex flex-wrap items-end gap-3">
            <div className="grid min-w-48 flex-1 gap-1.5">
              <Label htmlFor="sleeper-league">
                {found.display_name}&apos;s {found.season} leagues
              </Label>
              <select
                id="sleeper-league"
                className="h-9 rounded-md border border-input bg-transparent px-3 text-sm shadow-xs"
                value={leagueId}
                onChange={(e) => {
                  imported.reset();
                  setLeagueId(e.target.value);
                }}
              >
                {found.leagues.map((l) => (
                  <option key={l.league_id} value={l.league_id}>
                    {l.name} ({l.total_rosters} teams, {l.scoring})
                  </option>
                ))}
              </select>
            </div>
            <Button
              disabled={!leagueId || imported.isPending}
              onClick={() =>
                imported.mutate({ league: leagueId, user: found.user_id })
              }
            >
              {imported.isPending && (
                <Loader2 className="animate-spin" aria-hidden />
              )}
              Import league
            </Button>
          </div>
        )}

        {imported.isError && (
          <ErrorAlert
            title="Couldn't import that league"
            error={imported.error}
          />
        )}

        {imported.isSuccess && (
          <div className="text-sm text-muted-foreground" role="status">
            <p>
              Imported {count(imported.data.roster.roster.players, "player")}{" "}
              from {imported.data.league_name}, the best{" "}
              {count(imported.data.free_agents.roster.players, "free agent")},
              and its lineup settings.
            </p>
            {[
              ...imported.data.warnings,
              ...imported.data.roster.roster.warnings,
              ...imported.data.free_agents.roster.warnings,
            ].map((w) => (
              <p key={w} className="text-warning">
                {w}
              </p>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
