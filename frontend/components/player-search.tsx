"use client";

import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { useId, useState } from "react";

import { Input } from "@/components/ui/input";
import { searchPlayers, type PlayerHit } from "@/lib/api";
import { useDebounced } from "@/lib/use-debounced";

import { PlayerLabel } from "./player-label";

interface PlayerSearchProps {
  label: string;
  onSelect: (player: PlayerHit) => void;
  autoFocus?: boolean;
}

const MIN_QUERY = 2;

export function PlayerSearch({
  label,
  onSelect,
  autoFocus,
}: PlayerSearchProps) {
  const inputId = useId();
  const [query, setQuery] = useState("");
  const debounced = useDebounced(query.trim(), 250);
  const enabled = debounced.length >= MIN_QUERY;
  const results = useQuery({
    queryKey: ["player-search", debounced],
    queryFn: () => searchPlayers(debounced),
    enabled,
  });

  return (
    <div className="grid gap-2">
      <label htmlFor={inputId} className="sr-only">
        {label}
      </label>
      <div className="relative">
        <Search
          className="absolute top-2.5 left-2.5 size-4 text-muted-foreground"
          aria-hidden
        />
        <Input
          id={inputId}
          value={query}
          placeholder={label}
          autoFocus={autoFocus}
          autoComplete="off"
          className="pl-8"
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      {enabled && results.isPending && (
        <p className="text-sm text-muted-foreground">Searching…</p>
      )}
      {enabled && results.isError && (
        <p className="text-sm text-destructive">{results.error.message}</p>
      )}
      {enabled && results.isSuccess && results.data.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No players match “{debounced}”.
        </p>
      )}
      {enabled && results.isSuccess && results.data.length > 0 && (
        <ul className="grid gap-1" aria-label="Search results">
          {results.data.map((hit) => (
            <li key={hit.player_id}>
              <button
                type="button"
                className="w-full rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
                onClick={() => {
                  onSelect(hit);
                  setQuery("");
                }}
              >
                <PlayerLabel player={hit} showProjection />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
