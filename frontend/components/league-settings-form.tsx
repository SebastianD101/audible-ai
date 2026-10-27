"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { LeagueSettings, SlotName } from "@/lib/api";

const SLOTS: { slot: SlotName; label: string }[] = [
  { slot: "QB", label: "QB" },
  { slot: "RB", label: "RB" },
  { slot: "WR", label: "WR" },
  { slot: "TE", label: "TE" },
  { slot: "FLEX", label: "FLEX (RB/WR/TE)" },
  { slot: "SUPERFLEX", label: "Superflex (QB/RB/WR/TE)" },
];
const MAX_PER_SLOT = 4;

interface LeagueSettingsFormProps {
  settings: LeagueSettings;
  maxMoves: number;
  onSettingsChange: (settings: LeagueSettings) => void;
  onMaxMovesChange: (maxMoves: number) => void;
}

export function LeagueSettingsForm({
  settings,
  maxMoves,
  onSettingsChange,
  onMaxMovesChange,
}: LeagueSettingsFormProps) {
  function setSlot(slot: SlotName, raw: string) {
    const count = Math.max(0, Math.min(MAX_PER_SLOT, Number(raw) || 0));
    onSettingsChange({
      ...settings,
      slots: { ...settings.slots, [slot]: count },
    });
  }

  return (
    <fieldset className="grid gap-4">
      <legend className="sr-only">League settings</legend>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {SLOTS.map(({ slot, label }) => (
          <div key={slot} className="grid gap-1.5">
            <Label htmlFor={`slot-${slot}`}>{label} starters</Label>
            <Input
              id={`slot-${slot}`}
              type="number"
              inputMode="numeric"
              min={0}
              max={MAX_PER_SLOT}
              value={settings.slots[slot] ?? 0}
              onChange={(e) => setSlot(slot, e.target.value)}
            />
          </div>
        ))}
      </div>
      <div className="grid max-w-xs gap-1.5">
        <Label htmlFor="max-moves">Moves to consider</Label>
        <select
          id="max-moves"
          className="h-9 rounded-md border border-input bg-transparent px-3 text-sm shadow-xs"
          value={maxMoves}
          onChange={(e) => onMaxMovesChange(Number(e.target.value))}
        >
          <option value={1}>1 add/drop</option>
          <option value={2}>Up to 2 add/drops</option>
          <option value={3}>Up to 3 add/drops (not backtested)</option>
        </select>
      </div>
      <p className="text-xs text-muted-foreground">
        Scoring is full PPR. Kicker and defense slots are left out because they
        aren&apos;t projected.
      </p>
    </fieldset>
  );
}
