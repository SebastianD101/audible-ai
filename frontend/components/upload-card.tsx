"use client";

import { useMutation } from "@tanstack/react-query";
import { AlertCircle, ImageUp, Loader2 } from "lucide-react";
import { useId, useState, type DragEvent } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ACCEPTED_IMAGE_TYPES,
  ApiError,
  checkImage,
  parseScreenshot,
  type ParseKind,
  type ParseResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface UploadCardProps {
  kind: ParseKind;
  title: string;
  description: string;
  onParsed: (result: ParseResponse) => void;
}

export function UploadCard({
  kind,
  title,
  description,
  onParsed,
}: UploadCardProps) {
  const inputId = useId();
  const [clientError, setClientError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const mutation = useMutation({
    mutationFn: (file: File) => parseScreenshot(file, kind),
    onSuccess: (result) => onParsed(result),
  });

  function upload(file: File | undefined) {
    if (!file) return;
    const problem = checkImage(file);
    setClientError(problem);
    if (!problem) mutation.mutate(file);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    upload(event.dataTransfer.files[0]);
  }

  const error =
    clientError ?? (mutation.isError ? mutation.error.message : null);
  const retryAfter =
    mutation.error instanceof ApiError
      ? mutation.error.retryAfterSeconds
      : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <label
          htmlFor={inputId}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={cn(
            "flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 text-center text-sm text-muted-foreground transition-colors hover:bg-muted/50",
            dragging && "border-ring bg-muted/50",
            mutation.isPending && "pointer-events-none opacity-60",
          )}
        >
          {mutation.isPending ? (
            <Loader2 className="size-6 animate-spin" aria-hidden />
          ) : (
            <ImageUp className="size-6" aria-hidden />
          )}
          <span>
            {mutation.isPending
              ? "Reading screenshot… this can take 20-30 seconds."
              : "Drop a screenshot here, or click to choose one (PNG, JPEG, WebP, GIF; up to 5 MB)."}
          </span>
          <input
            id={inputId}
            type="file"
            className="sr-only"
            accept={ACCEPTED_IMAGE_TYPES.join(",")}
            aria-label={`${title} screenshot`}
            disabled={mutation.isPending}
            onChange={(e) => {
              upload(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </label>

        {error && (
          <Alert variant="destructive">
            <AlertCircle aria-hidden />
            <AlertTitle>Couldn&apos;t read that screenshot</AlertTitle>
            <AlertDescription>
              <p>{error}</p>
              {retryAfter && <p>Try again in about {retryAfter} seconds.</p>}
            </AlertDescription>
          </Alert>
        )}

        {mutation.isSuccess && !clientError && (
          <div className="text-sm text-muted-foreground" role="status">
            <p>
              Read {mutation.data.roster.players.length} player
              {mutation.data.roster.players.length === 1 ? "" : "s"}
              {mutation.data.roster.platform !== "unknown" &&
                ` from ${mutation.data.roster.platform}`}
              . Check the flagged rows below.
            </p>
            {mutation.data.roster.warnings.map((w) => (
              <p key={w} className="text-warning">
                {w}
              </p>
            ))}
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          The screenshot is sent to the vision model to read names and is not
          stored.
        </p>
      </CardContent>
    </Card>
  );
}
