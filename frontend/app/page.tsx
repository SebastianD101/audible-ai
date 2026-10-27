import { WaiverFlow } from "@/components/waiver-flow";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-8 px-4 py-12">
      <header className="grid gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">Audible AI</h1>
        <p className="text-muted-foreground">
          Upload a screenshot of your fantasy roster and your league&apos;s
          waiver wire. Check the players we read, then get the add/drop that
          most raises next week&apos;s projected lineup.
        </p>
      </header>
      <WaiverFlow />
    </main>
  );
}
