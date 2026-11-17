import { Scissors } from "lucide-react";

/*
  The signature motif. Circularity is a loop: the label feeds off the same
  period's accounting, which feeds the label. CRESTA cuts that loop and
  stretches it into a line — features observed at FY(t), label resolved at
  FY(t+1). This appears wherever a prediction is made.
*/
export default function Spine({
  year,
  className = "",
  severedCount,
}: {
  year: number;
  className?: string;
  /** Live count from /api/features. Omitted renders no claim rather
   *  than a stale number — it was hardcoded at 6; the guard severs 19. */
  severedCount?: number;
}) {
  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        <Node label="features" sub={`FY ${year}`} tone="plain" />

        <div className="relative flex flex-1 items-center">
          <span className="h-px flex-1 bg-ink-600" />
          <span className="mx-2 grid h-6 w-6 shrink-0 place-items-center rounded-full border border-amber-dim bg-ink-900 text-amber">
            <Scissors size={12} strokeWidth={2.2} />
          </span>
          <span className="h-px flex-1 bg-ink-600" />
          <span
            className="ml-1 border-y-[5px] border-l-[7px] border-y-transparent border-l-amber"
            aria-hidden
          />
        </div>

        <Node label="label" sub={`FY ${year + 1}`} tone="target" />
      </div>
      <p className="mt-1 text-center text-2xs text-ink-500">
        1-year lag
        {severedCount != null && ` · ${severedCount} circular features severed`}
      </p>
    </div>
  );
}

function Node({
  label,
  sub,
  tone,
}: {
  label: string;
  sub: string;
  tone: "plain" | "target";
}) {
  return (
    <div
      className={`shrink-0 rounded border px-2.5 py-1.5 text-center ${
        tone === "target"
          ? "border-amber-dim bg-amber-glow"
          : "border-ink-600 bg-ink-800"
      }`}
    >
      <p
        className={`text-2xs uppercase tracking-wider ${
          tone === "target" ? "text-amber" : "text-ink-400"
        }`}
      >
        {label}
      </p>
      <p className="tnum text-xs text-ink-100">{sub}</p>
    </div>
  );
}
