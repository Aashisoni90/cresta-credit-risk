import { ArrowDownRight, ArrowUpRight } from "lucide-react";

export default function StatTile({
  label,
  value,
  unit,
  context,
  delta,
  channel,
}: {
  label: string;
  value: string;
  unit?: string;
  context?: string;
  delta?: { value: string; positive: boolean };
  channel?: "a" | "b" | "amber";
}) {
  const accent =
    channel === "a"
      ? "var(--model-a)"
      : channel === "b"
        ? "var(--model-b)"
        : channel === "amber"
          ? "var(--amber)"
          : undefined;

  return (
    <div className="rounded-card border border-ink-700 bg-ink-850 p-4">
      <div className="flex items-center gap-2">
        {accent && (
          <span
            className="h-2.5 w-0.5 rounded-full"
            style={{ backgroundColor: accent }}
          />
        )}
        <p className="text-2xs uppercase tracking-[0.14em] text-ink-500">
          {label}
        </p>
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className="tnum font-display text-2xl font-semibold text-ink-50">
          {value}
        </span>
        {unit && <span className="text-xs text-ink-400">{unit}</span>}
      </div>
      <div className="mt-1 flex items-center gap-2">
        {delta && (
          <span
            className={`tnum inline-flex items-center gap-0.5 text-2xs font-medium ${
              delta.positive ? "text-pos" : "text-neg"
            }`}
          >
            {delta.positive ? (
              <ArrowUpRight size={11} />
            ) : (
              <ArrowDownRight size={11} />
            )}
            {delta.value}
          </span>
        )}
        {context && <span className="text-2xs text-ink-500">{context}</span>}
      </div>
    </div>
  );
}
