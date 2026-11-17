import { AlertTriangle, Loader2 } from "lucide-react";
import { RISK_META, type RiskBand } from "@/lib/format";

/* ---------- Panel ---------- */
export function Panel({
  children,
  className = "",
  pad = true,
}: {
  children: React.ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section
      className={`rounded-card border border-ink-700 bg-ink-850 shadow-panel ${
        pad ? "p-5" : ""
      } ${className}`}
    >
      {children}
    </section>
  );
}

/* ---------- Section header ---------- */
export function SectionHeader({
  eyebrow,
  title,
  hint,
  right,
}: {
  eyebrow?: string;
  title: string;
  hint?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-3 flex items-end justify-between gap-3">
      <div>
        {eyebrow && (
          <p className="text-2xs uppercase tracking-[0.18em] text-ink-500">
            {eyebrow}
          </p>
        )}
        <h2 className="font-display text-base font-semibold text-ink-50">
          {title}
        </h2>
        {hint && <p className="mt-0.5 text-xs text-ink-400">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

/* ---------- Chip ---------- */
export function Chip({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "a" | "b" | "amber" | "pos" | "neg";
}) {
  const tones: Record<string, string> = {
    neutral: "border-ink-600 bg-ink-800 text-ink-300",
    a: "border-model-a/40 bg-model-a/10 text-model-a",
    b: "border-model-b/40 bg-model-b/10 text-model-b",
    amber: "border-amber-dim bg-amber-glow text-amber",
    pos: "border-pos/40 bg-pos/10 text-pos",
    neg: "border-neg/40 bg-neg/10 text-neg",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-2xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

/* ---------- Risk badge (label always present, never colour alone) ---------- */
export function RiskBadge({
  band,
  size = "sm",
}: {
  band: RiskBand;
  size?: "sm" | "md";
}) {
  const m = RISK_META[band];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border font-semibold uppercase tracking-wider ${
        size === "md" ? "px-2.5 py-1 text-xs" : "px-1.5 py-0.5 text-2xs"
      }`}
      style={{
        color: m.color,
        borderColor: m.color,
        backgroundColor: `${m.color}1a`,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: m.color }}
      />
      {m.label}
    </span>
  );
}

/* ---------- Loading / Empty / Error ---------- */
export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-ink-400">
      <Loader2 size={16} className="animate-spin" />
      {label}…
    </div>
  );
}

export function Skeleton({
  className = "",
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={`animate-pulse rounded bg-ink-750 ${className}`}
      style={style}
      aria-hidden
    />
  );
}

/*
  §12.6: "Skeleton loaders; never a bare spinner on a data-dense view."

  A spinner tells the reader only that something is happening. A skeleton
  shaped like the content keeps the layout from jumping when data lands,
  which on a dense analytical screen is the difference between a usable wait
  and a page that reflows under the cursor.

  Each is marked aria-busy and labelled, so a screen reader hears "loading
  <what>" rather than silence followed by a sudden tableful of numbers.
*/
function SkeletonFrame({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div role="status" aria-busy="true" aria-label={`Loading ${label}`}>
      {children}
      <span className="sr-only">Loading {label}…</span>
    </div>
  );
}

/** Rows of label + value, for lists and metric stacks. */
export function SkeletonRows({
  rows = 6,
  label = "data",
}: {
  rows?: number;
  label?: string;
}) {
  return (
    <SkeletonFrame label={label}>
      <div className="space-y-2 py-1">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex items-center gap-3">
            <Skeleton className="h-3 flex-1" />
            <Skeleton className="h-3 w-16" />
          </div>
        ))}
      </div>
    </SkeletonFrame>
  );
}

/** A header strip plus body rows, for tables. */
export function SkeletonTable({
  rows = 8,
  cols = 5,
  label = "table",
}: {
  rows?: number;
  cols?: number;
  label?: string;
}) {
  return (
    <SkeletonFrame label={label}>
      <div className="space-y-2 py-1">
        <div className="flex gap-3">
          {Array.from({ length: cols }).map((_, i) => (
            <Skeleton key={i} className="h-2.5 flex-1" />
          ))}
        </div>
        {Array.from({ length: rows }).map((_, r) => (
          <div key={r} className="flex gap-3">
            {Array.from({ length: cols }).map((_, c) => (
              <Skeleton key={c} className="h-3 flex-1" />
            ))}
          </div>
        ))}
      </div>
    </SkeletonFrame>
  );
}

/** A plot-sized block, for charts. */
export function SkeletonChart({
  height = 240,
  label = "chart",
}: {
  height?: number;
  label?: string;
}) {
  return (
    <SkeletonFrame label={label}>
      <Skeleton className="w-full" style={{ height }} />
    </SkeletonFrame>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-card border border-neg/30 bg-neg/5 p-5">
      <div className="flex items-center gap-2 text-sm font-medium text-neg">
        <AlertTriangle size={16} />
        Something broke
      </div>
      <p className="text-sm text-ink-300">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="rounded border border-ink-600 px-3 py-1.5 text-xs text-ink-200 hover:bg-ink-800"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="rounded-card border border-dashed border-ink-600 bg-ink-900/40 p-8 text-center">
      <p className="font-display text-sm font-semibold text-ink-200">{title}</p>
      <p className="mx-auto mt-1 max-w-sm text-sm text-ink-400">{body}</p>
    </div>
  );
}
