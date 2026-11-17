"use client";

import { useApi, endpoints } from "@/lib/api";

/*
  The honesty strip. It sits under every page header and never goes away:
  which model version, which test window, how rare the event actually is, and
  what the service cannot speak about.

  Every figure is READ FROM THE API. It previously hardcoded "cresta-b-v1.3",
  a 0.2278% base rate and a "synthetic validation" badge — all three are now
  false, and a hardcoded honesty strip is the one thing on the page that must
  never drift from the model it describes.
*/
export default function StatusStrip() {
  const info = useApi("model-info", endpoints.modelInfo);
  const metrics = useApi("metrics", endpoints.modelMetrics);

  const version = info.data?.version ?? "—";
  const window =
    metrics.data == null
      ? "—"
      : `FY${metrics.data.test_window[0]}–${metrics.data.test_window[1]}`;
  const prevalence =
    metrics.data == null
      ? "—"
      : `${(metrics.data.test_prevalence * 100).toFixed(1)}%`;

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 border-b border-ink-700 bg-ink-900/60 px-5 py-2 text-2xs text-ink-400">
      <Item k="model" v={version} />
      <Item k="out-of-time" v={window} />
      <Item k="test base rate" v={prevalence} />
      <Item k="track B (ESG)" v="unavailable" />
      <span className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-amber-dim bg-amber-glow px-2 py-0.5 font-medium text-amber">
        <span className="h-1.5 w-1.5 rounded-full bg-amber" />
        analytical estimates · human review required
      </span>
    </div>
  );
}

function Item({ k, v }: { k: string; v: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="uppercase tracking-wider text-ink-500">{k}</span>
      <span className="tnum text-ink-200">{v}</span>
    </span>
  );
}
