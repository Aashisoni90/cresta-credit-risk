"use client";

import { fmtCount, fmtPct } from "@/lib/format";
import { useApi, endpoints } from "@/lib/api";
import { Skeleton } from "./primitives";

/*
  Out-of-time class balance, drawn literally: the amber band is the positive
  class as a share of the test set. This is why accuracy is a useless headline
  metric here — §8.6 reports it for completeness only.

  All three figures are READ FROM THE API. They were hardcoded at 264 events
  in 115,873 firm-years at 0.2278% — the synthetic testbed's balance, which is
  two orders of magnitude rarer than the real panel's. A hardcoded class
  balance beside a real model is a straightforwardly false claim about the
  data, so there is no default to fall back to.
*/
export default function BaseRateStrip() {
  const { data, error } = useApi("metrics", endpoints.modelMetrics);
  if (error) return null;
  if (!data) return <Skeleton className="h-28" />;

  const rate = data.test_prevalence * 100;
  const total = data.n_test_rows;
  const positives = Math.round(data.test_prevalence * total);

  return (
    <div className="rounded-card border border-ink-700 bg-ink-850 p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-2xs uppercase tracking-[0.16em] text-ink-500">
          out-of-time test set · class balance
        </p>
        <p className="tnum text-xs text-ink-400">
          <span className="text-amber">{fmtCount(positives)}</span> distress
          events in {fmtCount(total)} firm-years
        </p>
      </div>

      <div className="relative mt-3 h-6 overflow-hidden rounded bg-ink-900">
        <div className="absolute inset-0 opacity-40 [background-image:repeating-linear-gradient(90deg,var(--ink-700)_0_1px,transparent_1px_7px)]" />
        <div
          className="absolute inset-y-0 left-0 bg-amber shadow-lamp"
          style={{ width: `${Math.max(0.4, rate)}%` }}
        />
      </div>

      <p className="tnum mt-2 text-2xs text-ink-500">
        prevalence {fmtPct(rate, 1)} — a model that predicts &quot;healthy&quot;
        every time scores {fmtPct(100 - rate, 1)} accuracy and catches nothing
      </p>
    </div>
  );
}
