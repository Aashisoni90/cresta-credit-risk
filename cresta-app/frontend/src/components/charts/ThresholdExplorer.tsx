"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useApi, endpoints } from "@/lib/api";
import { AXIS, GRID, TOOLTIP_STYLE, COLORS } from "./chartTheme";
import {
  ErrorState,
  Panel,
  SkeletonChart,
} from "@/components/ui/primitives";
import ConfusionMatrix from "./ConfusionMatrix";
import { fmtCount, fmtPct } from "@/lib/format";
import type { ThresholdPoint } from "@/types/api";

/*
  Move the decision boundary and watch recall and accuracy trade off.
  This is the interactive form of the accuracy paradox: slide right and
  accuracy climbs toward 98% while recall falls off a cliff.
*/
export default function ThresholdExplorer() {
  const { data, error, isLoading, mutate } = useApi("threshold-99", () =>
    endpoints.threshold(99),
  );
  // Start at the model's own cost-optimal operating point, not 0.5. A slider
  // that opens at 0.5 implies the model's threshold IS 0.5, which it is not
  // (§8.4 — it is fitted on validation at the stated cost ratio).
  const [t, setT] = useState<number | null>(null);
  const operating = data?.operating_threshold ?? 0.5;
  const tv = t ?? operating;

  const rows = data?.sweep ?? [];
  const current: ThresholdPoint | undefined = useMemo(() => {
    if (!rows.length) return undefined;
    return rows.reduce((best, r) =>
      Math.abs(r.threshold - tv) < Math.abs(best.threshold - tv) ? r : best,
    );
  }, [rows, tv]);

  if (isLoading) return <SkeletonChart height={300} label="threshold sweep" />;
  if (error || !data)
    return <ErrorState message={error?.message ?? "No data"} onRetry={mutate} />;

  const chartData = rows.map((r) => ({
    threshold: r.threshold,
    recall: r.recall,
    accuracy: r.accuracy,
  }));

  // Reference marks come from the data, not from round numbers: the deployed
  // operating point and the F1-maximising point are the two that mean
  // something. A hardcoded "0.5 = default" marker would assert a default the
  // model does not have.
  const refs = [
    { v: data.operating_threshold, note: `operating (${data.cost_ratio_fn_fp}:1 cost)` },
    { v: data.max_f1_point.threshold, note: "max F1" },
  ];

  return (
    <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
      <Panel>
        <div className="mb-3 flex items-center justify-between">
          <span className="text-2xs uppercase tracking-[0.16em] text-ink-500">
            decision threshold
          </span>
          <span className="tnum font-display text-lg font-semibold text-amber">
            {tv.toFixed(2)}
          </span>
        </div>

        <input
          type="range"
          min={0.01}
          max={0.99}
          step={0.01}
          value={tv}
          onChange={(e) => setT(parseFloat(e.target.value))}
          aria-label="Decision threshold"
          className="w-full accent-amber"
        />
        <div className="mt-1 flex justify-between text-2xs text-ink-500">
          {refs.map((r) => (
            <button
              key={r.v}
              onClick={() => setT(r.v)}
              className="tnum rounded px-1 py-0.5 hover:bg-ink-800 hover:text-ink-200"
            >
              {r.v.toFixed(2)} · {r.note}
            </button>
          ))}
        </div>

        <div className="mt-4 h-[220px] w-full">
          <ResponsiveContainer>
            <LineChart
              data={chartData}
              margin={{ top: 6, right: 10, bottom: 2, left: -12 }}
            >
              <CartesianGrid {...GRID} />
              <XAxis
                dataKey="threshold"
                type="number"
                domain={[0, 1]}
                {...AXIS}
              />
              <YAxis domain={[0, 100]} {...AXIS} unit="%" />
              <Tooltip
                {...TOOLTIP_STYLE}
                formatter={(v: number) => `${v.toFixed(1)}%`}
              />
              {refs.map((r) => (
                <ReferenceLine
                  key={r.v}
                  x={r.v}
                  stroke="var(--ink-600)"
                  strokeDasharray="2 3"
                />
              ))}
              <ReferenceLine x={tv} stroke="var(--amber)" strokeWidth={1.5} />
              <Line
                dataKey="recall"
                name="recall"
                stroke={COLORS.b}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                dataKey="accuracy"
                name="accuracy"
                stroke={COLORS.muted}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Legend />
      </Panel>

      <div className="grid content-start gap-3">
        <div className="grid grid-cols-2 gap-2">
          <Mini label="recall" value={fmtPct(current?.recall ?? 0, 1)} tone="b" />
          <Mini
            label="precision"
            value={fmtPct(current?.precision ?? 0, 3)}
            tone="amber"
          />
          <Mini
            label="accuracy"
            value={fmtPct(current?.accuracy ?? 0, 1)}
            tone="muted"
          />
          <Mini
            label="events caught"
            value={`${current?.tp ?? 0} / ${data.n_positives}`}
            tone="b"
          />
        </div>
        {current && (
          <ConfusionMatrix
            tn={current.tn}
            fp={current.fp}
            fn={current.fn}
            tp={current.tp}
            threshold={current.threshold}
          />
        )}
        <p className="text-2xs leading-relaxed text-ink-500">
          {fmtCount(data.n_test_rows)} out-of-time test rows,{" "}
          {fmtCount(data.n_positives)} distress events. Slide toward 1.0 and
          accuracy climbs while the model quietly stops finding anything —
          which is why §8.6 refuses accuracy as a headline metric.
        </p>
      </div>
    </div>
  );
}

function Mini({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "b" | "amber" | "muted";
}) {
  const c =
    tone === "b"
      ? "var(--model-b)"
      : tone === "amber"
        ? "var(--amber)"
        : "var(--ink-300)";
  return (
    <div className="rounded border border-ink-700 bg-ink-900 p-2.5">
      <p className="text-2xs uppercase tracking-wider text-ink-500">{label}</p>
      <p
        className="tnum font-display text-base font-semibold"
        style={{ color: c }}
      >
        {value}
      </p>
    </div>
  );
}

function Legend() {
  const items = [
    { c: COLORS.b, label: "recall — share of real distress events caught" },
    { c: COLORS.muted, label: "accuracy — the misleading headline metric" },
  ];
  return (
    <div className="mt-2 flex flex-wrap gap-3 text-2xs text-ink-400">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span
            className="h-0.5 w-4"
            style={{ backgroundColor: i.c }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}
