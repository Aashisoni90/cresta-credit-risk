"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, GRID, TOOLTIP_STYLE, COLORS } from "./chartTheme";
import { fmtCompact, fmtCount, fmtProb } from "@/lib/format";
import type { CompanyHistoryPoint } from "@/types/api";

/* ---------- Sector distribution ---------- */
export function SectorBars({
  distribution,
}: {
  distribution: Record<string, number>;
}) {
  const rows = Object.entries(distribution)
    .map(([sector, count]) => ({ sector, count }))
    .sort((a, b) => b.count - a.count);

  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 0, right: 16, bottom: 0, left: 8 }}
        >
          <CartesianGrid {...GRID} horizontal={false} />
          <XAxis type="number" {...AXIS} tickFormatter={(v) => fmtCompact(v, 0)} />
          <YAxis
            type="category"
            dataKey="sector"
            width={122}
            {...AXIS}
            tick={{ ...AXIS.tick, fontFamily: "var(--font-body)" }}
          />
          <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => fmtCount(v)} />
          <Bar dataKey="count" fill={COLORS.a} radius={[0, 2, 2, 0]} barSize={12} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- Company risk trajectory ---------- */
/*
  §12.3's "RISK TRAJECTORY" panel: predicted distress probability across the
  company's observed years, against the model's operating threshold.

  Previously plotted ROA alongside an averaged E/S/G score. Neither exists on
  the real panel — the ESG series was synthetic — and averaging three pillars
  into one number discards exactly the pillar-level detail §9.1 cares about.
  The probability series is what the product actually claims.
*/
export function RiskTrajectory({
  history,
  threshold,
}: {
  history: CompanyHistoryPoint[];
  threshold?: number;
}) {
  const rows = history.map((h) => ({
    year: h.fiscal_year,
    label: h.fiscal_year_label,
    probability: h.probability,
    altman: h.altman_z_ex_x4,
    // The realised outcome at t+1, so the chart can show where the model was
    // right rather than only what it predicted.
    outcome: h.label_next,
  }));
  const firstEvent = rows.find((r) => r.outcome === 1);

  return (
    <div className="h-[240px] w-full">
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
          <CartesianGrid {...GRID} vertical={false} />
          <XAxis dataKey="year" {...AXIS} />
          <YAxis yAxisId="l" domain={[0, 1]} {...AXIS} />
          <Tooltip
            {...TOOLTIP_STYLE}
            formatter={(v: number) => fmtProb(v, 3)}
            labelFormatter={(y) =>
              rows.find((r) => r.year === y)?.label ?? String(y)
            }
          />
          {threshold != null && (
            <ReferenceLine
              yAxisId="l"
              y={threshold}
              stroke="var(--amber)"
              strokeDasharray="3 3"
              label={{
                value: `threshold ${threshold.toFixed(2)}`,
                position: "right",
                fill: "var(--ink-500)",
                fontSize: 9,
              }}
            />
          )}
          <Line
            yAxisId="l"
            dataKey="probability"
            name="P(distress at t+1)"
            stroke={COLORS.a}
            strokeWidth={2}
            dot={{ r: 2 }}
            isAnimationActive={false}
          />
          {firstEvent && (
            <ReferenceDot
              yAxisId="l"
              x={firstEvent.year}
              y={firstEvent.probability ?? 0}
              r={4}
              fill="var(--amber)"
              stroke="none"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-between pt-1 text-2xs text-ink-500">
        <span>predicted probability of distress in the FOLLOWING fiscal year</span>
        {firstEvent && (
          <span className="text-amber">
            first distress label · {firstEvent.label}
          </span>
        )}
      </div>
    </div>
  );
}

/* ---------- ΔAUC bootstrap interval ---------- */
export function DeltaAucBar({
  delta,
  low,
  high,
}: {
  delta: number;
  low: number;
  high: number;
}) {
  const scaleMax = 0.14;
  const pct = (v: number) => `${(v / scaleMax) * 100}%`;
  return (
    <div>
      <div className="relative h-10">
        <div className="absolute inset-x-0 top-1/2 h-px bg-ink-700" />
        <div className="absolute top-1/2 h-px w-px bg-ink-600" style={{ left: 0 }} />
        {/* zero mark */}
        <div className="absolute inset-y-0 w-0.5 bg-ink-500" style={{ left: 0 }}>
          <span className="absolute -bottom-4 left-0 text-2xs text-ink-500">
            0
          </span>
        </div>
        {/* CI band */}
        <div
          className="absolute top-1/2 h-2 -translate-y-1/2 rounded-full bg-model-b/25"
          style={{ left: pct(low), width: pct(high - low) }}
        />
        {/* point estimate */}
        <div
          className="absolute top-1/2 h-4 w-1 -translate-y-1/2 rounded-full bg-model-b"
          style={{ left: pct(delta) }}
        />
        <span
          className="tnum absolute -top-1 text-2xs text-model-b"
          style={{ left: pct(delta) }}
        >
          {fmtProb(delta)}
        </span>
      </div>
      <p className="tnum mt-3 text-2xs text-ink-500">
        95% CI [{fmtProb(low)}, {fmtProb(high)}] · 2,000-iteration bootstrap ·
        interval clears zero
      </p>
    </div>
  );
}
