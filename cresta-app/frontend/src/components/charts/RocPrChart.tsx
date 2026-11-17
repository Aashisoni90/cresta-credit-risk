"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, GRID, TOOLTIP_STYLE, COLORS } from "./chartTheme";
import type { ModelCurves } from "@/types/api";
import { fmtProb } from "@/lib/format";

/** ROC curve for Model A with the chance diagonal.
 *
 *  Previously overlaid Model A against Model B. There is no Model B: Track B
 *  is deferred, and the old second series was drawn from synthetic weights. */
export function RocCurves({ data, auc }: { data: ModelCurves; auc?: number }) {
  const rows = data.roc.map((p) => ({ a_fpr: p.fpr, a_tpr: p.tpr }));

  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer>
        <LineChart margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid {...GRID} />
          <XAxis
            type="number"
            dataKey="a_fpr"
            domain={[0, 1]}
            {...AXIS}
            label={{
              value: "false positive rate",
              position: "insideBottom",
              offset: -2,
              fill: "var(--ink-500)",
              fontSize: 10,
            }}
          />
          <YAxis type="number" domain={[0, 1]} {...AXIS} />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="var(--ink-600)"
            strokeDasharray="3 3"
          />
          <Tooltip
            {...TOOLTIP_STYLE}
            formatter={(v: number) => fmtProb(v, 3)}
          />
          <Line
            data={rows}
            dataKey="a_tpr"
            name={auc == null ? "Model A" : `Model A · AUC ${fmtProb(auc)}`}
            stroke={COLORS.a}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Precision–recall curve for Model B with the base-rate floor. */
export function PrCurve({
  data,
  baseRate,
}: {
  data: ModelCurves;
  baseRate: number;
}) {
  const rows = data.pr.map((p) => ({
    recall: p.recall,
    precision: p.precision,
  }));
  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer>
        <AreaChart
          data={rows}
          margin={{ top: 8, right: 12, bottom: 4, left: -8 }}
        >
          <defs>
            <linearGradient id="prfill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.b} stopOpacity={0.35} />
              <stop offset="100%" stopColor={COLORS.b} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid {...GRID} />
          <XAxis
            type="number"
            dataKey="recall"
            domain={[0, 1]}
            {...AXIS}
            label={{
              value: "recall",
              position: "insideBottom",
              offset: -2,
              fill: "var(--ink-500)",
              fontSize: 10,
            }}
          />
          <YAxis type="number" domain={[0, "auto"]} {...AXIS} />
          <ReferenceLine
            y={baseRate}
            stroke="var(--ink-500)"
            strokeDasharray="4 3"
            label={{
              value: `base rate ${fmtProb(baseRate)}`,
              fill: "var(--ink-500)",
              fontSize: 10,
              position: "insideTopRight",
            }}
          />
          <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => fmtProb(v, 4)} />
          <Area
            dataKey="precision"
            stroke={COLORS.b}
            strokeWidth={2}
            fill="url(#prfill)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
