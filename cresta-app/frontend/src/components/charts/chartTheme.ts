/* Shared Recharts styling so every chart reads as one instrument. */

export const AXIS = {
  stroke: "var(--ink-600)",
  tick: { fill: "var(--ink-400)", fontSize: 11, fontFamily: "var(--font-mono)" },
  tickLine: false,
} as const;

export const GRID = {
  stroke: "var(--grid)",
  strokeDasharray: "0",
} as const;

export const TOOLTIP_STYLE = {
  contentStyle: {
    background: "var(--ink-800)",
    border: "1px solid var(--ink-600)",
    borderRadius: 8,
    fontSize: 12,
    fontFamily: "var(--font-mono)",
    color: "var(--ink-100)",
    boxShadow: "var(--shadow-panel)",
  },
  labelStyle: { color: "var(--ink-400)", fontSize: 11 },
  itemStyle: { color: "var(--ink-100)" },
} as const;

export const COLORS = {
  a: "var(--model-a)",
  b: "var(--model-b)",
  amber: "var(--amber)",
  low: "var(--risk-low)",
  moderate: "var(--risk-moderate)",
  high: "var(--risk-high)",
  critical: "var(--risk-critical)",
  muted: "var(--ink-500)",
  pos: "var(--positive)",
  neg: "var(--negative)",
} as const;
