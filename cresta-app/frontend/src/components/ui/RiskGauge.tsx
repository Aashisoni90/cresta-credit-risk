import { bandFromProb, fmtProb, RISK_META, type RiskBand } from "@/lib/format";
import { RiskBadge } from "./primitives";

/*
  The core readout. A probability, a horizontal fill against the four risk
  zones, and the decision threshold drawn as a gate. No dial, no needle —
  this is an instrument panel, not a speedometer.
*/
export default function RiskGauge({
  probability,
  threshold = 0.5,
  label = "distress probability",
  channel = "b",
  compact = false,
  band: bandProp,
  bands,
}: {
  probability: number;
  threshold?: number;
  label?: string;
  channel?: "a" | "b";
  compact?: boolean;
  /** The server's own risk_category. Always pass it when you have it. */
  band?: RiskBand;
  /** Live band edges from /api/model/info, so the coloured zones line up
   *  with the threshold gate drawn on the same bar. */
  bands?: { low: number; moderate: number; high: number };
}) {
  const band = bandProp ?? bandFromProb(probability, bands);
  const channelColor = channel === "a" ? "var(--model-a)" : "var(--model-b)";
  const edges = bands ?? { low: 0.3, moderate: 0.5, high: 0.7 };
  const zones = [
    { to: edges.low, c: "var(--risk-low)" },
    { to: edges.moderate, c: "var(--risk-moderate)" },
    { to: edges.high, c: "var(--risk-high)" },
    { to: 1, c: "var(--risk-critical)" },
  ];

  return (
    <div
      className="w-full"
      role="img"
      aria-label={`${label}: ${fmtProb(probability)}, ${RISK_META[band].label} risk, decision threshold ${threshold.toFixed(2)}`}
    >
      <div className="flex items-baseline justify-between">
        <span className="text-2xs uppercase tracking-[0.16em] text-ink-500">
          {label}
        </span>
        {!compact && <RiskBadge band={band} />}
      </div>

      <div className="mt-1.5 flex items-end gap-3">
        <span
          className="tnum font-display text-3xl font-semibold leading-none"
          style={{ color: RISK_META[band].color }}
        >
          {fmtProb(probability)}
        </span>
        {compact && <RiskBadge band={band} />}
      </div>

      {/* track — decorative: the figure and the band label above already
          carry the information, so this is hidden from assistive tech
          rather than read out as a list of coloured segments. */}
      <div
        aria-hidden="true"
        className="relative mt-3 h-2.5 overflow-hidden rounded-full bg-ink-900"
      >
        {/* faint zone tint */}
        <div className="absolute inset-0 flex opacity-25">
          {zones.map((z, i) => (
            <div
              key={i}
              style={{
                width: `${(z.to - (zones[i - 1]?.to ?? 0)) * 100}%`,
                backgroundColor: z.c,
              }}
            />
          ))}
        </div>
        {/* fill */}
        <div
          className="absolute inset-y-0 left-0 origin-left rounded-full motion-safe:animate-fill-up"
          style={{
            width: `${Math.min(100, Math.max(2, probability * 100))}%`,
            backgroundColor: channelColor,
          }}
        />
        {/* threshold gate */}
        <div
          className="absolute inset-y-0 w-0.5 bg-ink-50"
          style={{ left: `${threshold * 100}%` }}
          title={`decision threshold ${fmtProb(threshold, 2)}`}
        />
      </div>

      {!compact && (
        <div className="mt-1.5 flex justify-between text-2xs text-ink-500">
          <span>0.00</span>
          <span className="tnum">
            threshold {fmtProb(threshold, 2)}
          </span>
          <span>1.00</span>
        </div>
      )}
    </div>
  );
}
