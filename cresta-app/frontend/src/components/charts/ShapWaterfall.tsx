import { humanizeFeature, fmtProb, isEsgFeature } from "@/lib/format";
import type { PredictionExplanation, ShapContribution } from "@/types/api";

/*
  Per-prediction attribution. SHAP contributions for a tree classifier are
  in log-odds space, so they are shown as relative pushes — amber toward
  distress, teal protective — rather than something that sums to a
  probability. The final row is the actual model probability.
*/
export default function ShapWaterfall({
  data,
  finalProbability,
  top = 9,
}: {
  data: PredictionExplanation;
  finalProbability?: number;
  top?: number;
}) {
  const ranked = data.contributions.slice(0, top);
  const rest = data.contributions
    .slice(top)
    .reduce((acc: number, f: ShapContribution) => acc + f.shap_log_odds, 0);
  const steps: ShapContribution[] = [...ranked];
  if (Math.abs(rest) > 0.0005) {
    steps.push({
      feature: `${data.contributions.length - top} smaller features`,
      value: null,
      value_missing: false,
      shap_log_odds: rest,
      direction: rest > 0 ? "increases risk" : "decreases risk",
    });
  }

  const maxMag = Math.max(...steps.map((s) => Math.abs(s.shap_log_odds)), 0.001);

  return (
    <div className="space-y-2">
      {steps.map((s, i) => {
        const toward = s.shap_log_odds > 0;
        return (
          <Row
            key={i}
            label={humanizeFeature(s.feature)}
            sub={
              s.value != null ? `value ${fmtProb(s.value, 2)}` : undefined
            }
            esg={isEsgFeature(s.feature)}
            value={`${toward ? "+" : ""}${fmtProb(s.shap_log_odds, 3)}`}
            bar={{ maxMag, mag: Math.abs(s.shap_log_odds), toward }}
          />
        );
      })}
      <div className="border-t border-ink-700 pt-2">
        <Row
          label="base rate"
          sub="the model's class-rebalanced prior, before any feature"
          value={fmtProb(data.base_rate)}
          bar={null}
        />
        <Row
          label="model probability"
          sub={data.prediction_horizon}
          value={fmtProb(finalProbability ?? data.probability)}
          strong
          bar={null}
        />
      </div>
      <p className="pt-0.5 text-2xs text-ink-500">
        bars are SHAP contributions in log-odds · amber pushes toward distress,
        teal is protective
      </p>
      {/* §8.8: attribution is not causation, and the caveat ships with the
          numbers rather than living in copy that a redesign can drop. */}
      <p className="text-2xs text-ink-500">{data.caveat}</p>
    </div>
  );
}

function Row({
  label,
  sub,
  value,
  esg,
  strong,
  bar,
}: {
  label: string;
  sub?: string;
  value: string;
  esg?: boolean;
  strong?: boolean;
  bar: { maxMag: number; mag: number; toward: boolean } | null;
}) {
  return (
    <div className="flex items-center gap-3">
      <span
        className={`w-40 shrink-0 truncate text-xs ${
          strong
            ? "font-semibold text-ink-50"
            : esg
              ? "font-medium text-model-b"
              : "text-ink-300"
        }`}
        title={label}
      >
        {label}
        {sub && (
          <span className="ml-1 text-2xs text-ink-500">· {sub}</span>
        )}
      </span>

      <div className="relative h-4 flex-1">
        {/* centre line */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-ink-700" />
        {bar && (
          <div
            className="absolute inset-y-1 rounded-sm"
            style={{
              left: bar.toward ? "50%" : undefined,
              right: bar.toward ? undefined : "50%",
              width: `${(bar.mag / bar.maxMag) * 45}%`,
              backgroundColor: bar.toward
                ? "var(--amber)"
                : "var(--model-b)",
            }}
          />
        )}
      </div>

      <span
        className={`tnum w-14 shrink-0 text-right text-xs ${
          strong ? "font-semibold text-ink-50" : "text-ink-300"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
