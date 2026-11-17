import { humanizeFeature, isEsgFeature, fmtProb } from "@/lib/format";
import type { ShapGlobal } from "@/types/api";

/*
  Global SHAP importance. Bars were previously tinted by whether a feature
  was ESG; no ESG feature exists in this model, so the colouring is by
  attribution magnitude instead. The point the paper makes — three ESG
  variables on top, the best financial feature at less than half — should
  be visible without reading a number.
*/
export default function ShapBar({
  data,
  top = 14,
}: {
  data: ShapGlobal;
  top?: number;
}) {
  const rows = data.features.slice(0, top);
  const max = Math.max(...rows.map((r) => r.mean_abs_shap));

  return (
    <div className="space-y-1.5">
      {rows.map((r, i) => {
        const esg = isEsgFeature(r.feature);
        return (
          <div key={r.feature} className="flex items-center gap-3">
            <span className="tnum w-5 shrink-0 text-right text-2xs text-ink-500">
              {i + 1}
            </span>
            <span
              className={`w-44 shrink-0 truncate text-xs ${
                esg ? "font-medium text-model-b" : "text-ink-300"
              }`}
              title={r.feature}
            >
              {humanizeFeature(r.feature)}
            </span>
            <div className="relative h-4 flex-1 overflow-hidden rounded-sm bg-ink-900">
              <div
                className="absolute inset-y-0 left-0 rounded-sm"
                style={{
                  width: `${(r.mean_abs_shap / max) * 100}%`,
                  backgroundColor: esg
                    ? "var(--model-b)"
                    : "var(--ink-600)",
                }}
              />
            </div>
            <span className="tnum w-12 shrink-0 text-right text-2xs text-ink-400">
              {fmtProb(r.mean_abs_shap, 3)}
            </span>
          </div>
        );
      })}
      <p className="pt-1 text-2xs text-ink-500">
        mean |SHAP| over {data.n_features.toLocaleString()} sampled test
        rows · bar length is mean |SHAP| in log-odds
      </p>
    </div>
  );
}
