import { fmtCount, fmtRatioPct } from "@/lib/format";

/*
  2x2 at a given threshold. The cell area is scaled to its share of the
  total, so the eye sees immediately that TN dwarfs everything and TP is
  a sliver — which is the whole story of a 0.23% base rate.
*/
export default function ConfusionMatrix({
  tn,
  fp,
  fn,
  tp,
  threshold,
}: {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  threshold: number;
}) {
  const total = tn + fp + fn + tp;
  const cell = (
    v: number,
    label: string,
    tone: "good" | "bad",
    sub: string,
  ) => {
    const share = v / total;
    return (
      <div className="relative overflow-hidden rounded border border-ink-700 bg-ink-900 p-3">
        <div
          className="absolute inset-x-0 bottom-0"
          style={{
            height: `${Math.max(2, Math.sqrt(share) * 100)}%`,
            backgroundColor:
              tone === "good" ? "var(--risk-low)" : "var(--risk-critical)",
            opacity: 0.14,
          }}
        />
        <div className="relative">
          <p className="text-2xs uppercase tracking-wider text-ink-500">
            {label}
          </p>
          <p className="tnum font-display text-xl font-semibold text-ink-50">
            {fmtCount(v)}
          </p>
          <p className="tnum text-2xs text-ink-500">{sub}</p>
        </div>
      </div>
    );
  };

  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5">
        <div />
        <ColHead>predicted healthy</ColHead>
        <ColHead>predicted distress</ColHead>

        <RowHead>actually healthy</RowHead>
        {cell(tn, "true negative", "good", fmtRatioPct(tn / total, 1))}
        {cell(fp, "false positive", "bad", fmtRatioPct(fp / total, 1))}

        <RowHead>actually distressed</RowHead>
        {cell(fn, "false negative", "bad", `${fn} of ${fn + tp} events missed`)}
        {cell(tp, "true positive", "good", `${tp} of ${fn + tp} events caught`)}
      </div>
      <p className="tnum mt-2 text-2xs text-ink-500">
        threshold = {threshold.toFixed(2)} · n = {fmtCount(total)}
      </p>
    </div>
  );
}

function ColHead({ children }: { children: React.ReactNode }) {
  return (
    <div className="pb-0.5 text-center text-2xs uppercase tracking-wider text-ink-400">
      {children}
    </div>
  );
}
function RowHead({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center pr-2 text-right text-2xs uppercase leading-tight tracking-wider text-ink-400">
      {children}
    </div>
  );
}
