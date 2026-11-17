"use client";

import AppShell from "@/components/shell/AppShell";
import { useApi, endpoints } from "@/lib/api";
import {
  Panel,
  SectionHeader,
  ErrorState,
  Chip,
  SkeletonRows,
  SkeletonTable,
  SkeletonChart,
} from "@/components/ui/primitives";
import StatTile from "@/components/ui/StatTile";
import { RocCurves, PrCurve } from "@/components/charts/RocPrChart";
import ThresholdExplorer from "@/components/charts/ThresholdExplorer";
import ShapBar from "@/components/charts/ShapBar";
import DataTable, { Column } from "@/components/ui/DataTable";
import { fmtProb, fmtPct, fmtCount } from "@/lib/format";
import type { ModelRow, StratumRow } from "@/types/api";

/*
  §12.4 — the evidence surface, and the page a model validator reads first.

  REWRITTEN IN PHASE 5. The previous version was built around Model A vs
  Model B and a ΔAUC from the ESG experiment. There is no Model B: Track B is
  deferred, and that comparison was computed from synthetic weights. The
  screen now shows the real §8.1 hierarchy — persistence, Altman Z″, logistic,
  XGBoost — which is a more demanding comparison, because the naive
  persistence baseline beats two of the four.

  The pooled AUC is NOT the headline and the page is laid out to say so: the
  stratified panel sits directly beneath the metric tiles, not below the fold.
*/

export default function ModelPage() {
  return (
    <AppShell
      title="Model performance"
      subtitle="The evidence surface. Out-of-time test window, scored once, financial-sector issuers excluded."
    >
      <Headline />

      <div className="mt-4">
        <Panel>
          <SectionHeader
            eyebrow="the number that matters"
            title="Pooled AUC is inflated by label persistence"
            hint="the early-warning stratum is the honest figure"
          />
          <StrataPanel />
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel>
          <SectionHeader
            eyebrow="discrimination"
            title="ROC curve"
            hint="out-of-time test split"
          />
          <RocPanel />
        </Panel>
        <Panel>
          <SectionHeader
            eyebrow="imbalance"
            title="Precision–recall"
            hint="more informative than ROC when positives are rare"
          />
          <PrPanel />
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel>
          <SectionHeader
            eyebrow="calibration"
            title="Predicted vs observed"
            hint="decile-binned — a probability shown to an analyst must mean something"
          />
          <CalibrationPanel />
        </Panel>
        <Panel>
          <SectionHeader
            eyebrow="stability"
            title="Rolling-origin folds"
            hint="§8.5 — a distribution, not one arbitrary split date"
          />
          <RollingOriginPanel />
        </Panel>
      </div>

      <div className="mt-4">
        <Panel pad={false}>
          <div className="p-5 pb-0">
            <SectionHeader
              eyebrow="§8.1 hierarchy"
              title="Every model, reported together"
              hint="reporting only the winner is a methodological failure"
            />
          </div>
          <ModelTable />
        </Panel>
      </div>

      <div className="mt-4">
        <Panel>
          <SectionHeader
            eyebrow="§8.4 cost assumption"
            title="Where to put the decision boundary"
            hint="the ratio is an assumption, and a consequential one"
          />
          <ThresholdExplorer />
        </Panel>
      </div>

      <div className="mt-4">
        <Panel>
          <SectionHeader
            eyebrow="§8.8 attribution"
            title="What the model leans on"
            hint="SHAP explains the model, not the world"
          />
          <ShapPanel />
        </Panel>
      </div>
    </AppShell>
  );
}

function Headline() {
  const { data, error, isLoading } = useApi("metrics", endpoints.modelMetrics);
  if (isLoading) return <SkeletonRows rows={4} label="headline metrics" />;
  if (error || !data)
    return <ErrorState message={error?.message ?? "No data"} />;

  const xgb = data.models.find((m) => m.model === "XGBoost");
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label="AUC-ROC"
          value={fmtProb(xgb?.auc_roc ?? 0)}
          channel="a"
          context={`FY${data.test_window[0]}–${data.test_window[1]}, scored once`}
        />
        <StatTile
          label="PR-AUC"
          value={fmtProb(xgb?.pr_auc ?? 0)}
          channel="a"
          context={`base rate ${fmtPct(data.test_prevalence * 100, 1)}`}
        />
        <StatTile label="KS" value={fmtProb(xgb?.ks ?? 0)} context="credit-scoring convention" />
        <StatTile
          label="Brier"
          value={fmtProb(xgb?.brier ?? 0)}
          context="calibration, lower is better"
        />
      </div>
      {/* The caveat ships with the numbers. Reporting the pooled AUC without
          it overstates the model to anyone reading this screen. */}
      <p className="rounded border border-amber-dim bg-amber-glow px-4 py-3 text-xs leading-relaxed text-ink-200">
        {data.headline_caveat}
      </p>
    </div>
  );
}

function StrataPanel() {
  const { data, error, isLoading } = useApi("strata", endpoints.modelStrata);
  if (isLoading) return <SkeletonRows rows={3} label="stratified performance" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;

  const label: Record<StratumRow["stratum"], string> = {
    all_test_rows: "All test rows",
    already_distressed_at_t: "Already distressed at t (predicting recovery)",
    healthy_at_t_early_warning: "Healthy at t → early warning",
  };

  return (
    <div className="space-y-2">
      {data.strata.map((s) => {
        const isHeadline = s.stratum === "healthy_at_t_early_warning";
        return (
          <div
            key={s.stratum}
            className={`flex items-baseline justify-between gap-4 rounded border px-3 py-2 ${
              isHeadline
                ? "border-amber-dim bg-amber-glow"
                : "border-ink-800 bg-ink-900"
            }`}
          >
            <div>
              <p className={`text-xs ${isHeadline ? "text-ink-100" : "text-ink-300"}`}>
                {label[s.stratum] ?? s.stratum}
              </p>
              <p className="tnum text-2xs text-ink-500">
                n = {fmtCount(s.n)} · prevalence {fmtPct(s.prevalence * 100, 1)}
              </p>
            </div>
            <span
              className={`tnum font-display text-lg font-semibold ${
                isHeadline ? "text-amber" : "text-ink-200"
              }`}
            >
              {fmtProb(s.auc)}
            </span>
          </div>
        );
      })}
      <p className="pt-1 text-2xs leading-relaxed text-ink-500">{data.note}</p>
    </div>
  );
}

function RocPanel() {
  const curves = useApi("curves", endpoints.modelCurves);
  const metrics = useApi("metrics", endpoints.modelMetrics);
  if (curves.isLoading || metrics.isLoading) return <SkeletonChart label="curve" />;
  if (curves.error || !curves.data)
    return <ErrorState message={curves.error?.message ?? "No data"} />;
  const auc = metrics.data?.models.find((m) => m.model === "XGBoost")?.auc_roc;
  return <RocCurves data={curves.data} auc={auc ?? undefined} />;
}

function PrPanel() {
  const curves = useApi("curves", endpoints.modelCurves);
  const metrics = useApi("metrics", endpoints.modelMetrics);
  if (curves.isLoading || metrics.isLoading) return <SkeletonChart label="curve" />;
  if (curves.error || !curves.data)
    return <ErrorState message={curves.error?.message ?? "No data"} />;
  return (
    <>
      <PrCurve
        data={curves.data}
        baseRate={metrics.data?.test_prevalence ?? 0}
      />
      <p className="pt-1 text-2xs text-ink-500">
        the floor is the base rate ({fmtPct((metrics.data?.test_prevalence ?? 0) * 100, 1)}
        ) — a random model sits on it
      </p>
    </>
  );
}

function CalibrationPanel() {
  const { data, error, isLoading } = useApi("curves", endpoints.modelCurves);
  if (isLoading) return <SkeletonRows rows={10} label="calibration" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;
  return (
    <div className="space-y-1.5">
      {data.calibration.map((d) => {
        const gap = d.observed - d.predicted;
        return (
          <div key={d.decile} className="flex items-center gap-3 text-xs">
            <span className="tnum w-6 text-ink-500">{d.decile + 1}</span>
            <span className="tnum w-14 text-right text-ink-400">
              {fmtProb(d.predicted, 3)}
            </span>
            <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
              <div
                className="h-full rounded-full bg-model-a/70"
                style={{ width: `${d.observed * 100}%` }}
              />
            </div>
            <span className="tnum w-14 text-right text-ink-200">
              {fmtProb(d.observed, 3)}
            </span>
            <span
              className={`tnum w-14 text-right ${
                Math.abs(gap) > 0.1 ? "text-amber" : "text-ink-600"
              }`}
            >
              {gap > 0 ? "+" : ""}
              {fmtProb(gap, 3)}
            </span>
          </div>
        );
      })}
      <p className="pt-1 text-2xs text-ink-500">
        decile · predicted · observed · gap — a well-calibrated model keeps the
        gap near zero
      </p>
    </div>
  );
}

function RollingOriginPanel() {
  const { data, error, isLoading } = useApi("rolling", endpoints.rollingOrigin);
  if (isLoading) return <SkeletonRows rows={4} label="rolling-origin folds" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;

  const series = ["xgboost", "logistic", "altman", "persistence"].filter(
    (k) => k in data.summary,
  );
  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        {series.map((k) => {
          const s = data.summary[k];
          const ew = data.summary[`${k}_EW`];
          return (
            <div key={k} className="flex items-baseline justify-between text-xs">
              <span className="text-ink-300">{k}</span>
              <span className="tnum text-ink-200">
                {fmtProb(s.mean)}{" "}
                <span className="text-ink-500">± {s.sd.toFixed(4)}</span>
                {ew && (
                  <span className="ml-3 text-amber">
                    EW {fmtProb(ew.mean)}{" "}
                    <span className="text-ink-600">± {ew.sd.toFixed(4)}</span>
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-2xs leading-relaxed text-ink-500">
        {data.folds.length} expanding-window folds. {data.note}
      </p>
    </div>
  );
}

function ModelTable() {
  const { data, error, isLoading } = useApi("metrics", endpoints.modelMetrics);
  if (isLoading) return <SkeletonTable rows={5} cols={7} label="model comparison" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;

  const num = (v: number | null, d = 4) =>
    v == null ? <span className="text-ink-600">—</span> : <span className="tnum">{fmtProb(v, d)}</span>;

  const columns: Column<ModelRow>[] = [
    {
      key: "model",
      header: "model",
      render: (r) => (
        <span className="flex items-center gap-2">
          <span className="text-ink-100">{r.model}</span>
          {r.model.startsWith("Persistence") && (
            <Chip tone="neutral">naive baseline</Chip>
          )}
          {r.model === "XGBoost" && <Chip tone="amber">deployed</Chip>}
        </span>
      ),
    },
    { key: "auc", header: "AUC", align: "right", sortable: true,
      sortValue: (r) => r.auc_roc ?? -1, render: (r) => num(r.auc_roc) },
    { key: "pr", header: "PR-AUC", align: "right", sortable: true,
      sortValue: (r) => r.pr_auc ?? -1, render: (r) => num(r.pr_auc) },
    { key: "ks", header: "KS", align: "right", render: (r) => num(r.ks) },
    { key: "brier", header: "Brier", align: "right", render: (r) => num(r.brier) },
    { key: "recall", header: "recall", align: "right", render: (r) => num(r.recall, 3) },
    { key: "precision", header: "precision", align: "right", render: (r) => num(r.precision, 3) },
  ];

  return (
    <>
      <DataTable rows={data.models} columns={columns} getKey={(r) => r.model} />
      <p className="p-5 pt-3 text-2xs leading-relaxed text-ink-500">
        {fmtCount(data.n_test_rows)} out-of-time test rows, scored once.
        Accuracy is deliberately absent: §8.6 refuses it as a headline metric
        under imbalance.
      </p>
    </>
  );
}

function ShapPanel() {
  const { data, error, isLoading } = useApi("shap", () => endpoints.shapGlobal(16));
  if (isLoading) return <SkeletonRows rows={10} label="SHAP attribution" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;
  return (
    <div className="space-y-2">
      <ShapBar data={data} />
      <p className="text-2xs leading-relaxed text-ink-500">{data.caveat}</p>
    </div>
  );
}
