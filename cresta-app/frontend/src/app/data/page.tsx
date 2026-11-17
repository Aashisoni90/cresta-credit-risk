"use client";

import { Scissors } from "lucide-react";
import AppShell from "@/components/shell/AppShell";
import { useApi, endpoints } from "@/lib/api";
import {
  Panel,
  SectionHeader,
  ErrorState,
  Chip,
  SkeletonRows,
} from "@/components/ui/primitives";
import { fmtCount, humanizeFeature } from "@/lib/format";

export default function DataPage() {
  return (
    <AppShell
      title="Data & features"
      subtitle="What entered the pipeline, what was rejected, and which features were pulled for being circular with the distress label."
    >
      <GateFlow />
      <div className="mt-4">
        <GateGrid />
      </div>
      <div className="mt-6">
        <FeatureSection />
      </div>
    </AppShell>
  );
}

function GateFlow() {
  const { data, error, isLoading } = useApi("dq", endpoints.dataQuality);
  if (isLoading) return <SkeletonRows rows={6} label="data quality" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;
  return (
    <Panel>
      <SectionHeader
        eyebrow="quality gating"
        title="Every row is accounted for"
        hint="rejections are logged, never silently dropped"
      />
      <div className="flex flex-wrap items-center gap-3">
        <FlowNode
          label="scored panel rows"
          value={fmtCount(data.pipeline_flow.raw_company_years)}
        />
        <span className="tnum text-ink-500">−</span>
        <FlowNode
          label="removed"
          value={fmtCount(data.pipeline_flow.removed)}
          tone="neg"
        />
        <span className="tnum text-ink-500">=</span>
        <FlowNode
          label="model rows"
          value={fmtCount(data.pipeline_flow.usable_company_years)}
          tone="pos"
        />
      </div>
      <p className="tnum mt-3 text-2xs leading-relaxed text-ink-500">
        {data.pipeline_flow.note}
      </p>
      <p className="mt-2 text-2xs leading-relaxed text-ink-500">
        {data.failed_ticker_note}
      </p>
    </Panel>
  );
}

function FlowNode({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "pos" | "neg";
}) {
  const c =
    tone === "pos"
      ? "var(--positive)"
      : tone === "neg"
        ? "var(--negative)"
        : "var(--ink-100)";
  return (
    <div className="rounded border border-ink-700 bg-ink-900 px-3 py-2">
      <p className="text-2xs uppercase tracking-wider text-ink-500">{label}</p>
      <p
        className="tnum font-display text-lg font-semibold"
        style={{ color: c }}
      >
        {value}
      </p>
    </div>
  );
}

function GateGrid() {
  const { data, error, isLoading } = useApi("dq", endpoints.dataQuality);
  if (isLoading) return <SkeletonRows rows={6} label="data quality" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;

  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {data.quality_gates.map((g) => {
        const warn = g.status === "warning";
        return (
          <div
            key={g.gate}
            className="rounded-card border border-ink-700 bg-ink-850 p-4"
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-2xs uppercase tracking-wider text-ink-500">
                  gate {g.gate}
                </p>
                <h3 className="font-display text-sm font-semibold text-ink-50">
                  {g.name}
                </h3>
              </div>
              <Chip tone={warn ? "amber" : "pos"}>
                {warn ? `${fmtCount(g.n_offending)} flagged` : "clean"}
              </Chip>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-400">
              {g.description}
            </p>
            <p className="mt-1.5 text-2xs uppercase tracking-wider text-ink-600">
              {g.kind === "hard" ? "hard reject" : "warn · row kept and marked"}
            </p>
          </div>
        );
      })}
    </div>
  );
}

function FeatureSection() {
  const { data, error, isLoading } = useApi("feat", endpoints.features);
  if (isLoading) return <SkeletonRows rows={6} label="data quality" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;

  return (
    <div className="space-y-4">
      <Panel>
        <SectionHeader
          eyebrow="feature engineering"
          title={`${data.n_features + data.circular_features.length} candidates → ${data.circular_features.length} severed → ${data.n_features} model inputs`}
        />
        <div className="flex flex-wrap items-center gap-3">
          <FlowNode
            label="candidates"
            value={String(data.n_features + data.circular_features.length)}
          />
          <span className="tnum text-ink-500">−</span>
          <FlowNode
            label="circular, removed"
            value={String(data.circular_features.length)}
            tone="neg"
          />
          <span className="tnum text-ink-500">=</span>
          <FlowNode
            label="final model inputs"
            value={String(data.n_features)}
            tone="pos"
          />
        </div>
        <p className="mt-3 text-2xs leading-relaxed text-ink-500">
          {data.excluded_as_circular}
        </p>
      </Panel>

      <div className="grid gap-3 md:grid-cols-2">
        {data.blocks.map((b) => (
          <div
            key={b.block}
            className="rounded-card border border-ink-700 bg-ink-850 p-4"
          >
            <div className="mb-2 flex items-center gap-2">
              <span className="tnum text-2xs text-ink-500">
                {b.block}
              </span>
              <h3 className="font-display text-sm font-semibold text-ink-50">
                {b.name}
              </h3>
            </div>
            <ul className="space-y-1">
              {b.features.map((f) => (
                <li
                  key={f.name}
                  className="flex items-baseline justify-between gap-3 border-b border-ink-800 pb-1 text-xs last:border-0"
                >
                  <span className="tnum text-ink-200">
                    {humanizeFeature(f.name)}
                  </span>
                  <span className="tnum text-right text-ink-500">
                    {f.shap_share_pct != null
                      ? `${f.shap_share_pct.toFixed(1)}% SHAP`
                      : "—"}
                    {f.null_rate_pct > 0
                      ? ` · ${f.null_rate_pct.toFixed(0)}% null`
                      : ""}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {data.circular_features.length > 0 && (
        <Panel className="border-amber-dim/50">
          <div className="mb-3 flex items-center gap-2">
            <Scissors size={15} className="text-amber" />
            <h3 className="font-display text-sm font-semibold text-ink-50">
              The {data.circular_features.length} features cut for circularity
            </h3>
          </div>
          <ul className="space-y-1.5">
            {data.circular_features.map((f) => (
              <li
                key={f.name}
                className="flex items-baseline justify-between gap-4 border-b border-ink-800 pb-1.5 text-xs last:border-0"
              >
                <span className="tnum text-amber line-through decoration-amber-dim">
                  {f.name}
                  {f.partial ? " (partial)" : ""}
                </span>
                <span className="text-right text-ink-400">{f.reason}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-2xs leading-relaxed text-ink-500">
            Each of these shares a deterministic accounting relationship with the
            distress definition. Left in, the model re-derives the label instead
            of forecasting it.
          </p>
        </Panel>
      )}
    </div>
  );
}
