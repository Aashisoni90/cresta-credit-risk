"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import AppShell from "@/components/shell/AppShell";
import { useApi, endpoints } from "@/lib/api";
import {
  Panel,
  SectionHeader,
  ErrorState,
  Skeleton,
  Chip,
  SkeletonRows,
} from "@/components/ui/primitives";
import RiskGauge from "@/components/ui/RiskGauge";
import Spine from "@/components/ui/Spine";
import ExportButton from "@/components/ui/ExportButton";
import ShapWaterfall from "@/components/charts/ShapWaterfall";
import { RiskTrajectory } from "@/components/charts/MiscCharts";
import { fmtCompact, fmtProb, fmtDelta } from "@/lib/format";

export default function CompanyDetailPage({
  params,
}: {
  params: { ticker: string };
}) {
  const ticker = decodeURIComponent(params.ticker);

  const detail = useApi(`company:${ticker}`, () => endpoints.company(ticker));
  const history = useApi(`history:${ticker}`, () =>
    endpoints.companyHistory(ticker),
  );

  const years: number[] = detail.data?.years_covered ?? [];
  // default to the second-to-last year so there is always a t+1 to forecast
  const defaultYear = years.length >= 2 ? years[years.length - 2] : years[0];
  const [year, setYear] = useState<number | null>(null);
  const activeYear = year ?? defaultYear ?? null;

  return (
    <AppShell
      title={detail.data?.name ?? ticker.split(" ")[0]}
      subtitle={
        detail.data
          ? `${ticker.split(" ")[0]} · ${detail.data.sector}`
          : "Loading company…"
      }
      actions={
        <Link
          href="/portfolio"
          className="inline-flex items-center gap-1.5 rounded-md border border-ink-600 px-3 py-1.5 text-xs text-ink-300 hover:bg-ink-850"
        >
          <ArrowLeft size={13} /> Portfolio
        </Link>
      }
    >
      {detail.isLoading ? (
        <SkeletonRows rows={5} label="company" />
      ) : detail.error || !detail.data ? (
        <ErrorState message={detail.error?.message ?? "Not found"} />
      ) : (
        <>
          {/* year selector */}
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <span className="text-2xs uppercase tracking-wider text-ink-500">
              feature year
            </span>
            {years.map((y) => (
              <button
                key={y}
                onClick={() => setYear(y)}
                className={`tnum rounded px-2 py-1 text-xs ${
                  y === activeYear
                    ? "bg-amber text-ink-950"
                    : "border border-ink-600 text-ink-300 hover:bg-ink-800"
                }`}
              >
                {y}
              </button>
            ))}
          </div>

          {activeYear != null && (
            <PredictionBlock ticker={ticker} year={activeYear} />
          )}

          <div className="mt-4 grid gap-4 lg:grid-cols-[1.3fr_1fr]">
            <Panel>
              <SectionHeader
                eyebrow="trajectory"
                title="ROA and ESG over time"
                hint="the distress event, where present, is marked in red"
              />
              {history.isLoading ? (
                <Skeleton className="h-[240px]" />
              ) : history.data ? (
                <RiskTrajectory history={history.data.history} />
              ) : (
                <ErrorState message="History unavailable" />
              )}
            </Panel>

            <Panel>
              <SectionHeader eyebrow="latest filing" title="Snapshot" />
              <Snapshot detail={detail.data} />
            </Panel>
          </div>
        </>
      )}
    </AppShell>
  );
}

function PredictionBlock({ ticker, year }: { ticker: string; year: number }) {
  // Null until the user moves the slider, so the first render uses the
  // model's own validated operating point rather than 0.5. Sending 0.5 made
  // the page report "not flagged" for companies the model flags, and then
  // labelled that 0.5 as cost-optimal — a provenance it does not have.
  const info = useApi("model-info", endpoints.modelInfo);
  const feats = useApi("features", endpoints.features);
  const [threshold, setThreshold] = useState<number | null>(null);
  const operating = info.data?.operating_threshold ?? null;
  const tv = threshold ?? operating;
  const isOverride = threshold != null && operating != null && threshold !== operating;

  const pred = useApi(`predict:${ticker}:${year}:${tv}`, () =>
    endpoints.predict({
      ticker,
      fiscal_year: year,
      ...(threshold != null ? { threshold } : {}),
    }),
  );

  return (
    <Panel>
      <SectionHeader
        eyebrow="forecast"
        title={`Distress forecast for FY ${year + 1}`}
        hint="features observed at fiscal year end drive a label one year out"
        right={
          <div className="flex items-center gap-2 text-2xs text-ink-400">
            <span>threshold</span>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.02}
              value={tv ?? 0.5}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-24 accent-amber"
              aria-label="Decision threshold"
            />
            <span className="tnum text-ink-200">
              {tv == null ? "—" : tv.toFixed(2)}
            </span>
          </div>
        }
      />

      <div className="flex justify-end">
        <ExportButton kind="company" params={{ ticker }} label="Export history CSV" />
      </div>
      <Spine severedCount={feats.data?.circular_features.length} year={year} className="my-4" />

      {pred.isLoading ? (
        <SkeletonRows rows={6} label="model estimate" />
      ) : pred.error || !pred.data ? (
        <ErrorState message={pred.error?.message ?? "Prediction failed"} />
      ) : (
        <>
          <div className="grid gap-5 md:grid-cols-2">
            <div className="rounded border border-ink-700 bg-ink-900 p-4">
              <div className="mb-2 flex items-center gap-2">
                <Chip tone="a">Model A</Chip>
                <span className="text-2xs text-ink-500">
                  financial only · {info.data?.n_features ?? "—"} features
                </span>
              </div>
              <RiskGauge
                probability={pred.data.model_a.probability}
                threshold={tv ?? 0.5}
                band={pred.data.model_a.risk_category}
                bands={
                  info.data
                    ? {
                        low: info.data.risk_bands.low,
                        moderate: info.data.risk_bands.moderate,
                        high: info.data.risk_bands.high,
                      }
                    : undefined
                }
                channel="a"
              />
            </div>
            {/* Track B is reported as unavailable, never as a number. A
                figure here would come from synthetic weights. */}
            <div className="rounded border border-dashed border-ink-700 bg-ink-900 p-4">
              <div className="mb-2 flex items-center gap-2">
                <Chip tone="neutral">Model B</Chip>
                <span className="text-2xs text-ink-500">ESG-augmented</span>
              </div>
              <p className="text-xs leading-relaxed text-ink-400">
                {pred.data.track_b_status.message}
              </p>
            </div>
          </div>

          <div className="mt-3 rounded border border-ink-700 bg-ink-900 px-4 py-3 text-xs leading-relaxed text-ink-300">
            <span className="tnum font-semibold text-amber">
              {fmtProb(pred.data.threshold_used, 3)}
            </span>{" "}
            decision threshold ·{" "}
            {isOverride
              ? `manual override — the model's validated operating point is ${operating?.toFixed(3)}`
              : pred.data.threshold_basis}
          </div>

          {pred.data.data_quality_flags.length > 0 && (
            <div className="mt-3 rounded border border-amber-dim bg-amber-glow px-4 py-3 text-xs text-ink-200">
              <span className="font-semibold text-amber">
                {pred.data.data_quality_flags.length} data-quality flag
                {pred.data.data_quality_flags.length > 1 ? "s" : ""}
              </span>{" "}
              · {pred.data.data_quality_flags.join(" · ")}
            </div>
          )}

          <div className="mt-4">
            <SectionHeader
              eyebrow="attribution"
              title="Why the model landed here"
              hint="SHAP contributions, base rate → prediction"
            />
            <ExplanationPanel ticker={ticker} year={year} />
          </div>

          <p className="mt-4 text-2xs leading-relaxed text-ink-500">
            {pred.data.disclaimer}
          </p>
        </>
      )}
    </Panel>
  );
}

function ExplanationPanel({ ticker, year }: { ticker: string; year: number }) {
  const { data, error, isLoading } = useApi(
    `explanation:${ticker}:${year}`,
    () => endpoints.predictionExplanation(ticker, year),
  );
  if (isLoading) return <SkeletonRows rows={8} label="SHAP attribution" />;
  if (error || !data)
    return <ErrorState message={error?.message ?? "No attribution"} />;
  return <ShapWaterfall finalProbability={data.probability} data={data} />;
}

function Snapshot({
  detail,
}: {
  detail: NonNullable<ReturnType<typeof useApi<import("@/types/api").CompanyDetail>>["data"]>;
}) {
  const snapshot = detail.snapshot ?? [];
  const flags = detail.data_quality_flags ?? [];
  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
        {snapshot.map((f) => (
          <div key={f.key} className="flex items-baseline justify-between border-b border-ink-800 pb-1">
            <dt className="text-xs text-ink-400">{f.label}</dt>
            <dd className="tnum text-sm text-ink-100">
              {f.value == null ? (
                // Not "0" and not blank: an undefined ratio is a finding.
                // debt/equity under negative equity is the canonical case.
                <span className="text-ink-500" title="not defined for this firm-year">
                  n/d
                </span>
              ) : (
                `${fmtCompact(f.value)}${f.unit ? ` ${f.unit}` : ""}`
              )}
            </dd>
          </div>
        ))}
      </dl>
      {flags.length > 0 && (
        <div>
          <p className="mb-1.5 text-2xs uppercase tracking-wider text-ink-500">
            data-quality flags
          </p>
          <ul className="space-y-1">
            {flags.map((f) => (
              <li key={f} className="text-xs text-amber">
                · {f}
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="text-2xs leading-relaxed text-ink-500">
        {detail.label_definition}
      </p>
    </div>
  );
}
