"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";
import AppShell from "@/components/shell/AppShell";
import { useApi, endpoints } from "@/lib/api";
import {
  Panel,
  SectionHeader,
  ErrorState,
  Skeleton,
  RiskBadge,
  SkeletonRows,
  SkeletonTable,
} from "@/components/ui/primitives";
import StatTile from "@/components/ui/StatTile";
import BaseRateStrip from "@/components/ui/BaseRateStrip";
import { SectorBars } from "@/components/charts/MiscCharts";
import ExportButton from "@/components/ui/ExportButton";
import DataTable, { Column } from "@/components/ui/DataTable";
import { fmtCompact, fmtCount, fmtProb, bandFromProb } from "@/lib/format";
import type { CompanyListItem } from "@/types/api";

export default function PortfolioPage() {
  return (
    <AppShell
      title="Portfolio overview"
      subtitle="Universe-level view: how the scored population sits by sector and risk band, and an explorer over every company-year the model scores."
    >
      <KpiRow />
      <div className="mt-4">
        <div className="flex justify-end">
          <ExportButton kind="portfolio" label="Export portfolio CSV" />
        </div>
        <BaseRateStrip />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel>
          <SectionHeader
            eyebrow="composition"
            title="Companies by sector"
            hint="latest year on record per firm"
          />
          <SectorCard />
        </Panel>
        {/* The ESG distribution panel is gone, not emptied: Track B is
            deferred and the previous chart was drawn from synthetic scores.
            The risk-band mix is what this screen can honestly show. */}
        <Panel>
          <SectionHeader
            eyebrow="composition"
            title="Risk bands"
            hint="latest fiscal year, at the operating threshold"
          />
          <RiskBandCard />
        </Panel>
      </div>
      <div className="mt-4">
        <Explorer />
      </div>
    </AppShell>
  );
}

function KpiRow() {
  const { data, error, isLoading } = useApi("dash", endpoints.dashboard);
  if (isLoading)
    return (
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    );
  if (error || !data)
    return <ErrorState message={error?.message ?? "No data"} />;
  const u = data.universe;
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <StatTile
        label="firms"
        value={fmtCount(u.n_tickers)}
        context="Indian listed, non-financial"
      />
      <StatTile
        label="firm-years"
        value={fmtCompact(u.n_company_years, 1)}
        context="after quality gating"
      />
      <StatTile
        label="firms ever labelled distressed"
        value={fmtCount(data.n_with_distress_label)}
        channel="amber"
        context={`${fmtProb((data.n_with_distress_label / u.n_tickers) * 100, 2)}% of universe`}
      />
      <StatTile
        label={`flagged · FY${data.latest_year}`}
        value={fmtCount(data.n_flagged)}
        channel="amber"
        context={`${data.pct_flagged}% at threshold ${data.threshold}`}
      />
    </div>
  );
}

function SectorCard() {
  const { data, error, isLoading } = useApi("dash", endpoints.dashboard);
  if (isLoading) return <SkeletonRows rows={5} label="composition" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;
  return <SectorBars distribution={data.sector_distribution} />;
}

function RiskBandCard() {
  const { data, error, isLoading } = useApi("dash", endpoints.dashboard);
  if (isLoading) return <SkeletonRows rows={5} label="composition" />;
  if (error || !data) return <ErrorState message={error?.message ?? "No data"} />;
  const order = ["critical", "high", "moderate", "low"] as const;
  const total = order.reduce((n, b) => n + (data.risk_bands[b] ?? 0), 0) || 1;
  return (
    <div className="space-y-2 pt-1">
      {order.map((b) => {
        const n = data.risk_bands[b] ?? 0;
        return (
          <div key={b} className="flex items-center gap-3">
            <RiskBadge band={b} />
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
              <div
                className="h-full rounded-full bg-amber/70"
                style={{ width: `${(n / total) * 100}%` }}
              />
            </div>
            <span className="tnum w-20 text-right text-2xs text-ink-300">
              {fmtCount(n)} · {((n / total) * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
      <p className="pt-1 text-2xs text-ink-500">
        Bands are derived from the model&apos;s own operating threshold
        ({data.threshold}), so a flagged firm can never be banded low.
      </p>
    </div>
  );
}

function Explorer() {
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [sector, setSector] = useState("");
  const [page, setPage] = useState(1);

  const { data: sectorsData } = useApi("sectors", endpoints.sectors);
  const key = `companies:${search}:${sector}:${page}`;
  const { data, error, isLoading } = useApi(key, () =>
    endpoints.companies({ search, sector, page, per_page: 15 }),
  );

  const columns: Column<CompanyListItem>[] = [
    {
      key: "ticker",
      header: "ticker",
      sortable: true,
      sortValue: (r) => r.ticker,
      render: (r) => (
        <span className="tnum text-ink-100">{r.ticker.split(" ")[0]}</span>
      ),
    },
    {
      key: "name",
      header: "name",
      render: (r) => <span className="text-ink-300">{r.name}</span>,
    },
    {
      key: "sector",
      header: "sector",
      render: (r) => <span className="text-ink-400">{r.sector}</span>,
    },
    {
      key: "assets",
      header: "total assets",
      align: "right",
      sortable: true,
      sortValue: (r) => r.total_assets ?? -1,
      render: (r) => (
        <span className="tnum text-ink-300">
          {r.total_assets != null ? fmtCompact(r.total_assets) : "—"}
        </span>
      ),
    },
    {
      key: "flag",
      header: "flagged",
      align: "right",
      render: (r) =>
        r.flagged ? (
          <RiskBadge band={r.risk_category} />
        ) : (
          <span className="text-2xs text-ink-500">below threshold</span>
        ),
    },
  ];

  return (
    <Panel pad={false}>
      <div className="flex flex-wrap items-center gap-3 border-b border-ink-700 p-4">
        <div className="flex-1">
          <SectionHeader
            eyebrow="explorer"
            title="Company universe"
            hint="pick a firm to score it and see the attribution"
          />
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-500"
            />
            <input
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              placeholder="ticker or name"
              className="tnum w-44 rounded-md border border-ink-600 bg-ink-900 py-1.5 pl-8 pr-2 text-xs text-ink-100 placeholder:text-ink-500 focus:border-amber"
            />
          </div>
          <select
            value={sector}
            onChange={(e) => {
              setSector(e.target.value);
              setPage(1);
            }}
            className="rounded-md border border-ink-600 bg-ink-900 px-2 py-1.5 text-xs text-ink-200 focus:border-amber"
          >
            <option value="">all sectors</option>
            {sectorsData?.sectors.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="p-4">
        {isLoading ? (
          <SkeletonTable rows={10} cols={6} label="companies" />
        ) : error || !data ? (
          <ErrorState message={error?.message ?? "No data"} />
        ) : (
          <>
            <DataTable
              columns={columns}
              rows={data.companies}
              getKey={(r) => r.ticker}
              onRowClick={(r) =>
                router.push(`/company/${encodeURIComponent(r.ticker)}`)
              }
            />
            <div className="mt-3 flex items-center justify-between text-2xs text-ink-500">
              <span className="tnum">
                {fmtCount(data.total)} firms · page {data.page} / {data.pages}
              </span>
              <div className="flex gap-1.5">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((p) => p - 1)}
                  className="rounded border border-ink-600 px-2 py-1 disabled:opacity-40 enabled:hover:bg-ink-800"
                >
                  prev
                </button>
                <button
                  disabled={page >= data.pages}
                  onClick={() => setPage((p) => p + 1)}
                  className="rounded border border-ink-600 px-2 py-1 disabled:opacity-40 enabled:hover:bg-ink-800"
                >
                  next
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </Panel>
  );
}
