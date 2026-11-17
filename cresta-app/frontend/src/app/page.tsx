"use client";

import Link from "next/link";
import { ArrowRight, FileText } from "lucide-react";
import CutLoopHero from "@/components/viz/CutLoopHero";
import { useApi, endpoints } from "@/lib/api";
import { fmtProb, fmtPct, fmtCount } from "@/lib/format";

/*
  Landing page.

  REWRITTEN IN PHASE 5. It previously led with "Model B ROC-AUC 0.6117,
  ΔAUC +0.0774" and a 0.2278% prevalence over 115,873 synthetic firm-years —
  the ESG result from the generated testbed, stated as the project's finding.
  None of it holds: Track B is deferred and the panel is real.

  Figures are now fetched, not hardcoded, so the landing page cannot assert a
  number the model no longer produces.
*/

export default function Landing() {
  return (
    <main className="min-h-screen bg-ink-950">
      <nav className="flex items-center justify-between border-b border-ink-700 px-6 py-4">
        <span className="font-display text-sm font-semibold tracking-[0.16em] text-ink-50">
          CRESTA
        </span>
        <div className="flex items-center gap-5 text-sm">
          <Link href="/portfolio" className="text-ink-300 hover:text-ink-50">
            Open instrument
          </Link>
        </div>
      </nav>

      <section className="mx-auto max-w-5xl px-6 pb-12 pt-16">
        <p className="text-2xs uppercase tracking-[0.22em] text-amber">
          Capstone · Indian listed equities · real Bloomberg panel
        </p>
        <h1 className="mt-4 max-w-3xl font-display text-4xl font-semibold leading-[1.1] tracking-tight text-ink-50 md:text-5xl">
          Most distress models score 0.95 by cheating.
          <span className="text-ink-400"> This one cuts the loop first.</span>
        </h1>
        <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-ink-300">
          When a distress label is built from the same accounting figures fed
          back in as features, the model isn&apos;t forecasting — it&apos;s
          re-deriving an identity. CRESTA forces a one-year gap between features
          and label, deletes every circular feature, then measures whether an
          deletes every circular feature, and benchmarks against a baseline
          that simply assumes nothing changes — which, on this data, beats
          Altman Z&Prime;. This instrument lets you take that apart.
        </p>

        <div className="mt-10 rounded-card border border-ink-700 bg-ink-900 p-6">
          <CutLoopHero />
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            href="/portfolio"
            className="inline-flex items-center gap-2 rounded-md bg-amber px-5 py-2.5 text-sm font-semibold text-ink-950 hover:bg-amber/90"
          >
            Open the instrument <ArrowRight size={16} />
          </Link>
          <Link
            href="/model"
            className="inline-flex items-center gap-2 rounded-md border border-ink-600 px-5 py-2.5 text-sm text-ink-200 hover:bg-ink-850"
          >
            <FileText size={15} /> Jump to model performance
          </Link>
        </div>
      </section>

      <section className="border-t border-ink-700 bg-ink-900/50">
        <HeadlineFigures />
      </section>

      <footer className="mx-auto max-w-5xl px-6 py-10 text-2xs leading-relaxed text-ink-500">
        Analytical estimates for prioritising analyst attention. Not investment
        advice, not a credit rating, and not a probability of default — the
        label is a financial-criteria distress proxy, because agency ratings
        resolved for only 4 of 3,000 tickers. Financial-sector issuers are out
        of scope. No output may drive a credit, lending or investment decision
        without qualified human review. ESG (Track B) is documented future
        work: the BRSR-mandated subset yields too few events per variable to
        support a credible incremental-power claim.
      </footer>
    </main>
  );
}

function HeadlineFigures() {
  const metrics = useApi("metrics", endpoints.modelMetrics);
  const strata = useApi("strata", endpoints.modelStrata);
  const info = useApi("model-info", endpoints.modelInfo);

  const xgb = metrics.data?.models.find((m) => m.model === "XGBoost");
  const ew = strata.data?.strata.find(
    (s) => s.stratum === "healthy_at_t_early_warning",
  );
  const persistence = metrics.data?.models.find((m) =>
    m.model.startsWith("Persistence"),
  );

  const figures = [
    {
      v: ew ? fmtProb(ew.auc) : "—",
      l: "early-warning AUC",
      n: ew
        ? `healthy at t, n=${fmtCount(ew.n)}, ${fmtPct(ew.prevalence * 100, 1)} prevalence`
        : "loading",
      c: "var(--amber)",
    },
    {
      v: xgb?.auc_roc ? fmtProb(xgb.auc_roc) : "—",
      l: "pooled AUC",
      n: "inflated by label persistence — see model performance",
      c: "var(--model-a)",
    },
    {
      v: persistence?.auc_roc ? fmtProb(persistence.auc_roc) : "—",
      l: "naive baseline",
      n: "\u201cassume nothing changes\u201d — the bar that matters",
      c: "var(--ink-300)",
    },
    {
      v: info.data ? fmtCount(info.data.n_tickers) : "—",
      l: "companies scored",
      n: info.data
        ? `${fmtCount(info.data.n_scored_rows)} company-years, financials excluded`
        : "loading",
      c: "var(--ink-300)",
    },
  ];

  return (
    <div className="mx-auto grid max-w-5xl grid-cols-2 gap-px overflow-hidden bg-ink-700 md:grid-cols-4">
      {figures.map((f) => (
        <div key={f.l} className="bg-ink-950 p-5">
          <p
            className="tnum font-display text-2xl font-semibold"
            style={{ color: f.c }}
          >
            {f.v}
          </p>
          <p className="mt-1 text-xs font-medium text-ink-200">{f.l}</p>
          <p className="mt-0.5 text-2xs leading-snug text-ink-500">{f.n}</p>
        </div>
      ))}
    </div>
  );
}
