/*
  Thin typed client for the CRESTA API. One fetch wrapper, one error type,
  and a set of endpoint functions. Everything client-side goes through
  `useApi` (below) so loading / error states are uniform.

  REWRITTEN IN PHASE 5 against the real-data backend. Four kinds of change:

  1. Dead endpoints removed. `/model/comparison`, `/model/classifiers`,
     `/model/pdp` and `/model/shap/individual` belonged to the synthetic
     backend and no longer exist.
  2. Query parameters corrected (`n_points`, not `steps`; `fiscal_year`, not
     `year`), which were silently ignored and would have served defaults.
  3. The hardcoded `threshold: 0.5` on /predict is GONE. The model's validated
     operating point is 0.200; sending 0.5 made the UI report "not flagged"
     for companies the model flags. Omitting the field lets the backend apply
     its own cost-optimal threshold, which is the only defensible default.
  4. ApiError now carries the STRUCTURED detail. The backend returns
     `{code, message}` objects, so the previous `String(body.detail)` rendered
     "[object Object]" — including for SECTOR_UNSUPPORTED, the one error state
     §12.6 explicitly requires real copy for.
*/
"use client";

import useSWR, { SWRConfiguration } from "swr";
import type {
  ApiErrorDetail,
  Counterfactual,
  CompanyDetail,
  CompanyHistory,
  CompanyListResponse,
  DashboardSummary,
  DataQualityReport,
  FeatureOverview,
  ModelCurves,
  ModelInfo,
  ModelMetrics,
  ModelStrata,
  PortfolioResponse,
  PredictionExplanation,
  PredictionResult,
  RollingOriginResponse,
  ShapBlocks,
  ShapGlobal,
  ThresholdSweep,
} from "@/types/api";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  /** Structured body when the backend sent one — switch on `detail.code`. */
  detail?: ApiErrorDetail;

  constructor(message: string, status: number, detail?: ApiErrorDetail) {
    super(message);
    this.status = status;
    this.detail = detail;
    this.name = "ApiError";
  }

  /** True when the issuer is out of scope rather than missing (§5.6). */
  get isSectorUnsupported() {
    return this.detail?.code === "SECTOR_UNSUPPORTED";
  }

  get isNoData() {
    return this.detail?.code === "NO_DATA";
  }

  /** True when a Phase 3 artefact has not been generated yet. */
  get isArtefactMissing() {
    return this.detail?.code === "ARTEFACT_MISSING";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(
      "Can't reach the CRESTA API. Start the backend with `uvicorn main:app` on port 8000.",
      0,
    );
  }

  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    let detail: ApiErrorDetail | undefined;
    try {
      const body = await res.json();
      const d = body?.detail;
      if (d && typeof d === "object") {
        detail = d as ApiErrorDetail;
        message = detail.message || message;
      } else if (typeof d === "string") {
        message = d;
      }
    } catch {
      /* keep the status line */
    }
    throw new ApiError(message, res.status, detail);
  }
  return res.json() as Promise<T>;
}

const swrDefaults: SWRConfiguration = {
  revalidateOnFocus: false,
  shouldRetryOnError: false,
  dedupingInterval: 10_000,
};

/** Generic hook: pass a key (or null to skip) and a fetcher. */
export function useApi<T>(
  key: string | null,
  fetcher: () => Promise<T>,
  config?: SWRConfiguration,
) {
  const { data, error, isLoading, mutate } = useSWR<T, ApiError>(
    key,
    fetcher,
    { ...swrDefaults, ...config },
  );
  return { data, error, isLoading, mutate };
}

/* ---- endpoint functions ---- */
/** Absolute URL for a download link. Exports are served as file attachments,
 *  so they are navigated to rather than fetched — letting the browser name
 *  and save the file from Content-Disposition. */
export function exportUrl(
  path: "portfolio" | "company",
  params: Record<string, string | number> = {},
): string {
  const q = new URLSearchParams(
    Object.entries(params).map(([k, v]) => [k, String(v)]),
  );
  const qs = q.toString();
  return `${API_BASE}/api/export/${path}.csv${qs ? `?${qs}` : ""}`;
}

export const endpoints = {
  health: () =>
    request<{ status: string; model_version: string; n_scored_rows: number }>(
      "/api/health",
    ),
  dashboard: () => request<DashboardSummary>("/api/dashboard/summary"),
  modelInfo: () => request<ModelInfo>("/api/model/info"),
  sectors: () => request<{ sectors: string[] }>("/api/companies/sectors"),

  companies: (p: {
    search?: string;
    sector?: string;
    page?: number;
    per_page?: number;
  }) => {
    const q = new URLSearchParams();
    if (p.search) q.set("search", p.search);
    if (p.sector) q.set("sector", p.sector);
    q.set("page", String(p.page ?? 1));
    q.set("per_page", String(p.per_page ?? 25));
    return request<CompanyListResponse>(`/api/companies?${q.toString()}`);
  },
  company: (ticker: string) =>
    request<CompanyDetail>(`/api/companies/${encodeURIComponent(ticker)}`),
  companyHistory: (ticker: string) =>
    request<CompanyHistory>(
      `/api/companies/${encodeURIComponent(ticker)}/history`,
    ),

  /** `threshold` is deliberately omitted by default — see note 3 at the top. */
  predict: (body: {
    ticker: string;
    fiscal_year?: number;
    threshold?: number;
  }) =>
    request<PredictionResult>("/api/predict", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  portfolio: (p: {
    limit?: number;
    min_probability?: number;
    fiscal_year?: number;
    sector?: string;
  } = {}) => {
    const q = new URLSearchParams();
    q.set("limit", String(p.limit ?? 50));
    if (p.min_probability != null) q.set("min_probability", String(p.min_probability));
    if (p.fiscal_year != null) q.set("fiscal_year", String(p.fiscal_year));
    if (p.sector) q.set("sector", p.sector);
    return request<PortfolioResponse>(`/api/predict/portfolio?${q.toString()}`);
  },

  /* model performance (§12.4) */
  modelMetrics: () => request<ModelMetrics>("/api/model/metrics"),
  /** The stratified view. Always show this beside the pooled metrics: the
   *  pooled AUC is largely label persistence. */
  modelStrata: () => request<ModelStrata>("/api/model/strata"),
  rollingOrigin: () => request<RollingOriginResponse>("/api/model/rolling-origin"),
  modelCurves: () => request<ModelCurves>("/api/model/curves"),
  modelCard: () =>
    request<{ format: string; content: string }>("/api/model/card"),
  registry: () =>
    request<{ versions: unknown[]; serving: string }>("/api/model/registry"),

  threshold: (nPoints = 60) =>
    request<ThresholdSweep>(`/api/threshold?n_points=${nPoints}`),

  /* explainability (§8.8) */
  shapGlobal: (topN = 20) =>
    request<ShapGlobal>(`/api/model/shap/global?top_n=${topN}`),
  shapBlocks: () => request<ShapBlocks>("/api/model/shap/blocks"),
  predictionExplanation: (ticker: string, fiscalYear?: number, topN = 8) => {
    const q = new URLSearchParams({ ticker, top_n: String(topN) });
    if (fiscalYear != null) q.set("fiscal_year", String(fiscalYear));
    return request<PredictionExplanation>(
      `/api/predict/explanation?${q.toString()}`,
    );
  },
  counterfactual: (ticker: string, fiscalYear?: number, topN = 5) => {
    const q = new URLSearchParams({ ticker, top_n: String(topN) });
    if (fiscalYear != null) q.set("fiscal_year", String(fiscalYear));
    return request<Counterfactual>(`/api/predict/counterfactual?${q.toString()}`);
  },

  /** The stamp every export carries — also used for the print/PDF header. */
  exportManifest: () =>
    request<Record<string, string | number | null>>("/api/export/manifest"),

  dataQuality: () => request<DataQualityReport>("/api/data-quality"),
  features: () => request<FeatureOverview>("/api/features"),
};
