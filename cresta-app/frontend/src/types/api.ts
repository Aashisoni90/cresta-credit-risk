/*
  Response types for the CRESTA API (real data, Model A v1.2.0).

  Rewritten in Phase 5: the previous definitions described the synthetic
  backend — every prediction carried a Model B and an `esg_impact` delta, and
  companies had `e_score` / `s_score` / `g_score`. None of that exists. Track B
  is deferred, so ESG is absent by design (invariant 6), and the types now say
  so rather than leaving optional fields that are always undefined.
*/

/* ---- shared ---- */
export type RiskCategory = "low" | "moderate" | "high" | "critical";

/** Structured error body. The code is what the UI switches on for its empty
 *  and error states; the message is analyst-facing copy from the backend. */
export interface ApiErrorDetail {
  code: "SECTOR_UNSUPPORTED" | "NO_DATA" | "NO_PEER" | "ARTEFACT_MISSING" | string;
  message: string;
  ticker?: string;
  exclusion_reason?: string | null;
}

export interface TrackBStatus {
  available: false;
  code: "TRACK_B_UNAVAILABLE";
  message: string;
}

/* ---- dashboard ---- */
export interface DashboardSummary {
  model_version: string;
  /** Distinct tickers per sector — universe composition (§12.5). */
  sector_distribution: Record<string, number>;
  universe: {
    n_tickers: number;
    n_company_years: number;
    fiscal_years: [number, number];
    n_sector_unsupported_tickers: number;
  };
  latest_year: number;
  prediction_horizon: string;
  threshold: number;
  n_flagged: number;
  n_with_distress_label: number;
  pct_flagged: number;
  risk_bands: Partial<Record<RiskCategory, number>>;
  esg_available: false;
  disclaimer: string;
}

/* ---- companies ---- */
export interface CompanyListItem {
  ticker: string;
  name: string;
  sector: string | null;
  latest_year: number;
  probability: number | null;
  risk_category: RiskCategory;
  flagged: boolean;
  total_assets: number | null;
  esg_average: null;
}

export interface CompanyListResponse {
  companies: CompanyListItem[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
  esg_available: false;
}

/** A financial-sector issuer returns `supported: false` instead of a score —
 *  out of scope (§5.6), which is a different answer from missing data. */
export interface CompanyDetail {
  ticker: string;
  name: string;
  sector: string | null;
  supported: boolean;
  code?: string;
  message?: string;
  latest_year?: number;
  years_covered?: number[];
  probability?: number | null;
  risk_category?: RiskCategory;
  label_next?: number | null;
  label_definition?: string;
  data_quality_flags?: string[];
  /** Headline financials. `value: null` means genuinely absent — a missing
   *  ratio is information and is never zero-filled (invariant 1). */
  snapshot?: { key: string; label: string; unit: string; value: number | null }[];
  esg: null;
  esg_status?: string;
}

export interface CompanyHistoryPoint {
  fiscal_year: number;
  fiscal_year_label: string;
  probability: number | null;
  risk_category: RiskCategory;
  flagged: boolean;
  label_next: number | null;
  altman_z_ex_x4: number | null;
}

export interface CompanyHistory {
  ticker: string;
  name: string;
  history: CompanyHistoryPoint[];
}

/* ---- prediction ---- */
export interface PredictionResult {
  ticker: string;
  company_name: string;
  sector: string | null;
  fiscal_year: number;
  fiscal_year_label: string;
  prediction_horizon: string;
  model_a: {
    probability: number;
    risk_category: RiskCategory;
    flagged: boolean;
    model_type: "financial_only";
    model_version: string;
  };
  /** Always null. A number here would be synthetic — see TrackBStatus. */
  model_b: null;
  track_b_status: TrackBStatus;
  threshold_used: number;
  threshold_basis: string;
  data_quality_flags: string[];
  disclaimer: string;
}

export interface PortfolioRow {
  ticker: string;
  company_name: string;
  sector: string | null;
  probability: number;
  risk_category: RiskCategory;
  flagged: boolean;
  n_quality_flags: number;
}

export interface PortfolioResponse {
  fiscal_year: number;
  prediction_horizon: string;
  threshold: number;
  n_returned: number;
  n_scored_in_year: number;
  companies: PortfolioRow[];
  disclaimer: string;
}

/* ---- model performance ---- */
export interface ModelRow {
  model: string;
  n_scored: number | null;
  coverage: number | null;
  auc_roc: number | null;
  pr_auc: number | null;
  ks: number | null;
  brier: number | null;
  threshold: number | null;
  recall: number | null;
  precision: number | null;
  f1: number | null;
  accuracy: number | null;
}

export interface ModelMetrics {
  model_version: string;
  test_window: [number, number];
  n_test_rows: number;
  test_prevalence: number;
  models: ModelRow[];
  /** Must be shown with the table: the pooled AUC is inflated by persistence. */
  headline_caveat: string;
  scored_once: boolean;
}

export interface StratumRow {
  stratum: "all_test_rows" | "already_distressed_at_t" | "healthy_at_t_early_warning";
  n: number;
  prevalence: number;
  auc: number;
}

export interface ModelStrata {
  model_version: string;
  strata: StratumRow[];
  note: string;
}

export interface RollingOriginResponse {
  folds: Record<string, number | null>[];
  summary: Record<string, { mean: number; sd: number }>;
  note: string;
}

export interface ModelCurves {
  roc: { fpr: number; tpr: number }[];
  pr: { recall: number; precision: number }[];
  calibration: { decile: number; predicted: number; observed: number; n: number }[];
  n_test_rows: number;
}

export interface ModelInfo {
  version: string;
  n_features: number;
  features: string[];
  operating_threshold: number;
  trained_utc: string | null;
  n_scored_rows: number;
  n_tickers: number;
  n_sector_unsupported_tickers: number;
  track_b_available: false;
  risk_bands: Record<string, number>;
}

/* ---- threshold ---- */
export interface ThresholdPoint {
  threshold: number;
  recall: number;
  precision: number;
  f1: number;
  /** §8.6: reported for completeness only, never as a headline. Served so the
   *  UI can demonstrate the accuracy paradox against recall. */
  accuracy: number;
  pct_flagged: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  expected_cost_per_row: number;
  is_operating_point: boolean;
}

export interface ThresholdSweep {
  operating_threshold: number;
  cost_ratio_fn_fp: number;
  cost_ratio_basis: string;
  max_f1_point: ThresholdPoint;
  sweep: ThresholdPoint[];
  n_test_rows: number;
  n_positives: number;
}

/* ---- explainability ---- */
export interface ShapGlobalFeature {
  feature: string;
  block: string;
  mean_abs_shap: number;
  share_pct: number;
  mean_signed_shap: number;
  /** null when the feature is constant — not defined, not "no relationship". */
  corr_feature_vs_shap: number | null;
}

export interface ShapGlobal {
  model_version: string;
  n_features: number;
  features: ShapGlobalFeature[];
  note: string;
  caveat: string;
}

export interface ShapBlocks {
  model_version: string;
  blocks: { block: string; mean_abs_shap: number; share_pct: number; n_features: number }[];
  base_rate: number | null;
  note: string;
  caveat: string;
}

export interface ShapContribution {
  feature: string;
  shap_log_odds: number;
  direction: "increases risk" | "decreases risk";
  value: number | null;
  value_missing: boolean;
}

export interface PredictionExplanation {
  ticker: string;
  company_name: string;
  fiscal_year: number;
  prediction_horizon: string;
  probability: number;
  risk_category: RiskCategory;
  base_value_log_odds: number;
  base_rate: number;
  base_rate_note: string;
  contributions: ShapContribution[];
  additivity_error: number;
  additivity_ok: boolean;
  data_quality_flags: string[];
  caveat: string;
}

export interface Counterfactual {
  ticker: string;
  company_name: string;
  fiscal_year: number;
  probability: number;
  peer: { ticker: string; company_name: string; probability: number };
  peer_selection: string;
  differences: {
    feature: string;
    delta_shap_log_odds: number;
    company_value: number | null;
    peer_value: number | null;
  }[];
  caveat: string;
}

/* ---- data quality & features ---- */
export interface DataQualityReport {
  universe: {
    n_tickers_scored: number;
    n_company_years: number;
    n_sector_unsupported_tickers: number;
    sector_unsupported_note: string;
  };
  pipeline_flow: {
    raw_company_years: number;
    removed: number;
    usable_company_years: number;
    coverage_flagged_kept: number;
    note: string;
  };
  quality_gates: {
    gate: string;
    name: string;
    /** "hard" rejects the row; "warn" keeps and marks it. */
    kind: "hard" | "warn";
    description: string;
    n_offending: number;
    status: "ok" | "warning";
  }[];
  n_quality_rejections: number | null;
  n_failed_tickers: number | null;
  failed_ticker_note: string;
  feature_null_rates: { feature: string; null_rate_pct: number; n_null: number }[];
  coverage_by_year_pct: Record<string, number>;
  missingness_policy: string;
  esg_coverage: null;
  esg_note: string;
}

export interface FeatureOverview {
  model_version: string;
  n_features: number;
  features: {
    name: string;
    block: string | null;
    shap_share_pct: number | null;
    null_rate_pct: number;
  }[];
  blocks: {
    block: string;
    name: string;
    features: {
      name: string;
      shap_share_pct: number | null;
      null_rate_pct: number;
    }[];
  }[];
  /** Read from the pipeline's own circularity guard, not restated. */
  circular_features: {
    name: string;
    criterion: string;
    partial: boolean;
    reason: string;
  }[];
  absent_blocks: Record<string, string>;
  excluded_as_circular: string;
}
