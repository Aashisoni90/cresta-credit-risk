/*
  Number formatting. Each shape of number gets its own function so a
  probability never renders like a percentage and a count never loses
  its separators.
*/

/** Probability on [0,1], shown to 4 decimals: 0.7412 */
export function fmtProb(v: number | null | undefined, digits = 4): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

/** A value already expressed as a percent number (e.g. 86.74) -> "86.74%" */
export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)}%`;
}

/** A ratio on [0,1] shown as a percent: 0.5152 -> "51.52%" */
export function fmtRatioPct(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

/** Integer count with thousands separators: 115873 -> "115,873" */
export function fmtCount(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return Math.round(v).toLocaleString("en-US");
}

/** Signed delta, keeps the + for positive: +0.0774 */
export function fmtDelta(v: number | null | undefined, digits = 4): string {
  if (v == null || Number.isNaN(v)) return "—";
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : s;
}

/** Compact magnitude for money-ish figures: 1.24e9 -> "1.24B" */
export function fmtCompact(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}${(abs / 1e12).toFixed(digits)}T`;
  if (abs >= 1e9) return `${sign}${(abs / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(digits)}K`;
  return `${sign}${abs.toFixed(digits)}`;
}

export type RiskBand = "low" | "moderate" | "high" | "critical";

export const RISK_META: Record<
  RiskBand,
  { label: string; color: string; short: string }
> = {
  low: { label: "Low", short: "LOW", color: "var(--risk-low)" },
  moderate: { label: "Moderate", short: "MOD", color: "var(--risk-moderate)" },
  high: { label: "High", short: "HIGH", color: "var(--risk-high)" },
  critical: { label: "Critical", short: "CRIT", color: "var(--risk-critical)" },
};

/**
 * Band a probability. PREFER the `risk_category` the API already returns —
 * the server derives bands from the model's own operating threshold, and a
 * client that re-bands with different cut-points will label a company "low"
 * while the flag beside it reads "flagged".
 *
 * Use this only where no server-side category is available, and pass the
 * live `risk_bands` from /api/model/info when you can. The defaults below are
 * a last resort and are deliberately NOT the model's real cut-points.
 */
export function bandFromProb(
  p: number,
  bands?: { low: number; moderate: number; high: number },
): RiskBand {
  const b = bands ?? { low: 0.3, moderate: 0.5, high: 0.7 };
  if (p < b.low) return "low";
  if (p < b.moderate) return "moderate";
  if (p < b.high) return "high";
  return "critical";
}

/** Turn snake_case feature ids into something readable, ESG stays upper. */
export function humanizeFeature(id: string): string {
  const special: Record<string, string> = {
    e_score: "E-score",
    s_score: "S-score",
    g_score: "G-score",
    carbon_intensity: "Carbon intensity",
    roa: "ROA",
    ebit_margin: "EBIT margin",
    altman_z_double_prime: "Altman Z″",
    altman_ems_score: "Altman EMS score",
    altman_z_3yr_trend: "Altman Z 3-yr trend",
    altman_z_yoy_change: "Altman Z YoY change",
    revenue_growth_yoy: "Revenue growth YoY",
    ic_yoy_change: "Interest coverage YoY change",
    roa_yoy_change: "ROA YoY change",
    cfo_to_assets: "CFO / assets",
    cfo_to_debt: "CFO / debt",
    log_mkt_cap: "Log market cap",
    log_total_assets: "Log total assets",
    consecutive_ic_below1_years: "Consecutive IC-below-1 years",
    consecutive_loss_years: "Consecutive loss years",
    is_financials_sector: "Financials-sector flag",
    equity_volatility: "Equity volatility",
  };
  if (special[id]) return special[id];
  return id
    .replace(/_/g, " ")
    .replace(/\byoy\b/gi, "YoY")
    .replace(/^\w/, (c) => c.toUpperCase());
}

export const ESG_FEATURES = ["e_score", "s_score", "g_score", "carbon_intensity"];
export function isEsgFeature(id: string): boolean {
  return ESG_FEATURES.includes(id);
}
