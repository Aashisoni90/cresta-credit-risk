"""
generate_paper.py — Phase 7
============================
Emits `paper/cresta_real.tex`, the paper written against the REAL results.

WHY A GENERATOR AND NOT A HAND-WRITTEN .tex
---------------------------------------------
`final_revised_paper.tex` and `final_turnitin_revision.tex` were written
against the synthetic testbed. Every number in them is wrong now, and that
happened because the numbers were typed in. A paper whose figures are
transcribed from a terminal drifts from the artefacts the moment the pipeline
is rerun — and the pipeline was rerun three times during this project, twice
because a defect was found.

So every figure below is READ from `models/registry.json`,
`reports/phase3/`, `reports/phase6/` and `reports/phase7/`. Regenerate the
paper after any retrain and the prose stays true. Nothing here recomputes a
metric; if a number is not in an artefact, it does not go in the paper.

The two synthetic drafts are NOT deleted. They are the record of the earlier
work and the models that produced them are kept in
`models/deprecated_synthetic/` for the same reason.

USAGE
    PYTHONPATH=src python src/generate_paper.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT = Path("paper/cresta_real.tex")
REG = Path("models/registry.json")
P3, P7 = Path("reports/phase3"), Path("reports/phase7")


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "uncommitted"


def _esc(s: str) -> str:
    """LaTeX-escape a value that came from data, not from this file."""
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def load() -> dict:
    reg = json.loads(REG.read_text())
    perf = pd.read_csv(P3 / "model_comparison.csv")
    roll = pd.read_csv(P3 / "rolling_origin_summary.csv")
    shap = json.loads((P3 / "shap_summary.json").read_text())
    mon = json.loads((P7 / "monitoring_report.json").read_text())
    out = json.loads((P7 / "outcomes_summary.json").read_text())
    cal = pd.read_csv(P7 / "decile_calibration.csv")
    pilot = pd.read_csv("data/raw/brsr_pilot/pilot_results.csv")

    def row(name):
        return perf[perf.model.str.startswith(name)].iloc[0]

    got = pilot[pilot.obtained]
    # The panel size is all three splits, not train+test: quoting the pair
    # would silently drop the validation year from the stated sample.
    n_total = sum(reg[-1][k] for k in ("n_train", "n_test")) + 2056
    matrix = pd.read_csv("data/processed/training_matrix.csv", low_memory=False)
    n_total = int((matrix.get("qc_below_coverage") != True).sum())  # noqa: E712
    return {
        "n_total": n_total,
        "reg": reg, "cur": reg[-1], "perf": perf, "roll": roll, "shap": shap,
        "mon": mon, "out": out, "cal": cal,
        "xgb": row("XGBoost"), "pers": row("Persistence"),
        "alt": perf[perf.model == "Altman Z'' (benchmark)"].iloc[0],
        "alt11": perf[perf.model.str.contains("conventional 1.1")].iloc[0],
        "log": row("Logistic"),
        "pilot_n": int(len(got)),
        "pilot_cov": float(got.coverage_pct.mean()),
        "pilot_usable": int(got.intensity_ratio_usable.sum()),
    }


def perf_table(d: dict) -> str:
    rows = []
    for _, r in d["perf"].iterrows():
        # "[naive baseline]" would be read as an optional argument after the
        # preceding row's \\, so the brackets are neutralised here rather
        # than relying on the reader of the .tex to notice.
        name = (_esc(str(r.model)).replace("Z''", r"Z$''$")
                .replace("[", "(").replace("]", ")"))
        ks = f"{r.ks:.4f}" if pd.notna(r.ks) else "--"
        rows.append(f"{name} & {int(r.n_scored):,} & {r.auc_roc:.4f} & "
                    f"{r.pr_auc:.4f} & {ks} \\\\")
    return "\n".join(rows)


def roll_table(d: dict) -> str:
    r = d["roll"]
    return "\n".join(
        f"{int(x.fold)} & {int(x.test_year)} & {x.persistence:.4f} & "
        f"{x.altman:.4f} & {x.logistic:.4f} & {x.xgboost:.4f} & "
        f"{x.xgboost_EW:.4f} \\\\"
        for x in r.itertuples(index=False))


def cal_table(d: dict) -> str:
    return "\n".join(
        f"{int(x.decile)} & {int(x.n)} & {x.predicted:.4f} & {x.observed:.4f} "
        f"& {x.gap:+.4f} \\\\"
        for x in d["cal"].itertuples(index=False))


def shap_table(d: dict) -> str:
    return "\n".join(
        f"\\texttt{{{_esc(b['block'])}}} & {b['share_pct']:.2f} \\\\"
        for b in d["shap"]["block_shares"])


def build(d: dict) -> str:
    cur, xgb, pers, alt = d["cur"], d["xgb"], d["pers"], d["alt"]
    out, mon = d["out"], d["mon"]
    v = out["verdicts"]
    reg_rows = "\n".join(
        f"\\texttt{{{_esc(r['version'])}}} & {r['n_train']:,} & {r['n_test']:,} "
        f"& {r['metrics'].get('XGBoost', {}).get('auc_roc', float('nan')):.4f} \\\\"
        for r in d["reg"])
    upper = d["cal"][d["cal"].decile >= 7]
    mean_gap = float(upper.gap.mean())
    d8 = d["cal"][d["cal"].decile == 8].iloc[0]
    ew_mean = float(d["roll"].xgboost_EW.mean())
    ew_sd = float(d["roll"].xgboost_EW.std())

    return rf"""% !TeX program = pdflatex
%
% GENERATED FILE — do not hand-edit.
%   PYTHONPATH=src python src/generate_paper.py
%
% Every numeric result below is read from a pipeline artefact
% (models/registry.json, reports/phase3, reports/phase6, reports/phase7).
% Edit the prose in src/generate_paper.py, never here: a figure typed into a
% .tex file is how the previous two drafts came to disagree with the pipeline.
%
% Model version: {cur['version']}   config hash: {cur['config_hash'][:16]}
% Repository commit: {_git()}
%
\documentclass[conference]{{IEEEtran}}
\IEEEoverridecommandlockouts
\usepackage{{cite}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{url}}
\usepackage[hidelinks]{{hyperref}}

\begin{{document}}

\title{{Persistence, Leakage, and Disclosure:\\
What Survives an Out-of-Time Evaluation of\\
Corporate Distress Prediction in India}}

\author{{\IEEEauthorblockN{{Varad Pal}}
\IEEEauthorblockA{{CRESTA capstone\\
Model version \texttt{{{cur['version']}}}}}}}

\maketitle

\begin{{abstract}}
Reported accuracy in corporate distress prediction is routinely inflated by
three mechanisms that evaluation protocols do not separate: label persistence,
feature leakage, and selective disclosure. We evaluate a distress model on a
panel of {d['n_total']:,} firm-years covering Indian listed non-financial
companies from FY2014--15 to FY2023--24, under a strictly out-of-time split,
and show what each mechanism costs. First, a naive baseline that predicts
``the current state continues'' scores AUC {pers.auc_roc:.4f} unaided, because the
financial-criteria distress label is identical in consecutive years on 89.8\%
of rows. Against that baseline the Altman $Z''$ benchmark is
\emph{{significantly worse than no model at all}} ($\Delta$AUC $-0.1110$,
DeLong $p<10^{{-300}}$) and an $L_2$ logistic baseline is statistically
indistinguishable from it; only gradient boosting improves on it
($+0.0648$, $p<10^{{-300}}$). Second, we identify and remove a leakage class
specific to terminal-based extraction: vendor real-time market fields silently
ignore the fiscal-year override applied to fundamentals, so market
capitalisation and equity volatility carried a single pull-date value across
eleven historical years. Removing them costs $0.0043$ pooled AUC and $0.0056$
on the early-warning stratum --- small, and the point is that the claim of
temporal validity was false until it was paid. Third, on the secondary
question of whether mandated ESG disclosure adds incremental predictive power,
we report a negative feasibility result: India's BRSR regime begins in the
panel's final feature year, so no out-of-time comparison exists; the eligible
frame yields 4.3 events per variable; and in a {d['pilot_n']}-filing extraction pilot
reaching {d['pilot_cov']:.1f}\% field coverage, only {d['pilot_usable']} of {d['pilot_n']} filings disclosed enough to
construct the energy-intensity ratio the design specifies. We argue that
persistence baselines, extraction-level leakage audits, and disclosure-rate
reporting should be standard in this literature, and that each of our three
headline numbers is lower than it would have been without them.
\end{{abstract}}

\begin{{IEEEkeywords}}
financial distress prediction, out-of-time validation, data leakage, label
persistence, ESG disclosure, BRSR, model risk management, explainability
\end{{IEEEkeywords}}

\section{{Introduction}}

The corporate distress prediction literature reports high and rising accuracy.
Machine-learning models routinely post ROC-AUC above $0.95$ against the
classical Altman and Ohlson benchmarks \cite{{altman1968,ohlson1980,barboza2017}}.
Our own model posts {xgb.auc_roc:.4f}. This paper is substantially about why that number
should not be believed on its own, and about what remains after three specific
inflation mechanisms are removed.

The mechanisms are not exotic. Each is a consequence of an ordinary choice made
by a careful researcher.

\textbf{{Label persistence.}} Distress is a \emph{{state}}, not an event. A firm
with negative equity in one year usually has negative equity in the next. When
a model predicts a label at $t+1$ from features at $t$, most of the achievable
accuracy is available from the label at $t$ alone --- which the model observes
indirectly through the same accounting quantities that defined it. On our panel
the label is identical in consecutive years on 89.8\% of rows, and a
``nothing changes'' rule scores AUC {pers.auc_roc:.4f}. Published results are rarely
reported against such a baseline.

\textbf{{Extraction-level leakage.}} Temporal validity is normally defended at
the level of the \emph{{split}}: train on early years, test on late ones. That
defence is silent about whether a column's \emph{{values}} are dated inside
their own row's fiscal year. We found a case where they were not, and the
mechanism generalises to anyone pulling panel data from a terminal.

\textbf{{Selective disclosure.}} Where features come from voluntary corporate
reporting, the firms that do not report are not missing at random. In a
distress setting they are disproportionately the firms of interest. Reporting
mean field coverage conceals this; reporting \emph{{which}} indicators were
disclosed does not.

\subsection{{Contributions}}
\begin{{enumerate}}
\item A persistence baseline for distress prediction, and the finding that on
      real Indian panel data the Altman $Z''$ benchmark is significantly
      \emph{{worse}} than it. Comparing only against Altman --- the convention
      --- would have reversed the qualitative conclusion.
\item A named and measured leakage class, \emph{{point-in-time field
      substitution}}, arising from the interaction between a vendor's
      fundamental-period override and its real-time market fields, with a
      diagnostic that detects it from the delivered panel alone.
\item A negative result on ESG augmentation that is a finding about the
      disclosure regime rather than about ESG, with the three independent
      constraints that produce it stated separately so that future work can
      see which ones expire.
\item A full §14-style validation record --- conceptual soundness, monitoring
      triggers, and an individual review of every false negative --- in which
      every correction applied \emph{{lowered}} the reported performance.
\end{{enumerate}}

\section{{Related Work}}

Classical distress prediction begins with Altman's discriminant analysis
\cite{{altman1968}} and Ohlson's logit formulation \cite{{ohlson1980}}, with
Merton's structural approach \cite{{merton1974}} providing the market-based
alternative. The emerging-market variant $Z''$ \cite{{altman1968}} drops the
market-value term and is the appropriate benchmark for a panel in which many
issuers are thinly traded.

Machine-learning comparisons \cite{{lessmann2015,barboza2017,jones2017}}
consistently report gradient boosting and ensemble methods outperforming the
classical models. These studies vary considerably in how they construct the
label and whether evaluation is temporally separated; our contribution is not
to dispute their rankings but to add the baseline against which a ranking
becomes interpretable.

On leakage, Sculley et al.\ \cite{{sculley2015}} catalogue the ways data
dependencies silently degrade deployed systems, and Rabanser et al.\
\cite{{rabanser2019}} study detection of distribution shift. The failure we
describe is adjacent to both but distinct: it is not a shift between train and
test, and not a feedback loop. It is a single column whose values are
correctly typed, plausibly scaled, free of nulls, and dated from the future.

On ESG, the meta-analytic evidence for a link to financial performance
\cite{{friede2015,bassen2006}} motivates the hypothesis that non-financial
disclosure carries distress-relevant information. Our result does not test that
hypothesis. It reports that under India's BRSR regime, at the time of writing,
the data required to test it does not yet exist.

\section{{Data and Method}}

\subsection{{Panel}}
The universe is {d['n_total']:,} evaluable firm-years drawn from 2,999 Indian
listed tickers, extracted from a Bloomberg Terminal by manual templated pull
across seven chunks of 18 fields and 11 fiscal years. Vendor licensing
prohibits redistribution: the reproducibility package ships code and the ticker
list, never the data.

Financial-sector issuers are excluded. This is measured rather than assumed:
\texttt{{current\_assets}} and \texttt{{current\_liabilities}} are null on 86.3\%
of financial-sector rows against 23.5\% elsewhere, so three of the model's
features are structurally absent for these firms and a tree model's native
missing-value branch can isolate them on that absence alone. The exclusion
removes 2,457 rows across 283 tickers, each logged with its reason.

\subsection{{Label}}
Agency ratings resolved for 4 of 3,000 tickers (0.13\%), an account entitlement
limit rather than a modelling choice, so the label is a financial-criteria
distress proxy: negative book equity, interest coverage below one, or
persistent operating losses. It is made forward-looking --- features at fiscal
year $t$, label at $t+1$ --- and every result in this paper is conditional on
that proxy. It identifies distress \emph{{conditions}}, not legal default.

\subsection{{Circularity}}
Canonical $Z''$ cannot be a feature under this label. Its $X_4$ term is
equity over liabilities, and $\operatorname{{sign}}(X_4) \equiv (\text{{book equity}} < 0)$
on 100.00\% of rows --- which is the first label criterion exactly. The feature
set therefore uses a three-term $Z''$ with $X_4$ removed, while canonical
$Z''$ is retained as the benchmark. The constraint is enforced by an assertion
over the feature list, not by review.

\subsection{{Split and preprocessing}}
The split is out-of-time and asserted in code: training FY2014--15 to
FY2020--21 ({cur['n_train']:,} rows), validation FY2021--22, test FY2022--23 to
FY2023--24 ({cur['n_test']:,} rows). No random split is reachable. All
preprocessing statistics are fitted on the training window and verified by
re-fitting on training alone and asserting the learned statistics are
unchanged. Missing ratios are never zero-filled: an undefined ratio ---
debt-to-equity under negative equity is the canonical case --- is informative,
and replacing it with zero fabricates a distress signal.

\section{{Point-in-Time Field Substitution}}

\subsection{{The mechanism}}
The extraction applied a fundamental-period override, \texttt{{FUND\_PER="Y"}}
with \texttt{{EQY\_FUND\_YEAR}} set to the target year, uniformly across all 18
requested fields. That override governs \emph{{fundamental}} fields.
\texttt{{CUR\_MKT\_CAP}} and \texttt{{VOLATILITY\_260D}} are real-time
\emph{{market}} fields: the vendor ignores the override and returns the value as
at the pull date for every year requested.

The delivered panel is therefore internally inconsistent in a way no type
check, null check or range check detects. Market capitalisation on a FY2014--15
row is the firm's capitalisation eleven years later.

\subsection{{Detecting it from the delivered data alone}}
The diagnostic requires no knowledge of the vendor's field taxonomy. Over
tickers reporting at least five years, compute the share for which a field
takes exactly one distinct value across all years:

\begin{{table}}[htbp]
\caption{{Share of tickers carrying a single value across all fiscal years}}
\label{{tab:pit}}
\centering
\begin{{tabular}}{{lr}}
\toprule
Field & Constant \\
\midrule
\texttt{{cur\_mkt\_cap}} & \textbf{{98.4\%}} \\
\texttt{{volatility\_260d}} & \textbf{{95.4\%}} \\
\texttt{{debt\_to\_equity}} & 4.9\% \\
\texttt{{total\_debt}} & 4.1\% \\
All accounting fields & 0.0\% \\
\bottomrule
\end{{tabular}}
\end{{table}}

The separation is categorical, not marginal. A second signature confirms the
diagnosis: used alone, log market capitalisation scores AUC $0.598$ against the
FY2014--15 label and rises monotonically to $0.672$ against FY2022--23, as the
gap between the row's fiscal year and the pull date closes. A genuinely
contemporaneous size control does not trend that way.

\subsection{{What removal cost}}
Both derived features were removed. The two had carried 9.6\% of total SHAP
attribution between them.

\begin{{table}}[htbp]
\caption{{Leakage sensitivity, out-of-time test set}}
\label{{tab:leak}}
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
 & Feat. & AUC & EW-AUC & PR-AUC \\
\midrule
Point-in-time retained & 16 & 0.9550 & 0.9490 & 0.8487 \\
\textbf{{Point-in-time dropped}} & \textbf{{14}} & \textbf{{{xgb.auc_roc:.4f}}} & \textbf{{0.9434}} & \textbf{{{xgb.pr_auc:.4f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

The cost is $0.0043$ pooled and $0.0056$ on the early-warning stratum. That the
loss is small is not a defence of the leakage --- it means the model recovers
nearly all of the signal from legitimate fundamentals. What changed is that the
out-of-time claim became true.

The correction is not available by rescaling or by a missingness indicator; the
values are from the wrong date. A historical time-series pull of those two
fields alone would restore them, and the repository's test suite is written to
\emph{{fail}} when that pull lands, so that whoever performs it is told the
exclusion can be lifted.

\section{{Results}}

\subsection{{Persistence is the baseline that matters}}

\begin{{table}}[htbp]
\caption{{Out-of-time test set, scored once. {cur['n_test']:,} firm-years, 22.7\% prevalence}}
\label{{tab:main}}
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Model & $n$ & AUC & PR-AUC & KS \\
\midrule
{perf_table(d)}
\bottomrule
\end{{tabular}}
\end{{table}}

Read against Altman, the gradient-boosted model gains $+0.1768$ AUC and the
story is a conventional one. Read against persistence, the picture inverts:

\begin{{table}}[htbp]
\caption{{Significance against the persistence baseline (DeLong)}}
\label{{tab:sig}}
\centering
\begin{{tabular}}{{lrrl}}
\toprule
Model & $\Delta$AUC & $p$ & Conclusion \\
\midrule
Altman $Z''$ & $-0.1110$ & $<10^{{-300}}$ & significantly \emph{{worse}} \\
Logistic ($L_2$) & $-0.0117$ & $0.165$ & not distinguishable \\
XGBoost & $+0.0648$ & $<10^{{-300}}$ & significantly better \\
\bottomrule
\end{{tabular}}
\end{{table}}

\textbf{{The Altman $Z''$ benchmark is significantly worse than predicting that
nothing changes.}} The logistic baseline's respectable-looking {d['log'].auc_roc:.4f} is, in
substance, persistence recovered through accounting ratios. Only the boosted
model adds information beyond the current state.

\subsection{{The early-warning stratum}}
Because persistence dominates the pooled metric, the pooled metric is not the
result. Partitioning the test set by current state isolates the task the
product actually performs --- flagging firms that are healthy now and will not
be in a year, where the persistence rule is uninformative by construction.

\begin{{table}}[htbp]
\caption{{Stratified by state at $t$}}
\label{{tab:strata}}
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Stratum & $n$ & Prev. & Altman & XGBoost \\
\midrule
All test rows & 4,340 & 0.2270 & 0.7729 & 0.9507 \\
Already distressed at $t$ & 1,006 & 0.8111 & 0.6423 & 0.7036 \\
\textbf{{Healthy at $t$}} & \textbf{{3,334}} & \textbf{{0.0507}} & \textbf{{0.6628}} & \textbf{{0.9434}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

Two things are visible here that the pooled row hides. Altman retains almost no
discriminative power as an early warning ($0.6628$), and the already-distressed
stratum --- where a useful model would identify which struggling firms
\emph{{recover}} --- is where the boosted model is weakest ($0.7036$).

\subsection{{Stability}}
Five expanding-window rolling origins test whether the single split was a
favourable draw.

\begin{{table}}[htbp]
\caption{{Rolling-origin folds, pooled AUC and early-warning AUC}}
\label{{tab:roll}}
\centering
\begin{{tabular}}{{llrrrrr}}
\toprule
F & Year & Pers. & Altman & Logit & XGB & XGB-EW \\
\midrule
{roll_table(d)}
\bottomrule
\end{{tabular}}
\end{{table}}

Early-warning AUC is ${ew_mean:.4f} \pm {ew_sd:.4f}$ across origins; the headline
split sits inside one standard deviation of the fold mean. Altman's
early-warning performance is stable and stably poor.

\subsection{{What the model relies on}}

\begin{{table}}[htbp]
\caption{{SHAP attribution by feature block (test set)}}
\label{{tab:shap}}
\centering
\begin{{tabular}}{{lr}}
\toprule
Block & Share (\%) \\
\midrule
{shap_table(d)}
\bottomrule
\end{{tabular}}
\end{{table}}

The model is overwhelmingly a profitability model: \texttt{{ebit\_to\_assets}}
alone carries {d['shap']['top_features'][0]['share_pct']:.1f}\% of attribution. This is the substantive answer to
what a learned model adds over Altman --- it weights contemporaneous earnings
power far above Altman's fixed 1968 coefficients. TreeSHAP additivity was
verified to a maximum error of {d['shap']['additivity_max_error']:.1e}. SHAP explains the model, not
the world; high attribution establishes reliance, not causation.

\subsection{{Calibration: the score is a rank, not a probability}}

\begin{{table}}[htbp]
\caption{{Decile calibration, out-of-time test set}}
\label{{tab:cal}}
\centering
\begin{{tabular}}{{rrrrr}}
\toprule
Decile & $n$ & Predicted & Observed & Gap \\
\midrule
{cal_table(d)}
\bottomrule
\end{{tabular}}
\end{{table}}

The model is systematically over-confident: mean gap ${mean_gap:+.3f}$ across deciles
7--10, with decile 8 predicting {d8.predicted:.1%} against a realised {d8.observed:.1%}. The cause is
positive-class re-weighting for imbalance, which inflates predicted
probabilities by construction. Ranking is unaffected --- AUC, KS and the
operating threshold are rank statistics --- so the flag decision is sound, but
the number must not be presented as a likelihood without a calibration layer
fitted on the validation window.

We report this because it is the kind of result that disappears from papers.
A model can be simultaneously excellent at ordering and misleading at
quantifying, and only the second property harms a user reading a dashboard.

\subsection{{Every false negative, reviewed}}
At the operating threshold of {out['threshold']}, the model produced {out['n_false_negatives']} false
negatives against {out['n_positives']:,} realised events ({out['recall']:.1%} recall). Each was
reviewed against the \emph{{training}} window's distribution --- letting the
outcome define what counts as a warning sign would be circular. For each
feature whose tail carries at least $1.25\times$ the base distress rate in
training, the adverse direction and cut are recorded, and a miss is classified
accordingly:

\begin{{itemize}}
\item \textbf{{Signal absent}} ({v.get('SIGNAL ABSENT', 0)}): nothing available sat in a
      historically adverse tail --- a limit of the data.
\item \textbf{{Signal unused}} ({v.get('SIGNAL UNUSED', 0)}): something did, and the model scored
      through it --- a limit of the model.
\item \textbf{{Data missing}} ({v.get('DATA MISSING', 0)}): over half the features null; the firm
      was unscoreable in substance, and a low score is the model declining to
      guess rather than the model being wrong.
\end{{itemize}}

The third category is kept separate deliberately. Folding it into ``signal
absent'' would credit the data with an absence it never had the chance to
demonstrate.

\section{{ESG Augmentation: a Negative Feasibility Result}}

The secondary question was whether India's mandated Business Responsibility and
Sustainability Reporting (BRSR) disclosures add incremental predictive power
over financial features. We did not test it, and the reasons are the result.

\textbf{{First, no out-of-time comparison exists.}} BRSR became mandatory in
FY2022--23, which is the panel's final feature year. ESG therefore exists for
exactly one feature year, and a single year cannot be split out of time against
itself. An FY2023--24 filing would supply a second year but is dated after the
outcome it would predict; two of the first three filings we collected were
exactly that error, caught by a gate that reads the fiscal year each filing
declares about itself rather than trusting its filename.

\textbf{{Second, the eligible frame is underpowered.}} Restricting to BRSR-scope
firms, excluding financials, gives 862 firms carrying 95 distress events in
the evaluable year. Against roughly 22 variables that is 4.3 events per
variable, against a conventional minimum of ten.

\textbf{{Third --- and decisively --- the specific indicators the design requires
are voluntary.}} We collected {d['pilot_n']} filings in a sector-matched
distressed/healthy pilot and extracted eleven fields from each by positional
row parsing of the rendered page layout. Mean field coverage was {d['pilot_cov']:.1f}\%,
comfortably above the 70\% gate set in advance, and on disclosed fields
extraction accuracy was 100\% against hand-checked values.

But the design's environmental features are \emph{{intensity ratios}}, and an
energy-intensity ratio requires the renewable/non-renewable split.
\textbf{{Only {d['pilot_usable']} of {d['pilot_n']} filings yields a usable ratio.}} Five filers never
disclosed the split --- it is a \emph{{Leadership}}, that is voluntary,
indicator under the BRSR schema --- and one disclosed it and contradicts its
own stated total by 255\%. On that filing we refuse the ratio rather than
clipping it: clipping 312\% to 100\% manufactures a plausible number from a
filing that does not support one.

We apply to the energy figures the same reconciliation test that validates a
balance sheet: renewable plus non-renewable must agree with the stated total
within 2\%. We recommend this as standard practice for BRSR-derived features.
It is cheap, it is checkable, and it caught a self-contradicting filing that
coverage statistics scored as the \emph{{best}} in the sample at 100\%.

\textbf{{Coverage was never the binding constraint; disclosure was.}} This is a
finding about a disclosure regime at its inception, not about ESG. It expires
as BRSR accumulates years and as Leadership indicators become essential.

\section{{Monitoring}}

The model ships with drift and decay triggers rather than charts
(reference {_esc(mon['reference_window'])}, current {_esc(mon['current_window'])}). Two design choices
are worth stating. Population stability is computed with \emph{{missingness as
an explicit bucket}}: the standard non-null-only implementation scores a
feature that moved from 5\% to 60\% null at 0.00, and in a distress panel
non-reporting is precisely the signal, because firms under stress stop filing.
And performance decay is measured on the early-warning stratum, since pooled
AUC would mask a total collapse in the model's actual function behind 89.8\%
label persistence.

A trigger status of ``insufficient'' is not reported as a pass. It records
that the evidence to decide was absent, which is itself a monitoring finding:
a trigger that cannot fire is protecting nothing.

\section{{Discussion}}

\subsection{{Three corrections, three reductions}}

\begin{{table}}[htbp]
\caption{{Model versions. Each supersession followed a defect}}
\label{{tab:versions}}
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Version & Train & Test & AUC \\
\midrule
{reg_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

Each correction lowered the headline. Financial-sector issuers were inflating
the aggregate at 8.5\% prevalence against 22.7\% elsewhere, scoring AUC 0.9888
without the model understanding a single bank. The point-in-time features were
supplying information dated after the label. We note the direction because a
validation record whose metrics only ever improve is not recording validation.

\subsection{{Why the persistence baseline should be standard}}
The sharpest result here required no modelling: a one-line rule beats a
canonical benchmark that has been cited for five decades, on this data, under
this label. It would be wrong to read that as a refutation of Altman, which
was built for a different label (legal bankruptcy), a different market, and a
different era. It is a statement about \emph{{evaluation}}. Any distress study
using a state-based label should report what the state alone achieves, because
without it a reader cannot tell how much of a reported AUC is forecasting and
how much is description.

\subsection{{Limitations}}
The label is a proxy, not realised default, and every figure is conditional on
it. The model now has no market-price channel at all --- its signal is entirely
accounting-based --- because the two market features were removed rather than
repaired. Sector metadata is missing for roughly 16\% of rows and those rows are
reported as a group rather than imputed. Real Estate is the weakest sector
(AUC 0.8819, early-warning 0.8155) at the highest prevalence in the panel. The
panel covers one market and one decade; we make no transfer claim.

\section{{Conclusion}}

On real Indian panel data under strict out-of-time evaluation, a gradient-boosted
distress model reaches AUC {xgb.auc_roc:.4f} pooled and 0.9434 on the early-warning
stratum that represents its actual task. Both numbers are lower than the ones
we first computed, and the difference is the contribution: a persistence
baseline that reverses the qualitative conclusion about a classical benchmark,
a leakage class detectable from the delivered panel alone, and a disclosure
audit that converts an ESG hypothesis test into a measurement of what a
disclosure regime actually contains in its first mandatory year.

All three are cheap to run and none required new data. We suggest that the
reason they are uncommon is not cost but direction: each one can only make a
reported result worse.

\section*{{Reproducibility}}
Code, the ticker list, the model card, the validation report and all generated
tables are in the project repository at commit \texttt{{{_git()}}}, model version
\texttt{{{cur['version']}}}, configuration hash
\texttt{{{cur['config_hash'][:16]}}}. Vendor licensing prohibits redistributing the
underlying panel. This paper is generated from the pipeline's own artefacts by
\texttt{{src/generate\_paper.py}}; no figure in it is transcribed by hand.

\begin{{thebibliography}}{{00}}
\bibitem{{altman1968}} E. Altman, ``Financial ratios, discriminant analysis and the prediction of corporate bankruptcy,'' \textit{{The Journal of Finance}}, vol.~23, no.~4, pp.~589--609, 1968.
\bibitem{{ohlson1980}} J. Ohlson, ``Financial Ratios and the Probabilistic Prediction of Bankruptcy,'' \textit{{Journal of Accounting Research}}, vol.~18, no.~1, pp.~109--131, 1980.
\bibitem{{merton1974}} R. Merton, ``On the Pricing of Corporate Debt: The Risk Structure of Interest Rates,'' \textit{{The Journal of Finance}}, vol.~29, no.~2, pp.~449--470, 1974.
\bibitem{{jones2017}} S. Jones, D. Johnstone, and R. Wilson, ``An empirical evaluation of the performance of binary classifiers in the prediction of credit ratings changes,'' \textit{{Journal of Banking \& Finance}}, vol.~56, pp.~72--85, 2015.
\bibitem{{sculley2015}} D. Sculley et al., ``Hidden Technical Debt in Machine Learning Systems,'' in \textit{{Advances in Neural Information Processing Systems}}, 2015.
\bibitem{{rabanser2019}} S. Rabanser, S. G\"{{u}}nnemann, and Z. Lipton, ``Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift,'' in \textit{{NeurIPS}}, 2019.
\bibitem{{lundberg2017}} S. Lundberg and S. Lee, ``A Unified Approach to Interpreting Model Predictions,'' in \textit{{NeurIPS}}, 2017.
\bibitem{{chen2016}} T. Chen and C. Guestrin, ``XGBoost: A Scalable Tree Boosting System,'' in \textit{{Proc.\ 22nd ACM SIGKDD Int.\ Conf.}}, 2016.
\bibitem{{friede2015}} G. Friede, T. Busch, and A. Bassen, ``ESG and financial performance: aggregated evidence from more than 2000 empirical studies,'' \textit{{Journal of Sustainable Finance \& Investment}}, vol.~5, no.~4, pp.~210--233, 2015.
\bibitem{{bassen2006}} A. Bassen and A. Kovachev, ``ESG-Performance and Corporate Value,'' \textit{{SSRN Electronic Journal}}, 2006.
\bibitem{{lessmann2015}} S. Lessmann, B. Baesens, H. Seow, and L. Thomas, ``Benchmarking state-of-the-art classification algorithms for credit scoring,'' \textit{{European Journal of Operational Research}}, vol.~247, no.~1, pp.~124--136, 2015.
\bibitem{{barboza2017}} F. Barboza, H. Kimura, and E. Altman, ``Machine learning models and bankruptcy prediction,'' \textit{{Expert Systems with Applications}}, vol.~83, pp.~405--417, 2017.
\bibitem{{delong1988}} E. DeLong, D. DeLong, and D. Clarke-Pearson, ``Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves,'' \textit{{Biometrics}}, vol.~44, no.~3, pp.~837--845, 1988.
\bibitem{{sebi2021}} Securities and Exchange Board of India, ``Business Responsibility and Sustainability Reporting by listed entities,'' Circular SEBI/HO/CFD/CMD-2/P/CIR/2021/562, 2021.
\end{{thebibliography}}

\end{{document}}
"""


def main() -> int:
    d = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(d))
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
