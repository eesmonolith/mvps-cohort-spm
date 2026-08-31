# MVPS: Multi-View Pattern Selection for Cohort-Robust Sequential Pattern Mining

Reference implementation and reproduction package for:

> Eunsang Eom, Jong-Kook Kim, Kiyoung Park.
> **"Multi-View Pattern Selection for Cohort-Robust Sequential Pattern Mining."**
> IEEE International Conference on Data Mining (ICDM), Shenyang, China, 2026.

MVPS mines sequential patterns that transfer across cohorts (e.g. academic
semesters, calendar months, course modules) by combining four
complementary pattern-selection views — stability, a variance-penalised
cohort-invariance score $S$, a CV-penalised variant $S_\text{CV}$, and
pooled discrimination — into a single deduplicated union, instead of
committing in advance to any one view. See the paper for the full method,
the joint-bound pruning result used to make $S$-mining tractable, and the
oracle/matched-budget analysis summarized below.

## Contents

- `src/` — mining algorithm (`c2dpm.py`), scoring functions, the joint
  bound, the LOCO transfer-evaluation harness, per-corpus adapters, the
  synthetic generator, and every script needed to reproduce the paper's
  tables and figures on the public corpora.
- `results/` — result JSON/CSV already produced by these scripts: full
  detail for the public corpora (BPI 2012, Sepsis, OULAD, Hospital
  Billing, RetailRocket, and the synthetic generator), plus **fold-level
  aggregate AUC only** for Edu/EduB (see "Data availability" below —
  no student-level data of any kind is included).
- `docs/budget70_detail.md` — the full per-corpus, per-method
  matched-budget-70 table referenced in the paper as "released with the
  code" (was cut from the PDF appendix for the page limit).
- `LICENSE` — MIT.

## Data availability

| Corpus | Public? | In this repo |
|---|---|---|
| BPI Challenge 2012 (+ amount-cohort variant) | Yes (4TU) | Adapter + download instructions below |
| Sepsis Cases | Yes (4TU) | Adapter + download instructions below |
| OULAD | Yes (Open University) | Adapter + download instructions below |
| Hospital Billing (+ monthly variant) | Yes (4TU) | Adapter + download instructions below |
| RetailRocket | Yes (Kaggle) | Adapter + download instructions below |
| Synthetic generator | N/A (generated) | `src/synthetic_experiment.py` |
| **Edu** (3-semester Korean intro-Python course log) | **No** | Not released (see below) |
| **EduB** (`codle_hashed`, second Codle-platform cohort) | **No** | Not released (see below) |

**Edu and EduB cannot be released, even in de-identified/hashed form, due
to institutional data-use constraints on the underlying student records.**
This applies to every derived form as well — no per-student sequences,
no per-pattern mining output, no cluster assignments. What *is* included
for these two corpora is the same class of artifact the paper reports:
**fold-level aggregate AUC numbers** (e.g. `results/multi_dataset_transfer.json`,
`results/r8_ensemble.json`, `results/camera_ready/*.json`) — a handful of
floating-point numbers per (corpus, method, LOCO fold), with no
patterns, no student identifiers, and no text/code content. These let you
verify every number quoted in the paper for Edu/EduB without touching
the raw data.

In their place, `src/synthetic_experiment.py` provides a generator with
the same $(\rho_c, \rho_d)$ cohort-shift / cluster-discriminability knobs
used throughout the paper's synthetic experiments (Section IV,
"Synthetic generator"; defaults $K{=}3$, $M{=}4$, $N{=}2{,}000$,
$V{=}10$, length $30$), so the qualitative $S$-view / MVPS behavior can
be explored without the original data.
**Caveat**: the paper states these parameters are "fit to reproduce
Edu's cohort-shift and cluster-discriminability statistics." We could
not locate an automated fitting script for that specific claim during
release preparation (see `camera-ready/RELEASE_AUDIT.md` in the source
project for the full note); what is released is the general-purpose
generator with the same default configuration used in every synthetic
experiment in the paper, swept over the same $(\rho_c, \rho_d)$ ranges.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11+. All experiments are deterministic given
`SEED = 42` (`src/config.py`).

## Data preparation (public corpora)

Download each corpus into `data/external/<name>/` (paths below match
what the adapters expect; `data/external/` is gitignored).

| Corpus | Download | Expected path |
|---|---|---|
| BPI Challenge 2012 | 4TU.ResearchData, search "BPI Challenge 2012" (`BPI_Challenge_2012.xes`) | `data/external/bpi2012/BPIC2012.xes` |
| Sepsis Cases | 4TU.ResearchData, search "Sepsis Cases - Event Log" | `data/external/sepsis/Sepsis_Cases.xes` |
| Hospital Billing | 4TU.ResearchData, search "Hospital Billing - Event Log" (Mannhardt, 2017) | `data/external/hospital_billing/HospitalBilling.xes` |
| OULAD | https://analyse.kmi.open.ac.uk/open_dataset (download `anonymisedData.zip`, unzip) | `data/external/OULAD/{studentInfo.csv,vle.csv,studentVle.csv}` |
| RetailRocket | Kaggle `retailrocket/ecommerce-dataset` (`events.csv`) | `data/external/retailrocket/events.csv` |

4TU.ResearchData occasionally renumbers dataset DOIs; if a direct link
goes stale, search the portal by the dataset title above — all four logs
are standard, widely-cited process-mining benchmarks (BPI Challenge
2012/2011 archive, Mannhardt's hospital billing and sepsis logs).

Each `.xes` log is read with `pm4py`; the CSV-based corpora (OULAD,
RetailRocket) are read directly with pandas — no conversion step needed
beyond placing the files at the paths above.

## Reproducing the paper: table/figure → command mapping

All commands are run from the repository root as `python -m src.<module>`.
Each writes its output(s) to `results/` (JSON/CSV) or `figures/`
(PDF/PNG), as noted below. Per-corpus adapters must be run once before
the corresponding transfer/mining scripts (they write
`data/processed/sequences_<corpus>.parquet` + cluster-label files).

| Paper item | What it shows | Exact command(s) | Requires |
|---|---|---|---|
| Table I (`tab:datasets`) | Dataset summary, 6 corpora + RetailRocket control | `python -m src.<corpus>_adapter` for each corpus prints trace count and the cohort×cluster ($K{\times}M$) breakdown | Per-corpus raw download |
| Table II (`tab:nonsep`) / Fig. 2 (`fig:nonsep`) | $S$ vs $S^{(v1)}$ rank correlation as $\lambda$ varies | `python -m src.make_figures_r1` (uses the paper's reported numbers as a documented fallback; the original scan requires Edu raw data via `non_separability_demo.py`, not included) | Edu raw (not public); figure reproduces from hardcoded values otherwise |
| Table III (`tab:transfer`) | Edu LOCO transfer AUC, K-top sweep | `python -m src.wilcoxon_audit_r4` (audits/reproduces `results/transfer_ktop_sweep.json` exactly; the sweep itself needs Edu raw) | Edu raw for regeneration; pre-computed JSON included for verification |
| Table IV (`tab:mining-stats`) | Mining stats with/without joint-bound prune | `python -m src.ablation_score` *(not included — Edu-only; see note)* | Edu raw (not public) |
| Table V (`tab:synthetic-bound`) / Section IV-C | Joint vs naive bound gap, synthetic sweep | `python -m src.synthetic_experiment` → `results/synthetic_bound.{json,csv}`; stress sweep: `python -m src.r1_synthetic_stress` → `results/r1_synthetic_stress.{json,csv}` | Nothing (fully synthetic) |
| Table VI (`tab:multi-dataset`) | LOCO transfer AUC, 7 baselines, 5 corpora | `python -m src.multi_dataset_transfer` (Edu, EduB, BPI, Sepsis) + `python -m src.run_oulad_transfer` (OULAD) → `results/multi_dataset_transfer.json`, `results/oulad_transfer.json` | Edu/EduB raw for those 2 of 5 columns; BPI/Sepsis/OULAD fully public |
| MVPS ensemble (Algorithm 1) | The 4-view union itself | `python -m src.run_r8_ensemble` → `results/r8_ensemble.json` | Per-corpus raw/adapter output |
| $\lambda$/K-top robustness (R7 sweep) | Robustness of $S$ across $\lambda$, $K_\text{top}$ | `python -m src.run_r7_sweep` → `results/r7_sweep.json` | Per-corpus raw/adapter output |
| BPI amount-cohort robustness check | Orthogonal cohort axis control | `python -m src.run_bpi_amount_transfer` → `results/bpi_amount_transfer.json` | BPI 2012 (public) |
| Hospital Billing transfer | 6th real corpus, quarterly cohorts | `python -m src.run_hb_transfer` → `results/hospital_billing_transfer.json` | Hospital Billing (public) |
| V-REx baseline | Invariant-risk-minimisation-style baseline comparison | `python -m src.run_vrex_baseline` → `results/vrex_baseline_lam500.json` | Per-corpus raw/adapter output |
| Table VII (`tab:sfamily-ablation`) | $S$-family ablation (Experiment A / A2) | `python -m src.camera_ready_experiments` → `results/camera_ready/ablation_minus_s.json`, `ablation_minus_s_family.json` | Per-corpus raw/adapter output (public corpora reproduce; Edu/EduB rows shipped pre-computed) |
| "Does a single view catch up at matched budget?" (Experiment B) | Matched-budget-70 detail | Same command as above → `results/camera_ready/budget70_baselines.json`; rendered as Markdown by `python -m src.make_budget70_table` → `docs/budget70_detail.md` | Same as above |
| "Equivalence to the per-fold oracle" | Edu +0.98pp / EduB +0.63pp / BPI 0.00pp / Sepsis −1.04pp, TOST $p{=}0.0003$ | `python -m src.oracle_tost_reproduction` → `results/camera_ready/oracle_tost_reproduction.json` | Nothing beyond the two JSON files already in `results/` (see note below) |
| Table V-R4 audit (`wilcoxon_audit_r4`) | Reviewer-4 statistical audit (n=3 floor, honest fold-level tests, MVPS-vs-$S$-view n=23 claim) | `python -m src.wilcoxon_audit_r4` → `results/camera_ready/wilcoxon_audit.json` | `results/transfer_ktop_sweep.json`, `results/wilcoxon_transfer.json`, `results/r8_ensemble.json` (all included) |
| Fig. 3 (`figR1_transfer`) / Fig. 2 (`figR1_nonsep`) | K-top sweep bar chart / rank-correlation bars | `python -m src.make_figures_r1` → `figures/figR1_transfer.{pdf,png}`, `figures/figR1_nonsep.{pdf,png}` | `results/transfer_ktop_sweep.json` (included) or Edu raw for full regeneration |

**Note on `Table IV` and `ablation_score.py`**: this specific script is
not included in the release because its sole data source is Edu raw
sequences and its output is a table of top-scoring *mined patterns*
(pattern content, not just AUC numbers) — outside the "fold-level
aggregate only" line drawn for Edu artifacts in this repo. The mining
statistics it reports (explored/pruned/qualified counts, mining time)
are visible in `results/external_baselines.json` in the private
research tree but are not shipped here since that file is also Edu-
pattern-adjacent; if you need this table reproduced, contact the
authors for a description of the exact bound-toggle setup.

**Note on the oracle/TOST reproduction script**
(`src/oracle_tost_reproduction.py`): this script was written during
release preparation, not extracted from the original experiment run.
A pre-release audit found no recoverable script or notebook for the
"Equivalence to the per-fold oracle" paragraph's numbers anywhere in
the project history. This script reconstructs the computation directly
from the paper's own definition (oracle = per-fold max over the seven
`tab:multi-dataset` baselines) applied to the two result files that
already ship in this repo (`multi_dataset_transfer.json`,
`r8_ensemble.json`). Running it reproduces the paper's per-corpus
deltas and TOST $p$-value to the reported precision; see the script's
docstring for one CI caveat (last-digit-only) that we chose to flag
rather than silently paper over.

## Determinism

All scripts seed `numpy`/`random` with `SEED = 42` from `src/config.py`.
Pinned dependency versions matching the paper's reported environment
(`polars 1.36`, `numpy 2`, `scikit-learn 1.6`, `scipy 1.13`) are in
`requirements.txt`.

## Citation

```bibtex
@inproceedings{eom2026mvps,
  title     = {Multi-View Pattern Selection for Cohort-Robust Sequential Pattern Mining},
  author    = {Eom, Eunsang and Kim, Jong-Kook and Park, Kiyoung},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
  address   = {Shenyang, China},
  note      = {To appear}
}
```

Update the `note` field (and add volume/page numbers) once the IEEE
proceedings entry is assigned a DOI.

## License

MIT — see `LICENSE`.
