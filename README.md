# MVPS: Multi-View Pattern Selection for Cohort-Robust Sequential Pattern Mining

Reference implementation accompanying the ICDM 2026 paper
*Multi-View Pattern Selection for Cohort-Robust Sequential Pattern Mining*.

Anonymous repository — author identities removed for double-blind review.

## What this is

A sequential pattern mining method that selects patterns robust to
cohort shift. Instead of optimising a single non-separable quality
measure, MVPS mines candidate patterns **once** and then **ranks them
under multiple complementary views**, taking the **union of the top-K
from each view** as the selected pattern set:

1. **stability** — low cross-cohort variance of per-cohort information gain;
2. **S (mean − λ·variance)** — the cohort-conditional mean-variance score;
3. **S_CV** — a coefficient-of-variation–penalised variant;
4. **discrim** — pooled discriminative information gain.

The union (mine-once, rank-many) gives downside protection: on the
adversarial synthetic stress test, the worst held-out cohort score for
stability-only collapses to ≈0.468 while the MVPS union stays ≈0.493,
i.e. MVPS never loses to the worst single view. Because all views are
computed from the same per-cohort × per-cluster count tensor, mining is
performed only once and the multi-view ranking adds negligible cost.

## Repository layout

```
src/
  config.py                 Project paths, event vocab, cohort registry
  data_loader.py            Raw CSV loaders (multi-line code handled)
  build_sequences.py        Per-student event-sequence construction
  cluster_labels.py         KMeans behavioural clusters

  scoring.py                v1 pooled-product score (baseline)
  scoring_r1.py             Cohort-conditional IG + mean-variance score S
  scoring_r7.py             S_CV (coefficient-of-variation) scoring
  joint_bound_r1.py         Anti-monotone joint bound (mean-var box max)
  joint_bound_facewise.py   Face-wise bound variant
  joint_bound_polyk.py      Polynomial-K bound (high-K corpora)
  c2dpm.py                  Dataset registry + level-wise miner

  run_r8_ensemble.py        MVPS multi-view union (mine-once, rank-many)
  run_r7_sweep.py           S_CV view sweep
  multi_ktop_lambda_sweep.py  K_top × λ grid for the union
  default_killer_v5.py      Adversarial synthetic downside-protection demo
  r1_synthetic_stress.py    Synthetic bound-gap stress test
  synthetic_experiment.py   Synthetic generator (generate / contains)

  transfer_experiment.py    Leave-one-cohort-out transfer AUC core
  multi_dataset_transfer.py Cross-domain transfer AUC across corpora
  run_edu_locoCluster_transfer.py  Edu leave-one-cluster-out transfer
  bench_facewise_walltime.py  Wall-clock benchmark (facewise vs vertex)
  bench_polyk_scaling.py    Poly-K vs face-wise bound scaling benchmark

  baselines.py              Apriori-only / single-axis / intersect baselines
  external_baselines.py     External miner baselines
  make_figures_r1.py        Figures for the paper

  *_adapter.py              Public-corpus adapters (see Datasets below)

data/
  raw/                      Private raw corpora (NOT redistributed; see env vars)
  processed/                Cached parquet outputs (gitignored)
  external/                 Public datasets (RetailRocket, OULAD, BPI, etc.)

results/                    CSV/JSON experiment outputs
paper/                      LaTeX source + figures
```

## Datasets

The synthetic experiments and the core algorithm require **no external
data**. Public-corpus adapters expect their files under
`data/external/<name>/`:

| Adapter | Corpus | Expected path |
|---------|--------|---------------|
| `retailrocket_adapter.py` | RetailRocket (Kaggle) | `data/external/retailrocket/events.csv` |
| `oulad_adapter.py` | OULAD | `data/external/OULAD/` |
| `bpi2012_adapter.py`, `bpi2012_amount_adapter.py` | BPI Challenge 2012 | `data/external/bpi2012/BPIC2012.xes` |
| `hospital_billing_adapter.py`, `hospital_billing_monthly_adapter.py` | Hospital Billing (4TU) | `data/external/hospital_billing/HospitalBilling.xes` |
| `sepsis_adapter.py` | Sepsis Cases (4TU) | `data/external/sepsis/Sepsis_Cases.xes` |

The private Edu corpora are not redistributed. Their location is
resolved from environment variables (default: `data/raw/`):

- `HPIC_DATA_RAW` — root of the private raw corpora (default `data/raw`).
- `HPIC_CODLE_RAW` — the hashed Codle K-12 Python corpus
  (default `$HPIC_DATA_RAW/codle_K12_python`).

## Reproducing the main results

All commands below run from the repository root with `PYTHONPATH=.`.
The synthetic results are deterministic given `seed=42` and need no
external data.

| Result | Command |
|--------|---------|
| K=6 downside-protection (stab worst ≈0.468 vs MVPS worst ≈0.493) | `python -m src.default_killer_v5` |
| Synthetic bound-gap table (rel_med 0.4–1.2%, strict>0 100%) | `python -m src.r1_synthetic_stress` |
| MVPS multi-view union (mine-once, rank-many) | `python -m src.run_r8_ensemble` |
| Cross-domain transfer AUC across corpora | `python -m src.multi_dataset_transfer` |
| Wall-clock benchmark (facewise vs vertex bound) | `python -m src.bench_facewise_walltime` |
| Poly-K bound scaling benchmark | `python -m src.bench_polyk_scaling` |

See `REPRODUCIBILITY.md` for the full reproduction protocol including
the (non-redistributed) Edu pipeline.

## Dependencies

```
numpy >= 2
polars >= 1.36
scipy >= 1.13
scikit-learn >= 1.6
matplotlib >= 3.8
pyarrow >= 14.0       # parquet I/O backend
pandas >= 2.1         # CSV loaders / public-corpus adapters
tqdm >= 4.66
pm4py >= 2.7          # XES parsing for BPI / Hospital Billing / Sepsis only
```

Install with `python -m pip install -r requirements.txt`. The core
mining and synthetic experiments depend only on numpy / polars / scipy /
scikit-learn; `pm4py` is needed solely for the XES process-mining
adapters. All results are deterministic given `seed=42`.

## License

The released code is available under the MIT licence; the RetailRocket
data is redistributed under the original Kaggle CC-BY-NC-SA-4.0 licence;
the Edu corpus is not redistributed.
