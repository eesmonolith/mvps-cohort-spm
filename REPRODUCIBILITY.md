# Reproducibility Checklist — MVPS

This document accompanies the paper *Multi-View Pattern Selection for
Cohort-Robust Sequential Pattern Mining* (ICDM 2026 submission).
The complete code, configuration manifest, and synthetic generator
are released at the anonymous URL given in §I of the paper.

## ICDM Reproducibility Checklist Responses

### Models and Algorithms

- **[YES]** Mathematical setting, algorithm, and model is clearly
  described. See §III (Problem statement, cohort-conditional IG,
  $S$ definition, joint anti-monotone bound, and Algorithm 1).
- **[YES]** Assumptions made (e.g., on input data, computational
  resources) are clearly stated. See §III.A (database schema),
  §III.E (choice of $\lambda$ scaling), and §V.E (limitations on
  $KM$ scale of the vertex-enumeration bound).
- **[YES]** Mathematical claims are formally proved.
  Theorem 1 (non-separability), Theorem 2 (anti-monotone),
  Theorem 3 (joint dominates naive), and Proposition 1
  (validity) all carry constructive or direct proofs in §III.
- **[YES]** Pseudocode is provided. Algorithm 1.
- **[YES]** Time- and space- complexity analysis is provided. §III.F.
- **[YES]** Source code is provided. See anonymous URL.
- **[YES]** Source code links to specific commit hashes / versions
  of any libraries used. See `requirements.txt` (polars 1.36,
  numpy 2.1.3, scipy 1.13.1, scikit-learn 1.6.1).

### Datasets

- **[YES]** The data used in the experiments are clearly described,
  including how data was acquired, what preprocessing was done.
  See §IV.A.
- **[YES]** Public datasets used are cited and a link is provided.
  RetailRocket: `retailrocket/ecommerce-dataset` on Kaggle.
- **[PARTIAL]** Non-public datasets are accompanied by a data
  statement. The Edu corpus is from a real LLM-assisted learning
  platform; identifiers are hashed; access to the raw event log
  requires a data-use agreement. The preprocessing script in the
  released repo reproduces the trajectory parquet files from the
  raw CSVs verbatim (deterministic given the seed) but the raw
  CSVs are not redistributed.
- **[YES]** The synthetic data generator is provided.
  See `src/synthetic_experiment.py` in the released repo.

### Experiments

- **[YES]** Random seed values are explicitly specified.
  $\mathrm{seed}=42$ throughout.
- **[YES]** Number of runs / repetitions is specified. Transfer
  AUC reported as mean ± std across 3 leave-one-out folds; bound
  tightness on 184 sampled patterns (Table III); synthetic stress
  test (omitted to fit page budget but in repo).
- **[YES]** Hardware/compute environment is specified. All
  reported runtimes measured on a single CPU thread (Python 3.9,
  Apple M-series); no GPU required.
- **[YES]** Train/validation/test splits and threshold values are
  specified. Leave-one-cohort-out for transfer (held-out semester
  rotates over $\{2024 \text{Spring}, 2024 \text{Fall}, 2025
  \text{Spring}\}$); thresholds
  $\theta_{\sup}=0.02, \lambda=50, \tau=0.06$ for the main mining
  runs (§IV.D).

### Reports

- **[YES]** Quantitative results are reported with confidence
  intervals or standard deviations.
- **[YES]** Tables and figures are self-contained with
  captions explaining the metric and any abbreviations.

## How to reproduce

All commands run from the repository root with `PYTHONPATH=.`.

### A. Synthetic results (no external data required)

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Downside-protection demo: K=6 worst-cohort scores
#   expect stab worst ≈ 0.468, MVPS worst ≈ 0.493
PYTHONPATH=. python3 -m src.default_killer_v5

# Bound-gap stress test (Table)
#   expect rel_med 0.4–1.2%, strict>0 = 100% across all (rho_c, rho_d)
PYTHONPATH=. python3 -m src.r1_synthetic_stress

# Poly-K bound scaling benchmark
PYTHONPATH=. python3 -m src.bench_polyk_scaling
```

### B. Corpus experiments (require the corresponding datasets)

```bash
# 1. Build sequences + cluster labels from the (private) Edu raw CSVs.
#    Set MVPS_DATA_RAW / MVPS_EDUB_RAW to point at the raw corpora.
PYTHONPATH=. python3 -m src.build_sequences
PYTHONPATH=. python3 -m src.cluster_labels

# 2. Public-corpus adapters (place files under data/external/<name>/ first):
PYTHONPATH=. python3 -m src.retailrocket_adapter
PYTHONPATH=. python3 -m src.oulad_adapter
PYTHONPATH=. python3 -m src.bpi2012_adapter
PYTHONPATH=. python3 -m src.hospital_billing_adapter
PYTHONPATH=. python3 -m src.sepsis_adapter

# 3. MVPS multi-view union (mine-once, rank-many)
PYTHONPATH=. python3 -m src.run_r8_ensemble

# 4. Cross-domain transfer AUC across corpora
PYTHONPATH=. python3 -m src.multi_dataset_transfer

# 5. Wall-clock benchmark (facewise vs vertex bound)
PYTHONPATH=. python3 -m src.bench_facewise_walltime

# 6. Paper figures
PYTHONPATH=. python3 -m src.make_figures_r1
```

All results are deterministic given seed=42.
