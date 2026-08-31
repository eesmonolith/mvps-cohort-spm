# Matched-budget-70 baseline detail

Full per-corpus, per-method detail behind experiments.tex's "Does a single view catch up at matched budget?" paragraph (Section IV). Each single-view baseline is re-evaluated at top-K=70 patterns (matching MVPS's realised union size of 50-70 patterns) instead of the paper's default top-50. Generated from `results/camera_ready/budget70_baselines.json` by `src/make_budget70_table.py` -- see `camera-ready/CHANGES_PHASE3.md` for why this table was moved here from the PDF appendix.

Bold marks the best single view at top-70 per corpus. `gap` = MVPS mean AUC minus the best single view at top-70, in percentage points (positive = MVPS still ahead at matched budget). `p` is the one-sided paired Wilcoxon signed-rank p-value for MVPS > best-at-70 on the same LOCO folds.

## Edu (edu_kor)

K=3  M=4  N=5059  folds=3

| Method | mean AUC@70 | std |
|---|---:|---:|
| freq-only | 0.6978 | 0.0254 |
| stab-only | 0.7759 | 0.0285 |
| discrim-only | 0.7775 | 0.0299 |
| intersect | 0.7760 | 0.0316 |
| $S^{(v1)}$ | 0.7766 | 0.0304 |
| $\min_c \mathrm{IG}_c$ | 0.7828 | 0.0247 |
| **$S$ ($\lambda{=}50$)** | 0.7859 | 0.0246 |
| **MVPS** (4-view union) | **0.7954** | 0.0196 |

Best single view @70: **r1_lam50** (0.7859). MVPS gap: **+0.95pp**. Wilcoxon one-sided p (MVPS > best@70): 0.1250 (n=3 LOCO folds).

## EduB (codle_hashed)

K=4  M=4  N=14281  folds=4

| Method | mean AUC@70 | std |
|---|---:|---:|
| freq-only | 0.9278 | 0.0107 |
| stab-only | 0.9293 | 0.0122 |
| discrim-only | 0.9129 | 0.0126 |
| **intersect** | 0.9374 | 0.0100 |
| $S^{(v1)}$ | 0.9129 | 0.0127 |
| $\min_c \mathrm{IG}_c$ | 0.9153 | 0.0130 |
| $S$ ($\lambda{=}50$) | 0.9238 | 0.0123 |
| **MVPS** (4-view union) | **0.9358** | 0.0115 |

Best single view @70: **intersect** (0.9374). MVPS gap: **-0.16pp**. Wilcoxon one-sided p (MVPS > best@70): 0.6875 (n=4 LOCO folds).

## BPI (bpi2012)

K=5  M=3  N=9259  folds=5

| Method | mean AUC@70 | std |
|---|---:|---:|
| freq-only | 0.9238 | 0.0064 |
| **stab-only** | 0.9845 | 0.0019 |
| discrim-only | 0.8240 | 0.0104 |
| intersect | 0.9845 | 0.0019 |
| $S^{(v1)}$ | 0.8240 | 0.0104 |
| $\min_c \mathrm{IG}_c$ | 0.8240 | 0.0104 |
| $S$ ($\lambda{=}50$) | 0.8240 | 0.0104 |
| **MVPS** (4-view union) | **0.9845** | 0.0019 |

Best single view @70: **stab_only** (0.9845). MVPS gap: **-0.00pp**. Wilcoxon one-sided p (MVPS > best@70): 0.6726 (n=5 LOCO folds).

## Sepsis (sepsis)

K=4  M=3  N=671  folds=4

| Method | mean AUC@70 | std |
|---|---:|---:|
| **freq-only** | 0.7407 | 0.0173 |
| stab-only | 0.7401 | 0.0182 |
| discrim-only | 0.7393 | 0.0176 |
| intersect | 0.7145 | 0.0202 |
| $S^{(v1)}$ | 0.7390 | 0.0178 |
| $\min_c \mathrm{IG}_c$ | 0.7389 | 0.0180 |
| $S$ ($\lambda{=}50$) | 0.7393 | 0.0178 |
| **MVPS** (4-view union) | **0.7348** | 0.0120 |

Best single view @70: **freq_only** (0.7407). MVPS gap: **-0.59pp**. Wilcoxon one-sided p (MVPS > best@70): 0.8125 (n=4 LOCO folds).

## OULAD (oulad)

K=7  M=4  N=28547  folds=7

| Method | mean AUC@70 | std |
|---|---:|---:|
| freq-only | 0.6170 | 0.0462 |
| stab-only | 0.6170 | 0.0463 |
| discrim-only | 0.6170 | 0.0462 |
| intersect | 0.5000 | 0.0000 |
| $S^{(v1)}$ | 0.6168 | 0.0463 |
| $\min_c \mathrm{IG}_c$ | 0.6168 | 0.0462 |
| **$S$ ($\lambda{=}50$)** | 0.6171 | 0.0462 |
| **MVPS** (4-view union) | **0.6175** | 0.0461 |

Best single view @70: **r1_lam50** (0.6171). MVPS gap: **+0.04pp**. Wilcoxon one-sided p (MVPS > best@70): 0.7656 (n=7 LOCO folds).

## HB (hospital_billing)

K=12  M=3  N=43829  folds=12

| Method | mean AUC@70 | std |
|---|---:|---:|
| freq-only | 0.9885 | 0.0096 |
| stab-only | 0.9885 | 0.0096 |
| discrim-only | 0.9885 | 0.0096 |
| intersect | 0.9885 | 0.0096 |
| $S^{(v1)}$ | 0.9885 | 0.0096 |
| **$\min_c \mathrm{IG}_c$** | 0.9888 | 0.0091 |
| $S$ ($\lambda{=}50$) | 0.9885 | 0.0096 |
| **MVPS** (4-view union) | **0.9883** | 0.0096 |

Best single view @70: **min_ig** (0.9888). MVPS gap: **-0.05pp**. Wilcoxon one-sided p (MVPS > best@70): 0.4646 (n=12 LOCO folds).
