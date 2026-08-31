"""
Reproduction script for the "Equivalence to the per-fold oracle" paragraph
of the MVPS paper (Section IV, "Equivalence to the per-fold oracle").

--------------------------------------------------------------------------
Provenance note (read this first)
--------------------------------------------------------------------------
The original camera-ready audit (CHANGES_PHASE4.md, "Should-fix 5") found
that no script or notebook in the project computed the oracle/TOST numbers
quoted in the paper (Edu +0.98pp, EduB +0.63pp, BPI 0.00pp, Sepsis -1.04pp,
pooled mean 0.874 for both MVPS and oracle, TOST p=0.0003 at epsilon=0.01).
Those numbers predate the results/camera_ready/ JSON-backed verification
pipeline and were "unchanged from the accepted submission" with no
recoverable original script.

This script is a *post-hoc reconstruction*, written for the public release,
that derives the oracle definition directly from the paper's own text
("the oracle single-view selector that picks, per fold, whichever
criterion scores best: oracle_f = max_V AUC_V(f) over the seven baselines
of Table tab:multi-dataset") and computes it from the two result files
that already ship with this repository:

  - results/multi_dataset_transfer.json  -- per-fold AUC for the seven
    single-view baselines (freq_only, stab_only, discrim_only, intersect,
    v1_pooled, r1_lam50, min_ig) on edu_kor, codle_hashed, bpi2012, sepsis.
  - results/r8_ensemble.json             -- per-fold AUC for MVPS
    (method "r8_union_K20each") on the same four corpora.

Running this script reproduces, from those two files alone:
  - per-corpus mean(MVPS - oracle) in percentage points: matches the
    paper's Edu +0.98, EduB +0.63, BPI 0.00, Sepsis -1.04 to the reported
    2 decimal places.
  - the pooled 16-fold mean AUC for both MVPS and oracle (0.874 for both).
  - a paired TOST equivalence test at epsilon=0.01 giving p=0.0003,
    matching the paper's p_TOST=0.0003 to 4 decimal places.

Caveat: the paper also reports a 95% CI on the pooled mean difference of
[-0.005, +0.006]. A paired-t CI computed here on the same 16 folds gives
a very close but not bit-identical interval (see the printed output and
the "ci_95_paired_t" field of the written JSON) -- the exact CI
methodology used in the original (pre-camera-ready) run was not
recoverable. The point estimate, the per-corpus deltas, and the TOST
p-value all reproduce; only the last-digit CI endpoints are an
approximation. This is flagged explicitly rather than silently rounded
to match.

Usage:
    python -m src.oracle_tost_reproduction

Requires: numpy, scipy (both already in requirements.txt; the paired
TOST test is implemented inline below with scipy.stats so this script
adds no new dependency).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.config import RESULTS

# The "seven baselines of Table tab:multi-dataset", as named in
# results/multi_dataset_transfer.json's per-fold records.
BASELINES = [
    "freq_only", "stab_only", "discrim_only", "intersect",
    "v1_pooled", "r1_lam50", "min_ig",
]

# The four "non-saturated" corpora used for the oracle comparison
# (OULAD and Hospital Billing are excluded as saturated regimes --
# see experiments.tex, "Saturation regimes").
CORPORA = ["edu_kor", "codle_hashed", "bpi2012", "sepsis"]
DISPLAY = {"edu_kor": "Edu", "codle_hashed": "EduB", "bpi2012": "BPI", "sepsis": "Sepsis"}

EPS_TOST = 0.01
OUT_PATH = RESULTS / "camera_ready" / "oracle_tost_reproduction.json"


def load(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def ttost_paired(x, y, low, upp):
    """Two one-sided tests (TOST) for paired equivalence, low/upp on
    (x - y). Returns (p_tost, (t1, p1), (t2, p2)). Equivalent to
    statsmodels.stats.weightstats.ttost_paired without adding a hard
    dependency; included inline so this script has no import surprises
    beyond numpy/scipy.
    """
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    # H0_1: mean <= low  vs  H1_1: mean > low
    t1 = (mean - low) / se
    p1 = stats.t.sf(t1, n - 1)
    # H0_2: mean >= upp  vs  H1_2: mean < upp
    t2 = (mean - upp) / se
    p2 = stats.t.cdf(t2, n - 1)
    p_tost = max(p1, p2)
    return p_tost, (t1, p1), (t2, p2)


def main():
    m = load("multi_dataset_transfer.json")
    r8 = load("r8_ensemble.json")

    per_corpus = {}
    all_mvps, all_oracle = [], []

    for corpus in CORPORA:
        folds = m[corpus]["folds"]
        oracle = np.array([max(f[b] for b in BASELINES) for f in folds])
        mvps = np.array(r8[corpus]["r8_union_K20each"]["per_fold"])
        assert len(mvps) == len(oracle), f"{corpus}: fold count mismatch"

        delta = mvps - oracle
        per_corpus[corpus] = {
            "display_name": DISPLAY[corpus],
            "n_folds": len(delta),
            "mvps_per_fold": mvps.tolist(),
            "oracle_per_fold": oracle.tolist(),
            "delta_per_fold_pp": (delta * 100).tolist(),
            "mean_delta_pp": float(delta.mean() * 100),
        }
        all_mvps.extend(mvps.tolist())
        all_oracle.extend(oracle.tolist())

    all_mvps = np.array(all_mvps)
    all_oracle = np.array(all_oracle)
    n_total = len(all_mvps)

    p_tost, (t1, p1), (t2, p2) = ttost_paired(all_mvps, all_oracle, -EPS_TOST, EPS_TOST)

    diffs = all_mvps - all_oracle
    mean_diff = float(diffs.mean())
    se_diff = float(diffs.std(ddof=1) / np.sqrt(n_total))
    ci_lo, ci_hi = stats.t.interval(0.95, n_total - 1, loc=mean_diff, scale=se_diff)

    result = {
        "description": (
            "Reconstruction of the 'Equivalence to the per-fold oracle' "
            "paragraph (experiments.tex). See module docstring for "
            "provenance and the CI caveat."
        ),
        "baselines_used": BASELINES,
        "corpora": CORPORA,
        "per_corpus": per_corpus,
        "pooled": {
            "n_folds": n_total,
            "mean_mvps": float(all_mvps.mean()),
            "mean_oracle": float(all_oracle.mean()),
            "mean_diff": mean_diff,
            "ci_95_paired_t": [float(ci_lo), float(ci_hi)],
            "ci_note": (
                "Paired-t 95% CI on (MVPS - oracle) over the pooled 16 "
                "folds. Close to but not bit-identical with the paper's "
                "reported [-0.005, +0.006] -- see module docstring."
            ),
        },
        "tost": {
            "epsilon": EPS_TOST,
            "p_tost": float(p_tost),
            "lower_test": {"t": float(t1), "p": float(p1)},
            "upper_test": {"t": float(t2), "p": float(p2)},
        },
        "matches_paper_claim": {
            "claimed_delta_pp": {"edu_kor": 0.98, "codle_hashed": 0.63,
                                  "bpi2012": 0.00, "sepsis": -1.04},
            "reproduced_delta_pp": {c: round(per_corpus[c]["mean_delta_pp"], 2)
                                     for c in CORPORA},
            "claimed_pooled_mean": 0.874,
            "reproduced_pooled_mean_mvps": round(float(all_mvps.mean()), 3),
            "reproduced_pooled_mean_oracle": round(float(all_oracle.mean()), 3),
            "claimed_p_tost": 0.0003,
            "reproduced_p_tost": round(float(p_tost), 4),
            "claimed_ci_95": [-0.005, 0.006],
            "reproduced_ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_PATH}")

    print("\n=== Per-corpus MVPS - oracle (pp) ===")
    for c in CORPORA:
        pc = per_corpus[c]
        print(f"  {DISPLAY[c]:6s} n={pc['n_folds']}  mean_delta={pc['mean_delta_pp']:+.2f}pp"
              f"  (paper: {result['matches_paper_claim']['claimed_delta_pp'][c]:+.2f}pp)")

    print("\n=== Pooled (16 folds) ===")
    print(f"  mean MVPS   = {all_mvps.mean():.4f}  (paper: 0.874)")
    print(f"  mean oracle = {all_oracle.mean():.4f}  (paper: 0.874)")
    print(f"  TOST p (eps={EPS_TOST}) = {p_tost:.6f}  (paper: 0.0003)")
    print(f"  95% CI on mean diff = [{ci_lo:.4f}, {ci_hi:.4f}]  (paper: [-0.005, 0.006])")


if __name__ == "__main__":
    main()
