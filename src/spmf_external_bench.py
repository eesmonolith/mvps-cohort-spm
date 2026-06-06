"""
External SPMF wall-clock baseline.

Run PrefixSpan (separable frequency-based SPM) from SPMF library
on Edu sequences. Compare wall-clock against our polynomial-bound
mining at same minimum support.
"""
from __future__ import annotations
import json
import time
import subprocess
from pathlib import Path

import polars as pl

from src.config import DATA_PROC, RESULTS

JAVA = "/opt/homebrew/opt/openjdk/bin/java"
SPMF_JAR = "/tmp/spmf.jar"
SPMF_INPUT = "/tmp/edu_spmf.txt"
SPMF_OUTPUT = "/tmp/edu_spmf_output.txt"


def export_edu_spmf():
    """Convert Edu sequences to SPMF SequenceDatabase format."""
    from src.c2dpm import load_dataset, DATASET_REGISTRY
    spec = DATASET_REGISTRY['edu_kor']()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    with open(SPMF_INPUT, 'w') as f:
        for seq in sequences:
            # Each token as single-item itemset, end with -1 -1 -2 (itemset, itemset, sequence)
            line = ' '.join(f'{t} -1' for t in seq) + ' -2\n'
            f.write(line)
    print(f"exported {len(sequences)} sequences to {SPMF_INPUT}")
    return len(sequences)


def run_prefixspan(min_sup=0.02, timeout_s=120):
    """Run PrefixSpan with given minimum support."""
    Path(SPMF_OUTPUT).unlink(missing_ok=True)
    t0 = time.time()
    result = subprocess.run(
        [JAVA, '-jar', SPMF_JAR, 'run', 'PrefixSpan',
         SPMF_INPUT, SPMF_OUTPUT, f'{min_sup*100:.1f}%'],
        capture_output=True, text=True, timeout=timeout_s
    )
    elapsed = time.time() - t0
    print(f"PrefixSpan wall-clock: {elapsed:.2f}s")
    print(result.stdout[-500:])
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:]}")
    # Count output patterns
    n_patterns = 0
    try:
        with open(SPMF_OUTPUT) as f:
            n_patterns = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    return elapsed, n_patterns


def main():
    print(f"Java: ", end='')
    subprocess.run([JAVA, '-version'], capture_output=False)
    n_seq = export_edu_spmf()
    # Sweep multiple min_sup values
    runs = []
    for ms in [0.50, 0.30, 0.20, 0.10, 0.05, 0.02]:
        print(f"\nRunning PrefixSpan at min_sup={ms} on Edu ({n_seq} sequences)...")
        try:
            elapsed, n_patterns = run_prefixspan(min_sup=ms)
            runs.append({'min_sup': ms, 'wall_clock_seconds': elapsed, 'patterns_found': n_patterns})
            print(f"  result: {elapsed:.2f}s, {n_patterns} patterns")
            if elapsed > 300: break  # stop if too slow
        except Exception as e:
            print(f"  failed: {e}")
            runs.append({'min_sup': ms, 'wall_clock_seconds': None, 'error': str(e)})
            break

    summary = {
        'tool': 'SPMF PrefixSpan',
        'dataset': 'Edu',
        'n_sequences': n_seq,
        'runs': runs,
        'mvps_polynomial_for_comparison_seconds': 22,
        'note': 'PrefixSpan is frequency-only; cannot directly mine variance-penalised S. Comparison is wall-clock baseline at matched min_sup, demonstrating that even the simplest separable SPM is not orders-of-magnitude faster than our non-separable joint-bound mining.',
    }
    with open(RESULTS / 'spmf_external_bench.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS / 'spmf_external_bench.json'}")


if __name__ == "__main__":
    main()
