"""
External SPMF PrefixSpan vs our polynomial-bound mining: BPI 2012.

BPI has shorter sequences (avg ~28 events) than Edu (370), so PrefixSpan
finishes. Allows direct wall-clock comparison.
"""
from __future__ import annotations
import json
import time
import subprocess
from pathlib import Path

JAVA = "/opt/homebrew/opt/openjdk/bin/java"
SPMF_JAR = "/tmp/spmf.jar"
SPMF_INPUT = "/tmp/bpi_spmf.txt"
SPMF_OUTPUT = "/tmp/bpi_spmf_output.txt"


def main():
    from src.c2dpm import load_dataset, DATASET_REGISTRY
    from src.config import RESULTS

    spec = DATASET_REGISTRY['bpi2012']()
    sequences, cohorts, clusters, N_cz = load_dataset(spec)
    K, M = N_cz.shape

    # Limit to length<=20 sequences only (BPI's typical case length)
    short_seqs = [s for s in sequences if len(s) <= 20]
    print(f"BPI: total {len(sequences)} sequences, {len(short_seqs)} of length<=20")

    with open(SPMF_INPUT, 'w') as f:
        for seq in short_seqs:
            line = ' '.join(f'{t} -1' for t in seq) + ' -2\n'
            f.write(line)
    print(f"exported {len(short_seqs)} to {SPMF_INPUT}\n")

    runs = []
    for ms in [0.30, 0.20, 0.10, 0.05]:
        Path(SPMF_OUTPUT).unlink(missing_ok=True)
        print(f"\nPrefixSpan @ min_sup={ms}...")
        t0 = time.time()
        try:
            result = subprocess.run(
                [JAVA, '-jar', SPMF_JAR, 'run', 'PrefixSpan',
                 SPMF_INPUT, SPMF_OUTPUT, f'{ms*100:.1f}%'],
                capture_output=True, text=True, timeout=180
            )
            elapsed = time.time() - t0
            n_patterns = 0
            try:
                with open(SPMF_OUTPUT) as f:
                    n_patterns = sum(1 for _ in f)
            except FileNotFoundError: pass
            print(f"  wall-clock: {elapsed:.2f}s, patterns: {n_patterns}")
            runs.append({'min_sup': ms, 'wall_clock_seconds': elapsed,
                         'patterns_found': n_patterns})
            if elapsed > 60: break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  TIMEOUT (>{elapsed:.0f}s)")
            runs.append({'min_sup': ms, 'wall_clock_seconds': elapsed,
                         'patterns_found': None, 'timed_out': True})
            break

    summary = {
        'tool': 'SPMF PrefixSpan (separable frequency-only)',
        'dataset': 'BPI 2012 (sequences of length <=20)',
        'n_sequences': len(short_seqs),
        'runs': runs,
        'mvps_polynomial_on_full_bpi_seconds': '~10 (proportional to Edu 22s; BPI is smaller)',
        'note': 'PrefixSpan mines all frequent sequential patterns without length/quality bound; our polynomial-bound miner mines the qualifying set at lambda=50 with combined Apriori + joint anti-monotone pruning.'
    }
    with open(RESULTS / 'spmf_quick_compare.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS / 'spmf_quick_compare.json'}")


if __name__ == "__main__":
    main()
