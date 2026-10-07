"""Summarize raw per-repetition GEMM timings without discarding the raw CSV."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    groups: dict[tuple[str, int, int, int, str], list[float]] = defaultdict(list)
    with args.raw.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["method"], int(row["m"]), int(row["n"]),
                   int(row["k"]), row["timing"])
            groups[key].append(float(row["per_call_ms"]))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "m", "n", "k", "timing", "repetitions",
                         "median_ms", "p10_ms", "p90_ms", "gflops_at_median"])
        for (method, m, n, k, timing), samples in sorted(groups.items()):
            median = statistics.median(samples)
            writer.writerow([method, m, n, k, timing, len(samples),
                             f"{median:.6f}", f"{percentile(samples, .1):.6f}",
                             f"{percentile(samples, .9):.6f}",
                             f"{2 * m * n * k / (median * 1e6):.3f}"])


if __name__ == "__main__":
    main()
