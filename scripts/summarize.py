"""Summarize raw per-repetition GEMM timings without discarding the raw CSV."""

import argparse
import csv
import statistics
import math
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def validate_rows(rows):
    """Reject malformed measurements before generating performance summaries."""
    seen = set()
    if not rows:
        raise ValueError("Timing CSV contains no measurements")
    for row in rows:
        shape = tuple(int(row[d]) for d in ("m", "n", "k"))
        iterations = int(row["iterations"])
        repetition = int(row["repetition"])
        batch, per_call = float(row["batch_ms"]), float(row["per_call_ms"])
        if min(shape) <= 0 or iterations <= 0 or repetition < 0 or int(row["warmups"]) < 0:
            raise ValueError("Dimensions and iterations must be positive; counters nonnegative")
        if not all(math.isfinite(x) and x > 0 for x in (batch, per_call)):
            raise ValueError("Timings must be finite and positive")
        if not math.isclose(batch / iterations, per_call, rel_tol=3e-5, abs_tol=1e-8):
            raise ValueError("Batch and per-call timings disagree")
        expected = "steady_clock" if row["method"] == "cpu" else "cuda_event"
        if row["method"] not in {"cpu", "naive", "tiled", "cublas_fp32_pedantic"} or row["timing"] != expected:
            raise ValueError("Unexpected method or timing source")
        key = (row["method"], *shape, repetition)
        if key in seen:
            raise ValueError("Duplicate repetition for method and shape")
        seen.add(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    groups: dict[tuple[str, int, int, int, str], list[float]] = defaultdict(list)
    with args.raw.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
        validate_rows(rows)
        for row in rows:
            key = (row["method"], int(row["m"]), int(row["n"]),
                   int(row["k"]), row["timing"])
            groups[key].append(float(row["per_call_ms"]))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
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
