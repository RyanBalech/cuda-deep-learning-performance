"""Plot the retained GPU summary; no new hardware measurements are generated."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "results/summary_gemm_2026-10-07.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for ax, size in zip(axes, (512, 1024)):
    subset = [next(row for row in rows if row["method"] == method and int(row["m"]) == size)
              for method in ("naive", "tiled", "cublas_fp32_pedantic")]
    med = [float(row["median_ms"]) for row in subset]
    errors = [[value - float(row["p10_ms"]) for value, row in zip(med, subset)],
              [float(row["p90_ms"]) - value for value, row in zip(med, subset)]]
    ax.bar(["Direct", "Tiled", "cuBLAS FP32"], med, yerr=errors,
           color=["#64748b", "#2563eb", "#0d9488"], capsize=4, width=.6)
    ax.set_title(f"{size} × {size} × {size}")
    ax.set_ylabel("Kernel latency (ms)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
fig.suptitle("RTX 4050 Laptop GPU · median and p10–p90 · 15 timing batches", fontsize=11)
fig.tight_layout()
plt.rcParams["svg.hashsalt"] = "gemm-study"
fig.savefig(ROOT / "docs/gemm_latency.svg", metadata={"Date": None})
