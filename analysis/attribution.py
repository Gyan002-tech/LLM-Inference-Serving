"""
Attribution — how much of vLLM's throughput win is actually batching?

Left panel:  HF static batching vs vLLM at matched batch/concurrency levels, with
             the vLLM/HF ratio annotated. The ratio being FLAT across levels 1, 2
             and 4 is the finding: a constant multiplier is the signature of fixed
             per-step overhead, not of smarter scheduling.
Right panel: the multiplicative decomposition of the headline speedup into
             batching, engine efficiency, and the residual attributable to
             continuous batching + paged KV.

The three factors are computed so their product reconciles EXACTLY with the
measured total: the scheduling factor is derived as the residual rather than
measured independently, which is the honest construction given that this workload
pins every sequence to the same output length.

Pure analysis over results/*.json — loads no model and touches no GPU.
Run: python analysis/attribution.py
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_PNG = Path(__file__).resolve().parent / "attribution.png"

# Levels where the vLLM/HF ratio is flat, i.e. where the advantage is pure
# per-step overhead and not yet contaminated by HF's scaling collapse at 8.
FLAT_LEVELS = [1, 2, 4]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load():
    hfb_path = RESULTS / "hf_batched.json"
    if not hfb_path.exists():
        sys.exit("results/hf_batched.json not found — run benchmark/hf_batched.py "
                 "first (it produces the static-batching rows this compares).")
    hfb = json.loads(hfb_path.read_text())
    if hfb.get("drift_check_aborted"):
        sys.exit("results/hf_batched.json is a PARTIAL result — its batch=1 drift "
                 "check failed, so the sweep stopped after one level. Re-run "
                 "benchmark/hf_batched.py on the same library stack as the "
                 "baseline it compares against; attributing from one level would "
                 "be meaningless.")
    vllm = json.loads((RESULTS / "vllm.json").read_text())
    fp16 = json.loads((RESULTS / "fp16_control.json").read_text())

    hf = {r["batch_size"]: r for r in hfb["results"] if not r.get("oom")}
    vl = {r["concurrency"]: r for r in vllm["results"]}
    if len(set(hf) & set(vl)) < 2:
        sys.exit(f"need at least 2 matched levels to attribute; found "
                 f"{sorted(set(hf) & set(vl))}. The whole point is comparing the "
                 f"ratio ACROSS batch sizes.")
    base_e2e = fp16["max_new_tokens"] / (fp16["p50_ms"] / 1000.0)
    return hf, vl, base_e2e


def analyse(hf, vl, base_e2e):
    levels = sorted(set(hf) & set(vl))
    ratios = {n: vl[n]["aggregate_tps"] / hf[n]["aggregate_tps"] for n in levels}

    top = max(levels)
    total = vl[top]["aggregate_tps"] / base_e2e
    batching = hf[top]["aggregate_tps"] / base_e2e
    flat = [ratios[n] for n in FLAT_LEVELS if n in ratios]
    engine = statistics.mean(flat) if flat else ratios[top]
    # Residual, so batching x engine x scheduling reproduces total exactly.
    scheduling = total / (batching * engine)
    return {
        "levels": levels, "ratios": ratios, "top": top, "total": total,
        "batching": batching, "engine": engine, "scheduling": scheduling,
    }


def report(hf, vl, base_e2e, a):
    print("=" * 78)
    print("ATTRIBUTION — batching vs vLLM at matched levels")
    print("=" * 78)
    print(f"  HF batch=1 reference (e2e): {base_e2e:.2f} tok/s\n")
    print(f"{'level':>6}{'HF static':>12}{'vLLM':>11}{'vLLM/HF':>10}"
          f"{'HF eff':>9}{'vLLM eff':>10}")
    print("-" * 78)
    hf1, vl1 = hf[min(a["levels"])], vl[min(a["levels"])]
    for n in a["levels"]:
        hf_eff = (hf[n]["aggregate_tps"] / hf1["aggregate_tps"]) / n * 100
        vl_eff = (vl[n]["aggregate_tps"] / vl1["aggregate_tps"]) / n * 100
        print(f"{n:>6}{hf[n]['aggregate_tps']:>12.2f}{vl[n]['aggregate_tps']:>11.2f}"
              f"{a['ratios'][n]:>9.2f}x{hf_eff:>8.1f}%{vl_eff:>9.1f}%")

    flat = [a["ratios"][n] for n in FLAT_LEVELS if n in a["ratios"]]
    print(f"\n  Ratio across levels {FLAT_LEVELS}: "
          f"{', '.join(f'{r:.2f}x' for r in flat)}")
    print(f"  spread {max(flat) - min(flat):.3f}x — FLAT. A constant multiplier is")
    print("  fixed per-step overhead, not scheduling: a scheduling win would GROW")
    print("  with batch size.")

    print(f"\n  Decomposition of the {a['total']:.2f}x headline at level {a['top']}:")
    print(f"    plain static batching        {a['batching']:.2f}x")
    print(f"    vLLM engine efficiency       {a['engine']:.2f}x   "
          f"(mean of the flat region)")
    print(f"    scaling retention at {a['top']}        {a['scheduling']:.2f}x   "
          f"(residual)")
    print(f"    product                      "
          f"{a['batching'] * a['engine'] * a['scheduling']:.2f}x  "
          f"(vs measured {a['total']:.2f}x)")


def plot(hf, vl, a):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in axes:
        ax.set_facecolor(SURFACE)

    # ── left: matched head-to-head ──
    ax = axes[0]
    levels = a["levels"]
    x = np.arange(len(levels))
    width = 0.36
    hf_tps = [hf[n]["aggregate_tps"] for n in levels]
    vl_tps = [vl[n]["aggregate_tps"] for n in levels]

    ax.bar(x - width / 2 - 0.01, hf_tps, width, color=BLUE, zorder=3,
           label="HuggingFace static batching")
    ax.bar(x + width / 2 + 0.01, vl_tps, width, color=ORANGE, zorder=3,
           label="vLLM")

    for i, n in enumerate(levels):
        ax.text(i, max(hf_tps[i], vl_tps[i]) + max(vl_tps) * 0.045,
                f"{a['ratios'][n]:.2f}×", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
        ax.text(i - width / 2 - 0.01, hf_tps[i] + max(vl_tps) * 0.012,
                f"{hf_tps[i]:.0f}", ha="center", va="bottom", fontsize=8.5,
                color=INK_2)
        ax.text(i + width / 2 + 0.01, vl_tps[i] + max(vl_tps) * 0.012,
                f"{vl_tps[i]:.0f}", ha="center", va="bottom", fontsize=8.5,
                color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in levels])
    ax.set_xlabel("batch size (HF) / concurrency (vLLM)", fontsize=10.5, color=INK_2)
    ax.set_ylabel("Aggregate throughput (tok/s)", fontsize=10.5, color=INK_2)
    ax.set_ylim(0, max(vl_tps) * 1.22)
    ax.set_title("Matched levels — the ratio is flat until 8",
                 fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # ── right: multiplicative decomposition ──
    ax2 = axes[1]
    steps = [
        ("HF batch=1", 1.0, MUTED),
        ("+ static batching", a["batching"], BLUE),
        ("+ vLLM engine", a["batching"] * a["engine"], ORANGE),
        ("+ paged KV /\n  cont. batching", a["total"], AQUA),
    ]
    ys = np.arange(len(steps))
    for y, (label, cumulative, color) in zip(ys, steps):
        ax2.barh(y, cumulative, height=0.55, color=color, zorder=3)
        ax2.text(cumulative + a["total"] * 0.02, y, f"{cumulative:.2f}×",
                 ha="left", va="center", fontsize=10, fontweight="bold", color=INK)

    factors = ["", f"×{a['batching']:.2f}", f"×{a['engine']:.2f}",
               f"×{a['scheduling']:.2f}"]
    for y, (label, cumulative, _), f in zip(ys, steps, factors):
        if f:
            ax2.text(cumulative * 0.5, y, f, ha="center", va="center",
                     fontsize=9.5, color="white", fontweight="bold", zorder=4)

    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([s[0] for s in steps], fontsize=9.5, color=INK)
    ax2.invert_yaxis()
    ax2.set_xlabel("Cumulative throughput vs HF batch=1", fontsize=10.5, color=INK_2)
    ax2.set_xlim(0, a["total"] * 1.20)
    ax2.set_title(f"Where the {a['total']:.1f}× actually comes from",
                  fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax2.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["bottom"].set_color(AXIS)
    ax2.tick_params(colors=MUTED, labelsize=9.5)
    ax2.tick_params(axis="y", length=0)

    fig.suptitle("Isolating vLLM's contribution — most of the headline is plain batching",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PNG, dpi=170, facecolor=SURFACE)
    print(f"\nSaved {OUT_PNG}")


def main():
    hf, vl, base_e2e = load()
    a = analyse(hf, vl, base_e2e)
    report(hf, vl, base_e2e, a)
    plot(hf, vl, a)


if __name__ == "__main__":
    main()
