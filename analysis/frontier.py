"""
Throughput-latency frontier for the vLLM concurrency sweep, with a programmatic
saturation-knee finding.

Pure analysis over results/*.json — loads no model and touches no GPU.

The frontier is the classic serving tradeoff: every point is one concurrency
level, x is the p99 latency you pay, y is the aggregate throughput you get.
Up-and-to-the-left is better. The HF baseline is plotted as a single reference
point so the served frontier is anchored against the non-served case.

Run: python analysis/frontier.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_PNG = Path(__file__).resolve().parent / "frontier.png"

# dataviz reference palette — light surface, categorical slots 1-2 + chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# A knee is called when marginal efficiency falls by more than this factor.
KNEE_DROP_FACTOR = 2.0


def load():
    vllm = json.loads((RESULTS / "vllm.json").read_text())
    fp16 = json.loads((RESULTS / "fp16_control.json").read_text())
    return vllm, fp16


def baseline_point(fp16: dict) -> dict:
    """HF baseline on the SAME basis as vLLM's aggregate_tps.

    fp16_control's "tps" is decode-only (prefill excluded by construction).
    aggregate_tps is output tokens / wall time and includes prefill, so the
    comparable baseline figure must be recomputed end-to-end from p50.
    """
    e2e_tps = fp16["max_new_tokens"] / (fp16["p50_ms"] / 1000.0)
    return {"aggregate_tps": e2e_tps, "p99_ms": fp16["p99_ms"],
            "decode_tps": fp16["tps"], "p50_ms": fp16["p50_ms"]}


def segment_metrics(rows: list) -> list:
    """Marginal cost/benefit of each step up in concurrency."""
    segs = []
    for a, b in zip(rows, rows[1:]):
        tps_gain_pct = (b["aggregate_tps"] - a["aggregate_tps"]) / a["aggregate_tps"] * 100
        p99_cost_pct = (b["p99_ms"] - a["p99_ms"]) / a["p99_ms"] * 100
        # Elasticity = % throughput bought per % p99 paid. A non-positive latency
        # cost means the step was free, which is not a knee — mark it infinite so
        # the knee search cannot mistake it for a collapse.
        elasticity = float("inf") if p99_cost_pct <= 0 else tps_gain_pct / p99_cost_pct
        segs.append({
            "from_c": a["concurrency"], "to_c": b["concurrency"],
            "tps_gain_pct": tps_gain_pct, "p99_cost_pct": p99_cost_pct,
            "tps_delta": b["aggregate_tps"] - a["aggregate_tps"],
            "p99_delta": b["p99_ms"] - a["p99_ms"],
            "elasticity": elasticity,
        })
    return segs


def find_knee(segs: list):
    """The concurrency beyond which marginal efficiency collapses.

    Compares each segment's elasticity with the previous segment's and returns
    the start of the segment with the largest drop. Segments whose predecessor
    was 'free' (infinite elasticity) are skipped, since no finite ratio exists.
    """
    knee, best_drop, prev_e, next_e = None, 0.0, None, None
    for prev, cur in zip(segs, segs[1:]):
        if prev["elasticity"] == float("inf") or cur["elasticity"] <= 0:
            continue
        drop = prev["elasticity"] / cur["elasticity"]
        if drop > best_drop:
            best_drop, knee = drop, cur["from_c"]
            prev_e, next_e = prev["elasticity"], cur["elasticity"]
    if knee is None or best_drop < KNEE_DROP_FACTOR:
        return None, best_drop, prev_e, next_e
    return knee, best_drop, prev_e, next_e


def report(rows, base, segs, knee, drop, prev_e, next_e):
    print("=" * 74)
    print("THROUGHPUT-LATENCY FRONTIER — vLLM concurrency sweep")
    print("=" * 74)
    print("\nHF fp16 baseline reference point")
    print(f"  decode-only tps (from JSON)  : {base['decode_tps']:.2f} tok/s")
    print(f"  p50                          : {base['p50_ms']:.1f} ms")
    print(f"  e2e tps = 256 / (p50/1000)   : {base['aggregate_tps']:.3f} tok/s"
          f"   <- the figure comparable to aggregate_tps")
    print(f"  p99 (x-position on the plot) : {base['p99_ms']:.1f} ms")

    c1 = rows[0]["aggregate_tps"]
    print(f"\n{'C':>3} {'p99 (ms)':>10} {'agg tok/s':>11} {'scaling':>9} "
          f"{'efficiency':>11} {'vs baseline':>12}")
    print("-" * 74)
    for r in rows:
        scaling = r["aggregate_tps"] / c1
        print(f"{r['concurrency']:>3} {r['p99_ms']:>10.1f} {r['aggregate_tps']:>11.2f} "
              f"{scaling:>8.3f}x {scaling / r['concurrency'] * 100:>10.1f}% "
              f"{r['aggregate_tps'] / base['aggregate_tps']:>11.2f}x")

    print("\nMarginal cost/benefit per step")
    print("-" * 74)
    for s in segs:
        e = "free (p99 fell)" if s["elasticity"] == float("inf") else f"{s['elasticity']:.2f}"
        print(f"  C={s['from_c']}->{s['to_c']}: "
              f"throughput {s['tps_gain_pct']:+7.2f}% ({s['tps_delta']:+7.2f} tok/s) | "
              f"p99 {s['p99_cost_pct']:+7.2f}% ({s['p99_delta']:+8.1f} ms) | "
              f"elasticity {e}")
        if s["elasticity"] != float("inf"):
            print(f"{'':>12}marginal rate: "
                  f"{s['tps_delta'] / s['p99_delta']:.4f} tok/s per ms of added p99")

    print("\nSATURATION KNEE")
    print("-" * 74)
    if knee is None:
        print("  No knee: marginal efficiency never dropped by more than "
              f"{KNEE_DROP_FACTOR}x.")
    else:
        print(f"  Knee at C={knee}.")
        print(f"  Below it, each 1% of added p99 latency buys {prev_e:.2f}% more "
              f"throughput.")
        print(f"  Above it, the same 1% buys only {next_e:.2f}%.")
        print(f"  Marginal efficiency collapses by {drop:.2f}x at this point.")
        print(f"  Read: C={knee} is the last level where throughput is essentially "
              f"free;\n        past it you are buying throughput with latency.")


def plot(rows, base, knee):
    xs = [r["p99_ms"] for r in rows]
    ys = [r["aggregate_tps"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(xs, ys, "-", color=BLUE, linewidth=2, zorder=2,
            label="vLLM (PagedAttention + continuous batching)")
    ax.plot(xs, ys, "o", color=BLUE, markersize=9, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=2)

    for r in rows:
        # label above-left so the connecting line stays legible
        ax.annotate(f"C={r['concurrency']}",
                    (r["p99_ms"], r["aggregate_tps"]),
                    textcoords="offset points", xytext=(-6, 12),
                    ha="right", fontsize=10, color=INK, fontweight="bold")
        ax.annotate(f"{r['aggregate_tps']:.0f} tok/s",
                    (r["p99_ms"], r["aggregate_tps"]),
                    textcoords="offset points", xytext=(10, -4),
                    ha="left", fontsize=9, color=INK_2)

    ax.plot([base["p99_ms"]], [base["aggregate_tps"]], "s", color=ORANGE,
            markersize=10, zorder=3, markeredgecolor=SURFACE, markeredgewidth=2,
            label="HuggingFace generate() baseline (batch=1)")
    ax.annotate(f"baseline\n{base['aggregate_tps']:.1f} tok/s",
                (base["p99_ms"], base["aggregate_tps"]),
                textcoords="offset points", xytext=(-12, 16),
                ha="right", fontsize=9.5, color=INK_2)

    if knee is not None:
        kx = next(r["p99_ms"] for r in rows if r["concurrency"] == knee)
        ky = next(r["aggregate_tps"] for r in rows if r["concurrency"] == knee)
        ax.plot([kx], [ky], "o", markersize=19, markerfacecolor="none",
                markeredgecolor=BLUE, markeredgewidth=1.6, zorder=1)
        ax.annotate(f"knee — saturation begins after C={knee}",
                    (kx, ky), textcoords="offset points", xytext=(18, -30),
                    ha="left", fontsize=9.5, color=BLUE,
                    arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.2))

    # placed in the empty upper-right quadrant: the data occupies the far left
    # (vLLM) and the low far right (baseline)
    ax.annotate("better", xy=(0.60, 0.80), xytext=(0.76, 0.63),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=9, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.2))

    ax.set_xlabel("p99 latency per request (ms)", fontsize=10.5, color=INK_2)
    ax.set_ylabel("Aggregate throughput (tokens/s)", fontsize=10.5, color=INK_2)
    ax.set_title("Throughput–latency frontier — Qwen2.5-1.5B-Instruct on Tesla T4\n"
                 "1024 input / 256 output tokens, greedy, closed-loop load",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=14)

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.set_xlim(3600, 13000)
    ax.set_ylim(0, max(ys) * 1.22)

    leg = ax.legend(loc="upper center", frameon=False, fontsize=9.5)
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=SURFACE)
    print(f"\nSaved {OUT_PNG}")


def main():
    vllm, fp16 = load()
    rows = sorted(vllm["results"], key=lambda r: r["concurrency"])
    base = baseline_point(fp16)
    segs = segment_metrics(rows)
    knee, drop, prev_e, next_e = find_knee(segs)

    report(rows, base, segs, knee, drop, prev_e, next_e)

    best = rows[-1]
    print("\nHeadline")
    print("-" * 74)
    print(f"  vLLM @ C={best['concurrency']} vs HF baseline: "
          f"{best['aggregate_tps'] / base['aggregate_tps']:.2f}x throughput "
          f"at {base['p99_ms'] / best['p99_ms']:.2f}x LOWER p99 latency.")

    plot(rows, base, knee)


if __name__ == "__main__":
    main()
