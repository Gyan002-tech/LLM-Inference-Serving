"""
Per-token time breakdown — where each millisecond of a decode step actually goes.

This is the visual form of the baseline finding: generating one token requires
streaming all the weights from HBM, which sets a hard floor in milliseconds. Any
time beyond that floor is software overhead. Splitting measured per-token time
into those two parts shows immediately that the batch=1 bottleneck is not the
hardware.

All three rows are put on ONE basis — output tokens / wall clock, which INCLUDES
prefill — because that is the only basis vLLM's aggregate_tps supports. For the
HF rows this adds ~1.1 ms/token versus their decode-only figures (prefill's
~279 ms spread over 256 tokens); ANALYSIS.md quotes the decode-only split.

Pure analysis over results/*.json — loads no model and touches no GPU.
Run: python analysis/overhead.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_PNG = Path(__file__).resolve().parent / "overhead.png"

# Same constant and derivation as roofline.py: 256-bit GDDR6 @ 10 Gbps.
MEM_BANDWIDTH_BPS = 320e9
GIB = 1024 ** 3

# dataviz reference palette — light surface, categorical slots 1-2 + chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load():
    fp16 = json.loads((RESULTS / "fp16_control.json").read_text())
    nf4 = json.loads((RESULTS / "nf4.json").read_text())
    vllm = json.loads((RESULTS / "vllm.json").read_text())
    c1 = next(r for r in vllm["results"] if r["concurrency"] == 1)
    return fp16, nf4, c1


def row(label, weights_gib, tokens_per_sec, note):
    """Split measured per-token time into its bandwidth floor and the remainder."""
    weight_bytes = weights_gib * GIB
    floor_ms = weight_bytes / MEM_BANDWIDTH_BPS * 1000.0
    total_ms = 1000.0 / tokens_per_sec
    return {
        "label": label,
        "note": note,
        "tokens_per_sec": tokens_per_sec,
        "weights_gib": weights_gib,
        "floor_ms": floor_ms,
        "overhead_ms": total_ms - floor_ms,
        "total_ms": total_ms,
        "overhead_share": (total_ms - floor_ms) / total_ms * 100,
    }


def report(rows):
    print("=" * 78)
    print("PER-TOKEN TIME BREAKDOWN — batch=1 / concurrency=1")
    print("=" * 78)
    print(f"  bandwidth constant: {MEM_BANDWIDTH_BPS:.3e} B/s (T4, 256-bit GDDR6 @ 10 Gbps)")
    print("  basis: output tokens / wall clock (INCLUDES prefill), so all three")
    print("         rows are directly comparable to vLLM's aggregate_tps.\n")
    print(f"{'configuration':<26}{'tok/s':>8}{'ms/tok':>9}{'floor':>8}"
          f"{'overhead':>10}{'share':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<26}{r['tokens_per_sec']:>8.2f}{r['total_ms']:>9.2f}"
              f"{r['floor_ms']:>8.2f}{r['overhead_ms']:>10.2f}"
              f"{r['overhead_share']:>7.1f}%")

    hf, nf4, vllm = rows
    print(f"\n  The floor is what the HARDWARE requires: {hf['weights_gib']:.3f} GiB of fp16")
    print(f"  weights streamed per token at {MEM_BANDWIDTH_BPS/1e9:.0f} GB/s = "
          f"{hf['floor_ms']:.2f} ms. Everything above")
    print("  it is software.\n")
    print(f"  HF fp16 spends {hf['overhead_share']:.1f}% of each token outside the floor.")
    print(f"  NF4 lowers the floor to {nf4['floor_ms']:.2f} ms — and pushes overhead UP to")
    print(f"  {nf4['overhead_ms']:.2f} ms, so its overhead share RISES to "
          f"{nf4['overhead_share']:.1f}%. Quantization")
    print("  optimised the part that was never the problem.")
    print(f"  vLLM runs the SAME weights, so the same {vllm['floor_ms']:.2f} ms floor, but")
    print(f"  carries only {vllm['overhead_ms']:.2f} ms of overhead — "
          f"{hf['overhead_ms']/vllm['overhead_ms']:.2f}x less than HF.")


def plot(rows):
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ys = range(len(rows))
    labels = [r["label"] for r in rows]

    floors = [r["floor_ms"] for r in rows]
    overheads = [r["overhead_ms"] for r in rows]

    ax.barh(ys, floors, height=0.52, color=BLUE, zorder=3,
            label="memory-bandwidth floor (hardware requirement)")
    # 2px surface gap between the two segments, per the mark spec
    ax.barh(ys, overheads, height=0.52, left=[f + 0.35 for f in floors],
            color=ORANGE, zorder=3, label="everything else (software overhead)")

    for y, r in zip(ys, rows):
        ax.text(r["floor_ms"] / 2, y, f"{r['floor_ms']:.1f}", ha="center",
                va="center", fontsize=8.5, color="white", fontweight="bold",
                zorder=4)
        ax.text(r["floor_ms"] + r["overhead_ms"] / 2, y,
                f"{r['overhead_ms']:.1f} ms  ({r['overhead_share']:.0f}%)",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold", zorder=4)
        ax.text(r["total_ms"] + 1.2, y,
                f"{r['total_ms']:.1f} ms/token  ·  {r['tokens_per_sec']:.1f} tok/s",
                ha="left", va="center", fontsize=9, color=INK_2)

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Time per output token (ms, wall clock — includes prefill)",
                  fontsize=10.5, color=INK_2)
    ax.set_xlim(0, max(r["total_ms"] for r in rows) * 1.42)
    ax.set_title("Where each token's time goes — the batch=1 bottleneck is software\n"
                 f"Tesla T4 · Qwen2.5-1.5B-Instruct · "
                 f"{MEM_BANDWIDTH_BPS / 1e9:.0f} GB/s memory bandwidth",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=14)

    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.tick_params(axis="y", length=0)

    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=SURFACE)
    print(f"\nSaved {OUT_PNG}")


def main():
    fp16, nf4, c1 = load()

    # HF rows: end-to-end tokens/sec from p50, so the basis matches vLLM's
    # aggregate_tps (output tokens / wall clock, prefill included).
    hf_e2e_tps = fp16["max_new_tokens"] / (fp16["p50_ms"] / 1000.0)
    nf4_e2e_tps = nf4["max_new_tokens"] / (nf4["p50_ms"] / 1000.0)

    rows = [
        row("HF fp16, batch=1", fp16["weights_vram_gb"], hf_e2e_tps, "baseline"),
        row("HF NF4, batch=1", nf4["weights_vram_gb"], nf4_e2e_tps, "quantized"),
        # vLLM runs the same fp16 weights, so it faces the identical floor.
        row("vLLM fp16, C=1", fp16["weights_vram_gb"], c1["aggregate_tps"], "served"),
    ]

    report(rows)
    plot(rows)


if __name__ == "__main__":
    main()
