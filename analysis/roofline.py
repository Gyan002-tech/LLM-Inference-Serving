"""
Roofline model for batch=1 decode on a Tesla T4, with the measured fp16 and NF4
points placed on it.

Pure analysis over results/*.json — loads no model and touches no GPU.

The argument this file makes: batch=1 decode sits far to the LEFT of the roofline
ridge point (deeply memory-bound), NF4 moves its point further right by shrinking
the weights, and yet NF4 measures SLOWER. That combination is only possible if the
regression is compute/overhead, not bandwidth — which is the finding.

Run: python analysis/roofline.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_PNG = Path(__file__).resolve().parent / "roofline.png"

# ── HARDWARE CONSTANTS — Tesla T4 (TU104, sm_75) ─────────────────────────────
# Every constant is stated with its derivation. None is a remembered figure.

# Memory bandwidth: 256-bit GDDR6 bus at 10 Gbps effective data rate.
#   256 bits * 10e9 /s / 8 bits-per-byte = 320e9 B/s
# NVIDIA's T4 datasheet quotes 320 GB/s. Some sources cite ~300 GB/s (a more
# conservative sustained figure). SENSITIVITY is computed below; the conclusion
# does not depend on which is used.
MEM_BANDWIDTH_BPS = 320e9

# FP16 compute peak: 65 TFLOP/s. This is the TENSOR-CORE figure from NVIDIA's T4
# datasheet (320 tensor cores). It is NOT the CUDA-core FP16 rate, which is only
# ~8.1 TFLOP/s. The tensor-core number is the generous choice: it places the ridge
# point as far RIGHT as possible, which is the hardest case for a "memory-bound"
# claim to survive. Sensitivity against the CUDA-core figure is computed below.
FP16_PEAK_FLOPS = 65e12

MEM_BANDWIDTH_ALT_BPS = 300e9   # conservative bandwidth, for sensitivity
FP16_PEAK_ALT_FLOPS = 8.1e12    # CUDA-core (non-tensor) fp16, for sensitivity

# results/*.json report VRAM as bytes / 1024**3, i.e. GiB despite the "_gb" name.
GIB = 1024 ** 3

# FLOPs per parameter per generated token. One multiply-accumulate = 2 FLOPs, and
# at batch=1 each weight participates in exactly one MAC per token.
FLOPS_PER_PARAM = 2

# dataviz reference palette — light surface, categorical slots 1-2 + chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def fmt_flops(flops: float) -> str:
    """Format a FLOP/s value in the largest sensible unit.

    The plot reports performance in FLOP/s throughout — the roofline convention
    (Williams/Waterman/Patterson) — so no axis, label, or legend entry mixes it
    with tok/s. Tokens/s stays in the stdout report and the prose, where it is
    the serving-relevant metric and there is no axis to be inconsistent with.
    The subtitle carries the exact conversion so a reader can move between them.
    """
    if flops >= 1e12:
        return f"{flops / 1e12:.3g} TFLOP/s"
    if flops >= 1e9:
        return f"{flops / 1e9:.3g} GFLOP/s"
    return f"{flops / 1e6:.3g} MFLOP/s"


def load():
    fp16 = json.loads((RESULTS / "fp16_control.json").read_text())
    nf4 = json.loads((RESULTS / "nf4.json").read_text())
    return fp16, nf4


def analyse(label, weights_gib, measured_tps, flops_per_token):
    """Place one measured configuration on the roofline."""
    weight_bytes = weights_gib * GIB
    intensity = flops_per_token / weight_bytes          # FLOP per byte
    ceiling_tps = MEM_BANDWIDTH_BPS / weight_bytes      # bandwidth-bound tok/s
    achieved_flops = measured_tps * flops_per_token
    achieved_bps = measured_tps * weight_bytes
    ms_per_token = 1000.0 / measured_tps
    ms_bandwidth_floor = weight_bytes / MEM_BANDWIDTH_BPS * 1000.0
    return {
        "label": label,
        "weights_gib": weights_gib,
        "weight_bytes": weight_bytes,
        "intensity": intensity,
        "ceiling_tps": ceiling_tps,
        "measured_tps": measured_tps,
        "pct_of_ceiling": measured_tps / ceiling_tps * 100,
        "achieved_flops": achieved_flops,
        "pct_of_compute_peak": achieved_flops / FP16_PEAK_FLOPS * 100,
        "achieved_bps": achieved_bps,
        "ms_per_token": ms_per_token,
        "ms_bandwidth_floor": ms_bandwidth_floor,
        "ms_non_bandwidth": ms_per_token - ms_bandwidth_floor,
    }


def report(fp16_r, nf4_r, params, flops_per_token, ridge, vllm_c1_tps):
    print("=" * 78)
    print("ROOFLINE — batch=1 decode, Qwen2.5-1.5B-Instruct on Tesla T4")
    print("=" * 78)

    print("\nHardware constants (see file header for derivations)")
    print("-" * 78)
    print(f"  memory bandwidth        : {MEM_BANDWIDTH_BPS:.3e} B/s  "
          f"(256-bit GDDR6 @ 10 Gbps)")
    print(f"  fp16 compute peak       : {FP16_PEAK_FLOPS:.3e} FLOP/s "
          f"(tensor core, NOT cuda core)")
    print(f"  RIDGE POINT = peak / bw : {ridge:.3f} FLOP/byte")

    print("\nModel arithmetic (derived from measured fp16 weight bytes)")
    print("-" * 78)
    print(f"  fp16 weights            : {fp16_r['weights_gib']} GiB = "
          f"{fp16_r['weight_bytes']:,.0f} bytes")
    print(f"  parameters = bytes / 2  : {params:,.0f}")
    print(f"  FLOPs/token = 2 * params: {flops_per_token:,.0f}  "
          f"({flops_per_token / 1e9:.4f} GFLOP)")
    print("  NOTE: 'read all weights once per token' is a good approximation for a")
    print("        TIED-embedding model — the lm_head matmul reads the embedding")
    print("        matrix, and the input-side embedding gather is negligible.")

    print("\nArithmetic intensity vs the ridge point")
    print("-" * 78)
    for r in (fp16_r, nf4_r):
        print(f"  {r['label']:<5} intensity = {flops_per_token:,.0f} FLOP / "
              f"{r['weight_bytes']:,.0f} B = {r['intensity']:.4f} FLOP/byte"
              f"   -> {ridge / r['intensity']:.1f}x BELOW the ridge")
    print("  Both sit far to the left of the ridge => MEMORY-BOUND regime confirmed.")

    print("\nBandwidth ceilings vs measured")
    print("-" * 78)
    print(f"  {'':<5} {'weights':>10} {'ceiling':>12} {'measured':>10} "
          f"{'% ceiling':>11} {'achieved BW':>13}")
    for r in (fp16_r, nf4_r):
        print(f"  {r['label']:<5} {r['weights_gib']:>8.3f}GiB "
              f"{r['ceiling_tps']:>9.2f}t/s {r['measured_tps']:>9.2f}t/s "
              f"{r['pct_of_ceiling']:>10.2f}% "
              f"{r['achieved_bps'] / 1e9:>10.2f}GB/s")

    print("\n  The same two rows in FLOP/s — the units roofline.png uses:")
    for r in (fp16_r, nf4_r):
        roof = min(MEM_BANDWIDTH_BPS * r["intensity"], FP16_PEAK_FLOPS)
        print(f"  {r['label']:<5} ceiling {fmt_flops(roof):>12} | "
              f"measured {fmt_flops(r['achieved_flops']):>12} | "
              f"{r['pct_of_ceiling']:.2f}%")
    print("  The percentage is identical in either unit — the FLOP-per-token")
    print("  conversion factor cancels in the ratio.")

    print("\nCompute utilisation (why the compute roof is irrelevant here)")
    print("-" * 78)
    for r in (fp16_r, nf4_r):
        print(f"  {r['label']:<5} achieved {r['achieved_flops']:.4e} FLOP/s = "
              f"{r['pct_of_compute_peak']:.4f}% of the {FP16_PEAK_FLOPS:.1e} peak")

    print("\n*** THE KEY RESULT ***")
    print("-" * 78)
    ceiling_ratio = nf4_r["ceiling_tps"] / fp16_r["ceiling_tps"]
    measured_ratio = nf4_r["measured_tps"] / fp16_r["measured_tps"]
    print(f"  NF4's bandwidth ceiling is {ceiling_ratio:.3f}x HIGHER than fp16's "
          f"({nf4_r['ceiling_tps']:.2f} vs {fp16_r['ceiling_tps']:.2f} tok/s)")
    print(f"  ...yet NF4 MEASURES {measured_ratio:.3f}x of fp16 "
          f"({nf4_r['measured_tps']:.2f} vs {fp16_r['measured_tps']:.2f} tok/s), "
          f"i.e. {1 / measured_ratio:.3f}x SLOWER.")
    print(f"  A purely bandwidth-bound NF4 would have been {ceiling_ratio:.2f}x FASTER.")
    print(f"  The regression therefore CANNOT be bandwidth. It is the NF4->fp16")
    print(f"  dequantisation compute added on top of a workload that could not use")
    print(f"  the bandwidth it freed.")

    print("\n  Per-token time decomposition (ms/token)")
    print(f"  {'':<5} {'total':>9} {'bandwidth floor':>17} {'non-bandwidth':>15}")
    for r in (fp16_r, nf4_r):
        print(f"  {r['label']:<5} {r['ms_per_token']:>9.3f} "
              f"{r['ms_bandwidth_floor']:>17.3f} {r['ms_non_bandwidth']:>15.3f}")
    bw_saved = fp16_r["ms_bandwidth_floor"] - nf4_r["ms_bandwidth_floor"]
    overhead_added = nf4_r["ms_non_bandwidth"] - fp16_r["ms_non_bandwidth"]
    net = nf4_r["ms_per_token"] - fp16_r["ms_per_token"]
    print(f"  NF4 bought {bw_saved:.3f} ms/token of bandwidth time")
    print(f"  NF4 paid   {overhead_added:.3f} ms/token of extra non-bandwidth time")
    print(f"  net        {net:+.3f} ms/token  (check: {overhead_added:.3f} - "
          f"{bw_saved:.3f} = {overhead_added - bw_saved:.3f})")

    print("\n  Cross-check against the vLLM row (same weights, same bandwidth floor)")
    vllm_ms = 1000.0 / vllm_c1_tps
    vllm_overhead = vllm_ms - fp16_r["ms_bandwidth_floor"]
    print(f"  vLLM @ C=1: {vllm_c1_tps:.2f} tok/s = {vllm_ms:.3f} ms/token "
          f"-> {vllm_overhead:.3f} ms/token non-bandwidth")
    print(f"  HF fp16 carries {fp16_r['ms_non_bandwidth'] / vllm_overhead:.2f}x "
          f"the per-token overhead of vLLM on identical weights,")
    print(f"  which is the same 'the bottleneck is not bandwidth' conclusion from "
          f"the other direction.")

    print("\nSENSITIVITY — does the conclusion depend on the constants?")
    print("-" * 78)
    for r in (fp16_r, nf4_r):
        alt_ceiling = MEM_BANDWIDTH_ALT_BPS / r["weight_bytes"]
        print(f"  bandwidth {MEM_BANDWIDTH_ALT_BPS / 1e9:.0f} GB/s: {r['label']:<5} "
              f"ceiling {alt_ceiling:>7.2f} tok/s, achieved "
              f"{r['measured_tps'] / alt_ceiling * 100:>5.2f}%")
    alt_ridge = FP16_PEAK_ALT_FLOPS / MEM_BANDWIDTH_BPS
    print(f"  cuda-core peak {FP16_PEAK_ALT_FLOPS:.1e} FLOP/s: ridge = "
          f"{alt_ridge:.3f} FLOP/byte")
    for r in (fp16_r, nf4_r):
        print(f"{'':>17}{r['label']:<5} still {alt_ridge / r['intensity']:.1f}x "
              f"below that ridge")
    print("  Neither alternative changes the ordering, the memory-bound verdict, or")
    print("  the fact that NF4 has the higher ceiling and the lower measurement.")

    print("\nUNIT NOTE (flagged, per the no-silent-numbers rule)")
    print("-" * 78)
    print("  results/*.json computes VRAM as bytes / 1024**3, so 'weights_vram_gb'")
    print("  is really GiB. Treating it as decimal GB would give ceilings of")
    print(f"  {MEM_BANDWIDTH_BPS / 1e9 / fp16_r['weights_gib']:.2f} and "
          f"{MEM_BANDWIDTH_BPS / 1e9 / nf4_r['weights_gib']:.2f} tok/s instead of "
          f"{fp16_r['ceiling_tps']:.2f} and {nf4_r['ceiling_tps']:.2f}")
    print("  (7.4% apart). This file uses the correct GiB conversion. The ceiling")
    print("  RATIO, and therefore the conclusion, is identical either way.")


def plot(fp16_r, nf4_r, ridge, flops_per_token):
    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.logspace(-1, 4, 400)
    roof = np.minimum(MEM_BANDWIDTH_BPS * x, FP16_PEAK_FLOPS)
    ax.plot(x, roof, color=INK_2, linewidth=2, zorder=3, label="Roofline (T4)")

    ax.axvline(ridge, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.annotate(f"ridge point\n{ridge:.0f} FLOP/byte",
                (ridge, FP16_PEAK_FLOPS * 0.30), textcoords="offset points",
                xytext=(10, 0), ha="left", fontsize=9, color=MUTED)

    ax.fill_betweenx([1e9, 1e14], 0.05, ridge, color=BLUE, alpha=0.05, zorder=0)
    ax.annotate("memory-bound", (0.13, 3.2e13), fontsize=9.5, color=MUTED)
    ax.annotate("compute-bound", (4.0e2, 3.2e13), fontsize=9.5, color=MUTED)

    for r, color, marker in ((fp16_r, BLUE, "o"), (nf4_r, ORANGE, "s")):
        roof_here = min(MEM_BANDWIDTH_BPS * r["intensity"], FP16_PEAK_FLOPS)
        ax.plot([r["intensity"], r["intensity"]], [r["achieved_flops"], roof_here],
                color=color, linewidth=1.2, linestyle="--", alpha=0.75, zorder=2)
        ax.plot([r["intensity"]], [roof_here], marker=marker, color=color,
                markersize=7, markerfacecolor="none", markeredgewidth=1.6, zorder=3)
        ax.plot([r["intensity"]], [r["achieved_flops"]], marker=marker, color=color,
                markersize=11, zorder=4, markeredgecolor=SURFACE, markeredgewidth=2,
                label=f"{r['label']} measured — {fmt_flops(r['achieved_flops'])} "
                      f"({r['pct_of_ceiling']:.1f}% of its ceiling)")
        ax.annotate(f"{r['label']}  ceiling {fmt_flops(roof_here)}",
                    (r["intensity"], roof_here), textcoords="offset points",
                    xytext=(9, 4), ha="left", fontsize=9, color=color)

    # sits in the gap between the measured points and the bandwidth roof, clear
    # of the legend (lower left) and the regime labels (top)
    ax.annotate("NF4 moves RIGHT (less to load)\nbut DOWN (dequant tax)",
                xy=(nf4_r["intensity"], nf4_r["achieved_flops"]),
                xytext=(30.0, 2.0e11), fontsize=9.5, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_2, linewidth=1.2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1e9, 2e14)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte of weights loaded)",
                  fontsize=10.5, color=INK_2)
    ax.set_ylabel("Attainable performance (FLOP/s)", fontsize=10.5, color=INK_2)
    # Performance is reported in FLOP/s throughout. The conversion constant is in
    # the subtitle so tok/s stays recoverable without putting a second unit on the
    # axes: tok/s = (FLOP/s) / (FLOP per token).
    ax.set_title("Roofline — batch=1 decode, Qwen2.5-1.5B-Instruct on Tesla T4\n"
                 f"{MEM_BANDWIDTH_BPS / 1e9:.0f} GB/s bandwidth · "
                 f"{FP16_PEAK_FLOPS / 1e12:.0f} TFLOP/s fp16 tensor peak · "
                 f"1 token = {flops_per_token / 1e9:.3f} GFLOP",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=14)

    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5)

    # lower left is the only region clear of the roof line, the measured points,
    # and the annotations
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=SURFACE)
    print(f"\nSaved {OUT_PNG}")


def main():
    fp16, nf4 = load()
    vllm = json.loads((RESULTS / "vllm.json").read_text())
    vllm_c1_tps = next(r["aggregate_tps"] for r in vllm["results"]
                       if r["concurrency"] == 1)

    params = fp16["weights_vram_gb"] * GIB / 2      # fp16 => 2 bytes per parameter
    flops_per_token = FLOPS_PER_PARAM * params
    ridge = FP16_PEAK_FLOPS / MEM_BANDWIDTH_BPS

    fp16_r = analyse("fp16", fp16["weights_vram_gb"], fp16["tps"], flops_per_token)
    nf4_r = analyse("NF4", nf4["weights_vram_gb"], nf4["tps"], flops_per_token)

    report(fp16_r, nf4_r, params, flops_per_token, ridge, vllm_c1_tps)
    plot(fp16_r, nf4_r, ridge, flops_per_token)


if __name__ == "__main__":
    main()
