"""
HuggingFace STATIC batching at matched batch sizes — isolating vLLM's real win.

Why this row exists: the existing comparison is vLLM (batched) versus HF (batch=1),
which conflates two different things — "batching" and "vLLM". HuggingFace can batch
too. Running HF `generate()` with static batching at the same batch sizes as the
vLLM concurrency levels splits the measured advantage into:

    (vLLM @ C=n)          (HF static @ batch=n)
    ------------------  =  ---------------------  x  ------------------------
    (HF batch=1)           (HF batch=1)              (vLLM / HF static @ n)
      total gap             what batching alone       what vLLM's engine and
                            recovers                  paged KV add on top

The timing discipline is imported from baseline.py so it cannot drift: same model,
same fp16, same KV cache, same greedy decoding, same 1024-token prompts, same
WARMUP_RUNS, same seed, same synchronize-before-every-timer-stop rule.

IMPORTANT CAVEAT, stated in the output too: every sequence here is pinned to
exactly MAX_NEW_TOKENS and every prompt is exactly INPUT_TOKENS long, so the batch
is perfectly rectangular. That removes padding waste and straggler effects — which
is precisely the case where CONTINUOUS batching has the least to offer, since no
sequence ever finishes early and leaves a slot idle. The vLLM-over-HF-static ratio
measured here therefore isolates vLLM's ENGINE efficiency (compiled scheduler, CUDA
graphs, paged KV) and is a CONSERVATIVE estimate of its value on real, ragged
traffic.

Writes results/hf_batched.json. Changes no prior file.
Run: python benchmark/hf_batched.py
"""

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the validated harness constants/helpers — do NOT re-derive the workload.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import (  # noqa: E402
    DEVICE,
    DTYPE,
    INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    WARMUP_RUNS,
    build_prompts,
    log_gpu,
    set_determinism,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BATCH_SIZES = [1, 2, 4, 8]

# Matches vllm_bench.py's REQUESTS_PER_LEVEL, so each level does the SAME total
# work as the vLLM sweep did: 32 requests x 256 output tokens.
REQUESTS_PER_LEVEL = 32

# Reference values from the completed, verified rows. Used for the batch=1 drift
# check and the head-to-head. Preferred from results/*.json when those files are
# present; these constants are the fallback for a fresh Colab session.
# Fallback reference only — at runtime this is read from fp16_control.json (or
# baseline.json) so the check uses a same-stack figure. Values are cu130
# baseline.json; see quantization_nf4.py on why 24.45 tok/s is treated as an
# outlier session rather than averaged in.
BASELINE_E2E_TPS = 22.43        # 256 / (11415.8 / 1000)
BASELINE_DECODE_TPS = 22.89     # baseline.json "tps" (decode-only)
BASELINE_PEAK_VRAM_GB = 3.06
VLLM_AGGREGATE_TPS = {1: 59.18, 2: 111.93, 4: 207.53, 8: 344.17}   # vllm.json
VLLM_P99_MS = {1: 4930.4, 2: 4614.5, 4: 4943.0, 8: 5962.8}         # vllm.json

# batch=1 must reproduce the existing baseline within this fraction, or we stop.
DRIFT_TOLERANCE = 0.10


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure, across torch versions."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def reference_rows():
    """vLLM and baseline reference numbers for the head-to-head.

    Reads the measured JSONs when available so the comparison traces to data;
    falls back to the documented constants if results/ was not re-uploaded to this
    session. Prints which source won.
    """
    vllm_tps, vllm_p99 = dict(VLLM_AGGREGATE_TPS), dict(VLLM_P99_MS)
    base_e2e, source = BASELINE_E2E_TPS, "documented constants"

    vllm_path = RESULTS_DIR / "vllm.json"
    if vllm_path.exists():
        data = json.loads(vllm_path.read_text())
        vllm_tps = {r["concurrency"]: r["aggregate_tps"] for r in data["results"]}
        vllm_p99 = {r["concurrency"]: r["p99_ms"] for r in data["results"]}
        source = vllm_path.name

    # fp16_control.json is preferred (same-session control from the NF4 run), but
    # fall back to baseline.json so this stage can drift-check without first
    # spending 35 minutes on the NF4 stage — which matters when the library stack
    # has changed and BOTH references need re-measuring.
    for name in ("fp16_control.json", "baseline.json"):
        path = RESULTS_DIR / name
        if path.exists():
            d = json.loads(path.read_text())
            base_e2e = d["max_new_tokens"] / (d["p50_ms"] / 1000.0)
            source += f" + {name}"
            break

    print(f"Reference numbers from: {source}")
    return vllm_tps, vllm_p99, base_e2e


def build_batch(prompts: list, batch_size: int) -> torch.Tensor:
    """Stack batch_size prompts into one (batch_size, INPUT_TOKENS) tensor.

    Cycles the same fixed prompts every prior row used. All are exactly
    INPUT_TOKENS tokens, so the batch is rectangular and needs NO padding — the
    attention mask is all ones and there is no padding waste inflating throughput.
    """
    rows = [prompts[i % len(prompts)] for i in range(batch_size)]
    batch = torch.cat(rows, dim=0)
    assert batch.shape == (batch_size, INPUT_TOKENS), batch.shape
    return batch


def timed_batch(model, tokenizer, input_ids) -> float:
    """One static-batching generate() call. Returns wall-clock seconds.

    min_new_tokens == max_new_tokens pins every sequence to exactly the same
    output length (matching the vLLM run's ignore_eos + min_tokens), so all
    sequences in the batch start together AND finish together.
    """
    torch.cuda.synchronize()        # drain prior work so t0 is a clean start line
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MAX_NEW_TOKENS,   # exact length, no early EOS
            do_sample=False,                 # greedy, as in every prior row
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()        # CUDA is async — without this we time launches
    wall_s = time.perf_counter() - t0

    n_new = out.shape[1] - input_ids.shape[1]
    assert n_new == MAX_NEW_TOKENS, f"expected {MAX_NEW_TOKENS} new tokens, got {n_new}"
    return wall_s


def measure_level(model, tokenizer, prompts, batch_size: int) -> dict:
    """One batch size. Returns a result row, or an OOM row if it does not fit."""
    iterations = REQUESTS_PER_LEVEL // batch_size
    print(f"\n── batch_size={batch_size}: {WARMUP_RUNS} warmup + {iterations} "
          f"measured batches ({iterations * batch_size} requests, "
          f"{INPUT_TOKENS} in / {MAX_NEW_TOKENS} out) ──")

    batch = build_batch(prompts, batch_size)
    try:
        # Warmup is per batch size: a new batch shape triggers fresh kernel
        # selection and allocator growth, which must not land in the timing.
        for _ in range(WARMUP_RUNS):
            timed_batch(model, tokenizer, batch)

        torch.cuda.reset_peak_memory_stats()   # peak for THIS batch size only
        latencies_ms, walls = [], []
        t_level = time.perf_counter()
        for i in range(iterations):
            wall_s = timed_batch(model, tokenizer, batch)
            walls.append(wall_s)
            # Static batching: every request in the batch finishes when the batch
            # finishes, so all batch_size requests share the batch's wall time.
            latencies_ms.extend([wall_s * 1000.0] * batch_size)
            print(f"  batch {i+1}/{iterations}: {wall_s:.3f} s "
                  f"({batch_size * MAX_NEW_TOKENS / wall_s:.2f} tok/s)")
        level_wall_s = time.perf_counter() - t_level
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    except Exception as exc:                    # noqa: BLE001 — re-raised if not OOM
        if not is_oom(exc):
            raise
        print(f"  OUT OF MEMORY at batch_size={batch_size} — recorded as OOM, "
              f"continuing.\n  {type(exc).__name__}: {str(exc)[:200]}")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return {"batch_size": batch_size, "aggregate_tps": None, "p50_ms": None,
                "p99_ms": None, "peak_vram_gb": None, "oom": True}

    # Token counting, matching vllm_bench.py exactly: OUTPUT tokens only, divided
    # by wall clock for the whole measured window (which includes each batch's
    # prefill). Input tokens are never counted.
    total_output_tokens = iterations * batch_size * MAX_NEW_TOKENS
    aggregate_tps = total_output_tokens / level_wall_s

    row = {
        "batch_size": batch_size,
        "aggregate_tps": round(aggregate_tps, 2),
        "p50_ms": round(statistics.median(latencies_ms), 1),
        "p99_ms": round(statistics.quantiles(latencies_ms, n=100,
                                             method="inclusive")[98], 1),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "oom": False,
    }
    print(f"  aggregate {row['aggregate_tps']:.2f} tok/s | "
          f"p50 {row['p50_ms']:.1f} ms | p99 {row['p99_ms']:.1f} ms | "
          f"peak VRAM {row['peak_vram_gb']:.3f} GB")
    return row


def check_drift(row: dict, base_e2e: float) -> bool:
    """batch=1 must reproduce the existing baseline, or the session has drifted."""
    print("\n" + "=" * 74)
    print("CONSISTENCY CHECK — batch=1 must reproduce the existing baseline")
    print("=" * 74)
    if row["oom"]:
        print("  batch=1 hit OOM. Something is badly wrong; stopping.")
        return False
    got = row["aggregate_tps"]
    drift = (got - base_e2e) / base_e2e
    print(f"  reference e2e throughput : {base_e2e:.2f} tok/s "
          f"(decode-only was {BASELINE_DECODE_TPS:.2f})")
    print(f"  this run, batch=1        : {got:.2f} tok/s")
    print(f"  drift                    : {drift * 100:+.2f}% "
          f"(tolerance +/-{DRIFT_TOLERANCE * 100:.0f}%)")
    if abs(drift) > DRIFT_TOLERANCE:
        print("\n  *** DRIFT EXCEEDS TOLERANCE — STOPPING. ***")
        print("  batch=1 static batching is just the baseline workload, so it must")
        print("  reproduce it. It did not, which means something changed in this")
        print("  session (GPU allocation, library stack, thermal state). Do not")
        print("  trust any batched number measured against this baseline until the")
        print("  cause is found.")
        return False
    print("  OK — within tolerance. Batched rows are comparable to the prior rows.")
    return True


def analyse(rows, vllm_tps, vllm_p99, base_e2e):
    ok_rows = [r for r in rows if not r["oom"]]

    print("\n" + "=" * 74)
    print("1. HF STATIC BATCHING — throughput scaling")
    print("=" * 74)
    if ok_rows:
        b1 = ok_rows[0]["aggregate_tps"]
        print(f"{'batch':>6} {'agg tok/s':>11} {'scaling':>9} {'efficiency':>11} "
              f"{'p50 ms':>9} {'p99 ms':>9} {'VRAM GB':>9}")
        print("-" * 74)
        for r in rows:
            if r["oom"]:
                print(f"{r['batch_size']:>6} {'OOM':>11} {'—':>9} {'—':>11} "
                      f"{'—':>9} {'—':>9} {'—':>9}")
                continue
            scaling = r["aggregate_tps"] / b1
            print(f"{r['batch_size']:>6} {r['aggregate_tps']:>11.2f} "
                  f"{scaling:>8.3f}x {scaling / r['batch_size'] * 100:>10.1f}% "
                  f"{r['p50_ms']:>9.1f} {r['p99_ms']:>9.1f} "
                  f"{r['peak_vram_gb']:>9.3f}")
        print("\n  NOTE on p50/p99: in static batching every request in a batch")
        print("  finishes when the batch finishes, so per-request latency IS the")
        print("  batch wall time. For batch >= 2 that makes p50 ~ p99 almost by")
        print("  construction, because the 32 latency samples are only 16/8/4")
        print("  DISTINCT values duplicated within each batch. batch=1 is the")
        print("  exception: there every sample is an independent batch, so its p50")
        print("  vs p99 spread is genuine run-to-run variance. In no case is the")
        print("  spread a queueing tail — nothing waits behind anything here.")
        print("\n  The 'scaling' column above is self-relative (against THIS run's")
        print(f"  batch=1 = {b1:.2f} tok/s). Section 2 compares against the STORED")
        print(f"  baseline of {base_e2e:.2f} tok/s instead, so the same quantity")
        print("  differs slightly between the two; each is labelled where used.")

    print("\n" + "=" * 74)
    print("2. HEAD-TO-HEAD — vLLM vs HF static at matched batch/concurrency")
    print("=" * 74)
    print(f"{'level':>6} {'HF tok/s':>10} {'vLLM tok/s':>11} {'vLLM/HF':>9} "
          f"{'HF p99':>10} {'vLLM p99':>10} {'HF/vLLM':>9}")
    print("-" * 78)
    for r in rows:
        n = r["batch_size"]
        v, vp = vllm_tps.get(n), vllm_p99.get(n)
        if r["oom"] or v is None:
            hf_cell = "OOM" if r["oom"] else "n/a"
            vllm_cell = f"{v:.2f}" if v is not None else "n/a"
            vp_cell = f"{vp:.1f}" if vp is not None else "n/a"
            print(f"{n:>6} {hf_cell:>10} {vllm_cell:>11} {'n/a':>9} "
                  f"{'n/a':>10} {vp_cell:>10} {'n/a':>9}")
            continue
        lat_ratio = f"{r['p99_ms'] / vp:.2f}x" if vp else "n/a"
        vp_cell = f"{vp:.1f}" if vp is not None else "n/a"
        print(f"{n:>6} {r['aggregate_tps']:>10.2f} {v:>11.2f} "
              f"{v / r['aggregate_tps']:>8.2f}x {r['p99_ms']:>10.1f} "
              f"{vp_cell:>10} {lat_ratio:>9}")

    print("\n  'vLLM/HF' is the ISOLATED value of vLLM's engine + paged KV over")
    print("  static batching at the SAME batch size — the number this row exists to")
    print("  produce. 'HF/vLLM' is how many times worse HF's per-request p99 is at")
    print("  that same level, which is the other half of the story: static batching")
    print("  buys throughput by making every request wait for the whole batch.")

    print(f"\n  What plain batching recovers on its own, measured against the STORED"
          f"\n  baseline (fp16_control = {base_e2e:.2f} tok/s), not this run's batch=1:")
    for r in rows:
        if r["oom"]:
            print(f"    batch={r['batch_size']}: OOM")
            continue
        v = vllm_tps.get(r["batch_size"])
        vllm_gap = f"{v / base_e2e:.2f}x" if v is not None else "n/a"
        print(f"    batch={r['batch_size']}: HF static "
              f"{r['aggregate_tps'] / base_e2e:.2f}x  |  vLLM {vllm_gap}")

    print("\n" + "=" * 74)
    print("3. VRAM — how peak memory grows with batch size")
    print("=" * 74)
    print(f"  reference baseline peak (batch=1): {BASELINE_PEAK_VRAM_GB:.3f} GB")
    print("  Weights are fixed; what grows with the batch is the KV cache plus the")
    print("  per-step activations. Which of those dominates is visible in the deltas")
    print("  below rather than assumed: the KV cache for this workload is only about")
    print("  35 MiB per sequence (GQA, 2 KV heads, 1280 tokens), so a delta much")
    print("  larger than ~35 MiB per added sequence is activation growth, not KV.")
    for r in rows:
        if r["oom"]:
            served = vllm_tps.get(r["batch_size"])
            print(f"  batch={r['batch_size']}: OOM"
                  + (f"  <-- but vLLM SERVED concurrency {r['batch_size']} at "
                     f"{served:.2f} tok/s, so the load itself fits on this GPU; it "
                     f"is HF's allocation pattern that did not."
                     if served else ""))
        else:
            delta = r["peak_vram_gb"] - BASELINE_PEAK_VRAM_GB
            print(f"  batch={r['batch_size']}: {r['peak_vram_gb']:.3f} GB "
                  f"({delta:+.3f} GB vs batch=1)")
    if not any(r["oom"] for r in rows):
        print("  No OOM at any batch size — with GQA (2 KV heads) the KV cache for")
        print("  1280 tokens is small, so a 1.5B model has ample headroom on 16 GB.")
        print("  The PagedAttention memory argument would only bite at much larger")
        print("  batch sizes or context lengths than this workload uses.")

    print("\n" + "=" * 74)
    print("4. HOW MUCH OF vLLM's ADVANTAGE SURVIVES ONCE HF IS ALSO BATCHED?")
    print("=" * 74)
    if not ok_rows:
        print("  No comparable HF rows — cannot attribute.")
        return
    top = max(ok_rows, key=lambda r: r["batch_size"])
    n = top["batch_size"]
    v = vllm_tps.get(n)
    if n == 1:
        print("  Only batch=1 produced a comparable number, so there is no batched")
        print("  level to attribute against. Nothing to decompose.")
        return
    if v is None:
        print(f"  No vLLM reference at concurrency {n}; cannot attribute.")
        return

    print(f"  Original headline (vLLM @ C={n} vs HF batch=1): "
          f"{v / base_e2e:.2f}x")
    print(f"  Of which plain static batching alone accounts for: "
          f"{top['aggregate_tps'] / base_e2e:.2f}x")
    print(f"  Remaining, attributable to vLLM at batch={n}:      "
          f"{v / top['aggregate_tps']:.2f}x")
    # share = (HF/base) / (vLLM/base), which reduces to HF/vLLM
    share = top["aggregate_tps"] / v * 100
    print(f"  => static batching alone reaches {share:.1f}% of vLLM's throughput at")
    print(f"     this level; vLLM's engine and paged KV supply the remaining "
          f"{v / top['aggregate_tps']:.2f}x.")
    print(f"  Latency side: HF static p99 {top['p99_ms']:.1f} ms vs vLLM "
          f"{vllm_p99.get(n, float('nan')):.1f} ms at the same level.")
    print("\n  READ THIS BEFORE QUOTING THE RATIO:")
    print("  All sequences here are pinned to exactly 256 tokens with identical")
    print("  1024-token prompts, so the batch is perfectly rectangular. No sequence")
    print("  finishes early, so no slot ever sits idle — which is exactly the")
    print("  benefit CONTINUOUS batching exists to capture. This setup is therefore")
    print("  the BEST case for static batching, and the vLLM/HF ratio above is a")
    print("  CONSERVATIVE floor on vLLM's advantage under real ragged traffic where")
    print("  output lengths vary. What it does cleanly isolate is vLLM's engine")
    print("  efficiency: compiled scheduling, CUDA graphs, and paged KV.")


def main():
    set_determinism()
    gpu_name = log_gpu()
    vllm_tps, vllm_p99, base_e2e = reference_rows()

    print(f"\nLoading {MODEL_ID} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    model.eval()
    prompts = build_prompts(tokenizer)
    print(f"Weights at rest: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")

    rows, aborted = [], False
    for batch_size in BATCH_SIZES:
        row = measure_level(model, tokenizer, prompts, batch_size)
        rows.append(row)
        if batch_size == 1 and not check_drift(row, base_e2e):
            aborted = True
            break

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu_name": gpu_name,
        "model": MODEL_ID,
        "dtype": "fp16",
        "backend": "huggingface_static_batching",
        "input_tokens": INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "requests_per_level": REQUESTS_PER_LEVEL,
        "drift_check_aborted": aborted,
        "results": rows,
    }
    out = RESULTS_DIR / "hf_batched.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out}")

    if aborted:
        print("\nStopped after the batch=1 drift check. No head-to-head produced.")
        return
    analyse(rows, vllm_tps, vllm_p99, base_e2e)


if __name__ == "__main__":
    main()
