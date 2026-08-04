"""
NF4 (bitsandbytes 4-bit) row + WikiText-2 perplexity.

The timing harness is IMPORTED from baseline.py rather than reimplemented —
consistency with the baseline row is the whole point of this comparison. That
means the same prompts, the same 1024-in / 256-out workload, the same 3 warmup
iterations, the same torch.cuda.synchronize() discipline, the same greedy
decoding, and the same seed. Nothing about the measurement changes; only the
weights' dtype does.

Two models are measured in ONE session:
  fp16  — control. Re-measured here so the NF4 comparison is same-session
          (catches Colab GPU roulette / thermal drift), and because the fp16
          perplexity must come from the SAME perplexity code as NF4's.
  NF4   — the row under test.

Writes:
  results/nf4.json           NF4 row, baseline schema + perplexity_wikitext2
  results/fp16_control.json   same-session fp16 re-measurement + perplexity
  results/baseline*.json      patched in place with perplexity_wikitext2

Run: python benchmark/quantization_nf4.py
"""

import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import the validated harness from baseline.py — do NOT re-implement timing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import (  # noqa: E402
    DEVICE,
    INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    N_REQUESTS,
    WARMUP_RUNS,
    build_prompts,
    log_gpu,
    set_determinism,
    timed_generate,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Sliding-window perplexity parameters (the standard HF recipe).
PPL_WINDOW = 1024
PPL_STRIDE = 512

# Set False to skip re-timing fp16 (~7 min) if Colab time is tight. Perplexity
# for fp16 is still computed either way, since the accuracy delta needs it.
MEASURE_FP16_TIMING = True

# Set True to reuse an existing results/fp16_control.json instead of re-measuring
# fp16 (~12 min saved on a retry). CAVEAT: the checkpoint then comes from a
# DIFFERENT process, so it is no longer a strictly same-session control — the
# drift check against CANONICAL_FP16 below is what keeps it honest.
REUSE_FP16_CHECKPOINT = False

# transformers refuses 4-bit below this bitsandbytes version. Colab often ships
# an older one, and a plain `pip install bitsandbytes` will NOT upgrade it.
MIN_BITSANDBYTES = "0.46.1"

# Reference baseline for drift detection: results/baseline.json, cu130 stack.
# Six fp16 decode-TPS samples now exist — 22.08, 22.37, 22.45 (cu128) and 24.45,
# 22.79, 22.89 (cu130). Five cluster within 3.7%; the 24.45 run was an outlier
# session roughly 8% fast, which is exactly the kind of thing the drift check
# below is meant to catch, so it is deliberately NOT averaged into the reference.
CANONICAL_FP16 = {
    "ttft_ms": 271.3, "tps": 22.89, "p50_ms": 11415.8,
    "p99_ms": 11674.0, "peak_vram_gb": 3.06,
}


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────
def preflight_bitsandbytes():
    """Verify the NF4 path is usable BEFORE spending ~12 minutes on fp16.

    Same lesson as the dataset preflight: bitsandbytes is the most fragile
    dependency here, and discovering it is missing only after the fp16 pass
    throws away all of that work.
    """
    try:
        import bitsandbytes
    except Exception as exc:
        raise RuntimeError(
            f"bitsandbytes is required for NF4 but could not be imported "
            f"({type(exc).__name__}: {exc}).\n"
            f'  Fix: pip install -U "bitsandbytes>={MIN_BITSANDBYTES}"') from exc

    version = getattr(bitsandbytes, "__version__", "unknown")
    if version != "unknown" and Version(version) < Version(MIN_BITSANDBYTES):
        raise RuntimeError(
            f"bitsandbytes {version} is too old for 4-bit quantization "
            f"(need >= {MIN_BITSANDBYTES}).\n"
            f'  Fix: pip install -U "bitsandbytes>={MIN_BITSANDBYTES}"  '
            f"— the -U matters; without it pip leaves the old version in place.")
    print(f"bitsandbytes {version} available")


def load_fp16():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=DEVICE)
    model.eval()
    return model


def load_nf4():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 is sm_75: fp16 compute, no bf16
        bnb_4bit_use_double_quant=True,        # quantizes the quant constants too
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        # dtype applies to the modules bnb does NOT quantize (embeddings,
        # layernorms). Without it they default to fp32 and inflate VRAM.
        dtype=torch.float16,
        # Pin every layer to GPU 0: any CPU offload would inject PCIe transfers
        # into the decode loop and make the timing meaningless.
        device_map={"": 0},
    )
    model.eval()
    return model


def free_vram():
    """Reclaim VRAM and reset the high-water mark. Caller must `del` the model
    first — freeing a reference inside a helper would not drop the caller's.

    The reset matters: max_memory_allocated() is a high-water mark, so without it
    the NF4 peak would still report fp16's ~3 GB and the headline VRAM finding
    would be wrong.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────
# Timing — aggregation mirrors baseline.main(); timing itself is imported
# ─────────────────────────────────────────────────────────────────────
def measure(model, tokenizer, prompts, label):
    print(f"\n── {label}: {WARMUP_RUNS} warmup + {N_REQUESTS} measured requests "
          f"({INPUT_TOKENS} in / {MAX_NEW_TOKENS} out, batch=1) ──")

    # TRAP 1: WARMUP — same count as baseline. First CUDA calls trigger kernel
    # compilation and lazy allocation; for NF4 they also trigger bitsandbytes'
    # dequant kernel autotuning.
    for i in range(WARMUP_RUNS):
        timed_generate(model, tokenizer, prompts[i % len(prompts)])

    ttft_s, e2e_s, tps = [], [], []
    peak_vram_gb = 0.0
    for i in range(N_REQUESTS):
        torch.cuda.reset_peak_memory_stats()  # per-run peak, not a stale high-water mark
        t_first, t_total = timed_generate(model, tokenizer, prompts[i % len(prompts)])
        peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 1024**3)

        ttft_s.append(t_first)
        e2e_s.append(t_total)
        # DECODE PHASE ONLY — identical formula to baseline.py: prefill time
        # (== TTFT) is subtracted and the first token is attributed to prefill,
        # so n-1 tokens are divided by the decode window alone.
        tps.append((MAX_NEW_TOKENS - 1) / (t_total - t_first))
        print(f"  req {i+1:>2}/{N_REQUESTS}: ttft {t_first*1000:7.1f} ms | "
              f"decode {tps[-1]:6.2f} tok/s | e2e {t_total:6.2f} s")

    e2e_ms = [t * 1000 for t in e2e_s]
    return {
        "ttft_ms": round(statistics.median(ttft_s) * 1000, 1),
        "tps": round(statistics.median(tps), 2),
        "p50_ms": round(statistics.median(e2e_ms), 1),
        "p99_ms": round(statistics.quantiles(e2e_ms, n=100, method="inclusive")[98], 1),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "n_requests": N_REQUESTS,
    }


# ─────────────────────────────────────────────────────────────────────
# Perplexity — sliding window
# ─────────────────────────────────────────────────────────────────────
def load_ppl_tokens(tokenizer):
    """Load and tokenize the WikiText-2 test split ONCE, before any model loads.

    Called up front on purpose: a hub/dataset failure here used to surface only
    AFTER the 30-request timing pass and threw away ~7 minutes of good
    measurements. Failing in the first few seconds is strictly better.

    "wikitext" alone is a legacy canonical dataset name that current
    huggingface_hub rejects (it requires a namespaced 'namespace/name' repo id);
    the dataset now lives at Salesforce/wikitext. The bare name is kept as a
    fallback for older datasets/huggingface_hub pins.
    """
    errors = []
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            print(f"WikiText-2 (raw, test) loaded from '{repo}'")
            break
        except Exception as exc:
            print(f"  '{repo}' unavailable: {type(exc).__name__}: {exc}")
            errors.append(exc)
    else:
        raise RuntimeError(
            f"could not load WikiText-2 from any known repo id: {errors}")

    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    print(f"  {ids.shape[1]:,} tokens; sliding window {PPL_WINDOW}, stride {PPL_STRIDE}")
    return ids


def perplexity_wikitext2(model, ids, label: str) -> float:
    """WikiText-2 (raw, test) perplexity by SLIDING WINDOW.

    Why not naive independent chunking: chunk boundaries would leave the first
    tokens of every chunk with no left context, so the model is scored on
    predictions it had no information for and perplexity is over-reported. Here a
    PPL_WINDOW-token window advances by PPL_STRIDE, so every SCORED token carries
    at least PPL_STRIDE tokens of real context, and the overlapping prefix is
    masked to -100 so it contributes context but is never scored twice.

    `ids` is pre-tokenized by load_ppl_tokens() so both models are scored on the
    byte-identical token stream (and it is not re-tokenized per model).
    """
    seq_len = ids.shape[1]

    nll_sum, n_scored, prev_end = 0.0, 0, 0
    t0 = time.perf_counter()
    for step, begin in enumerate(range(0, seq_len, PPL_STRIDE)):
        end = min(begin + PPL_WINDOW, seq_len)
        target_len = end - prev_end          # tokens not already scored by an earlier window
        assert target_len > 0, "window added no new tokens — stride/window mismatch"
        window = ids[:, begin:end].to(DEVICE)
        targets = window.clone()
        targets[:, :-target_len] = -100      # overlap = context only, never scored

        with torch.no_grad():
            loss = model(window, labels=targets).loss

        # loss is a MEAN over the tokens this window actually scored, so weight it
        # by that count before summing — the first and last windows score a
        # different number of tokens, and an unweighted mean-of-means would skew
        # the result. The exact denominator the model used is the number of
        # non-ignored labels AFTER its internal causal shift, i.e. targets[:, 1:]
        # (position 0 has no preceding logit to be predicted from).
        scored = int((targets[:, 1:] != -100).sum())
        nll_sum += loss.item() * scored
        n_scored += scored

        prev_end = end
        if step % 50 == 0:
            print(f"    {label} ppl window {begin:>7}/{seq_len} ...")
        if end == seq_len:
            break

    ppl = math.exp(nll_sum / n_scored)
    print(f"  {label} perplexity: {ppl:.4f}  "
          f"({n_scored:,} scored tokens, {time.perf_counter()-t0:.0f}s)")
    return round(ppl, 4)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    set_determinism()
    gpu_name = log_gpu()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = build_prompts(tokenizer)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Fail fast on BOTH fragile dependencies before spending ~12 minutes on the
    # fp16 pass — a late failure here throws away completed measurements.
    preflight_bitsandbytes()
    ppl_ids = load_ppl_tokens(tokenizer)

    def row(config, dtype, timing, weights_gb, ppl):
        return {
            "gpu_name": gpu_name, "model": MODEL_ID, "dtype": dtype,
            "config": config, "input_tokens": INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            **(timing or {}),
            "weights_vram_gb": weights_gb,
            "perplexity_wikitext2": ppl,
        }

    def save(name, payload):
        with open(RESULTS_DIR / name, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  → wrote results/{name}")

    # ── fp16 control ──
    checkpoint = RESULTS_DIR / "fp16_control.json"
    if REUSE_FP16_CHECKPOINT and checkpoint.exists():
        fp16_row = json.loads(checkpoint.read_text())
        print(f"\n=== fp16 control ===\nREUSING {checkpoint.name} from an earlier "
              f"process — this is NOT a same-session control. The drift check "
              f"against the canonical row below is what validates it.")
    else:
        print(f"\n=== fp16 control ===\nLoading {MODEL_ID} (fp16) ...")
        torch.cuda.reset_peak_memory_stats()
        model = load_fp16()
        fp16_weights_gb = round(torch.cuda.memory_allocated() / 1024**3, 3)
        print(f"Weights at rest: {fp16_weights_gb:.3f} GB")

        # Timing BEFORE perplexity: the perplexity forward materialises a
        # 1024 x 151936 logits tensor, a bigger allocation than decode ever makes,
        # which would pollute the peak-VRAM reading if it ran first.
        fp16_timing = measure(model, tokenizer, prompts, "fp16") if MEASURE_FP16_TIMING else None
        fp16_ppl = perplexity_wikitext2(model, ppl_ids, "fp16")
        del model          # drop the last reference BEFORE reclaiming, or NF4's
        free_vram()        # weights-at-rest reading would include fp16's 3 GB

        # Checkpoint immediately so a later failure cannot cost this work again.
        fp16_row = row("baseline_kvcache", "fp16", fp16_timing, fp16_weights_gb, fp16_ppl)
        save("fp16_control.json", fp16_row)

    fp16_ppl = fp16_row["perplexity_wikitext2"]

    # ── NF4 ──
    print(f"\n=== NF4 ===\nLoading {MODEL_ID} (NF4, double-quant, fp16 compute) ...")
    model = load_nf4()
    nf4_weights_gb = round(torch.cuda.memory_allocated() / 1024**3, 3)
    print(f"Weights at rest: {nf4_weights_gb:.3f} GB")

    nf4_timing = measure(model, tokenizer, prompts, "NF4")
    nf4_ppl = perplexity_wikitext2(model, ppl_ids, "NF4")
    del model
    free_vram()

    nf4_row = row("nf4_4bit_kvcache", "nf4", nf4_timing, nf4_weights_gb, nf4_ppl)
    save("nf4.json", nf4_row)

    # Perplexity is deterministic (no timing involved), so back-filling it into
    # the earlier baseline runs is legitimate and keeps the comparison on one
    # perplexity implementation.
    for path in sorted(RESULTS_DIR.glob("baseline*.json")):
        data = json.loads(path.read_text())
        data["perplexity_wikitext2"] = fp16_ppl
        path.write_text(json.dumps(data, indent=2))
        print(f"Patched {path.name} with perplexity_wikitext2={fp16_ppl}")

    # ── summary ──
    print(f"\n{'─'*78}")
    print(f"fp16 vs NF4 — {gpu_name} · {MODEL_ID} · {INPUT_TOKENS} in / "
          f"{MAX_NEW_TOKENS} out · greedy · batch=1")
    print(f"{'─'*78}")
    print(f"{'metric':<22}{'fp16':>14}{'NF4':>14}{'change':>16}")

    def line(name, key, better_lower=True):
        # keyed on availability, not on the timing flag: weights_vram_gb exists
        # for fp16 even when its timing pass was skipped
        if key not in fp16_row:
            print(f"{name:<22}{'(skipped)':>14}{nf4_row[key]:>14}{'':>16}")
            return
        a, b = fp16_row[key], nf4_row[key]
        pct = (b - a) / a * 100
        arrow = "better" if ((b < a) == better_lower) else "worse"
        print(f"{name:<22}{a:>14}{b:>14}{f'{pct:+.1f}% {arrow}':>16}")

    line("TTFT (ms)", "ttft_ms")
    line("TPS decode (tok/s)", "tps", better_lower=False)
    line("p50 e2e (ms)", "p50_ms")
    line("p99 e2e (ms)", "p99_ms")
    line("peak VRAM (GB)", "peak_vram_gb")
    line("weights VRAM (GB)", "weights_vram_gb")
    ppl_delta = (nf4_ppl - fp16_ppl) / fp16_ppl * 100
    print(f"{'WikiText-2 PPL':<22}{fp16_ppl:>14}{nf4_ppl:>14}{f'{ppl_delta:+.2f}%':>16}")

    if "tps" in fp16_row:
        print(f"\nfp16 control vs canonical baseline (drift check):")
        for k, canon in CANONICAL_FP16.items():
            got = fp16_row[k]
            print(f"  {k:<16} canonical {canon:>10}   this session {got:>10}   "
                  f"drift {(got-canon)/canon*100:+.1f}%")

    print(f"\nSaved results/nf4.json, results/fp16_control.json")


if __name__ == "__main__":
    main()
