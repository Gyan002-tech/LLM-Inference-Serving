"""
Baseline benchmark — ONE trustworthy row.

Config:   HuggingFace generate() · fp16 · KV cache ENABLED (use_cache=True) · greedy
Workload: LONG regime only — exactly 1024 input tokens, 256 new tokens, batch=1
Target:   Colab T4 (16 GB, sm_75, fp16 only) · Qwen/Qwen2.5-1.5B-Instruct

Metric definitions:
  TTFT (ms)     — from immediately before generate() is called to the FIRST token
                  arriving from a TextIteratorStreamer, inside the same measured
                  run as total time (NOT a separate max_new_tokens=1 call), so the
                  TTFT/TPS split is internally consistent.
  TPS (tok/s)   — DECODE PHASE ONLY: (n_new - 1) / (total_s - ttft_s).
  p50/p99 (ms)  — end-to-end per-request latency over N_REQUESTS >= 30 runs.
  Peak VRAM (GB)— torch.cuda.max_memory_allocated(), reset before each run.

Run: python benchmark/baseline.py
"""

import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda"
DTYPE = torch.float16      # T4 is sm_75: fp16 only (no bf16)
INPUT_TOKENS = 1024        # long regime; every prompt truncated to exactly this
MAX_NEW_TOKENS = 256
N_REQUESTS = 30            # p99 needs a tail — 30 is the floor
WARMUP_RUNS = 3
SEED = 42
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "baseline.json"

# Fixed prompt topics, cycled over the 30 requests — same prompts every run so
# rows from different sessions are comparable. Content is filler; only the
# (fixed) token count affects timing.
TOPICS = [
    "Summarize the following notes on KV caching in autoregressive decoding.",
    "Summarize the following notes on continuous batching in LLM serving.",
    "Summarize the following notes on weight quantization for inference.",
    "Summarize the following notes on PagedAttention memory management.",
    "Summarize the following notes on speculative decoding acceptance rates.",
    "Summarize the following notes on GPU memory bandwidth roofline limits.",
]
FILLER = (
    "Autoregressive transformer inference proceeds in two phases: a prefill "
    "phase that processes every prompt token in parallel and populates the "
    "key-value cache, and a decode phase that generates one token at a time, "
    "each step reading the cached keys and values of all previous tokens. "
    "Prefill is compute-bound while decode is memory-bandwidth-bound, because "
    "each decode step must stream the full model weights from GPU memory to "
    "produce a single token. "
)


# ── TRAP 3: DETERMINISM — fixed seeds + greedy + fixed output length,
#    otherwise output length varies and rows are not comparable. ──
def set_determinism():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ── TRAP 4: SESSION LOGGING — Colab reassigns GPUs between sessions; log the
#    device so every result can be traced to the hardware that produced it. ──
def log_gpu() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv"],
        capture_output=True, text=True, check=True,
    ).stdout
    print(f"nvidia-smi GPU check:\n{out.strip()}\n")
    gpu_name = out.strip().splitlines()[-1]  # line 0 is the csv header "name"
    if "T4" not in gpu_name:
        print(f"WARNING: expected Tesla T4, got '{gpu_name}' — numbers from "
              f"different GPUs are not comparable.\n")
    return gpu_name


def build_prompts(tokenizer) -> list:
    prompts = []
    for topic in TOPICS:
        text = topic + "\n\n" + FILLER * 40  # overshoot, then truncate to exact length
        ids = tokenizer(text, return_tensors="pt").input_ids[:, :INPUT_TOKENS]
        assert ids.shape[1] == INPUT_TOKENS, (
            f"prompt is {ids.shape[1]} tokens, need {INPUT_TOKENS}")
        prompts.append(ids.to(DEVICE))
    return prompts


def timed_generate(model, tokenizer, input_ids):
    """One measured request. Returns (ttft_s, total_s).

    TTFT and total time come from the SAME generation: generate() runs in a
    background thread and streams tokens; the main thread timestamps the first
    arrival. No separate max_new_tokens=1 call — that would measure a different
    run and break the TTFT/TPS split.
    """
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=600.0)
    result = {}

    def _worker():
        with torch.no_grad():
            result["out"] = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MAX_NEW_TOKENS,  # TRAP 3: exact length — no early EOS
                do_sample=False,                # TRAP 3: greedy — deterministic
                use_cache=True,                 # baseline config: KV cache ON
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

    torch.cuda.synchronize()      # TRAP 2: drain pending GPU work so t0 is a clean start line
    t_start = time.perf_counter()  # clock starts immediately before generate is launched
    thread = threading.Thread(target=_worker)
    thread.start()

    t_first = None
    for _ in streamer:            # blocks until each token's text reaches the CPU
        if t_first is None:
            t_first = time.perf_counter()  # first token arrival = TTFT stop; the token
            # crossed GPU→CPU to be decoded here, so it provably exists — an explicit
            # synchronize() from this thread would instead block on the WHOLE generation
    thread.join()

    torch.cuda.synchronize()      # TRAP 2: CUDA is async — without this the stop
    t_end = time.perf_counter()   # timestamp measures kernel LAUNCH, not execution

    n_new = result["out"].shape[1] - input_ids.shape[1]
    assert n_new == MAX_NEW_TOKENS, f"expected {MAX_NEW_TOKENS} new tokens, got {n_new}"
    return t_first - t_start, t_end - t_start


def main():
    set_determinism()
    gpu_name = log_gpu()

    print(f"Loading {MODEL_ID} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    prompts = build_prompts(tokenizer)

    # ── TRAP 1: WARMUP — first CUDA calls trigger kernel compilation, cuBLAS
    #    autotuning, and lazy allocator growth; timing them poisons the stats. ──
    print(f"Warmup: {WARMUP_RUNS} discarded generations ...")
    for i in range(WARMUP_RUNS):
        timed_generate(model, tokenizer, prompts[i % len(prompts)])

    ttft_s, e2e_s, tps = [], [], []
    peak_vram_gb = 0.0
    print(f"Measuring {N_REQUESTS} requests "
          f"({INPUT_TOKENS} in / {MAX_NEW_TOKENS} out, batch=1) ...")
    for i in range(N_REQUESTS):
        torch.cuda.reset_peak_memory_stats()  # per-run peak, not a stale high-water mark
        t_first, t_total = timed_generate(model, tokenizer, prompts[i % len(prompts)])
        peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 1024**3)

        ttft_s.append(t_first)
        e2e_s.append(t_total)
        # TPS is DECODE PHASE ONLY: prefill time (== TTFT) is subtracted, and the
        # first token belongs to prefill, hence n-1 tokens over the decode window.
        # Folding prefill in would contaminate TPS with prompt-length effects.
        tps.append((MAX_NEW_TOKENS - 1) / (t_total - t_first))
        print(f"  req {i+1:>2}/{N_REQUESTS}: ttft {t_first*1000:7.1f} ms | "
              f"decode {tps[-1]:6.2f} tok/s | e2e {t_total:6.2f} s")

    e2e_ms = [t * 1000 for t in e2e_s]
    row = {
        "gpu_name": gpu_name,
        "model": MODEL_ID,
        "dtype": "fp16",
        "config": "baseline_kvcache",
        "input_tokens": INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "ttft_ms": round(statistics.median(ttft_s) * 1000, 1),
        "tps": round(statistics.median(tps), 2),
        "p50_ms": round(statistics.median(e2e_ms), 1),
        "p99_ms": round(statistics.quantiles(e2e_ms, n=100, method="inclusive")[98], 1),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "n_requests": N_REQUESTS,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(row, f, indent=2)

    print(f"\n{'─' * 60}")
    print(f"BASELINE ROW — {gpu_name} · {MODEL_ID} · fp16 · KV cache ON · greedy")
    print(f"{'─' * 60}")
    print(f"  TTFT (median)      : {row['ttft_ms']:>10.1f} ms")
    print(f"  TPS, decode only   : {row['tps']:>10.2f} tok/s")
    print(f"  p50 e2e latency    : {row['p50_ms']:>10.1f} ms")
    print(f"  p99 e2e latency    : {row['p99_ms']:>10.1f} ms")
    print(f"  peak VRAM          : {row['peak_vram_gb']:>10.3f} GB")
    print(f"  requests           : {row['n_requests']:>10}")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
