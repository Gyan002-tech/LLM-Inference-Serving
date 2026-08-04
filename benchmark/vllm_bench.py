"""
vLLM concurrency sweep — where vLLM's advantage actually comes from on a T4.

This is deliberately NOT a batch=1 latency race. On a T4 (sm_75) vLLM cannot use
FlashAttention-2 and falls back to xformers/SDPA, so single-request decode is not
expected to beat the HuggingFace baseline by much. vLLM's real win is aggregate
throughput under concurrency, via PagedAttention + continuous batching — so the
experiment sweeps concurrency and watches aggregate throughput climb while
per-request latency degrades.

Workload is identical to baseline.py: the same prompts at exactly 1024 input
tokens, exactly 256 output tokens, greedy. Prompts are sent as TOKEN IDS and the
output length is pinned with min_tokens + ignore_eos, so neither the input nor
the output length can drift from the baseline row.

Load pattern: closed-loop, with `concurrency` worker coroutines pulling from a
shared queue. A worker starts its next request the instant the previous one
returns, so there are no idle barrier gaps between rounds and the measured window
is genuine steady state. This is the standard load-generator shape (wrk, k6 and
locust all work this way) and it matters here because a round-based alternative
would insert a barrier at every round boundary and understate throughput.

Writes results/vllm.json — one row per concurrency level.
Run: python benchmark/vllm_bench.py
"""

import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the exact workload definition. Deliberately NOT importing
# set_determinism(): generation happens in the SERVER process, and seeding torch
# here risks creating a CUDA context in this client that would take VRAM away
# from the server's KV cache. Determinism instead comes from temperature=0, the
# fixed prompts, and the pinned input/output lengths.
from baseline import (  # noqa: E402
    FILLER,
    INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    TOPICS,
    log_gpu,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SERVER_LOG = Path(__file__).resolve().parent.parent / "vllm_server.log"

CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 32   # divisible by every level, and >=30 latency samples for p99
WARMUP_REQUESTS = 4       # discarded — first requests pay allocator/graph warmup
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"

# Set False if you already have `vllm serve` running in another Colab cell.
LAUNCH_SERVER = True
SERVER_TIMEOUT_S = 1500   # model download + engine init + CUDA graph capture on a T4

# Escape hatch for pre-Ampere GPUs: "0" forces vLLM's older V0 engine.
# NOTE: the V0 engine was REMOVED around vLLM 0.11, so this only helps on older
# releases — on 0.11+ it is a no-op. On newer vLLM the equivalent lever is
# VLLM_ATTENTION_BACKEND (see ATTENTION_BACKEND below).
VLLM_USE_V1 = None

# Force a specific attention backend, e.g. "TRITON_ATTN_VLLM_V1". On sm_75 the
# FlashAttention and FlashInfer backends are unavailable; if vLLM cannot pick a
# working backend by itself, naming the Triton one here is the remaining option.
ATTENTION_BACKEND = None

# The validated HuggingFace baseline row (benchmark/baseline.py on a Tesla T4).
# Hardcoded rather than read from results/*.json so the printed comparison is
# always against the canonical row, not whichever baseline file was touched last.
# results/baseline.json on the cu130 stack. See quantization_nf4.py for why the
# 24.45 tok/s sample is treated as an outlier rather than averaged in.
CANONICAL_BASELINE = {
    "ttft_ms": 271.3, "tps": 22.89, "p50_ms": 11415.8,
    "p99_ms": 11674.0, "peak_vram_gb": 3.06,
}


# ─────────────────────────────────────────────────────────────────────
# Workload — identical to baseline.py
# ─────────────────────────────────────────────────────────────────────
def build_prompt_ids(tokenizer) -> list:
    """The same prompts as baseline.build_prompts(), as raw token id lists.

    Sending token IDs instead of text guarantees the server sees EXACTLY
    INPUT_TOKENS tokens. Sending text would let server-side tokenization or an
    injected chat template change the prompt length, and prefill cost scales with
    prompt length, so the comparison would quietly stop being apples-to-apples.
    """
    prompt_ids = []
    for topic in TOPICS:
        text = topic + "\n\n" + FILLER * 40   # overshoot, then truncate to exact length
        ids = tokenizer(text).input_ids[:INPUT_TOKENS]
        assert len(ids) == INPUT_TOKENS, f"prompt is {len(ids)} tokens, need {INPUT_TOKENS}"
        prompt_ids.append(ids)
    return prompt_ids


# ─────────────────────────────────────────────────────────────────────
# Server lifecycle
# ─────────────────────────────────────────────────────────────────────
def preflight_vllm():
    """Confirm vLLM is importable and report the GPU before launching the server.

    The import is probed in a SUBPROCESS on purpose: importing vllm in this client
    process can initialize CUDA and take VRAM away from the server it is about to
    start. nvidia-smi is used for the capability check for the same reason —
    torch.cuda.get_device_capability() would create a context here.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        stderr = probe.stderr.strip()
        cuda_mismatch = re.search(r"libcudart\.so\.(\d+)", stderr)
        if cuda_mismatch:
            want = cuda_mismatch.group(1)
            hint = (
                f"vLLM's compiled extension needs CUDA {want} "
                f"(libcudart.so.{want}), which this runtime does not have — the "
                f"wheel was built against a different CUDA major version than the "
                f"installed torch. Restarting will NOT fix this; install a vLLM "
                f"release built for this runtime's CUDA. See README → 'vLLM on a T4'.")
        else:
            hint = ("If you just pip-installed vLLM, RESTART the Colab runtime — it "
                    "replaces torch and the old one is still loaded in this process.")
        raise RuntimeError(f"vLLM is not importable.\n  {hint}\n\n"
                           f"  probe stderr (tail): {stderr[-800:]}")
    print(f"vLLM {probe.stdout.strip()} importable")

    # A shallow `import vllm` does NOT pull in what the server actually imports
    # (transformers' image stack, and through it torchvision). Probe the real
    # entrypoint so transitive breakage surfaces here in ~20s instead of as an
    # opaque "server exited with code 1" after the launch.
    deep = subprocess.run(
        [sys.executable, "-c", "import vllm.entrypoints.openai.api_server"],
        capture_output=True, text=True)
    if deep.returncode != 0:
        stderr = deep.stderr.strip()
        if "different CUDA major versions" in stderr or "torchvision" in stderr:
            hint = ("torch and torchvision were built against different CUDA major "
                    "versions — reinstalling torch alone desynchronizes them. "
                    "Reinstall BOTH from the same index, e.g.:\n"
                    "    pip install torch==<ver> torchvision "
                    "--index-url https://download.pytorch.org/whl/cu130")
        else:
            hint = "The vLLM server entrypoint could not be imported."
        raise RuntimeError(f"{hint}\n\n  probe stderr (tail): {stderr[-1200:]}")
    print("vLLM server entrypoint imports cleanly")

    cap = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print(f"GPU compute capability: {cap}")
    if cap and float(cap) < 8.0:
        print("  NOTE: pre-Ampere — no FlashAttention-2, so vLLM must fall back to "
              "another attention backend. This is expected and is exactly why this "
              "benchmark sweeps concurrency instead of racing batch=1 latency.\n"
              "  If the server fails to start, set ATTENTION_BACKEND at the top of "
              "this file (e.g. \"TRITON_ATTN_VLLM_V1\") and retry. VLLM_USE_V1=\"0\" "
              "does NOT help on vLLM 0.11+ — the V0 engine was removed.")


def start_server():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--dtype", "float16",              # T4 is sm_75: no bf16 support
        "--max-model-len", "2048",         # workload needs 1024 + 256; keeps KV cache modest
        "--gpu-memory-utilization", "0.85",
        "--port", str(PORT),
    ]
    env = os.environ.copy()
    if VLLM_USE_V1 is not None:
        env["VLLM_USE_V1"] = VLLM_USE_V1
        print(f"Forcing VLLM_USE_V1={VLLM_USE_V1}")
    if ATTENTION_BACKEND is not None:
        env["VLLM_ATTENTION_BACKEND"] = ATTENTION_BACKEND
        print(f"Forcing VLLM_ATTENTION_BACKEND={ATTENTION_BACKEND}")
    print(f"Launching vLLM server (log → {SERVER_LOG}):\n  {' '.join(cmd)}\n")
    log = open(SERVER_LOG, "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def wait_for_server(proc):
    deadline = time.time() + SERVER_TIMEOUT_S
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited with code {proc.returncode} before becoming "
                f"ready — see {SERVER_LOG}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as resp:
                if resp.status == 200:
                    print("Server is ready.\n")
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server not ready within {SERVER_TIMEOUT_S}s — see {SERVER_LOG}")


# ─────────────────────────────────────────────────────────────────────
# Load generation
# ─────────────────────────────────────────────────────────────────────
async def one_request(session, prompt_ids) -> tuple:
    """Issue one completion request. Returns (latency_ms, output_tokens)."""
    payload = {
        "model": MODEL_ID,
        "prompt": prompt_ids,              # token ids → exact input length
        "max_tokens": MAX_NEW_TOKENS,
        "min_tokens": MAX_NEW_TOKENS,      # vLLM extension: exact output length
        "ignore_eos": True,                # vLLM extension: never stop early
        "temperature": 0.0,                # greedy, matching the baseline row
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(f"{BASE_URL}/v1/completions", json=payload) as resp:
        body = await resp.json()
    # No torch.cuda.synchronize() here, and none is possible: the CUDA context
    # lives in the SERVER process. The server only writes the HTTP response after
    # generation has completed, so the response boundary is itself the
    # synchronization point and this client-side clock is honest.
    latency_ms = (time.perf_counter() - t0) * 1000

    if "usage" not in body:
        raise RuntimeError(f"unexpected server response: {body}")
    n_out = body["usage"]["completion_tokens"]
    assert n_out == MAX_NEW_TOKENS, (
        f"expected {MAX_NEW_TOKENS} output tokens, got {n_out} — output length "
        f"must be pinned or throughput is not comparable")
    return latency_ms, n_out


async def run_level(concurrency: int, prompt_ids: list, n_requests: int) -> dict:
    """Closed-loop: `concurrency` workers keep that many requests in flight."""
    queue = asyncio.Queue()
    for i in range(n_requests):
        queue.put_nowait(prompt_ids[i % len(prompt_ids)])

    latencies, tokens = [], []
    timeout = aiohttp.ClientTimeout(total=3600)   # long generations must not be cut off

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker():
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                latency_ms, n_out = await one_request(session, item)
                latencies.append(latency_ms)
                tokens.append(n_out)

        # Wall clock spans the whole steady-state window: workers start together
        # and each immediately picks up new work, so aggregate throughput is not
        # deflated by idle gaps.
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall_s = time.perf_counter() - t0

    return {
        "concurrency": concurrency,
        "aggregate_tps": round(sum(tokens) / wall_s, 2),
        "p99_ms": round(statistics.quantiles(latencies, n=100, method="inclusive")[98], 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "n_requests": n_requests,
        "wall_s": round(wall_s, 2),
    }


async def warmup(prompt_ids):
    """Discarded requests — the first few pay allocator growth and cache effects."""
    print(f"Warmup: {WARMUP_REQUESTS} discarded requests ...")
    await run_level(WARMUP_REQUESTS, prompt_ids, WARMUP_REQUESTS)


# ─────────────────────────────────────────────────────────────────────
# Baseline reference for an honest comparison
# ─────────────────────────────────────────────────────────────────────
def baseline_reference():
    """Baseline throughput on the SAME basis as vLLM's aggregate_tps.

    CANONICAL_BASELINE's "tps" is DECODE-ONLY — baseline.py excludes prefill by
    construction. vLLM's aggregate_tps here is output tokens / wall time, which
    INCLUDES each request's prefill. Quoting the decode-only number as vLLM's
    competition would flatter vLLM, so the comparison uses the baseline
    recomputed end-to-end: MAX_NEW_TOKENS / p50 seconds.
    """
    data = CANONICAL_BASELINE
    return {
        "decode_tps": data["tps"],
        "e2e_tps": round(MAX_NEW_TOKENS / (data["p50_ms"] / 1000), 2),
        "p99_ms": data["p99_ms"],
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    gpu_name = log_gpu()
    preflight_vllm()   # fail in seconds, not after a 25-minute server timeout
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = build_prompt_ids(tokenizer)

    proc = start_server() if LAUNCH_SERVER else None
    try:
        wait_for_server(proc)

        async def sweep():
            await warmup(prompt_ids)
            rows = []
            for concurrency in CONCURRENCY_LEVELS:
                print(f"Concurrency {concurrency}: {REQUESTS_PER_LEVEL} requests "
                      f"({INPUT_TOKENS} in / {MAX_NEW_TOKENS} out) ...")
                row = await run_level(concurrency, prompt_ids, REQUESTS_PER_LEVEL)
                print(f"  aggregate {row['aggregate_tps']:8.2f} tok/s | "
                      f"p50 {row['p50_ms']:8.1f} ms | p99 {row['p99_ms']:8.1f} ms | "
                      f"wall {row['wall_s']:.1f} s")
                rows.append(row)
            return rows

        rows = asyncio.run(sweep())
    finally:
        if proc is not None:
            print("\nShutting down vLLM server ...")
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    payload = {
        "gpu_name": gpu_name,
        "model": MODEL_ID,
        "dtype": "fp16",
        "backend": "vllm",
        # sm_75 has no FlashAttention-2, so vLLM is expected to fall back to
        # xformers/SDPA. The actual backend it chose is printed in vllm_server.log
        # ("Using ... backend") — check there rather than trusting this label.
        "attention_expected": "xformers/SDPA fallback (sm_75, no FlashAttention-2)",
        "input_tokens": INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "requests_per_level": REQUESTS_PER_LEVEL,
        "results": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "vllm.json", "w") as f:
        json.dump(payload, f, indent=2)

    ref = baseline_reference()
    print(f"\n{'─'*70}")
    print(f"vLLM CONCURRENCY SWEEP — {gpu_name} · {MODEL_ID} · fp16")
    print(f"{'─'*70}")
    print(f"{'concurrency':>12}{'aggregate tok/s':>18}{'p50 ms':>10}{'p99 ms':>10}"
          f"{'vs C=1':>9}")
    base_tps = rows[0]["aggregate_tps"]
    for r in rows:
        print(f"{r['concurrency']:>12}{r['aggregate_tps']:>18.2f}{r['p50_ms']:>10.1f}"
              f"{r['p99_ms']:>10.1f}{r['aggregate_tps']/base_tps:>8.2f}x")

    print(f"\nHF baseline reference (canonical benchmark/baseline.py row):")
    print(f"  decode-only TPS         : {ref['decode_tps']:>8.2f} tok/s")
    print(f"  end-to-end TPS          : {ref['e2e_tps']:>8.2f} tok/s  "
          f"← the number comparable to aggregate_tps")
    print(f"  p99 latency             : {ref['p99_ms']:>8.1f} ms")
    print(f"\n  vLLM @ C=1 vs baseline  : {base_tps/ref['e2e_tps']:.2f}x throughput")
    print(f"  vLLM @ C={rows[-1]['concurrency']} vs baseline  : "
          f"{rows[-1]['aggregate_tps']/ref['e2e_tps']:.2f}x throughput")

    print(f"\nSaved {RESULTS_DIR / 'vllm.json'}")


if __name__ == "__main__":
    main()
