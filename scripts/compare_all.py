"""
Single entry point — reproduce every row and every insight in this repo.

This file ORCHESTRATES; it contains no benchmark or plotting logic of its own.
Every stage is one of the existing scripts, invoked as-is:

    benchmark/baseline.py          the reference row (fp16, batch=1)
    benchmark/quantization_nf4.py  NF4 row + fp16 control + perplexity
    benchmark/hf_batched.py        HF static batching at matched batch sizes
    benchmark/vllm_bench.py        vLLM concurrency sweep
    analysis/frontier.py           throughput-latency frontier + knee  -> frontier.png
    analysis/roofline.py           decode roofline model               -> roofline.png

Why the GPU stages run as SUBPROCESSES rather than `import x; x.main()`:

  1. VRAM accounting. quantization_nf4 reads torch.cuda.memory_allocated()
     straight after loading, to report weights-at-rest. If baseline's model were
     still awaiting garbage collection in the same process, that reading would
     include it and the headline NF4 memory finding would be wrong. A fresh
     process makes the reading unconditionally correct.
  2. The vLLM stage launches its server as a child process sized by
     --gpu-memory-utilization. Any CUDA context held by THIS process would take
     VRAM away from that server. Keeping the orchestrator CUDA-free is the only
     way its numbers stay comparable.
  3. One stage crashing (a missing dependency, an OOM) then does not abort the
     rest of the suite.

The two analysis stages are pure post-processing — no GPU, no torch — so those
are imported and called directly.

Usage:
    python scripts/compare_all.py                  # run everything still missing
    python scripts/compare_all.py --list           # show the plan and exit
    python scripts/compare_all.py --only nf4       # run selected stages
    python scripts/compare_all.py --skip vllm      # run everything except one
    python scripts/compare_all.py --force          # re-run even if results exist
    python scripts/compare_all.py --analysis-only  # just regenerate the plots
"""

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "benchmark"
ANALYSIS_DIR = ROOT / "analysis"
RESULTS_DIR = ROOT / "results"


def hf_batched_is_complete(path: Path) -> bool:
    """An aborted hf_batched.json still exists on disk.

    hf_batched.py deliberately saves its partial result when the batch=1 drift
    check fails, so the measurement isn't lost. Without this check the file's mere
    existence would mark the stage complete and it would be skipped forever with
    only one of four batch sizes measured.
    """
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return (not data.get("drift_check_aborted", False)
            and len(data.get("results", [])) >= 2)


@dataclass
class Stage:
    key: str
    title: str
    path: Path
    outputs: list           # existing artifacts mean "already done"
    requires: list          # importable module names this stage needs
    minutes: int
    mode: str = "subprocess"   # "subprocess" (GPU) or "import" (CPU analysis)
    critical: bool = False     # abort the suite if this one fails
    note: str = ""
    # optional callable(Path) -> bool, for outputs that can exist but be partial
    validate: object = None


STAGES = [
    Stage(
        key="baseline",
        title="Baseline — HF fp16, KV cache on, batch=1",
        path=BENCHMARK_DIR / "baseline.py",
        outputs=[RESULTS_DIR / "baseline.json"],
        requires=["torch", "transformers", "numpy"],
        minutes=10,
        critical=True,
        note="Every other row is measured against this one.",
    ),
    Stage(
        key="nf4",
        title="NF4 4-bit quantization + WikiText-2 perplexity",
        path=BENCHMARK_DIR / "quantization_nf4.py",
        outputs=[RESULTS_DIR / "nf4.json", RESULTS_DIR / "fp16_control.json"],
        requires=["torch", "transformers", "datasets", "bitsandbytes"],
        minutes=35,
        note='Needs a current bitsandbytes: pip install -U "bitsandbytes>=0.46.1"',
    ),
    Stage(
        key="hf_batched",
        title="HF static batching at batch 1/2/4/8",
        path=BENCHMARK_DIR / "hf_batched.py",
        outputs=[RESULTS_DIR / "hf_batched.json"],
        requires=["torch", "transformers"],
        minutes=16,
        validate=hf_batched_is_complete,
        note="Must run on the SAME library stack as baseline — it drift-checks "
             "against it and stops if batch=1 moved more than 10%.",
    ),
    Stage(
        key="vllm",
        title="vLLM concurrency sweep at C=1/2/4/8",
        path=BENCHMARK_DIR / "vllm_bench.py",
        outputs=[RESULTS_DIR / "vllm.json"],
        requires=["vllm", "aiohttp"],
        minutes=20,
        note="RUNS LAST ON PURPOSE: installing vllm replaces torch, which can "
             "break the transformers/bitsandbytes stack the HF stages need. "
             "Install it, restart the runtime, then run --only vllm.",
    ),
    Stage(
        key="overhead",
        title="Analysis — per-token time breakdown (bandwidth floor vs overhead)",
        path=ANALYSIS_DIR / "overhead.py",
        outputs=[ANALYSIS_DIR / "overhead.png"],
        requires=["matplotlib"],
        minutes=1,
        mode="import",
        note="Reads fp16_control.json + nf4.json + vllm.json.",
    ),
    Stage(
        key="attribution",
        title="Analysis — batching vs vLLM attribution",
        path=ANALYSIS_DIR / "attribution.py",
        outputs=[ANALYSIS_DIR / "attribution.png"],
        requires=["matplotlib", "numpy"],
        minutes=1,
        mode="import",
        note="Reads hf_batched.json + vllm.json + fp16_control.json.",
    ),
    Stage(
        key="frontier",
        title="Analysis — throughput-latency frontier + saturation knee",
        path=ANALYSIS_DIR / "frontier.py",
        outputs=[ANALYSIS_DIR / "frontier.png"],
        requires=["matplotlib"],
        minutes=1,
        mode="import",
        note="Reads vllm.json + fp16_control.json.",
    ),
    Stage(
        key="roofline",
        title="Analysis — decode roofline model",
        path=ANALYSIS_DIR / "roofline.py",
        outputs=[ANALYSIS_DIR / "roofline.png"],
        requires=["matplotlib", "numpy"],
        minutes=1,
        mode="import",
        note="Reads fp16_control.json + nf4.json + vllm.json.",
    ),
]

STAGE_BY_KEY = {s.key: s for s in STAGES}
ANALYSIS_KEYS = [s.key for s in STAGES if s.mode == "import"]


# ─────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────
def missing_requirements(stage: Stage) -> list:
    """Which of a stage's dependencies are not importable.

    find_spec() locates without importing, so this cannot blow up on a package
    that is installed but broken (e.g. a vLLM built for the wrong CUDA). Each
    stage does its own deeper preflight; this is only about whether to attempt.
    """
    return [m for m in stage.requires if importlib.util.find_spec(m) is None]


def log_gpu_via_baseline():
    """Report the GPU using baseline.py's own logger rather than a second copy.

    Importing baseline pulls in torch but does NOT create a CUDA context, and
    log_gpu() only shells out to nvidia-smi — so the orchestrator stays
    CUDA-free, which the vLLM stage depends on.
    """
    sys.path.insert(0, str(BENCHMARK_DIR))
    try:
        from baseline import log_gpu
        return log_gpu()
    except Exception as exc:
        print(f"Could not read GPU via baseline.log_gpu(): "
              f"{type(exc).__name__}: {exc}")
        return "unknown"


def done(stage: Stage) -> bool:
    if not all(p.exists() for p in stage.outputs):
        return False
    if stage.validate is not None:
        return all(stage.validate(p) for p in stage.outputs)
    return True


# ─────────────────────────────────────────────────────────────────────
# Running
# ─────────────────────────────────────────────────────────────────────
def run_subprocess(stage: Stage) -> int:
    """Fresh interpreter per GPU stage — see the module docstring for why."""
    cmd = [sys.executable, str(stage.path)]
    print(f"  $ {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def run_import(stage: Stage) -> int:
    """In-process for the CPU-only analysis stages."""
    sys.path.insert(0, str(stage.path.parent))
    module_name = stage.path.stem
    print(f"  >>> import {module_name}; {module_name}.main()\n")
    module = importlib.import_module(module_name)
    module.main()
    return 0


def run_stage(stage: Stage) -> tuple:
    """Returns (status, seconds). status in {ok, failed, skipped-deps}."""
    header = f"  STAGE: {stage.key} — {stage.title}"
    print("\n" + "=" * 78)
    print(header)
    print("=" * 78)
    if stage.note:
        print(f"  note: {stage.note}")

    absent = missing_requirements(stage)
    if absent:
        print(f"  SKIPPED — missing dependencies: {', '.join(absent)}")
        return "skipped-deps", 0.0

    t0 = time.perf_counter()
    try:
        code = run_subprocess(stage) if stage.mode == "subprocess" else run_import(stage)
    except Exception as exc:                       # noqa: BLE001
        print(f"  FAILED — {type(exc).__name__}: {exc}")
        return "failed", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0

    if code != 0:
        print(f"\n  FAILED — exit code {code} after {elapsed / 60:.1f} min")
        return "failed", elapsed
    print(f"\n  OK — {elapsed / 60:.1f} min")
    return "ok", elapsed


# ─────────────────────────────────────────────────────────────────────
# Consolidated summary (reads the JSONs; no numbers are recomputed here)
# ─────────────────────────────────────────────────────────────────────
def read(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def summarise():
    print("\n" + "=" * 78)
    print("  CONSOLIDATED RESULTS")
    print("=" * 78)

    fp16 = read("fp16_control.json") or read("baseline.json")
    nf4 = read("nf4.json")
    hfb = read("hf_batched.json")
    vllm = read("vllm.json")

    print(f"{'configuration':<34}{'tok/s':>10}{'basis':>10}{'p50 ms':>10}"
          f"{'p99 ms':>10}{'VRAM':>8}")
    print("-" * 78)

    def line(label, tps, basis, p50, p99, vram, ppl=None):
        vram_cell = f"{vram:.2f}" if vram is not None else "n/a"
        print(f"{label:<34}{tps:>10.2f}{basis:>10}{p50:>10.0f}{p99:>10.0f}"
              f"{vram_cell:>8}" + (f"   ppl {ppl}" if ppl is not None else ""))

    # "tps" can be absent if a row was produced with its timing pass skipped
    # (quantization_nf4.py's MEASURE_FP16_TIMING = False), so check before use.
    if fp16 and "tps" in fp16:
        line("HF fp16 batch=1 (baseline)", fp16["tps"], "decode",
             fp16["p50_ms"], fp16["p99_ms"], fp16.get("peak_vram_gb"),
             fp16.get("perplexity_wikitext2"))
    if nf4 and "tps" in nf4:
        line("HF NF4 batch=1", nf4["tps"], "decode", nf4["p50_ms"],
             nf4["p99_ms"], nf4.get("peak_vram_gb"),
             nf4.get("perplexity_wikitext2"))
    if hfb:
        for r in hfb.get("results", []):
            if r.get("oom"):
                print(f"{'HF fp16 static batch=' + str(r['batch_size']):<34}"
                      f"{'OOM':>10}")
                continue
            line(f"HF fp16 static batch={r['batch_size']}", r["aggregate_tps"],
                 "aggregate", r["p50_ms"], r["p99_ms"], r.get("peak_vram_gb"))
    if vllm:
        for r in vllm.get("results", []):
            line(f"vLLM fp16 concurrency={r['concurrency']}", r["aggregate_tps"],
                 "aggregate", r["p50_ms"], r["p99_ms"], None)

    print("\n  'decode' excludes prefill: (n-1)/(total - TTFT).")
    print("  'aggregate' is output tokens / wall time and INCLUDES prefill.")
    print("  Never compare the two directly — see analysis/ANALYSIS.md.")
    print("  vLLM VRAM is omitted: its server preallocates a KV pool by design.")

    missing = [n for n, d in (("fp16_control.json/baseline.json", fp16),
                              ("nf4.json", nf4), ("hf_batched.json", hfb),
                              ("vllm.json", vllm)) if d is None]
    if missing:
        print(f"\n  Not yet measured: {', '.join(missing)}")

    print("\n  Plots:   analysis/frontier.png, analysis/roofline.png")
    print("  Write-up: analysis/ANALYSIS.md  (the frontier knee, the roofline")
    print("            argument, and the batching-vs-vLLM attribution)")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def build_plan(args) -> list:
    keys = [s.key for s in STAGES]
    if args.analysis_only:
        keys = list(ANALYSIS_KEYS)
    if args.only:
        keys = [k for k in keys if k in args.only]
    if args.skip:
        keys = [k for k in keys if k not in args.skip]
    return [STAGE_BY_KEY[k] for k in keys]


def main():
    parser = argparse.ArgumentParser(
        description="Run every benchmark and analysis stage in this repo.")
    choices = [s.key for s in STAGES]
    parser.add_argument("--only", nargs="+", choices=choices, metavar="STAGE",
                        help="run only these stages")
    parser.add_argument("--skip", nargs="+", choices=choices, metavar="STAGE",
                        help="run everything except these stages")
    parser.add_argument("--force", action="store_true",
                        help="re-run stages whose outputs already exist")
    parser.add_argument("--analysis-only", action="store_true",
                        help="only regenerate the plots from existing JSONs")
    parser.add_argument("--list", action="store_true",
                        help="print the plan and exit without running anything")
    args = parser.parse_args()

    plan = build_plan(args)

    print("=" * 78)
    print("  LLM Inference / Serving Benchmark — full reproduction")
    print("=" * 78)
    total_minutes = sum(s.minutes for s in plan
                        if args.force or not done(s))
    print(f"  stages selected : {len(plan)}")
    print(f"  estimated time  : ~{total_minutes} min "
          f"({'forcing re-runs' if args.force else 'skipping already-complete stages'})")
    print(f"\n{'stage':<12}{'mode':<12}{'est':>6}  status")
    print("-" * 78)
    for s in plan:
        absent = missing_requirements(s)
        if absent:
            status = f"deps missing: {', '.join(absent)}"
        elif done(s) and not args.force:
            status = "already complete (use --force to redo)"
        else:
            status = "will run"
        print(f"{s.key:<12}{s.mode:<12}{s.minutes:>4}m  {status}")

    if args.list:
        print("\n--list given; nothing run.")
        return

    if any(s.mode == "subprocess" for s in plan):
        print()
        log_gpu_via_baseline()

    outcomes = []
    for stage in plan:
        if done(stage) and not args.force:
            print(f"\n  SKIPPING {stage.key} — outputs already present "
                  f"({', '.join(p.name for p in stage.outputs)}). --force to redo.")
            outcomes.append((stage, "already-done", 0.0))
            continue
        status, elapsed = run_stage(stage)
        outcomes.append((stage, status, elapsed))
        if status == "failed" and stage.critical:
            print(f"\n  {stage.key} is critical — every other row is measured "
                  f"against it. Aborting the run.")
            break

    print("\n" + "=" * 78)
    print("  RUN SUMMARY")
    print("=" * 78)
    for stage, status, elapsed in outcomes:
        print(f"  {stage.key:<12}{status:<16}{elapsed / 60:>6.1f} min")

    if any(st == "skipped-deps" and s.key == "vllm" for s, st, _ in outcomes):
        print("\n  vLLM was skipped. To add that row:")
        print('    pip install -q vllm     # then RESTART the runtime')
        print("    python scripts/compare_all.py --only vllm frontier roofline")

    summarise()


if __name__ == "__main__":
    main()
