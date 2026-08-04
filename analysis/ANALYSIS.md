# Analysis — overhead, roofline, frontier, attribution

Tesla T4 (Colab) · `Qwen/Qwen2.5-1.5B-Instruct` · 1024 input / 256 output tokens · greedy ·
torch 2.11.0+cu130 for every row.

Every number traces to `results/fp16_control.json`, `results/nf4.json`, `results/vllm.json`,
`results/hf_batched.json`, `results/baseline.json`, or a hardware constant stated with its
derivation in `roofline.py`. Nothing is remembered or re-rounded from an earlier write-up.

Reproduce with:

```bash
python analysis/overhead.py       # per-token time breakdown  -> overhead.png
python analysis/roofline.py       # decode roofline model     -> roofline.png
python analysis/frontier.py       # frontier + knee           -> frontier.png
python analysis/attribution.py    # batching vs vLLM          -> attribution.png
```

Each script prints its full numeric derivation to stdout, so every figure below can be checked
against the code that produced it.

---

## 0. Where each token's time goes

![Per-token time breakdown](overhead.png)

Generating one token requires streaming all the weights from HBM, which sets a hard floor in
milliseconds. Everything above that floor is software. All three rows use one basis — output
tokens ÷ wall clock, which *includes* prefill — because that is the only basis vLLM's
`aggregate_tps` supports.

| Configuration | tok/s | ms/token | Bandwidth floor | Overhead | Overhead share |
|---|---:|---:|---:|---:|---:|
| HF fp16, batch=1 | 22.33 | 44.78 | 9.65 | 35.13 | **78.5%** |
| HF NF4, batch=1 | 16.71 | 59.83 | 3.64 | 56.19 | **93.9%** |
| vLLM fp16, C=1 | 60.46 | 16.54 | 9.65 | 6.89 | **41.7%** |

Three readings, in order of importance:

1. **HF fp16 spends 78.5% of each token outside the hardware floor.** The bottleneck at
   batch=1 is per-step software work, not memory.
2. **NF4 lowers the floor to 3.64 ms and pushes overhead up to 56.19 ms**, so its overhead
   share *rises* to 93.9%. It optimised the part that was never the problem.
3. **vLLM runs the identical weights**, so it faces the identical 9.65 ms floor, but carries
   only 6.89 ms of overhead — **5.10× less than HF**. Same hardware, same model, same floor;
   the entire difference is software.

---

## 1. Roofline — why NF4 got slower

![Roofline](roofline.png)

### Hardware constants

| Constant | Value | Derivation |
|---|---:|---|
| Memory bandwidth | 320 GB/s | 256-bit GDDR6 × 10 Gbps ÷ 8 = 320e9 B/s (T4 datasheet) |
| FP16 compute peak | 65 TFLOP/s | T4 **tensor-core** fp16. CUDA-core fp16 is only ~8.1 TFLOP/s |
| **Ridge point** | **203.125 FLOP/byte** | 65e12 ÷ 320e9 |

Choosing the tensor-core peak is deliberately the *hardest* case for a memory-bound claim: it
pushes the ridge as far right as possible. Both alternatives are tested below.

### Model arithmetic

Derived from the measured fp16 weight bytes rather than assumed:

- fp16 weights: 2.875 GiB = **3 087 007 744 bytes**
- parameters = bytes ÷ 2 = **1 543 503 872**
- FLOPs per token = 2 × params = **3 087 007 744** (3.087 GFLOP), since one
  multiply-accumulate is 2 FLOPs and at batch=1 each weight is used exactly once

"Read all weights once per token" is a good approximation here specifically because
Qwen2.5-1.5B **ties its embeddings**: the `lm_head` matmul reads the embedding matrix, and the
input-side gather touches one row and is negligible.

### Both configurations are deeply memory-bound

| | Weight bytes | Intensity | Distance below ridge |
|---|---:|---:|---:|
| fp16 | 3 087 007 744 | **1.0000** FLOP/byte | 203.1× |
| NF4 | 1 163 936 137 | **2.6522** FLOP/byte | 76.6× |

fp16 lands at exactly 1.0 FLOP/byte — 2 FLOPs per parameter over 2 bytes per parameter. Both
points sit two orders of magnitude left of the ridge, which is what "memory-bound" means.

Compute utilisation makes the same point from the other side: fp16 achieves 7.035e10 FLOP/s
and NF4 5.242e10 FLOP/s, which are **0.108%** and **0.081%** of the 65 TFLOP/s peak. The
compute roof is irrelevant at batch=1; the GPU is essentially idle.

### Ceilings versus measurements

| | Bandwidth ceiling | Measured | % of ceiling | Achieved bandwidth |
|---|---:|---:|---:|---:|
| fp16 | 103.66 tok/s | 22.79 | **21.99%** | 70.35 GB/s |
| NF4 | 274.93 tok/s | 16.98 | **6.18%** | 19.76 GB/s |

`roofline.png` plots the same two rows in FLOP/s, the roofline convention, so no label on the
figure mixes units with its FLOP/s y-axis. At 3.087 GFLOP per token:

| | Bandwidth ceiling | Measured | % of ceiling |
|---|---:|---:|---:|
| fp16 | 320 GFLOP/s | 70.4 GFLOP/s | **21.99%** |
| NF4 | 849 GFLOP/s | 52.4 GFLOP/s | **6.18%** |

The percentages are identical in either unit — the FLOP-per-token factor cancels in the ratio,
which is why the argument is unit-independent.

### The key result

**NF4's bandwidth ceiling is 2.652× higher than fp16's** — 274.93 versus 103.66 tok/s, because
there is 2.652× less weight data to stream per token. **Yet NF4 measures 1.342× slower** (16.98
versus 22.79 tok/s).

If decode were limited by bandwidth alone, NF4 would have been 2.65× *faster*. It came out
slower. That single comparison rules bandwidth out as the explanation: the regression must be
compute or overhead added on top of a workload that could not use the bandwidth NF4 freed. NF4
uses just **19.76 GB/s of the T4's 320 GB/s** — it made the data smaller and then left the
saved bandwidth unused.

The per-token decomposition quantifies the trade:

| | Total ms/token | Bandwidth floor | Non-bandwidth |
|---|---:|---:|---:|
| fp16 | 43.879 | 9.647 | 34.232 |
| NF4 | 58.893 | 3.637 | 55.256 |

**NF4 bought 6.010 ms/token of bandwidth time and paid 21.024 ms/token in extra non-bandwidth
time — a net loss of 15.014 ms/token.** That extra time is the NF4→fp16 dequantisation
`bitsandbytes` performs before every matmul, repeated for all **196 quantized linear layers**
(28 blocks × 7 per block, verified below) on every single token — about **77 µs per layer per
token**.

*(These figures use the decode-only basis, so they differ slightly from section 0's
end-to-end basis, which adds ~1 ms/token of amortised prefill.)*

#### Model structure — verified against sources, not assumed

The 196 figure is load-bearing for the per-layer number above, so it is derived rather than
asserted:

| Fact | Value | Source |
|---|---|---|
| `num_hidden_layers` | 28 | [`config.json`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json) |
| `tie_word_embeddings` | `true` | same |
| `hidden_size` / `intermediate_size` | 1536 / 8960 | same |
| Q / KV heads | 12 / 2 (`head_dim` 128) | same, and the model card |
| Attention `nn.Linear` per block | 4 — `q_proj`, `k_proj`, `v_proj`, `o_proj` | `Qwen2Attention.__init__`, transformers `modeling_qwen2.py` |
| MLP `nn.Linear` per block | 3 — `gate_proj`, `up_proj`, `down_proj` | `Qwen2MLP.__init__`, same file |

Q/K/V are three **separate, unfused** Linears (`bias=True`; `o_proj` has `bias=False`), so the
count is 4 + 3 = **7 per block → 28 × 7 = 196**. This matters because a fused QKV projection
would have identical parameter *count* but only 5 modules per block — parameter arithmetic
alone cannot distinguish the two, which is why the module list was checked directly.

`lm_head` is a 197th `nn.Linear`, but `tie_word_embeddings: true` means it shares the embedding
matrix, and transformers excludes tied `lm_head` from bitsandbytes conversion, so it stays fp16
and is **not** among the 196.

Summing the verified shapes gives 1 543 714 304 parameters (1 310 340 608 non-embedding),
matching the model card's stated "1.54B" and "1.31B" exactly. At 2 bytes each that is
3 087 428 608 B = **2.875392 GiB**, which rounds to the **2.875** recorded in
`fp16_control.json`. The parameter count used above (1 543 503 872, derived from the rounded
2.875 GiB field) is 0.014% below the exact figure — JSON rounding, and it moves no conclusion.

On the plot this is one motion: NF4 moves **right** (higher intensity, higher ceiling) and
**down** (lower achieved performance), ending further below its own roof than fp16 is below
its.

### Cross-check against the vLLM row

vLLM runs the *same* fp16 weights, so it faces the identical 9.647 ms/token bandwidth floor. At
concurrency 1 it achieves 60.46 tok/s = 16.540 ms/token, implying only **6.893 ms/token** of
non-bandwidth time. HuggingFace `generate()` carries **4.97× that overhead** (34.232 ms) on
identical weights. Two independent measurements therefore agree that the batch=1 bottleneck is
per-step overhead, not memory bandwidth — which is also why quantization, an intervention aimed
squarely at bandwidth, could not help.

### Sensitivity — the conclusion does not rest on the constants

| Alternative constant | Effect | Conclusion |
|---|---|---|
| Bandwidth 300 GB/s | fp16 ceiling 97.18 (23.45% achieved); NF4 257.75 (6.59%) | unchanged |
| CUDA-core peak 8.1 TFLOP/s | ridge falls to 25.31 FLOP/byte; fp16 still 25.3× below, NF4 9.5× below | unchanged |

Neither alternative changes the ordering, the memory-bound verdict, or the fact that NF4 holds
the higher ceiling and the lower measurement.

### Unit discrepancy, flagged

`results/*.json` computes VRAM as `bytes / 1024**3`, so the field named `weights_vram_gb` is
really **GiB**. Reading it as decimal GB would give ceilings of 111.30 and 295.20 tok/s instead
of the correct 103.66 and 274.93 — a 7.4% difference. This analysis uses the GiB conversion
throughout. The ceiling *ratio* (2.652×) is unit-independent, so the argument is unaffected
either way; the absolute ceilings are not, which is why it is stated rather than silently
absorbed.

---

## 2. Throughput–latency frontier

![Frontier](frontier.png)

`aggregate_tps` is output tokens ÷ wall time and therefore **includes prefill**. The comparable
baseline figure is not the decode-only 22.79 tok/s but the end-to-end recomputation
`256 ÷ (11463.4 / 1000) = 22.33 tok/s`, plotted at the baseline's own p99 of 12 031.2 ms.

| Concurrency | p99 (ms) | Aggregate tok/s | Scaling vs C=1 | Scaling efficiency | vs HF baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 367.0 | 60.46 | 1.000× | 100.0% | 2.71× |
| 2 | 4 562.3 | 112.65 | 1.863× | 93.2% | 5.04× |
| 4 | 4 917.4 | 208.44 | 3.448× | 86.2% | 9.33× |
| 8 | 5 932.6 | 345.85 | 5.720× | 71.5% | 15.49× |

### Marginal cost of each step

| Step | Throughput gain | p99 cost | Elasticity | Marginal rate |
|---|---:|---:|---:|---:|
| C=1→2 | +86.32% (+52.19 tok/s) | +4.47% (+195.3 ms) | **19.30** | 0.267 tok/s per ms |
| C=2→4 | +85.03% (+95.79 tok/s) | +7.78% (+355.1 ms) | **10.93** | 0.270 tok/s per ms |
| C=4→8 | +65.92% (+137.41 tok/s) | +20.65% (+1 015.2 ms) | **3.19** | 0.135 tok/s per ms |

### The saturation knee is C=4

Below the knee, 1% of added p99 latency buys **10.93%** more throughput. Above it, the same 1%
buys only **3.19%**. Marginal efficiency collapses by **3.42×** at that point; on the absolute
measure the rate halves, from 0.270 to 0.135 tok/s per millisecond of added p99.

Read plainly: **up to concurrency 4 throughput is nearly free** — the engine is filling idle
GPU time that a single stream leaves on the table, and p99 moves only 4367 → 4562 → 4917 ms
(a 12.6% band across a 3.45× throughput gain). **Past concurrency 4 you begin buying throughput
with latency**: C=8 still delivers a real 1.66× gain, but the queue is now deep enough that
requests wait on each other and p99 jumps 20.7%.

That makes C=4 the right operating point for a latency-sensitive deployment and C=8 a
deliberate throughput-for-latency trade. The trade remains strongly favourable in absolute
terms: even at C=8, p99 of 5 932.6 ms is **2.03× lower than the HF baseline's 12 031.2 ms**
while delivering 15.49× the throughput.

Aggregate throughput at C=8 **exceeds the batch=1 bandwidth ceiling of 103.66 tok/s by 3.34×**
— the point of batching. Against the batched ceiling (8 × 103.66 = 829.3 tok/s) it is 41.7%.

### Measurement caveats

- This is a **closed-loop saturation test**, not open-loop Poisson arrivals. At concurrency ≥ 2
  requests complete in lockstep (hence p99 ≈ p50), so p99 here means "latency under sustained
  saturation," not tail latency under bursty traffic.
- 32 requests per level means concurrency 8 is only 4 batch generations, so its p99 is
  effectively the slowest batch rather than a true distribution tail.
- vLLM peak VRAM is **not** comparable to the HF rows and is deliberately not reported: the
  server preallocates a KV-cache pool sized by `--gpu-memory-utilization 0.85`.
- The attention backend vLLM selected is recorded in `vllm_server.log` ("Using ... backend");
  `attention_expected` in the JSON is a hypothesis, not a measurement.

---

## 3. Attribution — how much of vLLM's win is actually batching?

![Attribution](attribution.png)

Sections 1 and 2 compare vLLM (batched) against HuggingFace at batch=1, which conflates two
different things: **batching**, and **vLLM**. `benchmark/hf_batched.py` runs HF `generate()`
with *static* batching at the same batch sizes, using the workload and timing harness imported
from `baseline.py`, so the measured gap splits multiplicatively.

The batch=1 drift check passed at **+1.65%** against the same-session fp16 control, so the
batched rows are comparable to the other rows.

### HF static batching scales well — until batch 8

| Batch | Aggregate tok/s | Scaling | Efficiency | p50 (ms) | p99 (ms) | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.70 | 1.000× | 100.0% | 11 241.4 | 12 206.1 | 3.060 GiB |
| 2 | 43.63 | 1.922× | **96.1%** | 11 704.2 | 12 824.4 | 3.230 GiB |
| 4 | 78.67 | 3.466× | **86.6%** | 13 016.7 | 13 091.1 | 3.572 GiB |
| 8 | 104.57 | 4.607× | **57.6%** | 19 565.9 | 19 660.5 | 4.253 GiB |

Per-batch wall time tells the story directly: p50 goes 11 241 → 11 704 → 13 017 → **19 566 ms**,
i.e. **+4.1%, +15.8%, +74.1%** for 2×, 4×, 8× the work. Doubling from 4 to 8 is where it breaks.

*Note on p50/p99: in static batching every request in a batch finishes when the batch finishes,
so per-request latency IS the batch wall time. For batch ≥ 2 that makes p99 ≈ p50, because the
32 latency samples are only 16/8/4 distinct values duplicated within batches. batch=1 is the
exception — every sample is an independent batch there, so its 8.6% p50→p99 spread is genuine
run-to-run variance, not a queueing tail.*

### Head-to-head at matched levels

| Level | HF static tok/s | vLLM tok/s | **vLLM / HF** | HF efficiency | vLLM efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.70 | 60.46 | **2.66×** | 100.0% | 100.0% |
| 2 | 43.63 | 112.65 | **2.58×** | 96.1% | 93.2% |
| 4 | 78.67 | 208.44 | **2.65×** | 86.6% | 86.2% |
| 8 | 104.57 | 345.85 | **3.31×** | 57.6% | **71.5%** |

### The ratio is flat — and that identifies the mechanism

**vLLM's advantage is constant at 2.58–2.66× across batch 1, 2 and 4 — a spread of 0.081×.** A
*constant* multiplier is the signature of fixed per-step overhead, not of smarter scheduling: a
scheduling win would grow with batch size. It does not, until batch 8.

This independently confirms the roofline decomposition in section 1, which found HF carrying
34.2 ms/token of non-bandwidth overhead against vLLM's 6.9 ms — a 4.97× overhead ratio that
surfaces here as a ~2.6× throughput ratio at every batch size.

**HuggingFace's batching is not the weak component.** Its scaling efficiency matches vLLM's at
batch 2 (96.1% vs 93.2%) and is level with it at batch 4 (86.6% vs 86.2% — a 0.4-point gap,
inside the noise floor and not a real difference). Only at batch 8 does HF fall behind, to
57.6% against 71.5%.

### At batch 8, HuggingFace leaves the hardware idle

The batch=8 collapse is not a hardware roof. At 104.57 tok/s, batch 8 runs 13.07 decode steps
per second, so weight traffic is 13.07 × 3.087 GB = **40.4 GB/s, just 12.6% of the T4's
320 GB/s**, and compute is 323 GFLOP/s = **0.50% of the 65 TFLOP/s peak**. vLLM at C=8 reaches
133.5 GB/s — **41.7%**. HuggingFace is idling three-quarters of the memory bus for a software
reason.

### A numerical coincidence to *not* over-read

HF static at batch 8 measures **104.57 tok/s**; the fp16 single-stream bandwidth ceiling from
section 1 is **103.66 tok/s**. They agree to 0.9%, which invites the conclusion "static batching
saturates the memory-bandwidth roofline."

That reading is wrong. The 103.66 ceiling assumes one weight load per token. At batch 8 a single
weight load produces 8 tokens, so the applicable ceiling is 8 × 103.66 = **829.3 tok/s**, and
104.57 is **12.6%** of it — exactly matching the 40.4 GB/s measured above. The coincidence has
no mechanism behind it and is recorded here only so nobody mistakes it for one.

### The attribution

At batch/concurrency 8, against the baseline of 22.33 tok/s:

| Component | Factor | What it actually targets |
|---|---:|---|
| Plain static batching | **4.68×** | amortises per-step overhead across sequences |
| vLLM engine efficiency (mean of the flat region) | **2.63×** | removes the per-step overhead itself |
| Scaling retention at batch 8 (residual) | **1.26×** | the part plausibly from continuous batching + paged KV |
| **Total** | **15.49×** | reconciles exactly with the measured 15.49× |

Static batching alone reaches **30.2%** of vLLM's throughput at batch 8. So most of the original
15.5× headline is *batching*, which HuggingFace does perfectly well; and most of what remains is
vLLM's *engine*, not its scheduler. Only the modest 1.26× is plausibly attributable to
continuous batching and paged KV.

The latency side is the other half: at batch 8 HF static p99 is **19 660 ms against vLLM's
5 933 ms — 3.31× worse**. Static batching buys throughput by making every request wait for the
entire batch, so it degrades the metric users actually feel.

### Caveat: this is the best case for static batching

Every sequence is pinned to exactly 256 output tokens with identical 1024-token prompts, so the
batch is perfectly rectangular. No sequence finishes early, so no batch slot ever sits idle —
**which is precisely the inefficiency continuous batching exists to eliminate.**

The data confirms this structurally. At level 8 both systems processed 32 requests as **four
lockstep waves of eight**: vLLM's 23.69 s wall time over four waves gives 5.92 s per wave,
matching its measured p50 of 5 921.9 ms, and HF's 78.34 s gives 19.59 s per wave, matching its
19 565.9 ms. Because both ran the same wave structure, the throughput ratio and the p99 ratio
come out to the *same* 3.31× — not a coincidence but an identity. With output lengths pinned,
vLLM's continuous batching has degenerated into static batching, and what the 3.31× measures is
purely how much faster vLLM executes an identical batch.

**These ratios are therefore a conservative floor** on vLLM's advantage under real, ragged
traffic where output lengths vary. What this row cannot measure, by construction, is the
continuous-batching refill benefit; a follow-up with sampled output lengths would be needed.

### VRAM: activations dominate, and PagedAttention's memory argument does not bite here

| Batch | Peak VRAM | Δ vs batch=1 | Δ per added sequence |
|---:|---:|---:|---:|
| 1 | 3.060 GiB | — | — |
| 2 | 3.230 GiB | +0.170 | 0.170 GiB (≈174 MiB) |
| 4 | 3.572 GiB | +0.512 | 0.171 GiB |
| 8 | 4.253 GiB | +1.193 | 0.170 GiB |

Growth is strikingly linear at ≈174 MiB per sequence. The KV cache for this workload is only
≈35 MiB per sequence (28 layers × 2 × 1280 tokens × 2 KV heads × 128 dim × 2 bytes, GQA), so
**≈139 MiB — about 80% of the growth — is activations, not KV cache.**

**No batch size hit OOM**; peak was 4.253 GiB on a 16 GiB card. The PagedAttention
memory-efficiency argument is real in general but does **not** apply at this scale, and the
honest reading is to concede that rather than imply HF was memory-limited.

---

## 4. Reproducibility — the same suite, run twice on two library stacks

Every row above was re-measured end to end after a torch upgrade (cu128 → cu130). The earlier
run is preserved in `results/*_v1*.json`. This is the most useful validity evidence in the
repo, because it re-derives every conclusion from independent measurements.

| Measurement | Run 1 (cu128) | Run 2 (cu130) | Change |
|---|---:|---:|---:|
| Baseline decode TPS | 22.45 | 22.89 | +1.96% |
| fp16 control decode TPS | 22.37 | 22.79 | +1.88% |
| NF4 decode TPS | 16.24 | 16.98 | +4.56% |
| NF4 weights (GiB) | 1.074 | 1.084 | +0.93% |
| vLLM C=1 | 59.18 | 60.46 | +2.16% |
| vLLM C=8 | 344.17 | 345.85 | +0.49% |
| HF static batch=8 | 103.79 | 104.57 | +0.75% |
| WikiText-2 PPL (fp16 / NF4) | 9.2123 / 9.9256 | 9.2123 / 9.9256 | **0.0000%** |
| HF static peak VRAM (b=1/2/4/8) | 3.06 / 3.23 / 3.572 / 4.253 | identical | **0%** |

**What survived unchanged:** every qualitative conclusion. NF4 is slower (−27.4% → −25.5%); the
knee is at C=4 in both runs; the vLLM/HF ratio is flat then jumps at 8 in both (2.68–2.75 →
2.58–2.66); the attribution barely moves (4.73/2.70/1.23 → 4.68/2.63/1.26).

**What the second run tightened:**

- **Perplexity is byte-identical across stacks**, confirming the accuracy path is fully
  deterministic while the timing path is not. Any perplexity difference would have been a bug.
- **Peak-VRAM figures are identical to three decimals** across both runs at every batch size,
  confirming the memory accounting is exactly reproducible.
- **A run-1 anomaly resolved.** In run 1, p99 *fell* from C=1 to C=2 (−6.41%), which was
  attributed to C=1's p99 being inflated by variance (its p50→p99 spread was 15.3% versus 1.7%
  at C≥2). Run 2 confirms that reading: C=1's spread is 2.5% (4258.8 → 4367.0) and p99 now rises
  monotonically across all three steps. The explanation was right, and the anomaly was noise.
- **A claim was retired.** Run 1 appeared to show HF *out-scaling* vLLM at batch 2 and 4 (97.1%
  vs 94.6%, 89.3% vs 87.7%). Run 2 gives 96.1% vs 93.2% and 86.6% vs 86.2% — at batch 4 that is
  a 0.4-point gap. The honest statement is that HF **matches** vLLM's scaling efficiency below
  batch 8, not that it beats it.

**One correction worth recording.** An intermediate baseline sample measured 24.45 tok/s and was
briefly treated as evidence that cu130 was ~9% faster. Four further samples (22.37, 22.45, 22.79,
22.89) place it as an outlier session roughly 8% fast; the real stack difference is ~2%. The
drift gate in `hf_batched.py` caught that session at +12.7% and refused to build a comparison on
it, which is exactly its purpose.

**Established noise floor:** six independent fp16 baseline measurements (22.08, 22.37, 22.45,
22.79, 22.89, 24.45) — five within 3.7%, one outlier. Within-run p50→p99 spread is 2–3%. So
cross-session differences below ~3% are not claimed as findings anywhere in this repo, and the
comparisons that matter most are same-session (NF4 vs its own control) or self-relative
(scaling efficiency within one run), both of which are tighter than that.

---

## What the four analyses say together

All four are the same finding approached from different directions.

The **overhead breakdown** shows HF fp16 spending 78.5% of every token outside the hardware
floor. The **roofline** confirms why: batch=1 decode leaves ~78% of the bandwidth ceiling and
~99.9% of the compute peak unused, so the bottleneck is per-step software. **Quantization**
attacks bandwidth, which was never the constraint, and made things worse by adding
dequantisation compute to a pipeline already dominated by overhead — its overhead share rises
to 93.9%. The **frontier** shows batching reclaiming that idle capacity, with aggregate
throughput at C=8 exceeding the single-stream ceiling by 3.34×. And the **attribution** row
confirms the diagnosis independently: the vLLM-over-HF ratio is flat at ~2.6× regardless of
batch size, which is what a fixed per-step overhead looks like.

Every lever measured, ranked, at batch/concurrency 8 against the HF batch=1 baseline:

| Lever | Factor | What it actually targets |
|---|---:|---|
| Static batching (plain HuggingFace) | **4.68×** | amortises per-step overhead across sequences |
| vLLM engine efficiency | **2.63×** | removes the per-step overhead itself (constant across batch sizes) |
| vLLM scaling retention at batch 8 | **1.26×** | the part plausibly attributable to continuous batching + paged KV |
| NF4 quantization | **0.75×** | bandwidth — which was never the constraint |

**Pick the optimization that targets your actual bottleneck.** On this hardware the bottleneck
was per-step software overhead — so the winning levers were batching and a faster engine, in
that order. It was *not* memory bandwidth, which is why quantization lost 25%. And it was *not*
scheduling either: that turned out to be the smallest factor of the four, and this workload's
pinned equal lengths mean even the 1.26× is a floor rather than a fair test of it.
