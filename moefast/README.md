# moefast-35B

**Benchmark — first-token latency (TTFT) on a single 16 GB GPU, MTP enabled in all three stacks**

Hardware: **RTX 5060 Ti 16 GB** (driver 580.126.20, CUDA 13.0) · **AMD Ryzen 9 9900X** (24 threads) ·
**30 GB RAM** · Ubuntu 24.04 (kernel 7.0.0-38). Model: our fine-tuned merge of
**Qwen3.6-35B-A3B UD-Q4_K_XL** (753 tensors, MTP block included, 22.85 GB).

Measurement: streaming request, greedy, `max_tokens=32`, fresh prompt per run (no prompt-cache
reuse), 3 runs per size, **median TTFT** (time until the first streamed token).

| Stack | Config (all with MTP) | TTFT ~50 tok | TTFT ~1K tok | TTFT ~4K tok |
|---|---|---:|---:|---:|
| **moefast (this fork)** | `-ngl 99 -ot exps=CPU -c 8192 -b 2048 -ub 2048 --moe-cache-mib auto` + heat pre-seed + `--spec-type draft-mtp --spec-draft-n-max 3` | **0.266 s** | **0.595 s** | **1.275 s** |
| **llama.cpp (vanilla, our patches off)** | same, but `--moe-cache-mib 0` (no expert cache, no heat) | 0.562 s | 0.962 s | 1.942 s |
| **Strata (v0.1.45, 16 GB instance)** | `serve/server.py`, `--max-context 8192 --prefill auto --mtp 2`, greedy sampling, same GGUF | **0.191 s** | 0.717 s | 1.417 s |

| | small | medium | large |
|---|---:|---:|---:|
| moefast vs vanilla llama.cpp | **2.1× faster** | **1.6× faster** | **1.5× faster** |
| moefast vs Strata | 1.4× slower | **1.2× faster** | **1.1× faster** |

Reading the table: Strata wins the tiny-prompt case (its engine prefill path has less fixed overhead),
while the MoE expert cache pays off as the prompt grows — moefast is the fastest from ~1K tokens up,
which is where real agent / document workloads live. Vanilla llama.cpp (experts on CPU, no cache) is
the slowest in every size.

---

A small set of patches on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) that make a
30B-class MoE model (Qwen3.6-35B-A3B, `qwen35moe`) run fast on a **single 16 GB consumer GPU**.

Everything here is build-from-source. There are no prebuilt binaries on purpose: they would have to
be built per CUDA arch (`sm_75/86/89/120`), per OS and per CUDA version.

## What it does

| Layer | Mechanism |
|---|---|
| Hot experts on the GPU | `--moe-cache-mib` (llama.cpp's own MoE expert cache, merged upstream in #29887 / #30112) keeps a LRU set of host-resident experts in VRAM |
| Cache pre-seeding (ours) | `LLAMA_EXPERT_HEAT=heat/heat_qwen36.bin` preloads the hottest experts per layer at startup, so the first token is already fast instead of waiting for the LRU to learn |
| Auto sizing (ours) | `--moe-cache-mib auto` sizes the cache from the **measured free VRAM** after KV/compute buffers, leaving `max(700 MiB, 15%)` headroom |
| Cold experts | stay in host RAM and are computed on the CPU (llama.cpp's normal offload path) — nothing is copied per token |
| Optional MTP | `--spec-type draft-mtp --spec-draft-n-max 3` (model must contain the `nextn` block) |

## Measured (RTX 5060 Ti 16 GB, Qwen3.6-35B-A3B UD-Q4_K_XL)

| Scenario | tok/s |
|---|---:|
| Plain llama.cpp, experts in CPU (`-ot exps=CPU`) | 58.5 |
| + MoE expert cache + heat pre-seed | **88.5** |
| + MTP (`--spec-draft-n-max 3`, short context) | **111.4** |
| prefill, `-ub 512` | 1,302 |
| prefill, `-ub 2048` | **2,270–2,573** |
| decode after a 4.2K prompt | 94.2 |
| 2 concurrent requests (2 slots) | 126.5 aggregate (1.61×) |

All numbers: same model file, greedy, measured with `llama-cli`/`llama-server` timings.

## Build

```bash
git clone <this fork>
cd <this fork>
./moefast/install.sh            # configures CUDA build and compiles llama-server
```

Requirements: CMake ≥ 3.24, CUDA toolkit, a C++17 compiler. No Python needed at runtime.

## Run

```bash
./moefast/run.sh /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
# then: http://127.0.0.1:8080/   (OpenAI-compatible API at /v1)
```

The script uses these settings:

```
-ngl 99 -ot "exps=CPU" -b 2048 -ub 2048 --moe-cache-mib auto --parallel 1
LLAMA_EXPERT_HEAT=heat/heat_qwen36.bin
```

Notes / gotchas:

- **`-ot "exps=CPU"`, not `-ncmoe`/`-cmoe`.** The latter also move the router (`ffn_gate_inp`) to the
  CPU, and then the MoE cache silently disables itself (it needs the router on the device).
- **`-ub 2048`** matters a lot for prompt processing with CPU-resident experts (1.3K → 2.3K tok/s).
- **MTP is optional and context-dependent**: +26% on short contexts, slightly negative past ~1K
  tokens. Enable with `MOEFAST_MTP=3 ./moefast/run.sh model.gguf`.
- The model file must contain the MTP block (`blk.<n>.nextn.*`, 753 tensors for the Unsloth MTP
  GGUF). A 733-tensor build without those tensors fails with
  `context type MTP requested but model doesn't contain MTP layers`.
- No `heat/heat_qwen36.bin`? The script still works — the cache falls back to pure LRU (slower cold
  start).

## Heat tables

`heat/heat_qwen36.bin` is a 40 KB binary: per-layer expert byte sizes plus routing counts gathered
from 128 prompts (4 classes × 32) on the Qwen3.6-35B-A3B base model with `llama-route-dump`
(see `docs/简报01-heat统计工具.md` in our project repo). It is model-family specific, not
checkpoint specific: fine-tuned merges of the same base have the same routing distribution.

## Status / limits

- Linux + NVIDIA CUDA is what we test. Windows and AMD are untested.
- The heat table is for 35B-A3B (`qwen35moe`). Other MoE families work with the plain LRU cache
  (just drop `LLAMA_EXPERT_HEAT`), but our pre-seeding is keyed to that family.
- Cache pre-seeding is a startup cost (a few seconds) paid once.

## Credits

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). The MoE expert cache itself is llama.cpp's work;
our patches add auto sizing and heat pre-seeding.
