# Qwen3.8-27B on RTX 3090s — one-command sglang container

Fastest validated recipe for running **Qwen/Qwen3.8-27B** (hybrid GDN vision-language model, 48 Gated-DeltaNet + 16 gated-attention layers) on Ampere consumer GPUs with [SGLang](https://github.com/sgl-project/sglang). Tested on 1×, 2×, and 4× RTX 3090 (24 GB each, PCIe, no NVLink).

## What's inside

- **Target model**: [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) — W4A16 INT4 quantization with Marlin kernels. FP8/NVFP4 have no fast compute path on Ampere (sm_86); INT4 cuts weight bandwidth ~4× → ~2× decode vs bf16.
- **Speculative decoding**: [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) — 1.36B DSpark draft model, accept length up to ~4 on coding/greedy content. Enabled by default on 2×/4× GPUs.
- **Multimodal**: image input (all configs) and video input (2×/4× GPUs) via torchcodec + FFmpeg.
- **Everything pinned**: the exact sglang commit, torch, triton, flashinfer, and CUDA nvcc versions are locked in the Dockerfile. Reproducible anywhere.

## Quick start

```bash
# 1× 3090 (default) — image input, up to 8k context
docker run --gpus all --shm-size 32g -p 8000:8000 \
  -v $PWD/models:/models \
  ghcr.io/0xsero/qwen38-3090-sglang:latest

# 2× 3090 — image + video, DSpark, up to 64k context
docker run --gpus all --shm-size 32g -p 8000:8000 \
  -v $PWD/models:/models -e TP=2 \
  ghcr.io/0xsero/qwen38-3090-sglang:latest

# 4× 3090 — image + video, DSpark, up to 128k context
docker run --gpus all --shm-size 32g -p 8000:8000 \
  -v $PWD/models:/models -e TP=4 \
  ghcr.io/0xsero/qwen38-3090-sglang:latest
```

Weights (~23 GB target + 2.6 GB draft) download automatically into the mounted `/models` volume on first start. First boot takes ~5 min (kernel JIT compilation).

Use it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"hello /no_think"}],"max_tokens":100}'
```

Image input:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}]}],"max_tokens":100}'
```

## Environment variables

| var | default | values | notes |
|---|---|---|---|
| `TP` | `1` | `1 \| 2 \| 4` | tensor parallel GPUs; presets tuned per count |
| `DSPARK` | `1` | `1 \| 0` | DSpark speculative decoding. Auto-disabled on TP1 (insufficient VRAM for draft model). Recommended on 2×/4×. |
| `PORT` | `8000` | | host port |
| `MODELS_DIR` | `/models` | | mount a volume here to cache weights across restarts |
| `TOOL_PARSER` | `hermes_xml` | | tool-call parser. `hermes_xml` (default) is required for Qwen3.8-27B's chat template, which renders tools as Hermes-2-Pro XML (`<tool_call><function=…><parameter=…>`); the stock `qwen`/`hermes` parsers expect JSON inside the envelope and silently drop XML calls. Any sglang parser name works (`qwen`, `hermes`, `auto`, …). |

## Pinned software stack

| component | version | why |
|---|---|---|
| sglang | `2e7c85da6` (git main) | exact commit validated for Qwen3.8-27B hybrid GDN arch + DSpark |
| torch | `2.13.0+cu130` | CUDA 13 runtime, GDN kernel support |
| triton | `3.7.1` | fixes `_grid_2` bug in 3.6 that broke GDN kernels |
| flashinfer | `0.6.17` | attention backend for hybrid model; cccl version guard patched for nvcc 13.3 |
| nvidia-cuda-nvcc | `13.3.73` (pip wheel) | JIT kernel compilation inside container |
| base image | `ubuntu:24.04` | python 3.12, ffmpeg 6.x for torchcodec video |
| uv | latest | fast package install |

## Validated presets per GPU count

| config | TP | DSpark | mem-fraction | mamba cache | prefill graph | decode max_bs | notes |
|---|---|---|---|---|---|---|---|
| 1× 3090 + DSpark | 1 | yes | 0.92 | 4 | disabled | 4 | very tight; draft + target = 21.7 GB on 24 GB card |
| 1× 3090 plain | 1 | no | 0.92 | 8 | disabled | 8 | recommended for TP1; 19 GB weights leave room for 14k KV pool |
| 2× 3090 + DSpark | 2 | yes | 0.85 | 16 | disabled | 16 | 118k KV pool, 48 max concurrent requests |
| 2× 3090 plain | 2 | no | 0.85 | 32 | disabled | 16 | larger KV pool without draft model |
| 4× 3090 + DSpark | 4 | yes | 0.72 | — | enabled | auto | 275k KV pool, production config |
| 4× 3090 plain | 4 | no | 0.80 | — | enabled | auto | max KV cache |

> **TP1 note**: DSpark on a single 3090 is technically possible but leaves almost no KV cache (only ~1 GB), making it impractical. The entrypoint auto-falls-back to no-DSpark on TP1. On 2×/4×, DSpark is enabled by default.

> **CUDA graph note**: TP1 and TP2 require `--disable-prefill-cuda-graph` and reduced `--cuda-graph-max-bs` to avoid OOM during graph capture. TP4 has enough headroom for full graphs.

---

## Benchmark results

All benchmarks run on RTX 3090 (Ampere sm_86), 24 GB VRAM per card, PCIe 3.0 ×16, no NVLink. Driver 610.x, CUDA 13 runtime. AWQ-INT4 target, DSpark draft (where enabled). Coding-flavored prompts (unique nonce per request to defeat radix cache), `temperature=0`, `max_tokens=256`, streaming with usage reporting.

- **TTFT**: time to first token (seconds) — lower is better
- **Prefill**: tokens/s during prompt processing — higher is better
- **Decode**: tokens/s during generation (aggregate across concurrent streams) — higher is better
- **E2E**: end-to-end tokens/s including prefill — higher is better

### 4× RTX 3090 (TP=4, DSpark on) — production config

KV pool: **275,018 tokens** (bf16), max **9 concurrent requests** (GDN state bound), max context **128k**.

| context | conc | TTFT (s) | prefill (tok/s) | decode (tok/s) | e2e (tok/s) |
|---|---|---|---|---|---|
| 8k | 1 | 5.53 | 1,581 | 136.8 | 34.6 |
| 8k | 2 | 5.85 | 1,495 | 103.8 | 60.8 |
| 8k | 4 | 6.46 | 1,354 | 80.5 | 102.2 |
| 8k | 8 | 7.74 | 1,130 | 60.5 | 169.8 |
| 16k | 1 | 10.77 | 1,616 | 129.4 | 20.1 |
| 16k | 2 | 11.96 | 1,455 | 97.1 | 35.0 |
| 16k | 4 | 12.71 | 1,369 | 66.9 | 61.7 |
| 16k | 8 | 15.24 | 1,142 | 45.6 | 97.7 |
| 32k | 1 | 22.00 | 1,577 | 127.1 | 10.7 |
| 32k | 2 | 23.21 | 1,494 | 93.4 | 19.7 |
| 32k | 4 | 25.54 | 1,358 | 59.6 | 34.0 |
| 32k | 8 | 26.68 | 1,300 | 39.2 | 60.8 |
| 64k | 1 | 46.34 | 1,495 | 113.9 | 5.3 |
| 64k | 2 | 48.57 | 1,426 | 100.6 | 10.0 |
| 64k | 4 | 50.32 | 1,377 | 55.0 | 18.4 |
| 64k | 8 | 50.61 | 1,369 | 38.0 | 35.2 |
| 128k | 1 | 98.88 | 1,400 | 105.0 | 2.5 |
| 128k | 2 | 102.09 | 1,356 | 74.2 | 4.8 |
| 128k | 4 | 103.48 | 1,338 | 47.0 | 9.4 |
| 128k | 8 | 103.48 | 1,338 | 29.3 | 18.2 |

**Highlights**: 136.8 tok/s single-stream decode at 8k, 169.8 tok/s aggregate at 8 concurrent, 128k context supported. Prefill sustains 1.3–1.6k tok/s across all context lengths.

### 2× RTX 3090 (TP=2, DSpark on)

KV pool: **118,693 tokens** (bf16), max **48 concurrent requests**, max context **64k** (128k prompts exceed KV pool).

| context | conc | TTFT (s) | prefill (tok/s) | decode (tok/s) | e2e (tok/s) |
|---|---|---|---|---|---|
| 8k | 1 | 5.90 | 1,484 | 112.9 | 31.4 |
| 8k | 2 | 6.13 | 1,426 | 94.7 | 57.9 |
| 8k | 4 | 7.58 | 1,154 | 79.2 | 81.3 |
| 8k | 8 | 10.02 | 873 | 73.3 | 118.9 |
| 16k | 1 | 11.96 | 1,455 | 103.2 | 17.7 |
| 16k | 2 | 12.73 | 1,366 | 73.8 | 31.5 |
| 16k | 4 | 14.58 | 1,193 | 67.4 | 49.8 |
| 16k | 8 | 17.30 | 1,005 | 65.2 | 80.0 |
| 32k | 1 | 25.04 | 1,385 | 91.5 | 9.2 |
| 32k | 2 | 26.34 | 1,317 | 79.5 | 17.3 |
| 32k | 4 | 28.92 | 1,199 | 62.0 | 29.1 |
| 32k | 8 | 31.54 | 1,100 | 65.1 | 51.7 |
| 64k | 1 | 53.44 | 1,296 | 100.9 | 4.6 |
| 64k | 2 | 56.49 | 1,226 | 73.1 | 8.5 |
| 64k | 4 | 58.46 | 1,185 | 73.2 | 16.1 |
| 64k | 8 | 61.08 | 1,134 | 59.2 | 29.4 |
| 128k | — | — | — | — | truncated (KV pool 118k < 138k prompt) |

**Highlights**: 112.9 tok/s single-stream decode with DSpark, 118.9 tok/s aggregate at 8 concurrent. ~83% of TP4 throughput at half the GPUs.

### 1× RTX 3090 (TP=1, no DSpark)

KV pool: **14,775 tokens** (bf16), max **8 concurrent requests**, max context **8k** (16k+ prompts exceed KV pool).

| context | conc | TTFT (s) | prefill (tok/s) | decode (tok/s) | e2e (tok/s) |
|---|---|---|---|---|---|
| 8k | 1 | 7.03 | 1,244 | 43.2 | 19.8 |
| 8k | 2 | 10.36 | 844 | 43.2 | 26.1 |
| 8k | 4 | 16.16 | 541 | 43.1 | 32.9 |
| 8k | 8 | 28.26 | 309 | 43.1 | 37.0 |
| 16k+ | — | — | — | — | truncated (KV pool 14k < prompt) |

**Highlights**: 43.2 tok/s decode (weight-bandwidth-bound, no DSpark). Context limited to 8k — the 19 GB AWQ weights + 3 GB mamba/graph leave only ~1.3 GB for KV cache on a single 24 GB card. Usable for short-context chat and image Q&A.

---

## Multimodal (image + video)

All configs support image input. 2×/4× configs also support video input (torchcodec + FFmpeg).

| config | image | video | image latency | video latency |
|---|---|---|---|---|
| 1× 3090 | ✅ | ✅* | 1.44 s | 1.91 s |
| 2× 3090 | ✅ | ✅ | 0.84 s | 1.42 s |
| 4× 3090 | ✅ | ✅ | 0.63 s | 1.29 s |

\* Video on 1× works in the container (Ubuntu ffmpeg 6.x). On bare-metal Arch (ffmpeg 9), torchcodec requires ffmpeg 4–8 shared libs — the container handles this automatically.

Test image: 1×1 red PNG. Test video: short mp4 clip. Both produce correct descriptions.

---

## Scaling summary

| metric | 1× 3090 | 2× 3090 | 4× 3090 |
|---|---|---|---|
| decode (8k, c1) | 43.2 tok/s | 112.9 tok/s | 136.8 tok/s |
| decode (8k, c8 aggregate) | 37.0 tok/s | 118.9 tok/s | 169.8 tok/s |
| prefill peak | 1,244 tok/s | 1,484 tok/s | 1,616 tok/s |
| max KV pool | 14,775 tokens | 118,693 tokens | 275,018 tokens |
| max context | 8k | 64k | 128k |
| max concurrent | 8 | 48 | 9 (GDN state bound) |
| DSpark | ✗ (OOM) | ✅ | ✅ |
| video | ✅ | ✅ | ✅ |
| VRAM per GPU | 23.3 GB | 21.9 GB | 21.5 GB |

**Why TP4 > TP2 > TP1**: the model is weight-bandwidth-bound. TP4 aggregates 4× 936 GB/s = 3.7 TB/s, roughly 2× TP2 and 4× TP1. DSpark adds another ~2× on coding content. TP4's lower per-GPU VRAM usage (21.5 GB vs 23.3 GB) actually leaves more room for KV cache despite splitting across 4 cards.

---

## Dashboard

A standalone HTML dashboard (`dashboard.html`) visualizes all benchmark data. Tokyo Night themed, no dependencies — open in any browser:

```bash
# serve locally
python3 -m http.server 8899 --directory .
# open http://localhost:8899/dashboard.html
```

The dashboard reads from `benchmarks/tp{1,2,4}-results.json`, `benchmarks/tp{1,2,4}-mm.json`, and `benchmarks/tp{1,2,4}-kv.txt` and renders:
- Per-GPU-count sections with KV cache capacity
- Decode/prefill/TTFT bar charts
- Full matrix table (context × concurrency)
- Image/video support badges

Raw benchmark JSON files are in `benchmarks/`.

---

## Why these choices (Ampere specifics)

- **AWQ INT4 W4A16** beats FP8 on 3090s: Ampere has no FP8 tensor cores, so FP8 only saves memory while paying dequant overhead. INT4 Marlin kernels cut weight bandwidth ~4× → ~2× decode speed.
- **DSpark acceptance is workload-dependent**: measured best with the AWQ target on greedy coding content (accept length 2.8–4.1, 128 tok/s). Sampled prose drafts worse (~1.7 accept length, 64 tok/s). The AWQ target actually drafts better than FP8 despite the draft model being trained on FP8.
- **GDN hybrid model**: 48 Gated-DeltaNet layers + 16 gated-attention layers. `--mamba-radix-cache-strategy extra_buffer` is required (wrong strategies deadlock CUDA graph capture). The mamba state pool sizes concurrency, not just KV — this is why TP4 has only 9 max concurrent despite 275k KV tokens.
- **flashinfer cccl patch**: flashinfer's bundled cccl version guard rejects nvcc 13.3 as "incompatible." The Dockerfile patches this guard to allow the build. No functional impact.
- **CUDA graph capture OOM**: TP1/TP2 require disabling prefill CUDA graphs (`--disable-prefill-cuda-graph`) and reducing decode graph max batch size. The breakable prefill graph capture allocates a large activation buffer that OOMs on tight VRAM. TP4 has enough headroom for full graphs.

## Optional: PCIe P2P (more speed)

On multi-3090 PCIe rigs, GPU-to-GPU P2P is driver-gated. The [aikitoria fork](https://github.com/aikitoria/open-gpu-kernel-modules) of the open NVIDIA kernel modules re-enables BAR1 P2P (`610.57.04-p2p` branch). Requires:
- BIOS: Resizable-BAR enabled, Above-4G enabled
- Kernel cmdline: `amd_iommu=on iommu=pt pcie_acs_override=downstream,multifunction`

This is host-kernel-level and not part of the container. Without Resizable BAR, BAR1 is only 256 MB and P2P falls back to PIO copies (no speedup).

## Build from source

```bash
git clone https://github.com/0xSero/qwen38-3090-sglang.git
cd qwen38-3090-sglang
docker build -t qwen38-3090 .
docker run --gpus all --shm-size 32g -p 8000:8000 -v $PWD/models:/models -e TP=4 qwen38-3090
```

## License

MIT
