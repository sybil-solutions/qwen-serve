# Qwen3.6-35B-A3B on Intel Arc B70 + RTX 3090

Running Qwen3.6-35B-A3B (35B MoE, 3B active, hybrid GDN architecture) on:
- **2× Intel Arc Pro B70** (Battlemage G31, 32GB VRAM each) via sglang XPU
- **4× RTX 3090** (Ampere sm_86, 24GB VRAM each) via sglang CUDA bf16 and EXL3

## Model

| Spec | Value |
|---|---|
| Total params | 35B |
| Active per token | 3B |
| Layers | 40 |
| Experts | 256 (8 routed + 1 shared) |
| Architecture | Hybrid GDN (30 Gated DeltaNet + 10 GQA full-attention, every 4th layer) |
| Context | 262K |
| model_type | `qwen3_5_moe` / `qwen3_5_moe_text` |
| Hidden size | 2048 |
| Attention heads | 16 Q, 2 KV, head_dim=256 |
| Linear heads | 32 value, 16 key, key_dim=128, value_dim=128 |

## Status

| Backend | Hardware | Status | Speed | Notes |
|---|---|---|---|---|
| sglang XPU online FP8 | 2× B70 (TP=2) | ✅ Validated | 114–116 tok/s | fp32 SSM state, native GDN fast path, XPU graphs |
| sglang CUDA bf16 | 4× 3090 (TP=4) | ✅ Working | ~14 tok/s | Fully coherent, Triton GDN kernels |
| sglang CUDA bf16 + MTP | 4× 3090 (TP=4) | ✅ Working | ~25 tok/s | Native MTP speculative decoding, 1.8× speedup |
| EXL3 (trellis 4.5bpw) | 1× 3090 | ✅ Working | ~28 tok/s | Requires Ampere patch (see below) |
| llama.cpp SYCL Q4_K | 1–2× B70 | 🧪 Untested | TBD | GDN fp32-safe (no !!! bug possible), EAGLE3 MTP works on SYCL. See [analysis](docs/ggml-sycl-b70-pr-analysis.md) |
| sglang XPU FP8 + MTP | 2× B70 (TP=2) | ❌ Blocked | — | sgl_kernel C++ ops (build_tree, verify_tree) not compiled for XPU |
| EXL3 TP=4 | 4× 3090 | ❌ TP bug | — | PR #290 OOB write in exl3_mgemm |
| DSpark spec decode | any | ❌ No sglang support | — | sglang has no DSpark; use vLLM with `--speculative-config` |

## Published image provenance

| Field | Value |
| --- | --- |
| B70/XPU image | `ghcr.io/0xsero/qwen36-b70-sglang` |
| Pinned registry digest | `sha256:d027487a07e4391562b4460da72bc78008be129d44af540148787e02e3045c08` |
| Registry-recorded launcher source commit | [`15166aed68bc25c411e16ddd81621242ba3d6298`](https://github.com/sybil-solutions/qwen36-b70/commit/15166aed68bc25c411e16ddd81621242ba3d6298) |
| Evidence | [`Registry recipe's launch.source`](https://github.com/sybil-solutions/local-ai-registry/blob/8cc343980abde31c4f881961763aa8850a3fa41e/data/registry/recipe/qwen36-35b-a3b-layered-fp8-arcb70-sglang-tp2.json); this is not an attested image-build revision |
| Pull access | **Private**; authenticated manifest fetch and digest match verified 2026-10-12; anonymous pull denied |
| Registry launch | [`sglang-qwen3.6-35b-a3b-online-fp8-weights-with-fp16-activations-8k`](https://github.com/sybil-solutions/local-ai-registry/blob/8cc343980abde31c4f881961763aa8850a3fa41e/registry/launches/sglang-qwen3.6-35b-a3b-online-fp8-weights-with-fp16-activations-8k.json) |

The registry proof is marked legacy. The image retains Intel base-image labels, including
`intel/containers@ac12083301cbdbe06135882d401f7877b0b512db`; those do **not** identify the custom
SGLang image's build commit. No exact custom image-build Git revision is attested by the inspected
manifest/config. The table identifies the recorded launcher source rather than inventing build provenance.
The image was published from the local `llm-scaler-sgl:bmg` build; it is not a publicly pullable substitute.

The local-build quick start below remains valid. To use the registry copy, first authenticate to GHCR
with an account authorized for the private package, then run:

```bash
IMAGE=ghcr.io/0xsero/qwen36-b70-sglang@sha256:d027487a07e4391562b4460da72bc78008be129d44af540148787e02e3045c08 \
  MODEL_ROOT=/path/to/models ./sglang/xpu/run_qwen3_6_service.sh
```

The CUDA examples below share the public
[`qwen38-3090-sglang` image and its build source commit](../../qwen3.8/3090-sglang/README.md#published-image-provenance);
they are pinned to that image's digest, not its mutable `latest` tag. This is a CUDA path, not a replacement
for the B70/XPU image.

## Quick start: sglang XPU FP8 on 2× Intel Arc B70

The B70 path uses the `llm-scaler-sgl:bmg` image (built locally from `intel/llm-scaler/sglang/docker/Dockerfile`). The model loads as bf16 to host RAM, then each module is quantized to FP8 on-device (`--load-format layered_fp8`), so a TP=2 split fits in 64 GB total VRAM.

```bash
MODEL_ROOT=/path/to/models ./sglang/xpu/run_qwen3_6_service.sh
```

Test:
```bash
curl http://localhost:8030/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Reply with exactly: Omarchy local AI is running on SGLang.","sampling_params":{"temperature":0,"max_new_tokens":64}}'
```

The retained sweep validates coherent 128-token generations at 128, 512, 2,048, and 8,192 input tokens. Decode stays between 113.9 and 116.3 tok/s at concurrency one. The model supports a larger context window, but this run does not claim it was exercised.

| Input tokens | Concurrency | TTFT p50 | Decode p50 | End-to-end output rate |
|---:|---:|---:|---:|---:|
| 128 | 1 | 2.19s | 114.9 tok/s | 38.8 tok/s |
| 512 | 1 | 8.69s | 116.2 tok/s | 13.1 tok/s |
| 2,048 | 1 | 33.06s | 113.9 tok/s | 3.7 tok/s |
| 8,192 | 1 | 130.86s | 116.3 tok/s | 1.0 tok/s |
| 512 | 2 | 15.72s | 85.7 tok/s | 14.9 tok/s |
| 512 | 4 | 33.69s | 67.8 tok/s | 14.4 tok/s |
| 512 | 8 | 66.86s | 41.8 tok/s | 14.6 tok/s |

Raw request-level results are in [`benchmarks/results/20260825-195430-b70-sglang.json`](benchmarks/results/20260825-195430-b70-sglang.json). Every request passed the corruption check: non-empty output, no `!!!`, and no token ID 0.

### The `!!!` bug and fix

The XPU GDN (Gated DeltaNet) path produced `!!!` output when the recurrent SSM state was fp16. Root cause (identified in
[intel/llm-scaler#559](https://github.com/intel/llm-scaler/issues/559)):

1. Qwen3.6's HF config sets `mamba_ssm_dtype=float32` for accuracy.
2. The fused ESIMD GDN kernel reads the SSM state as fp16 — if the state
   cache is also fp16, the recurrence drifts over sequences → NaN → `!!!`.
3. The fused-kernel gate checks `if ssm_state.dtype == fp16: use fused` —
   so forcing fp16 SSM state routes onto exactly the drifting kernel.

**Validated runtime contract:**
- Set `SGLANG_MAMBA_SSM_DTYPE=float32` explicitly.
- Enable the image's fp32-safe native path with `SGLANG_XPU_GDN_FAST_PATH=1`.
- Keep the incompatible ESIMD GDN gates off under the legacy names read directly by SGLang 0.5.13.
- Keep `SGLANG_MAMBA_CONV_DTYPE=float16` because the convolution state must match the float16 activations.
- Bridge the canonical `SGLANG_XPU_*` variables to `SGL_XPU_*` for this pinned image. This includes `SGL_XPU_ENABLE_GRAPH=1`; the accepted run reports `cuda graph: True` during decode.

An upstream kernel fix (analogous to
[vllm-xpu-kernels PR #437](https://github.com/vllm-project/vllm-xpu-kernels/pull/437))
is to make the ESIMD GDN kernel read state as fp32. Until that lands, the runtime contract above gives correct output and fast decode, but cold prefill remains the clear bottleneck.

## Quick start: sglang CUDA bf16 on 4× 3090

```bash
docker run -d --name sglang-cuda-bf16 --gpus all --shm-size=4g \
  --entrypoint "" \
  -v /path/to/models:/models \
  -p 8020:8020 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ghcr.io/0xsero/qwen38-3090-sglang@sha256:cc2bfc369c709c53d239ef2af41d4ae2530904975816bb2ca484188008586ac9 \
  python3 -m sglang.launch_server \
  --model-path /models/Qwen3.6-35B-A3B \
  --tp 4 --dtype bfloat16 --trust-remote-code \
  --host 0.0.0.0 --port 8020 \
  --mem-fraction-static 0.85 \
  --context-length 2048 --max-running-requests 2 \
  --attention-backend flashinfer \
  --schedule-conservativeness 2.0 \
  --max-mamba-cache-size 32 \
  --mamba-scheduler-strategy extra_buffer
```

Test:
```bash
curl http://localhost:8020/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_completion_tokens":30,"temperature":0}'
# → "Paris, a city renowned for its rich history..."
```

## Quick start: sglang CUDA bf16 + MTP on 4× 3090

Native MTP (Multi-Token Prediction) speculative decoding uses the MTP module
built into the Qwen3.6 checkpoint (`mtp_num_hidden_layers=1`, 19 weight keys
under `mtp.*`). sglang's `NEXTN` algorithm loads this as an EAGLE draft model
from the same checkpoint — no separate draft model needed.

```bash
docker run -d --name sglang-cuda-mtp --gpus all --shm-size=4g \
  --entrypoint "" \
  -v /path/to/models:/models \
  -v ./sglang/cuda/start_sglang_bf16_mtp.sh:/start.sh:ro \
  -p 8020:8020 \
  -e PORT=8020 \
  ghcr.io/0xsero/qwen38-3090-sglang@sha256:cc2bfc369c709c53d239ef2af41d4ae2530904975816bb2ca484188008586ac9 \
  bash /start.sh
```

Result: ~25 tok/s (1.8× over the 14 tok/s baseline), fully coherent.
Auto-tuned params: `num_steps=3, eagle_topk=1, num_draft_tokens=4`.

### MTP on XPU (blocked)

MTP on XPU requires `build_tree_kernel_efficient` and `verify_tree_greedy`
from `sgl_kernel`, which are CUDA-only C++ ops not compiled into the XPU
build. Torch fallbacks were written (`sglang/xpu/patches/`) but the draft
model's KV cache also has a shape mismatch in the XPU attention backend.
Fixing this requires patching `sgl-kernel-xpu` to compile the tree ops for
SYCL/XPU — a substantial kernel porting effort.

The patched files in `sglang/xpu/patches/` cover:
- `qwen3_5_mtp_xpu.py` — `torch.cuda` → `torch.xpu` in `set_embed_and_head`
- `eagle_utils_xpu.py` — torch fallbacks for `build_tree_kernel_efficient` and `verify_tree_greedy`
- `eagle_info_v2_xpu.py` — force greedy verification path on XPU

## Quick start: EXL3 on single 3090

EXL3 requires a patch on Ampere GPUs (sm_80/86/87). The CUDA `conv1d_update_kernel`
in `gdn.cu:1448` crashes with an illegal memory access. The patch forces the
Triton fallback for conv1d and disables the BC (batched CUDA) GDN path.

```bash
# Build EXL3 container
docker run -d --name exl3-test --gpus all \
  -v /path/to/models:/models \
  pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel \
  bash -c "pip install exllamav3 --break-system-packages && sleep infinity"

# Copy patch and run
docker cp exl3_ampere_patch.py exl3-test:/tmp/
docker cp run_exl3.py exl3-test:/tmp/
docker exec exl3-test bash -c \
  "export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib && \
   python3 /tmp/run_exl3.py --model /models/Qwen3.6-35B-A3B-EXL3 \
     --prompt 'The capital of France is' --max-new-tokens 30"
# → "The capital of France is Paris."
```

VRAM: 17.43 GB on single RTX 3090. Speed: ~28 tok/s.

## EXL3 Ampere patch details

**Root cause:** The `conv1d_update_kernel` in `gdn.cu` (line 1448) triggers an
illegal memory access on Ampere GPUs (sm_80/sm_86/sm_87) when running Qwen3.5-
architecture models with Gated DeltaNet layers.

**Misdiagnosis in exllamav3 issue #291:** Without `CUDA_LAUNCH_BLOCKING=1`, the
error surfaces as `coop_autotune.cu:464` because the conv1d crash corrupts CUDA
state and the next CUDA call (GEMM autotuner) fails with `CUBLAS_STATUS_NOT_INITIALIZED`.
The true first-failing kernel is `gdn.cu:1448`, not `coop_autotune.cu:464`.

**Fix (Python monkey-patch):**
1. Force Triton conv1d path (`causal_conv1d_update_slotted_triton`)
2. Disable BC GDN path (`self.bc = None` in `GatedDeltaNet.forward`)
3. Auto-alloc recurrent state and KV cache on first access (normally done by BC C++ path)

See `exl3/exl3_ampere_patch.py` for the full patch. Reported in
[exllamav3#291](https://github.com/turboderp-org/exllamav3/issues/291#issuecomment-5323838587).

## llama.cpp SYCL on Intel Arc B70 (alternative backend)

llama.cpp has full first-class support for `qwen3_5_moe` (Qwen3.6-35B-A3B) with
a fused `GGML_OP_GATED_DELTA_NET` op on the SYCL backend. Unlike sgl-kernel-xpu,
ggml enforces **fp32 GDN state at three levels** (cache allocation, op assertion,
kernel), making the `!!!` bug structurally impossible. It also supports EAGLE3
speculative decoding on SYCL (PR #24593) — the MTP path that's blocked in sglang
XPU works here.

A cluster of SYCL PRs (Jun–Aug 2026) optimizes every component of this model on
Battlemage: MoE prefill +70% (#23142), GQA flash attention via XMX (#25222),
fused GDN state writeback (#26643), fused FFN GLU +2.8% tg (#26779). Full
breakdown in [docs/ggml-sycl-b70-pr-analysis.md](docs/ggml-sycl-b70-pr-analysis.md).

### Quick start

```bash
# 1. Build llama.cpp with SYCL (oneAPI 2026 required)
bash llamacpp/build_b70_sycl.sh

# 2. Convert model to GGUF Q4_K_M (~18-20 GB, fits one B70)
python3 llama.cpp/convert_hf_to_gguf.py /models/Qwen3.6-35B-A3B \
    --outtype q4_k_m --outfile /models/Qwen3.6-35B-A3B-Q4_K_M.gguf

# 3. Launch (single B70, flash attention, f16 KV cache for GQA layers)
bash llamacpp/start_llamacpp_b70_sycl.sh
# → llama-server on port 8040

# For 2× B70: set ZE_AFFINITY_MASK=0,1 and add -sm row to split the model
```

**Tradeoff**: llama-server has a simpler API than sglang (no radix cache, less
sophisticated continuous batching). For raw single-model throughput on B70 it's
competitive; for production multi-request serving, sglang's scheduler is more
mature. Benchmark both and pick the winner for your workload.

**When to prefer llama.cpp over sglang on B70:**
- You need MTP/speculative decoding on XPU (blocked in sglang, works here)
- You want the `!!!` bug to be structurally impossible, not env-var-worked-around
- You want Q4_K quantization instead of FP8
- Single-card operation (Q4_K_M fits one 32 GB B70)

### ggml SYCL PR #27160 (the "yesterday" B70 speedup)

PR [#27160](https://github.com/ggml-org/llama.cpp/pull/27160) (merged
2026-08-17) fixed work-group under-subscription in ggml's SYCL quantized cpy
kernels — **7.83× speedup** on B70 for `q4_0 → f32` copies. This is a ggml
backend fix; it does **not** apply to our sglang/sgl-kernel-xpu path (different
kernel stack, audited — no equivalent bug found). It only benefits the llama.cpp
Q4_0 quant path; Q4_K (the recommended quant for this model) is unaffected.
Full analysis in [docs/ggml-sycl-b70-pr-analysis.md](docs/ggml-sycl-b70-pr-analysis.md).

## sglang XPU kernel rebuild (optional, for ESIMD GDN speedup)

The env-var fix above disables the ESIMD GDN kernels to avoid the fp16 SSM state
drift. To re-enable them with the fix applied, rebuild `sgl-kernel-xpu` from
source with the C++ kernel patch (analogous to
[vllm-xpu-kernels PR #437](https://github.com/vllm-project/vllm-xpu-kernels/pull/437)):

```bash
cd intel/llm-scaler
# Patch sgl-kernel-xpu/src/sycl/gdn_attn/chunk_gated_delta_rule_kernels_*.hpp
# to read SSM state as fp32 instead of fp16
./scripts/build_image.sh  # ~25 min, dominated by ESIMD AOT compile
```

Note: `vllm-xpu-kernels` v0.1.12.1 (which contains PR #437) cannot be
pip-installed into the sglang image — it's a different kernel stack (sglang
uses `sgl-kernel-xpu`, not `vllm-xpu-kernels`) and it links `libsycl.so.9`
(oneAPI 2026) while the image's torch 2.11+xpu links `libsycl.so.8` (oneAPI
2025). The fix must be applied to `sgl-kernel-xpu` source and rebuilt.

## Model variants

| Variant | Size | Format | Notes |
|---|---|---|---|
| Qwen3.6-35B-A3B | 67 GB | bf16 safetensors | Base checkpoint, for sglang bf16/FP8 |
| Qwen3.6-35B-A3B-FP8 | 36 GB | FP8 E4M3 | Pre-quantized, split in_proj scales incompatible with TP |
| Qwen3.6-35B-A3B-EXL3 | 19 GB | Trellis 4.5bpw | For EXL3, works on Ampere with patch |
| Qwen3.6-35B-A3B-GGUF | 18-20 GB | Q4_K_M | llama.cpp SYCL, fp32-safe GDN (no !!! bug) |
| Qwen3.6-35B-A3B-DSpark | 2.8 GB | DSpark draft | Speculative decoding; use vLLM, not sglang |

## Directory structure

```
qwen36-b70/
├── sglang/
│   ├── cuda/
│   │   ├── start_sglang_bf16.sh       # sglang CUDA bf16 (4x 3090, TP=4)
│   │   └── start_sglang_bf16_mtp.sh   # sglang CUDA bf16 + MTP spec decode (1.8× speedup)
│   └── xpu/
│       ├── start_qwen3_6_service.sh   # sglang XPU FP8 (2x B70, TP=2)
│       ├── start_qwen3_6_mtp.sh       # sglang XPU FP8 + MTP (blocked — see notes)
│       └── patches/
│           ├── qwen3_5_mtp_xpu.py    # torch.cuda→xpu patch for MTP model
│           ├── eagle_utils_xpu.py     # torch fallbacks for tree kernels
│           └── eagle_info_v2_xpu.py  # force greedy verify on XPU
├── llamacpp/
│   ├── build_b70_sycl.sh              # Build llama.cpp with SYCL for B70
│   └── start_llamacpp_b70_sycl.sh    # llama-server Q4_K on B70 (fp32-safe GDN)
├── exl3/
│   ├── exl3_ampere_patch.py           # Ampere conv1d crash workaround
│   └── run_exl3.py                   # EXL3 inference script
├── docs/
│   └── ggml-sycl-b70-pr-analysis.md  # ggml SYCL PR #27160 analysis + B70 path research
└── benchmarks/
    └── bench.py                      # sglang API benchmark script
```

## Hardware

- 2× Intel Arc Pro B70 (Battlemage G31, PCI 8086:e223, 32GB VRAM each)
- 4× NVIDIA RTX 3090 (Ampere sm_86, 24GB VRAM each, driver 610.57.04)
- Level Zero works in Docker (NEO 26.05), broken on host (NEO 26.27)

## Known issues

1. **FP8 on Ampere**: RTX 3090 (sm_86) doesn't support `fp8e4nv` format. No sglang
   FP8 on 3090s — hardware limitation. Use bf16 with TP=4 or EXL3 trellis quant.
2. **FP8 checkpoint TP sharding**: Pre-quantized FP8 checkpoint has separate
   `in_proj_qkv` + `in_proj_z` FP8 tensors with incompatible scale shapes for TP
   sharding. Use bf16 checkpoint with `--load-format layered_fp8` instead.
3. **XPU `!!!` bug**: FIXED via env-var workaround (fp32 SSM state + GDN ESIMD
   disabled). Permanent fix requires rebuilding `sgl-kernel-xpu` with C++ kernel
   patch. See "The `!!!` bug and fix" section above.
4. **EXL3 Ampere crash**: conv1d_update_kernel in gdn.cu:1448 crashes. Patched
   via Triton fallback.
5. **EXL3 TP**: PR #290 OOB write in exl3_mgemm with TP-sharded MoE at batch > 16.
6. **DSpark in sglang**: sglang has no DSpark support (only EAGLE/EAGLE3/NEXTN/
   STANDALONE/NGRAM). DSpark requires vLLM with `--speculative-config
   '{"method":"dspark",...}'` and vLLM ≥ mid-Aug 2026. The `speculators`
   package uses its own `SpeculatorModelConfig.from_pretrained()`, NOT
   `AutoConfig.from_pretrained` — AutoConfig is the wrong loader.
7. **MTP on XPU**: Blocked by `sgl_kernel` C++ ops (`build_tree_kernel_efficient`,
   `verify_tree_greedy`) not compiled for XPU. Torch fallbacks written but KV
   cache shape mismatch in draft model attention also needs fixing.
8. **EXL3 on XPU**: Feasible via PR #289 (Triton linear) + PR #283 (fallback
   pattern) but 2-4 weeks effort. The validated SGLang XPU path is the higher-ROI baseline.
9. **llama.cpp SYCL on B70**: Architecture fully supported (`qwen35moe`), GDN
   is fp32-safe (no `!!!` bug possible), EAGLE3 MTP works on SYCL. Untested on
   our hardware — needs a build + benchmark against the retained SGLang sweep. See [docs/ggml-sycl-b70-pr-analysis.md](docs/ggml-sycl-b70-pr-analysis.md).
   ggml PR #27160 (cpy work-group fix, 7.83× on B70 for Q4_0) does not apply to
   our sglang path (different kernel stack, no equivalent bug found in
   sgl-kernel-xpu).
