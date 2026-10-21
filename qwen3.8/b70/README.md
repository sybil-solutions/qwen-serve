# qwen38-b70

One-command Qwen3.8-27B inference server on **two Intel Arc Pro B70 GPUs**,
tensor-parallel llama.cpp (SYCL), 262k context, OpenAI-compatible API.

```bash
docker compose up -d
```

That is the whole setup. On first start it downloads the pinned,
SHA-256-verified model (~19 GB) into `./models`, then serves on port **8010**.

```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","max_tokens":64,
       "messages":[{"role":"user","content":"Hello"}]}'
```

## Measured performance (Q4_K_M, f16 KV)

### 2x B70, tensor parallel (`GPU_COUNT=2`)

| Prompt context | Prefill tok/s | TTFT    | Decode tok/s |
|----------------|--------------:|---------|-------------:|
| ~2.5k          | 955           | 2.1 s   | 51.1         |
| ~10k           | 881           | 9.1 s   | 50.9         |
| ~20k           | 801           | 20.0 s  | 51.1         |
| ~40k           | 694           | 46.2 s  | 49.7         |
| ~60k           | 563           | 106.6 s | 46.9         |
| ~160k          | 365           | 355.8 s | 42.0         |
| ~245k          | 275           | 727.3 s | 30.8         |

### 1x B70 (`GPU_COUNT=1`, max context 131,072)

| Prompt context | Prefill tok/s | TTFT    | Decode tok/s |
|----------------|--------------:|---------|-------------:|
| ~2.5k          | 1164          | 1.8 s   | 33.3         |
| ~10k           | 1018          | 7.9 s   | 33.4         |
| ~40k           | 785           | 40.8 s  | 33.4         |
| ~65k           | 625           | 96.1 s  | 31.9         |
| ~128k          | 426           | 282.1 s | 27.5         |

Single-GPU prefill **beats** TP2 at every context length (the all-reduce costs
more than the second card recovers); TP2 wins on decode and KV capacity.

### MTP speculative decoding (`ENABLE_MTP=1`)

| Config            | Task              | tok/s | Baseline | Draft acceptance |
|-------------------|-------------------|------:|---------:|-----------------:|
| 2x B70 (TP2)      | Easy (counting)   | 84.3  | 51.1     | 97.2%           |
| 2x B70 (TP2)      | Hard (random gen) | 49.0  | 51.1     | 37.5%           |
| 1x B70 (64k ctx)  | Easy (counting)   | 61.7  | 33.3     | 97.2%           |
| 1x B70 (64k ctx)  | Hard (random gen) | 28.2  | 33.3     | 29.6%           |

Note: on 1x B70 the draft model plus 131k context exceeds VRAM
(UR_RESULT_ERROR_DEVICE_LOST); 65536 is the validated MTP context on a single card.

Decode is nearly flat with context: only 16 of 64 layers are full attention
(Qwen3.8 hybrid GDN); the linear-attention layers are O(1) per token.

## What is inside

- **Source**: [mndodd/llama.cpp](https://github.com/mndodd/llama.cpp) @
  `4302fb5` + the
  [b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab)
  full TP2 stack (`dp4a2` patch, decoded SHA-256
  `f21e9b55…`) + Q4K increment (`0a278585…`). Both digests verified against
  the lab's published values.
- **Model**: `ggml-org/Qwen3.8-27B-GGUF` @ `0669b986`, `Qwen3.8-27B-Q4_K_M.gguf`,
  SHA-256 checked at every container start.
- **Build**: SYCL JIT via Intel oneAPI DPC++ 2025.3.3. AOT `bmg_g31` builds
  under oneAPI 2026.1.x require `/opt/intel/oneapi/umf/1.0/lib` on
  `LD_LIBRARY_PATH` (or `/etc/ld.so.conf.d/oneapi.conf` + `ldconfig`) so
  `libur_loader.so.0` and `libur_adapter_level_zero.so.0` can resolve
  `libumf.so.1` and enumerate B70 devices (`level_zero:gpu:0`, `level_zero:gpu:1`).
  However, on the `dp4a2` snapshot with `26.05.37020.4` compute runtime, enabling
  the `ggml_sycl_reordered` (`Reorder + MulMatVecQ`) AOT path hangs during
  `check_Prefill_Speed`, so 2025.3.3 JIT remains the default verified-working
  path and reaches decode parity with the lab's promoted record (51 tok/s vs 49.7).

## Options (docker-compose `environment`)

| Variable      | Default | Meaning                                          |
|---------------|---------|--------------------------------------------------|
| `GPU_COUNT`   | 2       | `1` or `2` Arc Pro B70s (1 GPU: higher prefill)  |
| `CTX_SIZE_OVERRIDE` | — | Blank: 262144 (2 GPU) / 131072 (1 GPU)        |
| `PARALLEL`    | 1       | Concurrent request slots                         |
| `ENABLE_MTP`  | 0       | `1` = speculative decoding (mtp-Q4_0 draft)      |
| `ENABLE_VISION` | 0     | `1` = image input (mmproj Q8_0, CPU encoder)     |
| `BATCH`/`UBATCH` | 8192 | Prefill batch; 8192 maximizes prefill throughput |

## Notes

- `/dev/dri/by-path` must be mounted (done in compose): the Intel level-zero
  driver enumerates GPUs through it and `--device /dev/dri` alone leaves the
  GPUs invisible.
- The Q4K reorder-family fused kernels are disabled by default: they assume the
  lab AOT build's reordered tensor layout and corrupt output otherwise.
- Tested on: Arch kernel 7.1.8 (xe driver), 2x Arc Pro B70 (Battlemage G31),
  oneAPI 2025.3.3 container image.

## sglang XPU — not viable (2026-08)

sglang 0.5.13 XPU (`llm-scaler-sgl:bmg`) was tested as an alternative path
for Qwen3.8-27B on 2× B70. It loads and prefills, but the GDN decode kernel
produces NaN after ~4 decode tokens → all-`!!!` output at 0.44 tok/s.

Root cause: **SSM state drift** in
`fused_sigmoid_gating_delta_rule_update_kernel`. The fp32 state overflows to
NaN as aggressive decay values (`g` reaching -33 → `exp(g) ≈ 5e-15`) interact
with growing `h` across decode steps, producing `inf - inf = NaN` in the delta
rule update `v -= sum(h * k)`. Related to sglang issue
[#35150](https://github.com/sgl-project/sglang/issues/35150). The ratio-3
(48 V : 16 K heads) split is handled by the fallback path, but the numerical
instability is independent of the ratio.

Additionally, SGLang cannot run quantized models on XPU: GPTQ/AutoRound uses
CUDA-only `gptq_shuffle`/`gptq_gemm` kernels, AWQ-INT4 uses CUDA-only marlin
path, and FP8 block quantization has TP2 shape mismatches.

This repo (llama.cpp SYCL) is unaffected: `GGML_OP_GATED_DELTA_NET` enforces
fp32 at cache, op, and kernel levels, making the `!!!` state drift
structurally impossible.

Full investigation in [`docs/sglang-xpu-qwen38-analysis.md`](docs/sglang-xpu-qwen38-analysis.md).

## vLLM TP2 — 140.8 tok/s on 2× B70 (2026-08-20)

Real tensor parallelism (TP2) with MTP5 speculative decoding on the lab vLLM
image (`intel/llm-scaler-vllm:0.21.0-b3`) achieves **140.8 tok/s aggregate**
with 4 concurrent request streams, exceeding the 120 tok/s target.

### Results

| Config | GPU(s) | Decode tok/s | Coherent | Notes |
|--------|---------|-------------:|----------|-------|
| vLLM PIECEWISE MTP5 fp16 | 1× B70 | **90.1** hard | ✅ | Target 60 exceeded (+50%) |
| vLLM TP2 eager MTP5 fp16, 4 concurrent | 2× B70 | **140.8** aggregate | ✅ | Target 120 exceeded (+17%) |

### How to run TP2

```bash
# 1. Copy the TP2 launch script to the host
scp lab_tp2_batch_mtp.sh omarchy:/tmp/lab_tp2_batch_mtp.sh

# 2. Launch (auto-retries on intermittent XCCL failures)
scp launch_tp2_retry.sh omarchy:/tmp/launch_tp2_retry.sh
ssh omarchy 'bash /tmp/launch_tp2_retry.sh'

# 3. Benchmark
scp bench_concurrent_stable.sh omarchy:/tmp/bench_concurrent_stable.sh
ssh omarchy 'bash /tmp/bench_concurrent_stable.sh 4 5'
```

### Patches applied (all in the launch script, no image rebuild needed)

1. **mamba_utils.py** — `ctypes.c_int64(state.data_ptr()).value` fixes XPU
   pointer overflow (0xFFFF... range exceeds int64 max).
2. **gdn_linear_attn.py** — try/except around `self.out_proj.weight` in
   `_gdn_outproj_esimd_eligible()` for quantized `RowParallelLinear` in TP2.
3. **utils.py** — `non_blocking=False` in `CpuGpuBuffer.copy_to_gpu()` to
   avoid level_zero async copy staging buffer OOM.
4. **xpu_communicator.py** — catch XCCL `OUT_OF_RESOURCES` and fall back to
   CPU gloo `all_reduce` (tensor.cpu() → dist.all_reduce → tensor.xpu()).
   Only needed during profile_run (large prefill tensors); decode all_reduce
   works fine via XCCL (small single-token tensors).
5. **CLI flags** — `--skip-mm-profiling` (bypasses vision encoder all_reduce
   crash), `--enforce-eager` (avoids TP1 inductor compilation hang),
   `--tensor-parallel-size 2`, `--max-num-seqs 4`, `--gpu-memory-utilization 0.45`.
