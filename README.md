# qwen-serve

One-command Qwen inference containers, **one folder per model and instance**: a model directory holds one
directory per hardware instance, and each instance directory is self-contained (its own `Dockerfile`, launch
scripts, benchmarks and README). Recipes in [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry)
pin every published image by digest.

| Model | Instance | Engine | Hardware | Folder | Image |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | 3090-sglang | SGLang, AWQ-INT4 + DSpark | 1× / 2× / 4× RTX 3090 | [`qwen3.8/3090-sglang`](qwen3.8/3090-sglang) | `ghcr.io/0xsero/qwen38-3090-sglang@sha256:cc2bfc36…` |
| Qwen3.8-27B | b70 | llama.cpp SYCL, Q4_K_M | 1× / 2× Intel Arc Pro B70 | [`qwen3.8/b70`](qwen3.8/b70) | `ghcr.io/0xsero/qwen38-b70-attested@sha256:37219eae…` / `…6ce7771e…` |
| Qwen3.8-Flash-Next | b70-offload | SGLang XPU + exl3xpu, experts on NVMe | 1× Intel Arc Pro B70 | [`qwen3.8-flash-next/b70-offload`](qwen3.8-flash-next/b70-offload) | `ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload` |
| Qwen3.8-Flash-Next | 3090-usb-offload | EXL3 3 bpw, experts over USB | 1× RTX 3090 + Mac mini M4 | [`qwen3.8-flash-next/3090-usb-offload`](qwen3.8-flash-next/3090-usb-offload) | — |
| Qwen3.6-35B-A3B | b70 | SGLang XPU FP8, SGLang CUDA, EXL3 | 2× B70, 4× RTX 3090 | [`qwen3.6/b70`](qwen3.6/b70) | — |

## Workflows

| Workflow | Instance | What it does |
| --- | --- | --- |
| [`.github/workflows/qwen3.8-3090-sglang-validate.yml`](.github/workflows/qwen3.8-3090-sglang-validate.yml) | 3090-sglang | entrypoint test + build only (`push: false`) |
| [`.github/workflows/qwen3.8-b70-release-image.yml`](.github/workflows/qwen3.8-b70-release-image.yml) | b70 | build, push and attest `ghcr.io/0xsero/qwen38-b70-attested:v1` |

## Folded in

These repositories are archived (read-only, still cloneable and still fetchable by commit) and their full history
lives here under the folder above:

| Archived repository | Folder |
| --- | --- |
| [qwen36-b70](https://github.com/sybil-solutions/qwen36-b70) | `qwen3.6/b70` |
| [qwen38-b70](https://github.com/sybil-solutions/qwen38-b70) | `qwen3.8/b70` |
| [qwen38-flash-next-b70-offload](https://github.com/sybil-solutions/qwen38-flash-next-b70-offload) | `qwen3.8-flash-next/b70-offload` |
| [qwen38-3090-sglang](https://github.com/sybil-solutions/qwen38-3090-sglang) (this repository, renamed) | `qwen3.8/3090-sglang` |

Every archived commit is an ancestor of this repository's `main`, so `git log qwen3.6/b70` and the like read the
original history. Images published before the fold keep their names and digests: the registry recipes pin the digest, and
the `qwen38-b70` publisher now runs from `.github/workflows/qwen3.8-b70-release-image.yml` with
`qwen3.8/b70` as its build context.
