# qwen-serve

One-command Qwen inference containers, **one folder per model and instance**: a model directory holds one
directory per hardware instance, and each instance directory is self-contained (its own `Dockerfile`, launch
scripts, benchmarks and README). [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry)
pins the published images used by its recipes; publication does not imply anonymous pull access.

| Model | Instance | Engine | Hardware | Folder | Image provenance |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | 3090-sglang | SGLang, AWQ-INT4 + DSpark | 1× / 2× / 4× RTX 3090 | [`qwen3.8/3090-sglang`](qwen3.8/3090-sglang) | [Public `qwen38-3090-sglang`, `cc2bfc36…`](qwen3.8/3090-sglang/README.md#published-image-provenance) |
| Qwen3.8-27B | b70 | llama.cpp SYCL, Q4_K_M | 1× / 2× Intel Arc Pro B70 | [`qwen3.8/b70`](qwen3.8/b70) | [Public `qwen38-b70-attested`, `6ce7771e…`](qwen3.8/b70/README.md#published-image-provenance) |
| Qwen3.8-Flash-Next | b70-offload | SGLang XPU + exl3xpu, experts on NVMe | 1× Intel Arc Pro B70 | [`qwen3.8-flash-next/b70-offload`](qwen3.8-flash-next/b70-offload) | [Public `qwen38-flash-next-b70-offload`, `d30df691…`](qwen3.8-flash-next/b70-offload/README.md#published-image-provenance) |
| Qwen3.6-35B-A3B | b70 | SGLang XPU FP8, SGLang CUDA, EXL3 | 2× B70, 4× RTX 3090 | [`qwen3.6/b70`](qwen3.6/b70) | [Private `qwen36-b70-sglang`, `d027487a…`](qwen3.6/b70/README.md#published-image-provenance) |

## Workflows

| Workflow | Instance | What it does |
| --- | --- | --- |
| [`.github/workflows/qwen3.8-3090-sglang-validate.yml`](.github/workflows/qwen3.8-3090-sglang-validate.yml) | 3090-sglang | entrypoint test + build only (`push: false`) |
| [`.github/workflows/qwen3.8-b70-release-image.yml`](.github/workflows/qwen3.8-b70-release-image.yml) | b70 | build, push and attest `ghcr.io/0xsero/qwen38-b70-attested:v1` |

### Verification on main (2026-10-12)

| Workflow | Run | Source commit | Conclusion |
| --- | --- | --- | --- |
| B70 release (dispatched) | [38225225421](https://github.com/sybil-solutions/qwen-serve/actions/runs/38225225421) | `5743df3774f11f5668469b89dfd37f341f65ad85` | **Failure**: build completed, GHCR push denied; attestation skipped |
| 3090 validation (dispatched) | [38225230088](https://github.com/sybil-solutions/qwen-serve/actions/runs/38225230088) | `5743df3774f11f5668469b89dfd37f341f65ad85` | **Success**: entrypoint validation and Linux amd64 image build |
| 3090 validation (push) | [38225729200](https://github.com/sybil-solutions/qwen-serve/actions/runs/38225729200) | `e21b0229727fed42e4d5f12ead5d9687b7ff9545` | **Success**: entrypoint validation and Linux amd64 image build |

The release error was `denied: permission_denied: The requested installation does not exist`.
The workflow uses this organization's `GITHUB_TOKEN` to push to the user-owned `0xsero` namespace;
the repository had no publishing secret configured at audit time. Publishing requires package write access
to that namespace, or an explicit migration to an organization-owned package. Neither a credential change
nor an image migration is part of this pin audit. The failed run published no replacement image, so the
previous public attested digest remains canonical. Validation is build-only, not a GPU inference check.

## Registry digest audit (2026-10-12)

Audited [local-ai-registry at `8cc343980abde31c4f881961763aa8850a3fa41e`](https://github.com/sybil-solutions/local-ai-registry/tree/8cc343980abde31c4f881961763aa8850a3fa41e).
Each instance README records the full image name, digest, source commit and evidence type. All five
distinct instance image manifests (including the historical private B70 image) returned HTTP 200 and
their downloaded bytes matched the pinned SHA-256. The three current public images were checked
anonymously; the two private packages required an authorized GHCR login.

The registry's B70 128k/256k launches already use the public `qwen38-b70-attested` image. The former
root reference paired its name with `37219eae…`, which belongs to the separate private `qwen38-b70`
package; it is replaced here with the canonical attested `6ce7771e…` pin. Qwen3.6's XPU package remains
private, with no proven public equivalent. Flash-Next B70-offload's public pin also resolves, but it is
not a registry recipe: its chat-gate limitation remains documented in the instance README.

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
