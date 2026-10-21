"""
SGLang out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on Intel XPU (Battlemage).

Registered through SGLang's `sglang.srt.plugins` entry point (runs in the launcher, the tokenizer/detokenizer and
every scheduler process), so stock SGLang serves `--quantization exl3` with no source edits. The kernels are the
same ESIMD/XMX op library the vLLM plugin uses (`torch.ops.exl3xpu_C.linear`); only the engine glue differs.

Storage per SGLang fused linear (qkv_proj, gate_up_proj, in_proj_qkvz, ...): each checkpoint matrix is one "group"
with its own input scale vector `suh`; trellis tiles are concatenated along n in partition order, `svh` likewise,
and `shard_of_nb` maps every 128-column output block to its group (exactly the vLLM plugin's layout).
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
from typing import Any

import torch

logger = logging.getLogger(__name__)
_done = False

_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")
_QKV = {"q": 0, "k": 1, "v": 2}
_CB = {"3inst": 0, "mcg": 1, "mul1": 2}
_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
TARGET_LM_HEADS: list = []
_SYNC_LOGITS = os.environ.get("EXL3_SYNC_LOGITS", "0") == "1"   # debug: torch.xpu.synchronize() after lm_head (eager only)
_MEM_TRACE = int(os.environ.get("EXL3_MEM_TRACE", "0"))      # log torch XPU memory every N target lm_head calls
_mem_n = [0, 0]


def _mem_trace(rows: int) -> None:
    """Host-side allocator counters only (no device sync)."""
    _mem_n[0] += 1
    _mem_n[1] = max(_mem_n[1], rows)
    if _mem_n[0] % _MEM_TRACE == 0:
        g = 2 ** 30
        free, total = torch.xpu.mem_get_info()
        logger.info("exl3xpu mem: alloc %.2f GiB, reserved %.2f GiB, peak alloc %.2f GiB, peak reserved %.2f GiB, "
                    "device free %.2f of %.2f GiB, max lm_head rows %d", torch.xpu.memory_allocated() / g,
                    torch.xpu.memory_reserved() / g, torch.xpu.max_memory_allocated() / g,
                    torch.xpu.max_memory_reserved() / g, free / g, total / g, _mem_n[1])
        _mem_n[1] = 0


def _dev() -> torch.device:
    return torch.device("xpu", torch.xpu.current_device())


def _empty_cache():
    try:
        torch.xpu.empty_cache()
    except Exception:
        pass


def norm_key(k: str) -> str:
    """Canonical module name shared by checkpoint keys and SGLang prefixes:
    'model.language_model.layers.3.mlp.gate_proj' / 'model.layers.3.mlp.gate_proj' -> 'layers.3.mlp.gate_proj';
    'mtp.layers.0.mlp.gate_proj' -> 'mtp.layers.0.mlp.gate_proj'; '...mtp.fc' -> 'mtp.fc'; '...lm_head' -> 'lm_head'."""
    parts = k.split(".")
    if "visual" in parts:          # vision tower: 'visual.blocks.N...' (the layers.N rule below would drop the block id)
        return "visual." + ".".join(parts[parts.index("visual") + 1:])
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


def _read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def scan_checkpoint(model_dir: str) -> dict[str, tuple[int, int, int, int]]:
    """{canonical module: (k, n, K bits, codebook id)} from the safetensors headers (never from JSON hints)."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.isfile(idx) else \
        [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    tensors: dict[str, list] = {}
    for fn in files:
        for name, meta in _read_header(os.path.join(model_dir, fn)).items():
            if name != "__metadata__":
                tensors[name] = meta["shape"]
    out = {}
    for name, shape in tensors.items():
        if not name.endswith(".trellis"):
            continue
        base = name[: -len(".trellis")]
        cb = "mcg" if base + ".mcg" in tensors else "mul1" if base + ".mul1" in tensors else "3inst"
        out[norm_key(base)] = (shape[0] * 16, shape[1] * 16, shape[2] // 16, _CB[cb])
    return out


def _partitions(sid) -> tuple[int, ...]:
    if sid is None:
        return ()
    if isinstance(sid, (tuple, list)):
        return tuple(sid)
    return (_QKV.get(sid, sid),)


def _unpack_signs(packed: torch.Tensor) -> torch.Tensor:
    bits = (packed.to(torch.int32).unsqueeze(1) >> torch.arange(16, device=packed.device)) & 1
    return (1.0 - 2.0 * bits.flatten()).to(torch.float16)


_MOE_STORE = None
_MOE_HOT = None
_MOE_CACHE = os.environ.get("EXL3_MOE_CACHE", "0") == "1"
_STAGE_MIN_M = int(os.environ.get("EXL3_MOE_STAGE_MIN_M", "512"))   # >= rows: prefill layer streaming (0 = off)
if _STAGE_MIN_M <= 0:
    _STAGE_MIN_M = 1 << 30


_NVTIER_ON = os.environ.get("EXL3_NVTIER", "0") == "1"      # N104 NVMe tier (VRAM slots / RAM memfd THP / NVMe store)
_NVTIER = None
_NVTIER_LAYERS: list = []                                      # registration order -> checkpoint layer number
_EXACT = None                                                  # exact decode miss path (EXL3_EXACT_MISS=1, exact_miss.py)


_STORE_META = None


def _store_meta() -> dict:
    global _STORE_META
    if _STORE_META is None:
        path = os.environ["EXL3_NVTIER_STORE"]
        _STORE_META = json.load(open(os.environ.get("EXL3_NVTIER_META", path.rsplit(".", 1)[0] + ".json")))
    return _STORE_META


def _mtp_store_layers(n_target: int) -> list:
    """Store layers that hold the MTP draft's MoE layer(s): meta `mtp_layers` (serve/pack_store.py), else every store
    layer beyond the target's (meta L > n_target). [] for a 48-layer store: the MTP layer then stays RAM-pinned."""
    m = _store_meta()
    if "mtp_layers" in m:
        return [int(c) for c in m["mtp_layers"]]
    return list(range(n_target, int(m.get("L", n_target))))


def _extra_on_nvme(li: int) -> bool:
    """Tier layer li (an extra layer registered after the target's, i.e. the MTP draft's MoE layer) has an NVMe record."""
    if _NVTIER is not None and 0 <= li < _NVTIER.L:
        return bool(_NVTIER.ckpt[li] >= 0)
    n_target = int(os.environ.get("EXL3_NVTIER_LAYERS", "48"))
    extra = int(os.environ.get("EXL3_NVTIER_EXTRA_LAYERS", "1"))
    return 0 <= li - n_target < min(extra, len(_mtp_store_layers(n_target)))


def _nvtier_init(store):
    global _NVTIER
    from .nvtier import NvTierFast, NvTierAsync
    path = os.environ["EXL3_NVTIER_STORE"]
    meta = _store_meta()
    budget = float(os.environ.get("EXL3_NVTIER_RAM_GB", "16")) * 2 ** 30
    _cls = NvTierAsync if os.environ.get("EXL3_NVTIER_ASYNC", "0") == "1" else NvTierFast
    # + EXL3_NVTIER_EXTRA_LAYERS layers registered after the target (default 1: the MTP draft's MoE layer). With a
    # 49-layer store (pack_store.py: MTP experts = store layer 48) it is an ordinary NVMe layer (ckpt 48, only its safe
    # expert pinned); with a 48-layer store it is RAM-only (ckpt -1, packed into its memfd pages, pin_host_layer).
    extra = int(os.environ.get("EXL3_NVTIER_EXTRA_LAYERS", "1"))
    n_target = len(_NVTIER_LAYERS)
    mtp = _mtp_store_layers(n_target)[:extra]
    if mtp and (max(mtp) >= int(meta["L"]) or set(mtp) & set(_NVTIER_LAYERS)):
        raise RuntimeError(f"exl3xpu: store meta: MTP layers {mtp} vs L={meta['L']} and target layers {sorted(_NVTIER_LAYERS)[:3]}...")
    _NVTIER = _cls(store, path, meta, budget, ckpt_layer=list(_NVTIER_LAYERS) + mtp + [-1] * (extra - len(mtp)),
                   standalone=range(n_target, n_target + len(mtp)))
    if extra:
        logger.info("exl3xpu: NVMe tier: %d extra (MTP) layer(s): %s", extra,
                    ", ".join(f"tier layer {n_target + i} -> " + (f"NVMe store layer {mtp[i]}" if i < len(mtp) else "RAM-pinned (no store record)")
                              for i in range(extra)))
    # N104: every prefill batch (rows >= 3) goes to the DPAS prefill kernel - it reads RAM-tier (SVM) pages at link
    # speed, while the GEMV path run non-cached over many rows is ~100x slower on SVM memory (test_pf.py)
    store.X.moe_set_prefill_min_m(int(os.environ.get("EXL3_NVTIER_PREFILL_M", "3")))
    prior = os.environ.get("EXL3_NVTIER_PRIOR")
    if prior:
        n, dt = _NVTIER.warm(json.load(open(prior))["keys"], budget * float(os.environ.get("EXL3_NVTIER_WARM_FRAC", "0.9")))
        logger.info("exl3xpu: NVMe tier warm-up: %d experts (%.1f GB) from the prior in %.1f s", n, n * 2 / 1024, dt)
    logger.info("exl3xpu: NVMe tier on: %d layers, RAM budget %.1f GB, store %s", len(_NVTIER_LAYERS), budget / 2 ** 30, path)
    if os.environ.get("EXL3_EXACT_MISS", "0") == "1":
        # exact decode misses (docs/exact-miss.md): after the warm-up, before graph capture (kernels bake in the setup)
        global _EXACT
        from .exact_miss import ExactMiss
        _EXACT = ExactMiss(_NVTIER, store)


def _moe_store(H: int, I: int, K: int, n_experts: int):
    """Process-wide ExpertStore (one slot arena shared by all MoE layers incl. the MTP layer)."""
    global _MOE_STORE
    if _MOE_STORE is None:
        from .moe_offload import ExpertStore
        if _NVTIER_ON:
            from .nvtier import TierStore
            _MOE_STORE = TierStore(H, I, K, n_experts, int(os.environ.get("EXL3_MOE_SLOTS", "0")),
                                   n_layers=int(os.environ.get("EXL3_NVTIER_MAX_LAYERS", "64")))
        else:
            _MOE_STORE = ExpertStore(H, I, K, n_experts, int(os.environ.get("EXL3_MOE_SLOTS", "0")), _dev())
        logger.info("exl3xpu: expert store: %d device slots x %d B (%.2f GB)", _MOE_STORE.n_slots, _MOE_STORE.blob,
                    _MOE_STORE.n_slots * _MOE_STORE.blob / 1e9)
    s = _MOE_STORE
    if (s.H, s.I, s.K, s.E) != (H, I, K, n_experts):
        raise NotImplementedError(f"exl3xpu: MoE layers with different geometry {(H, I, K, n_experts)} vs {(s.H, s.I, s.K, s.E)}")
    return s


def _moe_hot() -> dict:
    global _MOE_HOT
    if _MOE_HOT is None:
        p = os.environ.get("EXL3_MOE_HOT")
        _MOE_HOT = json.load(open(p)) if p else {}
    return _MOE_HOT


def _build_classes():
    from sglang.srt.layers.quantization.base_config import LinearMethodBase, QuantizationConfig
    from sglang.srt.utils.common import set_weight_attrs

    class Exl3XpuConfig(QuantizationConfig):
        def __init__(self, declared: dict | None = None, model_path: str | None = None):
            super().__init__()
            self.declared = {k: v for k, v in (declared or {}).items() if k not in ("hf_config", "packed_modules_mapping")}
            self.model_path = model_path
            self.modules = scan_checkpoint(model_path) if model_path else {}
            if model_path:
                bits = {}
                for v in self.modules.values():
                    bits[v[2]] = bits.get(v[2], 0) + 1
                logger.info("exl3xpu: %s: %d EXL3 linears, by bits %s, %d in the MTP head", model_path,
                            len(self.modules), bits, sum(1 for k in self.modules if k.startswith("mtp.")))

        def get_name(self) -> str:
            return "exl3"

        def get_supported_act_dtypes(self):
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 0

        @staticmethod
        def get_config_filenames() -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict[str, Any]):
            hf_config = config.get("hf_config")
            hf_path = getattr(hf_config, "_name_or_path", None)
            cli_path = None
            for idx, arg in enumerate(sys.argv[:-1]):
                if arg in ("--model-path", "--model") and os.path.isdir(sys.argv[idx + 1]):
                    cli_path = sys.argv[idx + 1]
                    break
            path = os.environ.get("EXL3_MODEL_PATH") or cli_path or hf_path
            if path and not os.path.isdir(path):
                from huggingface_hub import snapshot_download
                path = snapshot_download(path, local_files_only=True)
            if not path:
                raise ValueError("exl3xpu: model path unknown (set EXL3_MODEL_PATH)")
            cfg = cls(config, path)
            if config.get("packed_modules_mapping"):
                cfg.packed_modules_mapping = dict(config["packed_modules_mapping"])
            return cfg

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant):
            if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == "exl3" and user_quant in (None, "exl3"):
                return "exl3"
            return None

        def get_scaled_act_names(self):
            return []

        def __getstate__(self):
            return self.__dict__.copy()

        def lookup(self, prefix: str, draft: bool = False):
            key = norm_key(prefix)
            if draft and not key.startswith("mtp.") and key.startswith("layers."):
                key = "mtp." + key
            return self.modules.get(key)

        def _sources(self, prefix: str) -> list[str]:
            parent, _, leaf = prefix.rpartition(".")
            packed = (getattr(self, "packed_modules_mapping", None) or _PACKED).get(leaf) or _PACKED.get(leaf)
            return [f"{parent}.{s}" if parent else s for s in packed] if packed else [prefix]

        def get_quant_method(self, layer: torch.nn.Module, prefix: str):
            from sglang.srt.layers.linear import LinearBase
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
            try:
                from sglang.srt.layers.radix_attention import RadixAttention
                if isinstance(layer, RadixAttention):
                    # k_scale/v_scale = 1.0 (no calibrated scales in EXL3 checkpoints): what fp8 KV needs on XPU
                    from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
                    return BaseKVCacheMethod(self)
            except ImportError:  # pragma: no cover
                pass
            try:
                from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
                if isinstance(layer, FusedMoE):
                    base = norm_key(prefix) + "."
                    experts = {k: v for k, v in self.modules.items() if k.startswith(base)}
                    return Exl3XpuMoEMethod(prefix, experts) if experts else None
            except ImportError:  # pragma: no cover
                pass
            parts = prefix.split(".")
            if ("visual" in parts or "vision_tower" in parts) and isinstance(layer, LinearBase):
                infos = [self.lookup(p) for p in self._sources(prefix)]
                # Flash-Next quantizes the ViT (vision_bits 5): EXL3 there; bf16 checkpoints' ViTs stay unquantized
                return Exl3XpuLinearMethod(prefix, infos) if infos and all(infos) else UnquantizedLinearMethod()
            draft = prefix.startswith("mtp") or ".mtp." in prefix or getattr(layer, "_exl3_draft", False)
            if isinstance(layer, ParallelLMHead):
                info = self.lookup(prefix, draft)
                if info is None and not draft:
                    info = self.modules.get("lm_head")
                return Exl3XpuLinearMethod(prefix, [info]) if info else None
            if not isinstance(layer, LinearBase):
                return None
            infos = [self.lookup(p, draft) for p in self._sources(prefix)]
            if not any(infos):
                return UnquantizedLinearMethod()
            if not all(infos):
                raise ValueError(f"exl3xpu: {prefix} mixes EXL3 and unquantized sources {self._sources(prefix)}")
            return Exl3XpuLinearMethod(prefix, infos)

    class Exl3XpuLinearMethod(LinearMethodBase):
        def __init__(self, prefix: str, infos: list[tuple]):
            self.prefix = prefix
            self.infos = infos       # per checkpoint matrix: (k, n, K, cb)

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                           params_dtype, **extra):
            if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
                raise NotImplementedError(f"exl3xpu: {self.prefix}: tensor parallel EXL3 is not supported (use DP)")
            layer.exl3_out_sizes = list(output_partition_sizes)
            layer.exl3_in_size = input_size
            layer.exl3_shards = {s: {} for s in _SUFFIXES}
            for suffix in _SUFFIXES:
                p = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": self._loader(layer, suffix), "exl3_placeholder": True})
                layer.register_parameter(suffix, p)
            if not hasattr(layer, "weight"):
                # ParallelLMHead: code paths read `.weight` (shape/dtype); zero-width placeholder
                w = torch.nn.Parameter(torch.empty((sum(output_partition_sizes), 0), dtype=params_dtype),
                                       requires_grad=False)
                set_weight_attrs(w, {"weight_loader": lambda *a, **k: None, "exl3_placeholder": True})
                layer.register_parameter("weight", w)

        def _loader(self, layer, suffix):
            def load(param, loaded_weight, loaded_shard_id=None, *a, **k):
                store = layer.exl3_shards[suffix]
                key = tuple(loaded_shard_id) if isinstance(loaded_shard_id, list) else loaded_shard_id
                if key in store:
                    raise ValueError(f"exl3xpu: {self.prefix}.{suffix}: shard {key!r} loaded twice")
                store[key] = loaded_weight.to(_dev(), copy=True)
            return load

        def process_weights_after_loading(self, layer) -> None:
            if not hasattr(layer, "exl3_shards"):
                return
            got = layer.exl3_shards
            ids = sorted(got["trellis"], key=lambda sid: _partitions(sid) or (0,))
            if not ids:
                logger.info("exl3xpu: %s: no EXL3 tensors delivered (shared from the target?)", self.prefix)
                layer.exl3_empty = True
                return
            if len(ids) != len(self.infos):
                raise ValueError(f"exl3xpu: {self.prefix}: expected {len(self.infos)} matrices, got {len(ids)} ({ids})")
            trellis, suh, svh, bounds, Ks, cbs = [], [], [], [0], set(), set()
            for i, sid in enumerate(ids):
                su, sv = got["suh"].get(sid), got["svh"].get(sid)
                if su is None and sid in got["su"]:
                    su, sv = _unpack_signs(got["su"][sid]), _unpack_signs(got["sv"][sid])
                if su is None or sv is None:
                    raise ValueError(f"exl3xpu: {self.prefix}: shard {sid!r} has no scale vectors")
                t = got["trellis"][sid]
                cb = 1 if sid in got["mcg"] else 2 if sid in got["mul1"] else 0
                k, n, K, cb_decl = self.infos[i]
                if (t.shape[0] * 16, t.shape[1] * 16, t.shape[2] // 16, cb) != (k, n, K, cb_decl):
                    raise ValueError(f"exl3xpu: {self.prefix}: shard {sid!r} {tuple(t.shape)} cb={cb} != header {(k, n, K, cb_decl)}")
                parts = _partitions(sid)
                width = sum(layer.exl3_out_sizes[p] for p in parts) if parts else sum(layer.exl3_out_sizes)
                if width != n:
                    raise ValueError(f"exl3xpu: {self.prefix}: shard {sid!r} is {n} wide, SGLang expects {width}")
                trellis.append(t); suh.append(su.to(torch.float16)); svh.append(sv.to(torch.float16))
                bounds.append(bounds[-1] + n); Ks.add(K); cbs.add(cb)
            if len(Ks) != 1 or len(cbs) != 1:
                raise ValueError(f"exl3xpu: {self.prefix}: fused group mixes bitrates/codebooks {Ks} {cbs}")
            if bounds[-1] % 128 or layer.exl3_in_size % 128:
                raise ValueError(f"exl3xpu: {self.prefix}: dims must be multiples of 128 ({layer.exl3_in_size}, {bounds})")
            for suffix in _SUFFIXES:
                delattr(layer, suffix)
            del layer.exl3_shards
            dev = trellis[0].device
            layer.register_buffer("exl3_trellis", torch.cat(trellis, 1).contiguous() if len(trellis) > 1 else trellis[0].contiguous(), persistent=False)
            layer.register_buffer("exl3_suh", torch.stack(suh).contiguous(), persistent=False)
            layer.register_buffer("exl3_svh", torch.cat(svh).contiguous(), persistent=False)
            sonb = torch.empty(bounds[-1] // 128, dtype=torch.int32)
            for g in range(len(bounds) - 1):
                sonb[bounds[g] // 128: bounds[g + 1] // 128] = g
            layer.register_buffer("exl3_shard_of_nb", sonb.to(dev), persistent=False)
            layer.exl3_bounds = bounds
            layer.exl3_K = Ks.pop()
            layer.exl3_cb = cbs.pop()
            del trellis
            _empty_cache()
            from . import ops
            E = ops._get_esimd()
            if not (E and hasattr(E, "linear") and E.exl3_supported(layer.exl3_K, layer.exl3_cb)):
                raise RuntimeError(f"exl3xpu: {self.prefix}: K={layer.exl3_K} cb={layer.exl3_cb} not built into _C.so "
                                   "(rebuild with EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS)")
            if self.prefix.endswith("lm_head") and not self.prefix.startswith("mtp"):
                TARGET_LM_HEADS.append(layer)
                if os.environ.get("EXL3_DRAFT_VOCAB"):
                    _build_draft_head(layer, os.environ["EXL3_DRAFT_VOCAB"])

        def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            from . import ops
            y = torch.ops.exl3xpu_C.linear(x, layer.exl3_trellis, layer.exl3_suh, layer.exl3_svh,
                                           layer.exl3_shard_of_nb, layer.exl3_bounds, layer.exl3_K, layer.exl3_cb,
                                           ops.SMALL_M_MAX, ops.RECON_SLICE_N)
            if _SYNC_LOGITS and self.prefix.endswith("lm_head") and not torch.xpu.is_current_stream_capturing():
                torch.xpu.synchronize()          # spike hunt: host-wait on every queue right after the logits
            if _MEM_TRACE and self.prefix.endswith("lm_head") and not torch.xpu.is_current_stream_capturing():
                _mem_trace(x.shape[0])
            return y if bias is None else y + bias

        def embedding(self, layer, input_):
            raise NotImplementedError("exl3xpu: quantized input embeddings are not supported")

    class Exl3Dense(torch.nn.Module):
        """Drop-in for a plain nn.Linear whose checkpoint tensors are EXL3 (the MTP head's `fc`)."""

        def __init__(self, in_features: int, out_features: int, info: tuple, prefix: str, params_dtype=torch.float16):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.prefix = prefix
            self.method = Exl3XpuLinearMethod(prefix, [info])
            self.method.create_weights(self, in_features, [out_features], in_features, out_features, params_dtype)

        def process_weights_after_loading(self):
            self.method.process_weights_after_loading(self)

        def forward(self, x):
            return self.method.apply(self, x)


    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

    class Exl3XpuMoEMethod(FusedMoEMethodBase):
        """EXL3 routed experts on XPU through the two-tier ExpertStore (moe_offload.py): every expert in USM host
        memory (zero-copy), EXL3_MOE_SLOTS device slots shared by all MoE layers (the expert cache), grouped
        kernels addressing experts through a per-layer pointer table. TP = EP = 1, SiLU-gated, no fused shared
        expert (--disable-shared-experts-fusion), routed_scaling_factor 1.
        Env: EXL3_MOE_SLOTS (default 0: zero-copy only), EXL3_MOE_RESIDENT_PER_LAYER (fill slots with the first N
        experts of each layer at load, default 0), EXL3_MOE_HOT (JSON {layer_key: [expert ids]} placed first),
        EXL3_MOE_CACHE=1 (decode through the device-managed LRU cache: misses written through into slots)."""

        _KEY_RE = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
        _ROLE = {"w1": "gate", "w3": "up", "w2": "down"}

        def __init__(self, prefix: str, experts: dict):
            self.prefix = prefix
            self.key = norm_key(prefix)
            self.runner = None
            self.moe_runner_config = None
            self.infos = {}
            for k, info in experts.items():
                m = self._KEY_RE.search(k)
                if m is None:
                    raise ValueError(f"exl3xpu: {prefix}: unexpected EXL3 matrix under the experts: {k}")
                self.infos[(m.group(2)[:-5], int(m.group(1)))] = info

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            if getattr(layer, "moe_tp_size", 1) != 1 or getattr(layer, "moe_ep_size", 1) != 1:
                raise NotImplementedError(f"exl3xpu: {self.prefix}: TP/EP EXL3 MoE is not implemented")
            if getattr(layer, "num_fused_shared_experts", 0):
                raise NotImplementedError(f"exl3xpu: {self.prefix}: run with --disable-shared-experts-fusion")
            missing = [(r, e) for r in ("gate", "up", "down") for e in range(num_experts) if (r, e) not in self.infos]
            if missing:
                raise ValueError(f"exl3xpu: {self.prefix}: no EXL3 tensors for {len(missing)} expert matrices, e.g. {missing[0]}")
            layer.exl3_store = {}
            layer.exl3_num_experts, layer.exl3_hidden, layer.exl3_inter = num_experts, hidden_size, intermediate_size_per_partition
            for w in ("w13", "w2"):
                for suffix in _SUFFIXES:
                    p = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
                    set_weight_attrs(p, {"weight_loader": self._loader(layer, suffix), "exl3_placeholder": True})
                    layer.register_parameter(f"{w}_{suffix}", p)

        def _loader(self, layer, suffix):
            def load(param, loaded_weight, weight_name=None, shard_id=None, expert_id=None, *a, **k):
                role = self._ROLE.get(shard_id)
                if role is None or expert_id is None:
                    raise ValueError(f"exl3xpu: {self.prefix}.{suffix}: unexpected shard {shard_id!r} / expert {expert_id!r}")
                st = layer.exl3_store.setdefault(int(expert_id), {}).setdefault(role, {})
                if suffix in st:
                    raise ValueError(f"exl3xpu: {self.prefix}: {role} {suffix} of expert {expert_id} loaded twice")
                st[suffix] = loaded_weight.detach().to("cpu", copy=True)
                self._maybe_pack(layer, int(expert_id))
            return load

        def _maybe_pack(self, layer, e: int) -> None:
            """Pack an expert into its host blob as soon as all its tensors arrived (bounded CPU staging)."""
            st = layer.exl3_store.get(e)
            if not st or any(r not in st or not {"trellis", "suh", "svh"} <= set(st[r]) for r in ("gate", "up", "down")):
                return
            for r in ("gate", "up", "down"):
                t = st[r]["trellis"]
                k_, n_, K_, cb_ = self.infos[(r, e)]
                if (t.shape[0] * 16, t.shape[1] * 16, t.shape[2] // 16) != (k_, n_, K_) or cb_ != 2:
                    raise ValueError(f"exl3xpu: {self.prefix}: {r}_proj expert {e}: {tuple(t.shape)} cb={cb_} "
                                     f"(need mul1, header {(k_, n_, K_)})")
            K = st["gate"]["trellis"].shape[2] // 16
            if st["down"]["trellis"].shape[2] // 16 != K or st["up"]["trellis"].shape[2] // 16 != K:
                raise NotImplementedError(f"exl3xpu: {self.prefix}: mixed K inside expert {e}")
            store = _moe_store(layer.exl3_hidden, layer.exl3_inter, K, layer.exl3_num_experts)
            if _NVTIER_ON:
                # tier mode: weights come from the packed NVMe store; nothing is packed into host RAM at load
                if self.key.startswith("mtp."):
                    if self.key not in store.layer_index:
                        store.add_layer(self.key)
                    if _extra_on_nvme(store.layer_index[self.key]):
                        pass        # 49-layer store: the MTP experts are NVMe store layer 48, nothing to pack
                    else:
                        # 48-layer store: no NVMe record - pack the experts straight into their RAM-tier pages
                        # (pinned by pin_host_layer() once the layer is complete)
                        import ctypes as _ct
                        from .moe_offload import pack_expert
                        a = store.addr(store.layer_index[self.key] * store.E + e)
                        out = torch.frombuffer((_ct.c_char * store.blob).from_address(a), dtype=torch.uint8)
                        pack_expert(st["gate"], st["up"], st["down"], K, out=out)
                elif self.key not in store.layer_index:
                    import re as _re
                    store.add_layer(self.key)
                    _NVTIER_LAYERS.append(int(_re.search(r"layers\.(\d+)\.", self.prefix + ".").group(1)))
            else:
                if self.key not in store.host:
                    store.add_layer(self.key)
                from .moe_offload import pack_expert
                pack_expert(st["gate"], st["up"], st["down"], K, out=store.host_view(self.key)[e])
            layer.exl3_K = K
            layer.exl3_packed = getattr(layer, "exl3_packed", 0) + 1
            del layer.exl3_store[e]

        def create_moe_runner(self, layer, moe_runner_config):
            self.moe_runner_config = cfg = moe_runner_config
            if cfg.activation != "silu" or not getattr(cfg, "is_gated", True):
                raise NotImplementedError(f"exl3xpu: {self.prefix}: only SiLU-gated experts (got {cfg.activation})")
            if cfg.apply_router_weight_on_input:
                raise NotImplementedError(f"exl3xpu: {self.prefix}: apply_router_weight_on_input unsupported")
            if cfg.routed_scaling_factor not in (None, 1.0):
                raise NotImplementedError(f"exl3xpu: {self.prefix}: routed_scaling_factor {cfg.routed_scaling_factor}")

        def process_weights_after_loading(self, layer) -> None:
            if not hasattr(layer, "exl3_store"):
                return
            n = layer.exl3_num_experts
            if layer.exl3_store or getattr(layer, "exl3_packed", 0) != n:
                raise ValueError(f"exl3xpu: {self.prefix}: {getattr(layer, 'exl3_packed', 0)} of {n} experts complete "
                                 f"({len(layer.exl3_store)} partial)")
            del layer.exl3_store
            for w in ("w13", "w2"):
                for suffix in _SUFFIXES:
                    delattr(layer, f"{w}_{suffix}")
            store = _moe_store(layer.exl3_hidden, layer.exl3_inter, layer.exl3_K, n)
            hot = _moe_hot().get(self.key, [])
            per = int(os.environ.get("EXL3_MOE_RESIDENT_PER_LAYER", "0"))
            want = list(dict.fromkeys(list(hot) + list(range(n))))[:max(per, len(hot))] if (per or hot) else []
            want = want[:len(store.free_slots)]
            if want and not _NVTIER_ON:
                store.make_resident(self.key, want)
            layer.exl3_moe_store = store
            if _NVTIER_ON and _NVTIER is None and len(_NVTIER_LAYERS) == int(os.environ.get("EXL3_NVTIER_LAYERS", "48")):
                _nvtier_init(store)          # before graph capture: captured kernels bake in the tier tensors
            if _NVTIER_ON and self.key.startswith("mtp."):
                if _NVTIER is None:
                    raise RuntimeError("exl3xpu: MTP layer loaded before the NVMe tier was set up")
                li = store.layer_index[self.key]
                if not len(_NVTIER_LAYERS) <= li < _NVTIER.L:
                    raise RuntimeError(f"exl3xpu: {self.prefix}: tier layer {li} outside the extra layers "
                                       f"[{len(_NVTIER_LAYERS)}, {_NVTIER.L}) (EXL3_NVTIER_EXTRA_LAYERS)")
                if _NVTIER.ckpt[li] >= 0:
                    logger.info("exl3xpu: %s: %d MTP experts served from NVMe store layer %d (ordinary tier layer)",
                                self.prefix, n, int(_NVTIER.ckpt[li]))
                else:
                    _NVTIER.pin_host_layer(li)
                    logger.info("exl3xpu: %s: %d MTP experts pinned in the RAM tier (no NVMe record)", self.prefix, n)
            logger.info("exl3xpu: %s: %d experts K=%d packed to host USM (%.3f GB), %d resident in device slots "
                        "(%d slots free)", self.prefix, n, layer.exl3_K, n * store.blob / 1e9,
                        store.resident_count(self.key), len(store.free_slots))

        def apply(self, layer, dispatch_output):
            from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
            x = dispatch_output.hidden_states
            tk = dispatch_output.topk_output
            flat = x.reshape(-1, x.shape[-1])
            store = layer.exl3_moe_store
            # tier mode: every PREFILL batch (rows > max decode batch: EXL3_NVTIER_PREFILL_M, default 3 = max-running 2 + 1)
            # fills its experts first - masking prompt experts wrecks the prompt (N104 srv4: ~70% of a cold 24-token prompt's
            # picks masked -> degenerate output). Also: calls the kernel runs non-cached (M >= 256) read host addresses
            # host addresses unmasked, so the batch's experts must be RAM-resident first - else it reads unfilled holes
            # (zero pages: wrong outputs and runaway shmem). Not just the >= _STAGE_MIN_M streaming threshold.
            if _NVTIER is not None and flat.shape[0] >= int(os.environ.get("EXL3_NVTIER_PREFILL_M", "3")) \
                    and not torch.xpu.is_current_stream_capturing():
                if os.environ.get("EXL3_NVTIER_STAGE", "0") == "1" and flat.shape[0] >= int(os.environ.get("EXL3_NVTIER_STAGE_M", "256")) \
                        and (not self.key.startswith("mtp.") or _extra_on_nvme(store.layer_index[self.key])):
                    # d1b4: big prefill batch -> NVMe -> pinned host -> copy engine -> VRAM staging (no SVM first touch)
                    y = _NVTIER.stage_forward(store, self.key, store.layer_index[self.key], flat,
                                              tk.topk_ids.reshape(flat.shape[0], -1), tk.topk_weights.reshape(flat.shape[0], -1))
                    return StandardCombineInput(hidden_states=y.view_as(x))
                # tier mode prefill: make this layer's experts RAM-resident from NVMe, then zero-copy prefill kernel
                _NVTIER.ensure_for_prefill(store.layer_index[self.key], tk.topk_ids)
                y = store.forward(self.key, flat, tk.topk_ids.reshape(flat.shape[0], -1), tk.topk_weights.reshape(flat.shape[0], -1))
                return StandardCombineInput(hidden_states=y.view_as(x))
            if flat.shape[0] >= _STAGE_MIN_M and not torch.xpu.is_current_stream_capturing() \
                    and not self.key.startswith("mtp."):
                # prefill: stream the layer's non-resident experts through the copy engine (double buffer)
                fwd = store.forward_prefill_staged
            else:
                store.prefill_reset()
                fwd = store.forward_cached if _MOE_CACHE else store.forward
                if _EXACT is not None and _EXACT.predict and not self.key.startswith("mtp.") and self.key in store.layer_index:
                    # next-layer prediction: hints for layer L+ahead before this layer's MoE (and its wait) is queued
                    _EXACT.hint_next(store.layer_index[self.key], flat)
            y = fwd(self.key, flat, tk.topk_ids.reshape(flat.shape[0], -1), tk.topk_weights.reshape(flat.shape[0], -1))
            return StandardCombineInput(hidden_states=y.view_as(x))

        def get_triton_quant_info(self, layer):
            raise NotImplementedError("EXL3 experts do not run on the Triton MoE runner")

    globals()["Exl3XpuMoEMethod"] = Exl3XpuMoEMethod

    return Exl3XpuConfig, Exl3XpuLinearMethod, Exl3Dense


_CLASSES = None


def classes():
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = _build_classes()
    return _CLASSES


# ---------------------------------------------------------------------------------------------------------------
# Pruned-vocabulary MTP draft head (same idea as the vLLM plugin): the drafter proposes from a subset of 128-token
# lm_head blocks; the target verifies with the full head, so outputs are unchanged. EXL3_DRAFT_VOCAB=<json>.

def _build_draft_head(layer, path):
    with open(path) as f:
        spec = json.load(f)
    blocks = torch.tensor(spec["blocks"], dtype=torch.long)
    nb = layer.exl3_svh.shape[0] // 128
    blocks = blocks[blocks < nb]
    dev = layer.exl3_trellis.device
    tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
    layer.exl3_draft = dict(
        trellis=layer.exl3_trellis.index_select(1, tiles).contiguous(),
        svh=layer.exl3_svh.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous(),
        idx=(blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev),
        shard=torch.zeros(len(blocks), dtype=torch.int32, device=dev),
        bounds=[0, len(blocks) * 128])
    logger.info("exl3xpu: pruned draft head: %d of %d vocab blocks", len(blocks), nb)


def _patch_mtp() -> None:
    """Qwen3.5 MTP draft: `fc` is a plain nn.Linear upstream but EXL3 checkpoints quantize `mtp.fc`; the draft's own
    embed_tokens is replaced by the target's (share it from construction so its bf16 copy never occupies VRAM)."""
    try:
        from sglang.srt.models import qwen3_5_mtp as m
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: MTP shim not installed (%s)", e)
        return
    cls = m.Qwen3_5ForCausalLMMTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig_init, orig_load = cls.__init__, cls.load_weights

    def __init__(self, config, quant_config=None, prefix="", *a, **k):
        orig_init(self, config, quant_config, prefix, *a, **k)
        Cfg, _, Dense = classes()
        qc = getattr(self, "quant_config", quant_config)
        if not isinstance(qc, Cfg):
            return
        emb = self.model.embed_tokens
        w = emb.weight
        emb.weight = torch.nn.Parameter(torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device), requires_grad=False)
        for name, val in vars(w).items():
            if name != "data" and not hasattr(emb.weight, name):
                setattr(emb.weight, name, val)
        self._exl3_skip_embed = True
        del w
        _empty_cache()
        info = qc.modules.get("mtp.fc")
        if info is not None and isinstance(getattr(self, "fc", None), torch.nn.Linear):
            self.fc = Dense(self.fc.in_features, self.fc.out_features, info, "mtp.fc", params_dtype=self.fc.weight.dtype)
            logger.info("exl3xpu: MTP fc served as an EXL3 linear %s", info)

    def load_weights(self, weights, *a, **k):
        if getattr(self, "_exl3_skip_embed", False):
            weights = ((n, w) for n, w in weights if not n.endswith("embed_tokens.weight"))
        out = orig_load(self, weights, *a, **k)
        fc = getattr(self, "fc", None)
        if fc is not None and hasattr(fc, "exl3_shards"):
            fc.process_weights_after_loading()
        return out

    orig_set_head = getattr(cls, "set_lm_head_from_target", None)

    def set_lm_head_from_target(self, target_lm_head):
        if orig_set_head is not None:
            orig_set_head(self, target_lm_head)
        if getattr(target_lm_head, "exl3_draft", None) is not None:
            self.lm_head = _DraftHead(target_lm_head)
            logger.info("exl3xpu: MTP draft proposes from the pruned head (%d vocab blocks)",
                        target_lm_head.exl3_draft["bounds"][1] // 128)

    cls.__init__, cls.load_weights, cls._exl3_patched = __init__, load_weights, True
    cls.set_lm_head_from_target = set_lm_head_from_target


class _Exl3Dense3D(torch.nn.Module):
    """Wraps an Exl3Dense so it accepts [..., in] (Qwen4-Exp's MTP applies fc_hidden to a [N, hc_count, H] view)."""

    def __init__(self, dense):
        super().__init__()
        self.inner = dense
        self.in_features, self.out_features = dense.in_features, dense.out_features

    def forward(self, x):
        y = self.inner(x.reshape(-1, x.shape[-1]))
        return y.view(*x.shape[:-1], y.shape[-1])


def _patch_qwen4_mtp() -> None:
    """Qwen4-Exp (Flash-Next) MTP draft (from qwen-serve N113). Its __init__ does not run Qwen3.5's (patched by
    _patch_mtp), so: serve `mtp.fc_embedding` / `mtp.fc_hidden` / `mtp.fc` (EXL3 in the checkpoint, plain nn.Linear
    upstream) as EXL3 linears, and drop the draft's own bf16 embed_tokens at construction: the EAGLE worker shares
    the target's."""
    try:
        from sglang.srt.models import qwen4_exp_mtp as m
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: Qwen4-Exp MTP shim not installed (%s)", e)
        return
    cls = m.Qwen4ExpForCausalLMMTP
    if cls.__dict__.get("_exl3_patched", False):      # not inherited from the Qwen3.5 class _patch_mtp marks
        return
    orig_init = cls.__init__
    orig_load = cls.load_weights

    def __init__(self, config, quant_config=None, prefix="", *a, **k):
        orig_init(self, config, quant_config, prefix, *a, **k)
        Cfg, _, Dense = classes()
        qc = getattr(self, "quant_config", quant_config)
        if not isinstance(qc, Cfg):
            return
        emb = self.model.embed_tokens
        w = emb.weight
        emb.weight = torch.nn.Parameter(torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device), requires_grad=False)
        for name, val in vars(w).items():
            if name != "data" and not hasattr(emb.weight, name):
                setattr(emb.weight, name, val)
        self._exl3_skip_embed = True
        del w
        _empty_cache()
        self._exl3_dense = []
        for attr in ("fc_embedding", "fc_hidden", "fc"):
            lin = getattr(self, attr, None)
            info = qc.modules.get(f"mtp.{attr}")
            if info is None or not isinstance(lin, torch.nn.Linear):
                continue
            setattr(self, attr, Dense(lin.in_features, lin.out_features, info, f"mtp.{attr}", params_dtype=lin.weight.dtype))
            self._exl3_dense.append(attr)
            logger.info("exl3xpu: Qwen4-Exp MTP %s served as an EXL3 linear %s", attr, info)

    def load_weights(self, weights, *a, **k):
        if getattr(self, "_exl3_skip_embed", False):
            weights = ((n, w) for n, w in weights if not n.endswith("embed_tokens.weight"))
        out = orig_load(self, weights, *a, **k)
        for attr in getattr(self, "_exl3_dense", []):
            d = getattr(self, attr)
            if hasattr(d, "exl3_shards"):
                d.process_weights_after_loading()
            if getattr(d, "exl3_empty", False):
                raise RuntimeError(f"exl3xpu: Qwen4-Exp MTP {attr}: no EXL3 tensors loaded")
            setattr(self, attr, _Exl3Dense3D(d))
        return out

    cls.__init__, cls.load_weights, cls._exl3_patched = __init__, load_weights, True


class _DraftHeadMethod:
    """quant_method of the pruned draft head: EXL3 GEMM over the kept 128-token blocks only, scattered into a
    full-vocab -inf row (the draft's top-1 can only be a kept token; the target verifies with the full head)."""

    def apply(self, layer, x, bias=None):
        from . import ops
        t = layer.target
        d = t.exl3_draft
        sub = torch.ops.exl3xpu_C.linear(x, d["trellis"], t.exl3_suh, d["svh"], d["shard"], d["bounds"],
                                         t.exl3_K, t.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        out = x.new_full((x.shape[0], t.exl3_svh.shape[0]), float("-inf"))
        out.index_copy_(1, d["idx"], sub)
        return out


class _DraftHead(torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        object.__setattr__(self, "target", target)      # not a submodule: no double registration / state_dict
        self.weight = target.weight                     # zero-width placeholder (logits code reads .weight)
        self.quant_method = _DraftHeadMethod()
        for a in ("org_vocab_size", "num_embeddings", "num_embeddings_padded", "vocab_size"):
            if hasattr(target, a):
                setattr(self, a, getattr(target, a))


_TOPK_FAST = int(os.environ.get("EXL3_SGL_SAMPLE_TOPK", "256"))   # 0 disables the top-k fast path


def _sample_rows(logits: torch.Tensor, temps: torch.Tensor, top_ks: torch.Tensor | None,
                 top_ps: torch.Tensor | None) -> torch.Tensor:
    """One token per row from softmax(logits / T) with SGLang's semantics: top-k renormalize, then top-p on the
    renormalized probabilities (top_k_renorm_prob -> top_p_renorm_prob). When every row's top-k is <= _TOPK_FAST
    (generation-config default top_k=20) only the top _TOPK_FAST logits are sorted: 0.3 vs 5.5 ms for 32 rows of the
    248K vocab. Otherwise the whole row is sorted."""
    x = logits.float() / temps.float().view(-1, 1)
    if top_ks is not None and _TOPK_FAST > 0 and bool((top_ks <= _TOPK_FAST).all()):
        sv, si = torch.topk(x, _TOPK_FAST, dim=-1)                   # sorted descending
    else:
        sv, si = x.sort(dim=-1, descending=True)
    ranks = torch.arange(sv.shape[1], device=sv.device).view(1, -1)
    if top_ks is not None:
        sv = sv.masked_fill(ranks >= top_ks.view(-1, 1).clamp_min(1), float("-inf"))
    sp = torch.softmax(sv, dim=-1)                                   # renormalized over the kept top-k
    if top_ps is not None:
        keep = (sp.cumsum(-1) - sp) < top_ps.float().view(-1, 1)
        keep[:, 0] = True
        sp = sp * keep
    pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)
    return si.gather(1, pick)


def _patch_xpu_spec_sampling() -> None:
    """SGLang 0.5.20 on XPU verifies speculative drafts greedily whatever the request's temperature (eagle_sample:
    `is_all_greedy or ... or _is_xpu`), i.e. sampling is silently turned off whenever MTP is on. Restore it: sample one
    target token per verify row from the request's distribution (temperature, top-k, top-p) and make it the row's
    argmax, so the stock greedy chain verify accepts a draft token iff it equals the sampled target token. For a
    deterministic top-1 draft chain this is lossless (every emitted token is a sample of the target distribution).
    Penalties/logit bias are applied by eagle_sample after this and are not reflected in the sample (not used here).
    EXL3_SGL_SPEC_SAMPLE=0 restores the stock behaviour."""
    if os.environ.get("EXL3_SGL_SPEC_SAMPLE", "1") != "1":
        return
    try:
        from sglang.srt.speculative import eagle_worker_common as ewc
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: spec sampling patch not installed (%s)", e)
        return
    if getattr(ewc, "_exl3_sample_patched", False) or not hasattr(ewc, "eagle_sample"):
        return
    orig = ewc.eagle_sample

    def eagle_sample(verify_input, batch, logits_output, *a, **k):
        si = getattr(batch, "sampling_info", None)
        logits = getattr(logits_output, "next_token_logits", None)
        if (si is not None and logits is not None and logits.device.type == "xpu" and not si.is_all_greedy
                and not batch.forward_mode.is_idle()):
            n = verify_input.draft_token_num
            rep = lambda t: None if t is None else torch.repeat_interleave(t.view(-1), n, dim=0)
            tok = _sample_rows(logits, rep(si.temperatures),
                               rep(si.top_ks) if si.need_top_k_sampling else None,
                               rep(si.top_ps) if si.need_top_p_sampling else None)
            logits.scatter_(1, tok, torch.finfo(logits.dtype).max)
        return orig(verify_input, batch, logits_output, *a, **k)

    ewc.eagle_sample, ewc._exl3_sample_patched = eagle_sample, True
    logger.info("exl3xpu: XPU speculative verify samples the target distribution (was greedy-only)")


def _patch_xpu_gdn_verify() -> None:
    """SGLang 0.5.20's XPU copy of the fused sigmoid-gating delta-rule wrapper
    (hardware_backend/xpu/kernels/fla) is stale against the shared Triton kernel: it omits `stride_h0_source`
    (TypeError on the first MTP verify step) and passes the runtime draft count instead of the per-request pitch of
    the intermediate-state buffer. Re-define the XPU wrapper with both fixed and its XPU launch shape kept (BV=16,
    fewer registers than the generic wrapper's BV=32). EXL3_SGL_GDN_FIX=0 disables; =generic uses the generic wrapper."""
    mode = os.environ.get("EXL3_SGL_GDN_FIX", "1")
    if mode == "0":
        return
    try:
        if not torch.xpu.is_available():
            return
        import inspect
        import textwrap
        from sglang.srt.layers.attention.linear.kernels import gdn_triton
        if mode == "generic":
            from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
                fused_sigmoid_gating_delta_rule_update as generic)
            gdn_triton.fused_sigmoid_gating_delta_rule_update = generic
            return
        from sglang.srt.hardware_backend.xpu.kernels.fla import fused_sigmoid_gating_recurrent as xm
        src = textwrap.dedent(inspect.getsource(xm.fused_sigmoid_gating_delta_rule_update))
        a1 = "h0_indices=initial_state_indices,"
        a2 = "cache_steps=0 if cache_steps is None else cache_steps,"
        if a1 not in src or a2 not in src or "stride_h0_source" in src:
            logger.warning("exl3xpu: XPU GDN wrapper layout changed; GDN verify fix not applied")
            return
        line1 = next(l for l in src.splitlines() if l.strip() == a1)
        ind = line1[: len(line1) - len(line1.lstrip())]
        src = src.replace(line1, line1 + "\n" + ind +
                          "stride_h0_source=(initial_state_source.stride(0) if initial_state_source is not None else 0),", 1)
        src = src.replace(a2, "cache_steps=(intermediate_states_buffer.stride(0) // (HV * K * V) "
                              "if intermediate_states_buffer is not None else (cache_steps or 0)),", 1)
        ns: dict = {}
        exec(compile(src, "<exl3xpu:xpu fused_sigmoid_gating_delta_rule_update>", "exec"), xm.__dict__, ns)
        fn = ns["fused_sigmoid_gating_delta_rule_update"]
        xm.fused_sigmoid_gating_delta_rule_update = fn
        gdn_triton.fused_sigmoid_gating_delta_rule_update = fn
        logger.info("exl3xpu: XPU GDN verify wrapper fixed (stride_h0_source, per-request pitch)")
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: GDN verify fix not installed (%s)", e)


def _patch_xpu_replayssm() -> None:
    """--enable-linear-replayssm-spec (replaces the per-draft full-state snapshots, 4.8 GB for 16 streams on the 27B,
    with a raw-input window): its verify kernel does tl.dot over a [BS, K] x [K, BS] tile with BS = next_pow2(draft
    tokens) = 4, and XPU Triton requires dot dims >= 16. Pad the spec tile to 16 (rows beyond the draft are masked).
    EXL3_SGL_RSSM_FIX=0 disables."""
    if os.environ.get("EXL3_SGL_RSSM_FIX", "1") != "1":
        return
    try:
        if not torch.xpu.is_available():
            return
        from sglang.kernels.ops.attention.fla import gdn_replayssm_spec_decode as g
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: ReplaySSM XPU fix not installed (%s)", e)
        return
    orig = getattr(g, "_launch_gdn_spec", None)
    if orig is None or getattr(orig, "_exl3", False):
        return
    import inspect
    sig = inspect.signature(orig)
    warps = int(os.environ.get("EXL3_RSSM_WARPS", "4"))
    dotp = os.environ.get("EXL3_RSSM_DOT", "ieee")

    def _launch_gdn_spec(*args, **kw):
        ba = sig.bind(*args, **kw)
        ba.arguments["bs_min"] = max(16, ba.arguments.get("bs_min") or 0)
        ba.arguments["num_warps"] = warps
        ba.arguments["dot_precision"] = dotp
        return orig(*ba.args, **ba.kwargs)

    _launch_gdn_spec._exl3 = True
    g._launch_gdn_spec = _launch_gdn_spec


_ONEDNN_MIN_Q = int(os.environ.get("EXL3_ONEDNN_MIN_Q", "64"))
_ONEDNN_MIN_SEQ = int(os.environ.get("EXL3_ONEDNN_MIN_SEQ", "4096"))


def _patch_xpu_attn_routes() -> None:
    """Two re-routes of sgl_kernel's XPU flash_attn_with_kvcache, installed on the name xpu_backend calls.

    1. EXL3_SGL_VERIFY_AS_DECODE=1 (default): speculative verify / short uniform extends (<= 8 queries per sequence)
       take the kernel's prefill path, which costs 3.0 ms per layer at 16K fp8 keys for 4 queries vs 0.08 ms for one
       decode query (tests/bench_sgl_verify_attn.py): SGLang's step time grew 3.5 ms per 1K tokens of context. Each
       verify row becomes its own decode query over the same pages with cache_seqlens = L - (nq - 1 - j), which is
       exactly the bottom-right causal mask. Tensor ops only (no host sync), so it is captured into the verify graph.
    2. EXL3_ONEDNN_ATTN=1: long single-sequence prefill chunks over the fp8 paged KV cache run through the oneDNN
       Graph fused SDPA (exl3xpu_C::exl3_sdpa_len, fp16, one call per KV head over [0, seq_len), bottom-right causal),
       the path the vLLM recipe uses."""
    verify_route = os.environ.get("EXL3_SGL_VERIFY_AS_DECODE", "1") == "1"
    onednn_route = os.environ.get("EXL3_ONEDNN_ATTN", "0") == "1"
    if not (verify_route or onednn_route):
        return
    try:
        from sglang.srt.layers.attention import xpu_backend as xb
        from . import fp8kv_prefill as fk
        from .ops import _get_esimd
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: XPU attention routes not installed (%s)", e)
        return
    orig = xb.flash_attn_with_kvcache
    if getattr(orig, "_exl3", False):
        return
    state = {"E": None, "n": 0, "v": 0}

    def _common_ok(kw):
        return (kw.get("q") is not None and kw.get("k_cache") is not None and kw.get("page_table") is not None
                and kw.get("cache_seqlens") is not None and kw.get("cu_seqlens_q") is not None
                and kw.get("causal", False) and tuple(kw.get("window_size", (-1, -1))) == (-1, -1)
                and not kw.get("return_softmax_lse", False) and not kw.get("softcap")
                and kw.get("sinks") is None and kw.get("k") is None and kw.get("cache_batch_idx") is None
                and kw.get("cache_leftpad") is None)

    def flash_attn_with_kvcache(*args, **kw):
        if args or not _common_ok(kw):
            return orig(*args, **kw)
        q, cs, pt = kw["q"], kw["cache_seqlens"], kw["page_table"]
        nq = kw.get("max_seqlen_q")
        bs = cs.shape[0]
        if (verify_route and isinstance(nq, int) and 1 < nq <= 8 and q.shape[0] == bs * nq
                and pt.shape[0] == bs):
            dev = q.device
            j = torch.arange(nq, device=dev, dtype=cs.dtype).repeat(bs)
            kw2 = dict(kw)
            kw2["cache_seqlens"] = cs.repeat_interleave(nq) - (nq - 1 - j)
            kw2["page_table"] = pt.repeat_interleave(nq, dim=0)
            kw2["cu_seqlens_q"] = torch.arange(bs * nq + 1, device=dev, dtype=kw["cu_seqlens_q"].dtype)
            kw2["max_seqlen_q"] = 1
            for d in ("k_descale", "v_descale"):
                if kw.get(d) is not None:
                    kw2[d] = kw[d].reshape(-1)[:1].reshape(()).expand(bs * nq, kw[d].shape[-1])
            if state["v"] == 0:
                logger.info("exl3xpu: %d-query verify attention served as %d decode queries", nq, bs * nq)
            state["v"] += 1
            return orig(**kw2)
        kc = kw["k_cache"]
        if not (onednn_route and kc.dtype == torch.float8_e4m3fn and kw["cu_seqlens_q"].numel() == 2
                and q.shape[0] >= _ONEDNN_MIN_Q and "out" not in kw and not torch.xpu.is_current_stream_capturing()):
            return orig(**kw)
        if state["E"] is None:
            E = _get_esimd()
            state["E"] = E if (E and hasattr(E, "exl3_sdpa_len")) else False
        if not state["E"]:
            return orig(**kw)
        seq_len = int(cs[0].item())
        if seq_len < _ONEDNN_MIN_SEQ:
            return orig(**kw)
        kd, vd = kw.get("k_descale"), kw.get("v_descale")
        ks = float(kd.reshape(-1)[0].item()) if kd is not None else 1.0
        vs = float(vd.reshape(-1)[0].item()) if vd is not None else 1.0
        out = torch.empty(q.shape, dtype=torch.float16, device=q.device)
        fk.prefill_attention_onednn(state["E"], q.to(torch.float16), kc, kw["v_cache"], pt[0], seq_len, ks, vs,
                                    float(kw["softmax_scale"]), out)
        if state["n"] == 0:
            logger.info("exl3xpu: oneDNN fused SDPA serving a %d-query prefill chunk at %d keys", q.shape[0], seq_len)
        state["n"] += 1
        return out.to(q.dtype)

    flash_attn_with_kvcache._exl3 = True
    xb.flash_attn_with_kvcache = flash_attn_with_kvcache
    logger.info("exl3xpu: XPU attention routes: verify-as-decode=%s oneDNN-prefill=%s", verify_route, onednn_route)


def _patch_gdn_replayssm_fold() -> None:
    """EXL3_SGL_GDN_FOLD=1 (with --enable-linear-replayssm-spec): use SGLang's raw-input "fold-every-commit" ReplaySSM
    protocol for GDN (SGLang 0.5.20 gates it to KDA and uses the compact circular kernel for GDN, which Intel Triton
    cannot compile). Verify ring-writes the raw k/v/g/beta of the draft window (generic fused recurrent kernel, same
    one the snapshot path uses), the commit folds the accepted prefix into the state. Drops the per-draft full-state
    snapshots: 4.8 GB at 16 streams on Qwen3.8-27B."""
    if os.environ.get("EXL3_SGL_GDN_FOLD", "0") != "1":
        return
    try:
        import inspect
        import textwrap
        from sglang.srt.mem_cache import memory_pool as m
        cls = m.MambaPool
        if getattr(cls.__init__, "_exl3", False):
            return
        src = textwrap.dedent(inspect.getsource(cls.__init__))
        new, n1 = re.subn(r"enable_linear_replayssm_spec and cache_params\.is_kda(\s*\n\s*\))",
                          r"enable_linear_replayssm_spec\1", src, count=1)
        new, n2 = re.subn(r"if enable_linear_replayssm_spec and cache_params\.is_kda:",
                          "if enable_linear_replayssm_spec and (cache_params.is_kda or self.replayssm_spec_fold):",
                          new, count=1)
        if (n1, n2) != (1, 1):
            logger.warning("exl3xpu: GDN fold patch not applied (MambaPool layout changed: %d %d)", n1, n2)
            return
        ns: dict = {}
        exec(compile(new, "<exl3xpu:MambaPool.__init__>", "exec"), m.__dict__, ns)
        ns["__init__"]._exl3 = True
        cls.__init__ = ns["__init__"]
        logger.info("exl3xpu: GDN ReplaySSM fold-every-commit enabled (no per-draft state snapshots)")
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: GDN fold patch failed (%s)", e)


def _patch_nvtier_step() -> None:
    """N104: run the NVMe tier's host step (fill logged misses, enforce the RAM budget) after every model step."""
    from sglang.srt.model_executor import model_runner as mr
    orig = mr.ModelRunner.forward

    def forward(self, *a, **k):
        out = orig(self, *a, **k)
        if _NVTIER is not None and not torch.xpu.is_current_stream_capturing():
            _NVTIER.step()
            every = int(os.environ.get("EXL3_NVTIER_LOG_EVERY", "200"))
            _NVTIER.log_stats(logger, every)
            if _EXACT is not None and _NVTIER.stats["steps"] % every == 0:
                _EXACT.log_stats(logger, every)
        return out
    mr.ModelRunner.forward = forward
    logger.info("exl3xpu: NVMe tier step hook installed on ModelRunner.forward")


def _patch_exact_gates() -> None:
    """EXL3_EXACT_MISS + EXL3_EXACT_PREDICT: register the target model's router gates for next-layer prediction right
    after load_model (before graph capture). The draft (MTP) runner is skipped."""
    from sglang.srt.model_executor import model_runner as mr
    orig = mr.ModelRunner.load_model

    def load_model(self, *a, **k):
        r = orig(self, *a, **k)
        if _EXACT is not None and _EXACT.predict and not getattr(self, "is_draft_worker", False):
            if _EXACT.register_gates(self.model) == 0:
                logger.warning("exl3xpu: exact prediction on but no router gates found: prediction stays off")
        return r

    mr.ModelRunner.load_model = load_model


def _patch_xpu_graph_warm_replay() -> None:
    """Replay every captured XPU graph once right after capture (EXL3_SGL_GRAPH_WARM_REPLAY=1, default). A SYCL graph
    allocates device memory lazily on its first replay; at serve time that first replay comes when concurrency first
    reaches a new batch size, after the torch cache has taken the headroom -> UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY in
    graph replay. Replaying at startup makes the cost deterministic and visible in the post-capture free memory.
    Capture already runs the forward twice eagerly with the same dummy inputs, so the extra replay is equivalent."""
    if os.environ.get("EXL3_SGL_GRAPH_WARM_REPLAY", "1") != "1":
        return
    try:
        from sglang.srt.hardware_backend.xpu.graph_runner import xpu_full_graph_backend as gb
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: graph warm replay not installed (%s)", e)
        return
    cls = gb.FullXPUGraphBackend
    orig = cls.capture_one
    if getattr(orig, "_exl3", False):
        return

    def capture_one(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        r = orig(self, shape_key, forward_fn, capture_inputs, post_warmup_hook)
        g = self._graphs.get(shape_key)
        if g is not None:
            self._device_module.synchronize()
            g.replay()
            self._device_module.synchronize()
        return r

    capture_one._exl3 = True
    cls.capture_one = capture_one


def install_mem_probe(model: torch.nn.Module, calls: int = 3, thresh_mb: int = 64) -> None:
    """Debug (EXL3_MEM_PROBE=1): log modules whose forward grows device memory held outside the torch allocator."""
    g = 2 ** 20
    state = {"n": 0}

    def nontorch():
        torch.xpu.synchronize()
        f, t = torch.xpu.mem_get_info()
        return (t - f - torch.xpu.memory_reserved()) / g

    def pre(mod, args):
        if state["n"] < calls * 1000:
            mod._exl3_nt = nontorch()

    def post(mod, args, out):
        if state["n"] < calls * 1000 and hasattr(mod, "_exl3_nt"):
            d = nontorch() - mod._exl3_nt
            state["n"] += 1
            if d > thresh_mb:
                logger.info("exl3xpu mem probe: %s (%s) +%.0f MiB outside torch", mod._exl3_name,
                            type(mod).__name__, d)

    for name, mod in model.named_modules():
        mod._exl3_name = name
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
    logger.info("exl3xpu mem probe installed on %d modules", sum(1 for _ in model.named_modules()))


def _patch_mem_probe() -> None:
    if os.environ.get("EXL3_MEM_PROBE", "0") != "1":
        return
    from sglang.srt.model_executor import model_runner as mr
    orig = mr.ModelRunner.load_model

    def load_model(self, *a, **k):
        r = orig(self, *a, **k)
        try:
            install_mem_probe(self.model)
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: mem probe failed (%s)", e)
        return r

    mr.ModelRunner.load_model = load_model


def _patch_step_timing() -> None:
    """Debug (EXL3_STEP_TIMING=N): synchronized wall time of every XPU graph replay (by runner class and batch shape)
    and of each speculative decode step (EAGLEWorkerV2.forward_batch_generation, decode batches only); logs means
    every N steps. Adds syncs, so absolute step time rises slightly; the split is what matters."""
    n_every = int(os.environ.get("EXL3_STEP_TIMING", "0"))
    if not n_every:
        return
    import collections
    import time as _t
    from sglang.srt.hardware_backend.xpu.graph_runner import xpu_full_graph_backend as gb
    from sglang.srt.speculative import eagle_worker_v2 as ew
    acc = collections.defaultdict(lambda: [0, 0.0])
    step = {"n": 0}
    ctx = {"runner": ""}
    orig_replay = gb.FullXPUGraphBackend.replay

    def replay(self, shape_key, static_forward_batch, **kw):
        torch.xpu.synchronize(); t = _t.perf_counter()
        r = orig_replay(self, shape_key, static_forward_batch, **kw)
        torch.xpu.synchronize()
        k = f"graph:{getattr(self, '_exl3_owner', '?')}:{shape_key}"
        acc[k][0] += 1; acc[k][1] += _t.perf_counter() - t
        return r

    orig_init = gb.FullXPUGraphBackend.__init__

    def init(self, runner, *a, **k):
        orig_init(self, runner, *a, **k)
        self._exl3_owner = type(runner).__name__

    orig_fbg = ew.EAGLEWorkerV2.forward_batch_generation

    def forward_batch_generation(self, batch, *a, **k):
        is_dec = hasattr(batch, "forward_mode") and not batch.forward_mode.is_extend()
        torch.xpu.synchronize(); t = _t.perf_counter()
        r = orig_fbg(self, batch, *a, **k)
        torch.xpu.synchronize()
        if is_dec:
            bs = getattr(batch, "batch_size", None)
            bs = bs() if callable(bs) else bs
            key = f"step:bs{bs}"
            acc[key][0] += 1; acc[key][1] += _t.perf_counter() - t
            step["n"] += 1
            if step["n"] % n_every == 0:
                logger.info("exl3xpu step timing: " + "; ".join(
                    f"{k} n={c} {1000 * s / c:.2f} ms" for k, (c, s) in sorted(acc.items())))
                acc.clear()
        return r

    gb.FullXPUGraphBackend.replay = replay
    gb.FullXPUGraphBackend.__init__ = init
    ew.EAGLEWorkerV2.forward_batch_generation = forward_batch_generation


def _patch_xpu_mamba_extra_buffer() -> None:
    """Prefix caching for GDN hybrids on XPU (EXL3_SGL_XPU_EXTRA_BUFFER=1, default). SGLang 0.5.20 refuses the mamba
    radix `extra_buffer` strategy on XPU outright (`supports_mamba_cache_extra_buffer`: `if is_xpu: return False`),
    and the alternative `no_buffer` needs page size 1, which the intel_xpu attention backend forbids (64/128) -> no
    prefix cache at all. extra_buffer only needs the FLA/Triton GDN kernels, which run on XPU; lift the platform gate
    (same arch/backend rule as CUDA)."""
    if os.environ.get("EXL3_SGL_XPU_EXTRA_BUFFER", "1") != "1":
        return
    try:
        from sglang.srt.arg_groups import overrides as ov
        from sglang.srt.arg_groups import mamba_hook as mh
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: extra_buffer gate patch not installed (%s)", e)
        return
    archs = ov._MAMBA_EXTRA_BUFFER_ARCHS

    def supports_mamba_cache_extra_buffer(view, model_arch):
        if model_arch in archs:
            return view.linear_attn_backend == "triton"
        return False

    ov.supports_mamba_cache_extra_buffer = supports_mamba_cache_extra_buffer
    mh.supports_mamba_cache_extra_buffer = supports_mamba_cache_extra_buffer
    logger.info("exl3xpu: mamba radix extra_buffer allowed on XPU (prefix caching for GDN hybrids)")


def _allow_xpu_in(module, names: list[str]) -> list[str]:
    """Re-define `module.<name>` from its source with every `<t>.is_cuda` test accepting XPU tensors as well
    (SGLang guards some device-agnostic Triton launchers with CUDA-only checks)."""
    import inspect
    import textwrap
    done = []
    for name in names:
        fn = getattr(module, name, None)
        if fn is None or getattr(fn, "_exl3_xpu_ok", False):
            continue
        src = textwrap.dedent(inspect.getsource(fn))
        new = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)\.is_cuda\b", r'(\1.device.type in ("cuda", "xpu"))', src)
        if new == src:
            continue
        ns: dict = {}
        exec(compile(new, f"<exl3xpu:{module.__name__}.{name}>", "exec"), module.__dict__, ns)
        ns[name]._exl3_xpu_ok = True
        setattr(module, name, ns[name])
        done.append(name)
    return done


def _patch_xpu_mamba_scatter() -> None:
    """MTP verify on hybrid GDN models commits the accepted step's recurrent/conv state with Triton scatter kernels
    whose Python launchers refuse non-CUDA tensors. EXL3_SGL_SCATTER_FIX=0 disables."""
    if os.environ.get("EXL3_SGL_SCATTER_FIX", "1") != "1":
        return
    try:
        if not torch.xpu.is_available():
            return
        from sglang.kernels.ops.mamba import mamba_state_scatter_triton as m
        done = _allow_xpu_in(m, ["fused_mamba_state_scatter_with_mask", "fused_conv_window_scatter_with_mask"])
        logger.info("exl3xpu: XPU allowed in mamba state scatter launchers %s", done)
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: mamba scatter XPU fix not installed (%s)", e)


_FP8_EXTEND = [
    "k_descale, v_descale = None, None",
    'if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256 and layer.k_scale is not None:  # exl3xpu',
    "    _ds = (forward_batch.batch_size, layer.tp_k_head_num)",
    "    k_descale = layer.k_scale.expand(_ds)",
    "    v_descale = layer.v_scale.expand(_ds)",
    "    if __EXL3_FP8_Q__:",
    "        q = q.to(self.kv_cache_dtype)",
    "        q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None",
    "        k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None",
    'if self.kv_cache_dtype_str != "auto" and k_descale is None:',
    '    raise RuntimeError(f"exl3xpu fp8 extend: layer {layer.layer_id} {type(layer).__name__} qm={type(layer.quant_method).__name__} k_scale={layer.k_scale!r} head_dim={layer.head_dim}")',
]


def _patch_xpu_fp8_extend() -> None:
    """intel_xpu attention: fp8 KV descale is wired for decode only; forward_extend (prefill chunks and the
    speculative target-verify) passes k_descale=None to a kernel that requires it with an fp8 cache. Wire it the
    same way as forward_decode. EXL3_SGL_FP8_Q=1 also casts q to fp8 in extend (as decode does)."""
    if os.environ.get("EXL3_SGL_FP8_EXTEND", "1") != "1":
        return
    try:
        import inspect
        import textwrap
        from sglang.srt.layers.attention import xpu_backend as xb
        cls = xb.XPUAttentionBackend
        fn = cls.forward_extend
        if getattr(fn, "_exl3_fp8", False):
            return
        src = textwrap.dedent(inspect.getsource(fn))
        anchor = "k_descale, v_descale = None, None"
        if anchor not in src or "super()" in src:
            logger.warning("exl3xpu: fp8 extend patch not applied (source layout changed)")
            return
        line = next(l for l in src.splitlines() if l.strip() == anchor)
        ind = line[: len(line) - len(line.lstrip())]
        block = "\n".join(ind + l for l in _FP8_EXTEND).replace(
            "__EXL3_FP8_Q__", str(os.environ.get("EXL3_SGL_FP8_Q", "0") == "1"))
        new = src.replace(line, block, 1)
        ns: dict = {}
        exec(compile(new, "<exl3xpu:xpu_backend.forward_extend>", "exec"), xb.__dict__, ns)
        ns["forward_extend"]._exl3_fp8 = True
        cls.forward_extend = ns["forward_extend"]
        # forward_decode casts q to fp8 before the kernel, which (sgl_kernel xpu 0.2.0) only accepts fp16/bf16 q
        # ("mha_fwd only supports Half and BFloat16"); it takes a bf16 q with an fp8 cache + descale.
        if os.environ.get("EXL3_SGL_FP8_Q", "0") != "1":
            dsrc = textwrap.dedent(inspect.getsource(cls.forward_decode))
            if "super()" not in dsrc:
                dnew = dsrc
                for pat in ("q = q.to(self.kv_cache_dtype)",
                            "q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None",
                            "k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None"):
                    dnew = dnew.replace(pat, "pass  # exl3xpu: keep q in the model dtype")
                if dnew != dsrc:
                    ns = {}
                    exec(compile(dnew, "<exl3xpu:xpu_backend.forward_decode>", "exec"), xb.__dict__, ns)
                    cls.forward_decode = ns["forward_decode"]
        logger.info("exl3xpu: intel_xpu forward_extend passes fp8 KV descale (q cast: %s)",
                    os.environ.get("EXL3_SGL_FP8_Q", "0") == "1")
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: fp8 extend patch failed (%s)", e)


def _cuda_compat_shim() -> None:
    """SGLang's qwen4_exp / hyperconnection / QSA code calls a few torch.cuda.* helpers directly (device placement,
    side streams, empty_cache). On an XPU-only torch they raise; map them onto torch.xpu (only when CUDA is absent)."""
    if torch.cuda.is_available() or not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return
    c = torch.cuda
    if getattr(c, "_exl3_xpu_shim", False):
        return
    c.current_device = lambda: torch.device("xpu", torch.xpu.current_device())
    c.current_stream = lambda device=None: torch.xpu.current_stream()
    c.stream = torch.xpu.stream
    c.Stream = torch.xpu.Stream
    c.Event = torch.xpu.Event
    c.empty_cache = torch.xpu.empty_cache
    c.synchronize = lambda device=None: torch.xpu.synchronize()
    if hasattr(torch.xpu, "is_current_stream_capturing"):
        c.is_current_stream_capturing = torch.xpu.is_current_stream_capturing
    else:
        c.is_current_stream_capturing = lambda: False
    c._exl3_xpu_shim = True
    # MultiPlatformOp subclasses that only implement forward_cuda (QSA indexer) dispatch XPU to forward_native, which
    # raises: use forward_cuda instead -- those bodies guard their CUDA-only kernels with tensor.is_cuda checks.
    try:
        from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
        native = MultiPlatformOp.forward_native

        def forward_xpu(self, *a, **k):
            if type(self).forward_native is native:
                return self.forward_cuda(*a, **k)
            return self.forward_native(*a, **k)
        MultiPlatformOp.forward_xpu = forward_xpu
    except ImportError:  # pragma: no cover
        pass
    logger.info("exl3xpu: torch.cuda.{current_device,current_stream,stream,Stream,Event,empty_cache,synchronize,"
                "is_current_stream_capturing} mapped to torch.xpu (XPU-only torch)")


def activate() -> None:
    """sglang.srt.plugins entry point."""
    global _done
    if _done:
        return
    _done = True
    try:
        from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: SGLang not importable (%s)", e)
        return
    Cfg, _, _ = classes()
    QUANTIZATION_METHODS["exl3"] = Cfg
    try:
        from sglang.srt.arg_groups.choices import QUANTIZATION_CHOICES, add_quantization_method_choices
        if "exl3" not in QUANTIZATION_CHOICES:
            add_quantization_method_choices(["exl3"])
    except Exception as e:  # pragma: no cover
        logger.warning("exl3xpu: could not add 'exl3' to --quantization choices (%s)", e)
    if os.environ.get("EXL3_CUDA_SHIM", "1") == "1":
        _cuda_compat_shim()
        if os.environ.get("EXL3_HC_XPU", "0") == "1":
            try:
                import sys as _sys
                _sys.path.insert(0, os.environ.get("EXL3_HC_DIR", "/hc"))
                import patch_hc
                patch_hc.install(import_now=True)
            except Exception as e:  # n107-hc
                logger.warning("exl3xpu: n107-hc not installed (%s)", e)

    if os.environ.get("EXL3_QSA_XPU", "1") == "1" and hasattr(torch, "xpu") and torch.xpu.is_available() \
            and not torch.cuda.is_available():
        try:
            from . import qsa_xpu
            qsa_xpu.install()
            import sys as _sys
            print("EXL3 qsa_xpu installed", file=_sys.stderr, flush=True)
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: QSA XPU shim not installed (%s)", e)
    if os.environ.get("EXL3_MODTIME", "0") == "1":
        try:
            from . import modtime
            modtime.install()
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: modtime not installed (%s)", e)
    if os.environ.get("EXL3_NGRAM_HOST", "1") == "1":
        # Qwen3.8-Flash-Next EXL3 n-gram table: USM host memory + zero-copy gather/decode (ngram_host.py)
        try:
            from . import ngram_host
            ngram_host.install()
        except ImportError as e:  # pragma: no cover - SGLang without qwen4_exp
            logger.info("exl3xpu: n-gram host table shim not installed (%s)", e)
    try:
        # image input: EXL3 ViT on the XPU kernels, q/k/v routing, PLE placeholder remap, XPU vision attention
        from . import vision_xpu
        vision_xpu.install()
    except ImportError as e:  # pragma: no cover - SGLang without qwen4_exp / qwen3_vl
        logger.info("exl3xpu: vision shim not installed (%s)", e)
    _patch_mtp()
    _patch_qwen4_mtp()
    _patch_xpu_spec_sampling()
    _patch_xpu_gdn_verify()
    _patch_xpu_mamba_scatter()
    _patch_xpu_fp8_extend()
    _patch_xpu_replayssm()
    _patch_xpu_attn_routes()
    _patch_gdn_replayssm_fold()
    _patch_xpu_graph_warm_replay()
    _patch_mem_probe()
    _patch_xpu_mamba_extra_buffer()
    _patch_step_timing()
    if _NVTIER_ON:
        _patch_nvtier_step()
        if os.environ.get("EXL3_EXACT_MISS", "0") == "1":
            _patch_exact_gates()
    frac = os.environ.get("EXL3_TORCH_MEM_FRACTION")
    if frac:
        # cap the torch caching allocator so the Level Zero driver keeps room for its own allocations (kernel scratch
        # for spilling kernels on first launch, command lists): SGLang's static fraction does not bound the cache
        try:
            torch.xpu.set_per_process_memory_fraction(float(frac))
            logger.info("exl3xpu: torch XPU allocator capped at %.3f of the card", float(frac))
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: could not cap the torch allocator (%s)", e)
    logger.info("exl3xpu: registered SGLang quantization method 'exl3' (XPU)")
