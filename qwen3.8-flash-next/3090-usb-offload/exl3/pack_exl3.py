"""M4 step 1: the Strata EXL3 expert pack (experts.bin), cut from the converter's out/l<L>/e<E>.npz.

    python pack_exl3.py write L [L ...]     # copy converted layers into pack/experts.bin (sparse until all are in)
    python pack_exl3.py verify L [N]        # bytes == npz for all 512 experts; N sampled experts through the NEON tier
    python pack_exl3.py status
    python pack_exl3.py prune L [L ...]     # delete out/l<L>/e*.npz once the layer is packed, verified ok and re-hashed
    python pack_exl3.py sync                # write + verify + prune every converted, not yet packed layer; then marker
    python pack_exl3.py marker              # write pack/exl3_experts.txt (Strata's completion marker), all layers only

Disk: out/ (0.91 GB/layer) + experts.bin (0.95 GB/layer) do not both fit for 48 layers on the mini, so a layer's npz are
deleted once its pack section is proven equal to them (verify) and still hashes to the manifest (prune re-hashes); the
converter's check/quant jsons stay in out/l<L>/ so its resume logic still skips the layer.

Format "exl3-k3-mul1-v1" (Strata ExpertLayout terms: one fixed blob size for every layer, layer l at
l * 512 * BLOB, expert e at layer_offset + e * BLOB). Expert ids are the ORIGINAL 0..511 (full model, no pruning).
Blob = verbatim converter arrays, every field 256-byte aligned so a VRAM slot base + fixed offsets gives the
exl3_moe_coop pointer tables (g/u/d_trellis, _suh, _svh) and the NEON tier reads the mmap directly:

    off       bytes   field          dtype  shape
    0         614400  gate_trellis   u16    [H/16, I/16, 48]   (K3 MUL1, H 2560, I 640)
    614400    614400  up_trellis     u16    [H/16, I/16, 48]
    1228800   614400  down_trellis   u16    [I/16, H/16, 48]
    1843200   5120    gate_suh       f16    [H]
    1848320   1280    gate_svh       f16    [I]
    1849600   5120    up_suh         f16    [H]
    1854720   1280    up_svh         f16    [I]
    1856000   1280    down_suh       f16    [I]
    1857280   5120    down_svh       f16    [H]
    BLOB = 1862400 (= 7275 x 256); layer = 953,548,800 B; 48 layers = 45.77 GB.

W_orig = had(suh) . W_inner . had(svh) with g_scale folded into suh/svh by the encoder (exl3_ref.weight_orig_f64).
pack/exl3_pack.json records the layout and, per layer, the sha256 of its section and the check json it was cut after.
pack/exl3_experts.txt is what Strata's expert_layout_load reads (kExl3LayoutVersion 1): written last, via tmp + rename,
once all 48 layers are packed; until then the engine refuses the directory as a pack still being written.
"""
import ctypes, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, PACK = os.path.join(HERE, "out"), os.path.join(HERE, "pack")
BIN, MANIFEST = os.path.join(PACK, "experts.bin"), os.path.join(PACK, "exl3_pack.json")
N_LAYERS, N_EXPERT, H, I, K, CB = 48, 512, 2560, 640, 3, 2
TR = (H // 16) * (I // 16) * 16 * K * 2                                # 614400
FIELDS = [("gate_trellis", np.uint16, (H // 16, I // 16, 16 * K)), ("up_trellis", np.uint16, (H // 16, I // 16, 16 * K)),
          ("down_trellis", np.uint16, (I // 16, H // 16, 16 * K)), ("gate_suh", np.float16, (H,)),
          ("gate_svh", np.float16, (I,)), ("up_suh", np.float16, (H,)), ("up_svh", np.float16, (I,)),
          ("down_suh", np.float16, (I,)), ("down_svh", np.float16, (H,))]
OFF, o = {}, 0
for name, dt, shp in FIELDS:
    assert o % 256 == 0, name
    OFF[name] = o
    o += int(np.prod(shp)) * np.dtype(dt).itemsize
BLOB = o
assert BLOB == 1862400 and BLOB % 256 == 0 and OFF["down_svh"] == 1857280
LAYER = N_EXPERT * BLOB
FORMAT = "exl3-k3-mul1-v1"


def layer_done(l):
    return next((os.path.join(OUT, f"l{l}", f"check_{s}.json") for s in ("sample32", f"0_{N_EXPERT}")
                 if os.path.exists(os.path.join(OUT, f"l{l}", f"check_{s}.json"))), None)


def manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    return dict(format=FORMAT, n_layers=N_LAYERS, n_expert=N_EXPERT, n_embd=H, n_ff=I, K=K, cb=CB, blob=BLOB,
                layer_bytes=LAYER, total=N_LAYERS * LAYER, fields={n: dict(off=OFF[n], dtype=np.dtype(d).name,
                shape=list(s)) for n, d, s in FIELDS}, layers={})


def blob_bytes(z):
    parts = []
    for name, dt, shp in FIELDS:
        a = z[name]
        assert a.dtype == dt and a.shape == shp, (name, a.dtype, a.shape)
        parts.append(np.ascontiguousarray(a).tobytes())
    b = b"".join(parts)
    assert len(b) == BLOB
    return b


def cmd_write(layers):
    os.makedirs(PACK, exist_ok=True)
    m = manifest()
    fd = os.open(BIN, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size < N_LAYERS * LAYER:
        os.ftruncate(fd, N_LAYERS * LAYER)                               # sparse on APFS until every layer is in
    try:
        for l in layers:
            chk = layer_done(l)
            if chk is None:
                print(f"layer {l}: not converted (no check json), skipped", flush=True); continue
            if not json.load(open(chk))["ok"]:
                raise SystemExit(f"layer {l}: check not ok ({chk})")
            t0, h = time.perf_counter(), hashlib.sha256()
            buf = bytearray()
            for e in range(N_EXPERT):
                with np.load(os.path.join(OUT, f"l{l}", f"e{e}.npz")) as z:
                    buf += blob_bytes(z)
            assert len(buf) == LAYER
            h.update(buf)
            os.pwrite(fd, bytes(buf), l * LAYER)
            os.fsync(fd)
            m["layers"][str(l)] = dict(sha256=h.hexdigest(), check=os.path.relpath(chk, HERE),
                                       written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            json.dump(m, open(MANIFEST + ".tmp", "w"), indent=1); os.replace(MANIFEST + ".tmp", MANIFEST)
            print(f"layer {l}: {LAYER} B at {l * LAYER}, sha256 {h.hexdigest()[:16]}, {time.perf_counter() - t0:.1f} s",
                  flush=True)
    finally:
        os.close(fd)


# ---- verify: the bytes, then the CPU tier reading the blob in place
lib = None


def load_lib():
    global lib
    lib = ctypes.CDLL(os.path.join(HERE, "cpu", "libexl3cpu.dylib"))
    P, Ii, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    lib.exl3_cpu_decode.argtypes = [P, L, Ii, Ii, P]
    lib.exl3_cpu_had_in.argtypes = [P, P, Ii, Ii, P]
    lib.exl3_cpu_gemv.argtypes = [P, P, P, Ii, Ii, Ii, Ii, Ii, P, P, P, Ii]


def addr(a): return ctypes.c_void_p(a.ctypes.data)


class BlobExpert:
    """One expert straight from the experts.bin mmap: every pointer is base + OFF[field] (no copies)."""
    def __init__(self, mm, l, e):
        base = l * LAYER + e * BLOB
        self.f = {n: np.ndarray(shp, dt, mm, base + OFF[n]) for n, dt, shp in FIELDS}
        self.scr = {}

    def proj(self, q, A, threads=1):
        k, n = (H, I) if q != "down" else (I, H)
        m = A.shape[0]
        A_had = np.empty_like(A)
        lib.exl3_cpu_had_in(addr(A), addr(self.f[f"{q}_suh"]), m, k, addr(A_had))
        C, y = np.empty((m, n), np.float32), np.empty((m, n), np.float32)
        xq = np.empty(m * k * 2 + 64, np.uint16)
        xq = xq[(-xq.ctypes.data // 2) % 32:][:m * k * 2]
        assert lib.exl3_cpu_gemv(addr(A_had), addr(self.f[f"{q}_trellis"]), addr(self.f[f"{q}_svh"]), m, k, n, K, CB,
                                 addr(C), addr(y), addr(xq), threads) == 0
        return C

    def forward(self, x, threads=1):
        g, u = self.proj("gate", x, threads), self.proj("up", x, threads)
        z = (g / (1 + np.exp(-g)) * u).astype(np.float16)
        return self.proj("down", z, threads)


def cmd_verify(l, n_sample):
    import exl3_ref as ref
    load_lib()
    m = manifest()
    rec = dict(layer=l, format=FORMAT, blob=BLOB, ok=True)
    mm = np.memmap(BIN, np.uint8, "r")
    assert mm.size == N_LAYERS * LAYER
    # 1) every byte of the layer == its npz fields (hash only once the npz are pruned), section hash == manifest
    t0, h, bad = time.perf_counter(), hashlib.sha256(), []
    pruned = "npz_pruned" in m["layers"].get(str(l), {})
    for e in range(N_EXPERT):
        b = mm[l * LAYER + e * BLOB:l * LAYER + (e + 1) * BLOB]
        h.update(b)
        if pruned:
            continue
        with np.load(os.path.join(OUT, f"l{l}", f"e{e}.npz")) as z:
            for name, dt, shp in FIELDS:
                if not np.array_equal(np.frombuffer(b, dt, int(np.prod(shp)), OFF[name]).reshape(shp), z[name]):
                    bad.append([e, name])
    rec["bytes"] = dict(experts=N_EXPERT, npz_compared=not pruned, field_mismatches=len(bad), first=bad[:5],
                        sha256_matches_manifest=h.hexdigest() == m["layers"].get(str(l), {}).get("sha256"),
                        s=round(time.perf_counter() - t0, 1))
    rec["ok"] &= not bad and rec["bytes"]["sha256_matches_manifest"]
    # 2) sampled experts through the NEON tier, pointers into the mmap
    rng = np.random.default_rng([20261011, l])
    sample = sorted({0, N_EXPERT - 1} | set(rng.choice(N_EXPERT, max(n_sample - 2, 0), replace=False).tolist()))
    rows = []
    for e in sample:
        be = BlobExpert(mm, l, e)
        r, Wi = dict(expert=e), {}
        for q in ("gate", "up", "down"):                  # NEON decode of the blob trellis == numpy reconstruct (bit-exact)
            tr = be.f[f"{q}_trellis"]
            tk, tn, _ = tr.shape
            dec = np.empty((tk * tn, 256), np.uint16)
            assert lib.exl3_cpu_decode(addr(tr), tk * tn, K, CB, addr(dec)) == 0
            t = np.empty_like(dec)
            t[:, ref.tensor_core_perm()] = dec
            Wn = t.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)
            Wi[q] = ref.reconstruct(tr, K, CB)
            r[f"{q}_decode_mismatch"] = int(np.count_nonzero(Wn != Wi[q].view(np.uint16)))
        W = {q: ref.weight_orig_f64(Wi[q], be.f[f"{q}_suh"], be.f[f"{q}_svh"]) for q in ("gate", "up", "down")}
        for mrows in (1, 4):
            x = rng.standard_normal((mrows, H)).astype(np.float16)
            yh = be.forward(x, threads=1).astype(np.float64)
            xd = x.astype(np.float64)
            g, u = xd @ W["gate"], xd @ W["up"]
            y = (g / (1 + np.exp(-g)) * u) @ W["down"]
            rr = float(np.sqrt(np.square(yh - y).mean() / np.square(y).mean()))
            r[f"fwd_m{mrows}_rel_rms"] = rr
            r["ok"] = r.get("ok", True) and bool(np.isfinite(yh).all()) and rr <= 5e-3
        r["ok"] &= all(r[f"{q}_decode_mismatch"] == 0 for q in ("gate", "up", "down"))
        rows.append(r)
        rec["ok"] &= r["ok"]
        print(r, flush=True)
    rec["sample"] = rows
    # 3) timing: one expert forward at m1 from the (page-cache warm) mmap, 1 thread, = one P-core's share
    be = BlobExpert(mm, l, sample[1])
    x = rng.standard_normal((1, H)).astype(np.float16)
    be.forward(x); ts = []
    for _ in range(20):
        t = time.perf_counter(); be.forward(x); ts.append(time.perf_counter() - t)
    rec["fwd_m1_us"] = dict(min=round(min(ts) * 1e6, 1), median=round(float(np.median(ts)) * 1e6, 1))
    # after prune the npz-compared receipt (prune's precondition) is kept; hash-only reruns go to verify_l<L>_hash.json
    json.dump(rec, open(os.path.join(PACK, f"verify_l{l}{'_hash' if pruned else ''}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "sample"}), flush=True)
    return rec["ok"]


def cmd_status():
    m = manifest()
    done = sorted(int(k) for k in m["layers"])
    conv = [l for l in range(N_LAYERS) if layer_done(l)]
    print(json.dumps(dict(format=m["format"], packed=done, converted=conv, unpacked=[l for l in conv if l not in done],
                          disk_used_gb=round(os.stat(BIN).st_blocks * 512 / 1e9, 2) if os.path.exists(BIN) else 0)))


def cmd_prune(layers):
    m = manifest()
    mm = np.memmap(BIN, np.uint8, "r")
    for l in layers:
        ent, vf = m["layers"].get(str(l)), os.path.join(PACK, f"verify_l{l}.json")
        if ent is None or "npz_pruned" in ent:
            print(f"layer {l}: {'not packed' if ent is None else 'already pruned'}, skipped", flush=True); continue
        v = json.load(open(vf)) if os.path.exists(vf) else {}
        if not (v.get("ok") and v.get("bytes", {}).get("npz_compared") is not False
                and v.get("bytes", {}).get("field_mismatches") == 0):
            raise SystemExit(f"layer {l}: no passing npz-compared verify ({vf}), refusing to prune")
        h = hashlib.sha256(mm[l * LAYER:(l + 1) * LAYER]).hexdigest()
        if h != ent["sha256"]:
            raise SystemExit(f"layer {l}: pack section sha256 {h[:16]} != manifest {ent['sha256'][:16]}, refusing")
        d = os.path.join(OUT, f"l{l}")
        npz = [f for f in os.listdir(d) if f.startswith("e") and f.endswith(".npz")]
        if len(npz) != N_EXPERT:
            raise SystemExit(f"layer {l}: {len(npz)} npz in {d}, expected {N_EXPERT}")
        freed = sum(os.stat(os.path.join(d, f)).st_size for f in npz)
        for f in npz:
            os.remove(os.path.join(d, f))
        ent["npz_pruned"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json.dump(m, open(MANIFEST + ".tmp", "w"), indent=1); os.replace(MANIFEST + ".tmp", MANIFEST)
        print(f"layer {l}: sha256 ok, removed {len(npz)} npz ({freed / 1e9:.2f} GB)", flush=True)


def cmd_sync():
    """Every converted (check json) but unpacked layer: write, verify 8 sampled experts, prune its npz. Card not used."""
    m = manifest()
    todo = [l for l in range(N_LAYERS) if layer_done(l) and str(l) not in m["layers"]]
    for l in todo:
        cmd_write([l])
        if not cmd_verify(l, 8):
            raise SystemExit(f"layer {l}: verify failed, npz kept")
        cmd_prune([l])
    cmd_status()
    if len(manifest()["layers"]) == N_LAYERS:
        cmd_marker()


MARKER = os.path.join(PACK, "exl3_experts.txt")
COOP_ORDER = ["gate_trellis", "gate_suh", "gate_svh", "up_trellis", "up_suh", "up_svh", "down_trellis", "down_suh",
              "down_svh"]                                               # Strata Exl3Field / MoeCoopParams order


def marker_text(m):
    """exl3_experts.txt v1 for the layers in manifest m (Strata refuses it unless all n_layers are listed)."""
    s = [f"# strata exl3 experts v1: {m['format']} (n_expert {m['n_expert']}, n_embd {m['n_embd']}, n_ff {m['n_ff']}, "
         f"K {m['K']}, cb {m['cb']}, blob {m['blob']})",
         "# field <name> <offset in blob> <bytes>"]
    for n in COOP_ORDER:
        f = m["fields"][n]
        s.append(f"field {n} {f['off']} {int(np.prod(f['shape'])) * np.dtype(f['dtype']).itemsize}")
    s.append("# <layer> <offset in experts.bin> <blob> <sha256 of the layer section>")
    for l in sorted(int(k) for k in m["layers"]):
        s.append(f"{l} {l * m['layer_bytes']} {m['blob']} {m['layers'][str(l)]['sha256']}")
    return "\n".join(s) + "\n"


def cmd_marker():
    m = manifest()
    missing = [l for l in range(N_LAYERS) if str(l) not in m["layers"]]
    if missing:
        raise SystemExit(f"{len(missing)} of {N_LAYERS} layers not packed yet ({missing[:8]}...), no marker")
    if os.stat(BIN).st_size != m["total"]:
        raise SystemExit(f"experts.bin is {os.stat(BIN).st_size} B, manifest total {m['total']}")
    open(MARKER + ".tmp", "w").write(marker_text(m))
    os.replace(MARKER + ".tmp", MARKER)
    print(f"marker {MARKER}: {N_LAYERS} layers, {m['total']} B", flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    if cmd == "write": cmd_write(args)
    elif cmd == "verify": sys.exit(0 if cmd_verify(args[0], args[1] if len(args) > 1 else 8) else 1)
    elif cmd == "sync": cmd_sync()
    elif cmd == "status": cmd_status()
    elif cmd == "prune": cmd_prune(args)
    elif cmd == "marker": cmd_marker()
    else: raise SystemExit(__doc__)
