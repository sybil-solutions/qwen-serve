"""pack_td.py - the Strata EXL3 expert pack cut from turboderp's published EXL3 quant instead of our own converter.

Source: https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3  branch 3.05bpw_h5_ng5  commit 69e33439ae950f17bcbe95c98f117d80f759ab6d
(exllamav3 1.4.4: K=3 experts, codebook mul1, head 5 bit, vision 5 bit, MTP 3 bit). Every layer has all 512 experts as
model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.{trellis I16, suh F16, svh F16, mul1 I32 scalar}; one expert's 12
tensors are ONE contiguous 1,862,412 B range of a shard (4 B mul1 x3 more than our 1,862,400 B blob) and a layer's 512 experts are
one contiguous ~909 MiB range, so a layer is range-fetched with a few HTTP Range requests (no shard is stored).

    python pack_td.py index                 # headers of the 7 shards -> ~/exl3/td/experts_index.json (done once)
    python pack_td.py check L E [E ...]     # NO writes: fetch the experts, decode, compare with the blob now in experts.bin
    python pack_td.py sync [L ...]          # fetch + write layers (default all not yet from turboderp), verify, manifest
    python pack_td.py verify [L ...]        # re-fetch 6 experts per layer (default all layers), byte-compare; section sha256 vs manifest
    python pack_td.py status

Blob layout is pack_exl3.FIELDS (gate/up/down trellis u16, suh/svh f16, 256-byte aligned, BLOB 1,862,400). Fields are checked on the
way in: dtype/shape per tensor, mul1 value (must be one value for all 73,728), mcg absent, finite suh/svh, trellis not constant.
"""
import hashlib, json, os, struct, sys, threading, time, urllib.request, queue
import numpy as np
import pack_exl3 as px

REV = "69e33439ae950f17bcbe95c98f117d80f759ab6d"
BASE = f"https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/{REV}/"
TD = os.path.join(px.HERE, "td")
IDX = os.path.join(TD, "experts_index.json")
VIA = f"turboderp/Qwen3.8-Flash-Next-exl3@{REV[:8]} 3.05bpw_h5_ng5"
GROUP = 32                                                  # experts per HTTP Range request (~60 MB)
SPAN = px.BLOB + 12


def http_range(shard, a, b, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(BASE + shard, headers={"Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = r.read()
            if len(d) != b - a:
                raise IOError(f"short read {len(d)} != {b - a}")
            return d
        except Exception as e:
            if t == tries - 1:
                raise
            print(f"  range {shard} {a}-{b} failed ({type(e).__name__}: {e}); retry {t + 1}", flush=True)
            time.sleep(3 * (t + 1))


def cmd_index():
    import subprocess
    subprocess.check_call([sys.executable, os.path.join(px.HERE, "td_hdr.py")])


def load_index():
    return json.load(open(IDX))


def expert_blob(rec, raw, base):
    """rec: index entry {proj_field: [shard, a, b, dtype, shape]}; raw: bytes of the fetched range starting at file offset `base`.
    Returns (blob bytes, mul1 values)."""
    parts, mul = {}, []
    for q in ("gate", "up", "down"):
        for f in ("trellis", "suh", "svh", "mul1"):
            sh, a, b, dt, shp = rec[f"{q}_{f}"]
            buf = raw[a - base:b - base]
            if f == "mul1":
                assert dt == "I32" and shp == [] and len(buf) == 4
                mul.append(struct.unpack("<i", buf)[0]); continue
            parts[f"{q}_{f}"] = (buf, dt, shp)
    out = []
    for name, dt, shp in px.FIELDS:
        buf, sdt, sshp = parts[name]
        want = ("I16" if dt == np.uint16 else "F16", list(shp))
        assert (sdt, sshp) == want, (name, sdt, sshp, want)
        assert len(buf) == int(np.prod(shp)) * np.dtype(dt).itemsize, name
        out.append(bytes(buf))
    b = b"".join(out)
    assert len(b) == px.BLOB
    return b, mul


def check_fields(blob):
    for name in ("gate_suh", "gate_svh", "up_suh", "up_svh", "down_suh", "down_svh"):
        a = np.frombuffer(blob, np.float16, int(np.prod(dict((n, s) for n, d, s in px.FIELDS)[name])), px.OFF[name])
        if not np.isfinite(a).all() or not np.any(a):
            return f"{name} not finite/zero"
    for name in ("gate_trellis", "up_trellis", "down_trellis"):
        a = np.frombuffer(blob, np.uint16, px.TR // 2, px.OFF[name])
        if a.min() == a.max():
            return f"{name} constant"
    return None


def layer_runs(idx, l):
    """[(shard, start, end, [experts])] covering the layer in GROUP-expert Range requests."""
    recs = [(e, idx[f"{l}.{e}"]) for e in range(px.N_EXPERT)]
    spans = [(e, r["down_suh"][0], min(v[1] for v in r.values()), max(v[2] for v in r.values())) for e, r in recs]
    spans.sort(key=lambda s: (s[1], s[2]))
    runs = []
    for e, sh, a, b in spans:
        assert b - a == SPAN, (l, e, b - a)
        if runs and runs[-1][0] == sh and runs[-1][2] == a and len(runs[-1][3]) < GROUP:
            runs[-1][2] = b; runs[-1][3].append(e)
        else:
            runs.append([sh, a, b, [e]])
    return runs


def cmd_check(l, es):
    import exl3_ref as ref
    idx = load_index()
    mm = np.memmap(px.BIN, np.uint8, "r")
    for e in es:
        rec = idx[f"{l}.{e}"]
        sh = rec["down_suh"][0]; a = min(v[1] for v in rec.values()); b = max(v[2] for v in rec.values())
        raw = http_range(sh, a, b)
        blob, mul = expert_blob(rec, raw, a)
        print(f"L{l} E{e}: mul1 {mul}, field check: {check_fields(blob)}")
        ours = bytes(mm[l * px.LAYER + e * px.BLOB:l * px.LAYER + (e + 1) * px.BLOB])
        W = lambda bl, q: ref.weight_orig_f64(ref.reconstruct(np.frombuffer(bl, np.uint16, px.TR // 2, px.OFF[f"{q}_trellis"]).reshape(
            dict((n, s) for n, d, s in px.FIELDS)[f"{q}_trellis"]), px.K, px.CB),
            np.frombuffer(bl, np.float16, dict((n, s) for n, d, s in px.FIELDS)[f"{q}_suh"][0], px.OFF[f"{q}_suh"]),
            np.frombuffer(bl, np.float16, dict((n, s) for n, d, s in px.FIELDS)[f"{q}_svh"][0], px.OFF[f"{q}_svh"]))
        for q in ("gate", "up", "down"):
            wt, wo = W(blob, q), W(ours, q)
            n = lambda x: float(np.sqrt(np.square(x).mean()))
            print(f"   {q}: |W_td| rms {n(wt):.4e} |W_ours| rms {n(wo):.4e}  rel diff td vs ours {n(wt - wo) / n(wo):.3f}  "
                  f"corr {float(np.corrcoef(wt.ravel(), wo.ravel())[0, 1]):.4f}", flush=True)
        # a control: the same projection of a DIFFERENT expert must not correlate
        o2 = bytes(mm[l * px.LAYER + ((e + 1) % 512) * px.BLOB:l * px.LAYER + ((e + 1) % 512 + 1) * px.BLOB])
        wt, wo = W(blob, "gate"), W(o2, "gate")
        print(f"   control (td gate vs ours of expert {(e + 1) % 512}): corr {float(np.corrcoef(wt.ravel(), wo.ravel())[0, 1]):.4f}", flush=True)


def fetcher(idx, l, q):
    for sh, a, b, es in layer_runs(idx, l):
        q.put((sh, a, es, http_range(sh, a, b), idx))
    q.put(None)


def cmd_sync(layers, neon=True, expect=None):
    ref = None
    if neon:
        import exl3_ref as ref
        px.load_lib()
    idx = load_index()
    m = px.manifest()
    os.makedirs(px.PACK, exist_ok=True)
    fd = os.open(px.BIN, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size < px.N_LAYERS * px.LAYER:
        os.ftruncate(fd, px.N_LAYERS * px.LAYER)
    mul_seen = set()
    for l in layers or range(px.N_LAYERS):
        if m["layers"].get(str(l), {}).get("via", "").startswith("turboderp"):
            print(f"layer {l}: already from turboderp, skipped", flush=True); continue
        t0 = time.perf_counter()
        qu = queue.Queue(maxsize=3)
        th = threading.Thread(target=fetcher, args=(idx, l, qu), daemon=True); th.start()
        h, nb, fetched = hashlib.sha256(), 0, 0
        blobs = {}
        while True:
            it = qu.get()
            if it is None: break
            sh, a, es, raw, _ = it
            fetched += len(raw)
            for e in es:
                rec = idx[f"{l}.{e}"]
                ea = min(v[1] for v in rec.values())
                blob, mul = expert_blob(rec, raw, a)
                mul_seen.update(mul)
                bad = check_fields(blob)
                if bad: raise SystemExit(f"layer {l} expert {e}: {bad}")
                os.pwrite(fd, blob, l * px.LAYER + e * px.BLOB)
                nb += 1
        th.join()
        assert nb == px.N_EXPERT, nb
        os.fsync(fd)
        mm = np.memmap(px.BIN, np.uint8, "r")
        sec = mm[l * px.LAYER:(l + 1) * px.LAYER]
        # the NEON tier on 4 sampled experts vs the f64 reference of the same fetched tensors (skipped with --no-neon: libexl3cpu is macOS-only)
        rng = np.random.default_rng([20261011, l, 3])
        fwd = []
        for e in (rng.choice(px.N_EXPERT, 4, replace=False).tolist() if neon else []):
            be = px.BlobExpert(mm, l, e)
            W = {q: ref.weight_orig_f64(ref.reconstruct(be.f[f"{q}_trellis"], px.K, px.CB), be.f[f"{q}_suh"], be.f[f"{q}_svh"])
                 for q in ("gate", "up", "down")}
            x = rng.standard_normal((4, px.H)).astype(np.float16)
            yh = be.forward(x).astype(np.float64)
            g, u = x.astype(np.float64) @ W["gate"], x.astype(np.float64) @ W["up"]
            y = (g / (1 + np.exp(-g)) * u) @ W["down"]
            fwd.append(float(np.sqrt(np.square(yh - y).mean() / np.square(y).mean())))
        sha = hashlib.sha256(sec).hexdigest()
        ref_sha = None if expect is None else json.load(open(expect))["layers"].get(str(l), {}).get("sha256")
        ok = (not neon or max(fwd) <= 5e-3) and len(mul_seen) == 1 and (expect is None or sha == ref_sha)
        print(f"layer {l}: {nb} experts, fetched {fetched / 1e9:.2f} GB, mul1 values {sorted(mul_seen)}, " + (f"fwd rel rms max {max(fwd):.2e}, " if neon else "no NEON forward check, ") +
              f"sha {sha[:16]}, {time.perf_counter() - t0:.1f} s -> {'ok' if ok else 'FAIL'}", flush=True)
        if not ok: raise SystemExit(f"layer {l}: verify failed")
        m["layers"][str(l)] = dict(sha256=sha, via=VIA, written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        json.dump(m, open(px.MANIFEST + ".tmp", "w"), indent=1); os.replace(px.MANIFEST + ".tmp", px.MANIFEST)
    os.close(fd)


def cmd_verify(per_layer=4, layers=None, sha_only=False):
    """Independent re-fetch: per_layer random experts of every layer, one HTTP Range each, bytes == experts.bin; plus a whole-pack
    sha256 vs the manifest for every layer. sha_only skips the re-fetch (offline: only the section sha256 vs the manifest).
    Returns the number of failures (0 = "VERIFY OK")."""
    idx = None if sha_only else load_index()
    m = px.manifest()
    mm = np.memmap(px.BIN, np.uint8, "r")
    bad = 0
    for l in layers or range(px.N_LAYERS):
        if not sha_only:
            rng = np.random.default_rng([7, l])
            for e in [0, px.N_EXPERT - 1] + rng.choice(px.N_EXPERT, per_layer, replace=False).tolist():
                rec = idx[f"{l}.{e}"]
                sh = rec["down_suh"][0]; a = min(v[1] for v in rec.values()); b = max(v[2] for v in rec.values())
                blob, _ = expert_blob(rec, http_range(sh, a, b), a)
                ok = bytes(mm[l * px.LAYER + e * px.BLOB:l * px.LAYER + (e + 1) * px.BLOB]) == blob
                bad += not ok
                if not ok: print(f"MISMATCH layer {l} expert {e}", flush=True)
        sha = hashlib.sha256(mm[l * px.LAYER:(l + 1) * px.LAYER]).hexdigest()
        ent = m["layers"].get(str(l), {})
        good = sha == ent.get("sha256") and ent.get("via", "").startswith("turboderp")
        bad += not good
        print(f"layer {l}: " + ("" if sha_only else f"{per_layer + 2} experts re-fetched byte-equal, ") +
              f"section sha {'matches manifest' if good else 'MISMATCH'}", flush=True)
    print("VERIFY", "OK" if bad == 0 else f"FAILED ({bad})", flush=True)
    return bad


def cmd_compare(a_path, b_path, layers):
    """Offline: the per-layer section sha256 of manifest A (e.g. a fresh fetch) must equal manifest B's (the committed one). Returns the failure count."""
    a, b = json.load(open(a_path))["layers"], json.load(open(b_path))["layers"]
    bad = 0
    for l in layers or sorted(b, key=int):
        x, y = a.get(str(l), {}).get("sha256"), b.get(str(l), {}).get("sha256")
        good = x is not None and x == y
        bad += not good
        print(f"layer {l} sha256 {(x or 'missing')[:16]} vs {(y or 'missing')[:16]} -> {'MATCH' if good else 'MISMATCH'}", flush=True)
    print("COMPARE", "OK" if bad == 0 else f"FAILED ({bad})", flush=True)
    return bad


def cmd_status():
    m = px.manifest()
    td = [l for l, v in m["layers"].items() if v.get("via", "").startswith("turboderp")]
    print(f"{len(td)}/{px.N_LAYERS} layers from turboderp: {sorted(map(int, td))}")
    return 0


def main(argv):
    """Exit status: 0 = ok, 1 = a verification failed (VERIFY FAILED / COMPARE FAILED / a layer's sha256 differs from --expect), 2 = usage."""
    c = argv[0] if argv else ""
    flags = [a for a in argv[1:] if a.startswith("--")]
    rest = [a for a in argv[1:] if not a.startswith("--")]
    if c == "index": cmd_index(); return 0
    if c == "check": cmd_check(int(rest[0]), [int(x) for x in rest[1:]]); return 0
    if c == "sync":
        expect = next((f.split("=", 1)[1] for f in flags if f.startswith("--expect=")), None)
        cmd_sync([int(x) for x in rest], neon="--no-neon" not in flags, expect=expect); return 0
    if c == "verify": return 1 if cmd_verify(layers=[int(x) for x in rest], sha_only="--sha-only" in flags) else 0
    if c == "compare": return 1 if cmd_compare(rest[0], rest[1], [int(x) for x in rest[2:]]) else 0
    if c == "status": return cmd_status()
    print("usage: pack_td.py index | check L E.. | sync [--no-neon] [--expect=REF.json] [L..] | verify [--sha-only] [L..] | compare A.json B.json [L..] | status", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
