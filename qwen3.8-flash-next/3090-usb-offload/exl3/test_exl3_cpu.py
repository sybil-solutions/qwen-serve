"""M2 gate + bench: NEON EXL3 decode / had_in / GEMV on the M4 CPU (cpu/exl3_cpu.c -> cpu/libexl3cpu.dylib) vs exl3_ref.

Run on the mini (CPU only, no GPU access; the Strata engine may keep running):
  zsh exl3/cpu/build.sh                       # -> exl3/cpu/libexl3cpu.dylib
  python exl3/test_exl3_cpu.py [receipt]      # default exl3/cpu_receipt.json; exit 0 iff all pass
  env: EXL3_PACK_BIN (pack for the real-expert cases when quant_gpu.npz is absent), EXL3_TEST_NOBENCH=1 (skip the bench)

Gate:
  - decode: fp16 values bit-identical to exl3_ref.codebook_lut(cb)[decode_states(trellis, K)] for K 1..8 x 3INST/MCG/MUL1
    (random trellises; K8 runs cover ~all 65536 states per codebook) and for the 18 real K3 MUL1 expert trellises in
    quant_gpu.npz (also == the encoder's own Q_r and == exl3_ref.reconstruct after the tensor-core permutation)
  - had_in: bit-identical to exl3_ref.had_in
  - GEMV: C vs exl3_ref.gemv_f64 (same fp16 A_had and W_inner, exact accumulate) within the M1 GEMV gate
    rel RMS <= 2e-3, max |err| <= 2e-2 * RMS(C); m 1/3/8, real expert tensors + random 4096x4096, 1 and 4 threads
  - negative control: decoding a MUL1 trellis with the MCG codebook must fail the GEMV gate
Bench: GB/s of trellis bytes (3 bpw) per core and on 4 / 10 threads, decode-only and GEMV m 1/8, expert and 8192^2 shapes.
"""
import ctypes, json, os, sys, time
import numpy as np
import exl3_ref as ref

HERE = os.path.dirname(os.path.abspath(__file__))
REL_RMS_MAX, MAX_ERR_REL = 2e-3, 2e-2
out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cpu_receipt.json")
lib = ctypes.CDLL(os.path.join(HERE, "cpu", "libexl3cpu.dylib"))
P, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
lib.exl3_cpu_decode.argtypes = [P, L, I, I, P]
lib.exl3_cpu_had_in.argtypes = [P, P, I, I, P]
lib.exl3_cpu_gemv.argtypes = [P, P, P, I, I, I, I, I, P, P, P, I]
rng = np.random.default_rng(20261012)
rec, ok_all = dict(decode=[], real=[], had_in=[], gemv=[], bench=[]), True


def ptr(a): return a.ctypes.data_as(P)


def decode(tr, K, cb):
    flat = np.ascontiguousarray(tr.reshape(-1, 16 * K))
    out = np.empty((len(flat), 256), np.uint16)
    assert lib.exl3_cpu_decode(ptr(flat), len(flat), K, cb, ptr(out)) == 0
    return out


def tiles_to_w(vals, tk, tn):
    t = np.empty_like(vals)
    t[:, ref.tensor_core_perm()] = vals
    return t.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def had_in(A, suh):
    out = np.empty_like(A)
    lib.exl3_cpu_had_in(ptr(A), ptr(suh), A.shape[0], A.shape[1], ptr(out))
    return out


class Gemv:
    def __init__(self, tr, svh, m, k, n, K=3, cb=2):
        self.tr, self.svh, self.m, self.k, self.n, self.K, self.cb = np.ascontiguousarray(tr), svh, m, k, n, K, cb
        self.C, self.y = np.empty((m, n), np.float32), np.empty((m, n), np.float32)
        self.xq = np.empty(m * k * 2 + 64, np.uint16)
        self.xq = self.xq[(-self.xq.ctypes.data // 2) % 32:][:m * k * 2]                  # 64-byte aligned
    def __call__(self, A_had, threads=1, cb=None):
        assert lib.exl3_cpu_gemv(ptr(A_had), ptr(self.tr), ptr(self.svh), self.m, self.k, self.n, self.K,
                                 self.cb if cb is None else cb, ptr(self.C), ptr(self.y), ptr(self.xq), threads) == 0
        return self.C


def score(C, C_ref):
    err = C.astype(np.float64) - C_ref
    rms = float(np.sqrt((C_ref ** 2).mean()))
    rr, mx = float(np.sqrt((err ** 2).mean())) / rms, float(np.abs(err).max()) / rms
    return dict(rel_rms=rr, max_err_over_rms=mx), bool(np.isfinite(C).all() and rr <= REL_RMS_MAX and mx <= MAX_ERR_REL)


def signs(n, lo, hi): return (rng.choice([-1.0, 1.0], n) * rng.uniform(lo, hi, n)).astype(np.float16)


# 1) decode, random trellises, every K x codebook
for K in range(1, 9):
    for cb in (0, 1, 2):
        nt = 4096 if K == 8 else 512
        tr = rng.integers(0, 65536, (nt, 16 * K), dtype=np.uint16)
        got = decode(tr, K, cb)
        st = ref.decode_states(tr, K)
        want = ref.codebook_lut(cb)[st].view(np.uint16)
        r = dict(K=K, cb=cb, tiles=nt, mismatches=int(np.count_nonzero(got != want)), states_covered=int(len(np.unique(st))))
        r["pass"] = r["mismatches"] == 0
        ok_all &= r["pass"]; rec["decode"].append(r)
    print("decode K", K, [(d["cb"], d["mismatches"]) for d in rec["decode"][-3:]], flush=True)

# 2) real K3 MUL1 expert trellises: either the M3 encoder fixture quant_gpu.npz (not in the repo; also checked against the encoder's own Q_r),
#    or, when it is absent, 6 experts of layer 24 read from the pack (EXL3_PACK_BIN, default <repo>/exl3/pack/experts.bin, read-only mmap)
real = []
fixture = os.path.join(HERE, "quant_gpu.npz")
if os.path.exists(fixture):
    z = np.load(fixture)
    meta = json.loads(str(z["meta"]))
    items = []
    for c in meta["experts"]:
        tag, k, n = f"e{c['expert']}_{c['proj']}", c["k"], c["n"]
        items.append((tag, k, n, z[f"{tag}_trellis"], z[f"{tag}_suh"], z[f"{tag}_svh"], z[f"{tag}_Qr"], z[f"{tag}_wgpu"]))
else:
    import pack_exl3 as px
    mm = np.memmap(os.environ.get("EXL3_PACK_BIN", px.BIN), np.uint8, "r")
    items = []
    for e in (0, 1, 2, 3, 255, 511):
        be = px.BlobExpert(mm, 24, e)
        for q in ("gate", "up", "down"):
            k, n = (px.H, px.I) if q != "down" else (px.I, px.H)
            items.append((f"L24e{e}_{q}", k, n, np.array(be.f[f"{q}_trellis"]), np.array(be.f[f"{q}_suh"]), np.array(be.f[f"{q}_svh"]), None, None))
for tag, k, n, tr, suh, svh, Qr, wgpu in items:
    W = tiles_to_w(decode(tr, 3, 2).view(np.float16), k // 16, n // 16)
    W_ref = ref.reconstruct(tr, 3, 2)
    r = dict(tensor=tag, k=k, n=n, mismatch_vs_ref=int(np.count_nonzero(W.view(np.uint16) != W_ref.view(np.uint16))))
    r["pass"] = r["mismatch_vs_ref"] == 0
    if Qr is not None:
        r["mismatch_vs_encoder_Qr"] = int(np.count_nonzero(W.view(np.uint16) != Qr.view(np.uint16)))
        r["pass"] = r["pass"] and r["mismatch_vs_encoder_Qr"] == 0
    ok_all &= r["pass"]; rec["real"].append(r); real.append((tag, k, n, tr, suh, svh, W_ref, wgpu))
print("real decode", sum(r["pass"] for r in rec["real"]), "/", len(rec["real"]), flush=True)

# 3) had_in bit-exact
for m, k in ((1, 2560), (8, 2560), (3, 640), (2, 8192)):
    A = (rng.standard_normal((m, k)) * rng.choice([1e-3, 1.0, 30.0])).astype(np.float16)
    suh = signs(k, 0.005, 0.03)
    got, want = had_in(A, suh), ref.had_in(A, suh)
    r = dict(m=m, k=k, mismatches=int(np.count_nonzero(got.view(np.uint16) != want.view(np.uint16))))
    r["pass"] = r["mismatches"] == 0
    ok_all &= r["pass"]; rec["had_in"].append(r)
print("had_in", [r["mismatches"] for r in rec["had_in"]], flush=True)


# 4) GEMV gate
def gemv_case(name, tr, suh, svh, W_inner, m, threads, cb_run=2, w_orig_gpu=None, negative=False):
    global ok_all
    k, n = W_inner.shape
    A = rng.standard_normal((m, k)).astype(np.float16)
    A_had = had_in(A, suh)
    C_ref = ref.gemv_f64(A_had, W_inner, svh)
    g = Gemv(tr, svh, m, k, n)
    C = g(A_had, threads, cb_run).copy()
    s, ok = score(C, C_ref)
    r = dict(case=name, m=m, k=k, n=n, threads=threads, cb_run=cb_run, **s)
    if w_orig_gpu is not None:   # cross-device: CPU C vs A @ (3090 recon_had W_orig, fp16), informational
        r["vs_gpu_recon_had_rel_rms"] = score(C, A.astype(np.float64) @ w_orig_gpu.astype(np.float64))[0]["rel_rms"]
    if negative:
        r["passes_gate"] = ok; r["pass"] = not ok
    else:
        r["pass"] = ok
    ok_all &= r["pass"]; rec["gemv"].append(r)
    return r


for idx, (tag, k, n, tr, suh, svh, W_ref, wgpu) in enumerate(real):
    for m in (1, 3, 8):
        gemv_case(tag, tr, suh, svh, W_ref, m, 1 if (idx + m) % 2 else 4, w_orig_gpu=wgpu)
tr = rng.integers(0, 65536, (256, 256, 48), dtype=np.uint16)
suh, svh = signs(4096, 0.005, 0.03), signs(4096, 0.5, 1.5)
Wr = ref.reconstruct(tr, 3, 2)
for m, th in ((1, 1), (1, 4), (8, 10)):
    gemv_case("random4096", tr, suh, svh, Wr, m, th)
for cb in (0, 1):                           # other codebooks through the GEMV path
    tr2 = rng.integers(0, 65536, (40, 160, 48), dtype=np.uint16)
    W2 = ref.reconstruct(tr2, 3, cb)
    g = Gemv(tr2, signs(2560, 0.5, 1.5), 1, 640, 2560, cb=cb)
    A_had = had_in(rng.standard_normal((1, 640)).astype(np.float16), signs(640, 0.005, 0.03))
    s, ok = score(g(A_had, 1).copy(), ref.gemv_f64(A_had, W2, g.svh))
    rec["gemv"].append(dict(case=f"random640x2560_cb{cb}", m=1, k=640, n=2560, threads=1, cb_run=cb, **s, **{"pass": ok}))
    ok_all &= ok
neg = gemv_case("negative: MCG codebook on MUL1 trellis", *real[0][3:7], 1, 1, cb_run=1, negative=True)
print("gemv", sum(r["pass"] for r in rec["gemv"]), "/", len(rec["gemv"]),
      "max rel RMS", max(r["rel_rms"] for r in rec["gemv"] if not r["case"].startswith("negative")),
      "negative", round(neg["rel_rms"], 3), flush=True)


# 5) bench (trellis GB/s = trellis bytes / wall)
def best(fn, reps):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def bench_gemv(name, k, n, m, threads, reps):
    tr = rng.integers(0, 65536, (k // 16, n // 16, 48), dtype=np.uint16)
    g = Gemv(tr, signs(n, 0.5, 1.5), m, k, n)
    A_had = had_in(rng.standard_normal((m, k)).astype(np.float16), signs(k, 0.005, 0.03))
    tmin, tmed = best(lambda: g(A_had, threads), reps)
    r = dict(op="gemv", shape=name, k=k, n=n, m=m, threads=threads, us_min=round(tmin * 1e6, 1),
             us_med=round(tmed * 1e6, 1), trellis_GBps=round(tr.nbytes / tmin / 1e9, 3),
             weights_Gps=round(k * n / tmin / 1e9, 2))
    rec["bench"].append(r); print(r, flush=True)


def bench_decode(threads_unused, ntiles, reps):
    tr = rng.integers(0, 65536, (ntiles, 48), dtype=np.uint16)
    out = np.empty((ntiles, 256), np.uint16)
    tmin, tmed = best(lambda: lib.exl3_cpu_decode(ptr(tr), ntiles, 3, 2, ptr(out)), reps)
    r = dict(op="decode_only", tiles=ntiles, threads=1, us_min=round(tmin * 1e6, 1), trellis_GBps=round(tr.nbytes / tmin / 1e9, 3),
             weights_Gps=round(ntiles * 256 / tmin / 1e9, 2))
    rec["bench"].append(r); print(r, flush=True)


if not os.environ.get("EXL3_TEST_NOBENCH"):        # EXL3_TEST_NOBENCH=1: gates only (the bench runs 10 threads for ~1 min)
    bench_decode(1, 6400, 50)
    for name, k, n in (("expert gate/up 2560x640", 2560, 640), ("expert down 640x2560", 640, 2560), ("8192x8192", 8192, 8192)):
        for m in (1, 8):
            for th in (1, 4, 10):
                bench_gemv(name, k, n, m, th, 30 if k * n < 1e7 else 8)

rec["gate"] = dict(gemv_rel_rms_max=REL_RMS_MAX, gemv_max_err_over_rms=MAX_ERR_REL, decode="bit-exact", had_in="bit-exact")
rec["load_avg"] = os.getloadavg()
n_pass = sum(r["pass"] for key in ("decode", "real", "had_in", "gemv") for r in rec[key])
n_all = sum(len(rec[key]) for key in ("decode", "real", "had_in", "gemv"))
rec["all_pass"], rec["n_pass"], rec["n_cases"] = bool(ok_all), n_pass, n_all
json.dump(rec, open(out_path, "w"), indent=1)
print(("ALL PASS" if ok_all else "FAIL"), f"{n_pass}/{n_all} ->", out_path)
sys.exit(0 if ok_all else 1)
