"""Parity / negative / determinism tests of libexl3metal against libexl3cpu (NEON) and the float64 numpy reference.
Run on the mini:  python exl3/metal/test_exl3_metal.py [--lib L] [--cpu-lib C] [--pack experts.bin] [--receipt out.json]
Pipeline per expert (identical to Strata pool.cpp exl3_task): x -> f16 -> had_in(suh) -> gemv(gate/up) -> silu(g)*u -> f16 -> had_in(dsuh)
-> gemv(down); y = sum_pairs w * out. Gate: rel RMS <= 1e-3 vs the NEON fp32 pipeline."""
import argparse, ctypes, json, mmap, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # exl3_ref.py
import exl3_ref as ref

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=os.path.join(HERE, "libexl3metal.dylib"))
ap.add_argument("--cpu-lib", default=os.path.join(HERE, "..", "cpu", "libexl3cpu.dylib"))
ap.add_argument("--pack", default=os.environ.get("EXL3_PACK_BIN", os.path.join(HERE, "..", "pack", "experts.bin")))
ap.add_argument("--receipt", default=None)
args = ap.parse_args()

H, FF, BLOB, LAYER = 2560, 640, 1862400, 953548800
O = dict(gt=0, ut=614400, dt=1228800, gsuh=1843200, gsvh=1848320, usuh=1849600, usvh=1854720, dsuh=1856000, dsvh=1857280)
GATE = 1e-3

cpu = ctypes.CDLL(args.cpu_lib)
mtl = ctypes.CDLL(args.lib)
vp, i32, i64, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float
mtl.exl3m_create.restype = vp; mtl.exl3m_create.argtypes = [i32, i32]
mtl.exl3m_last_error.restype = ctypes.c_char_p
mtl.exl3m_register.restype = i64; mtl.exl3m_register.argtypes = [vp, vp, ctypes.c_size_t]
mtl.exl3m_moe_rows.argtypes = [vp, vp, i32, i32, vp, vp, vp, vp, vp]
mtl.exl3m_moe_prefill.argtypes = [vp, vp, i32, i64, vp, vp, vp, vp, vp]
mtl.exl3m_set_param.argtypes = [vp, ctypes.c_char_p, i64]
cpu.exl3_cpu_gemv_blocks.argtypes = [vp] * 3 + [i32] * 7 + [vp, vp]
P = lambda a: ctypes.c_void_p(a.ctypes.data)

pack = np.memmap(args.pack, np.uint8, "r")
ctx = mtl.exl3m_create(4, 2)
assert ctx, mtl.exl3m_last_error()

def blob_u8(layer, e): return np.array(pack[layer * LAYER + e * BLOB: layer * LAYER + (e + 1) * BLOB])  # private copy
def fld(b, name, dt, n): return b[O[name]:O[name] + n * np.dtype(dt).itemsize].view(dt)

# ----- NEON reference: exactly pool.cpp's phases ---------------------------------------------------------------------------------
def neon_expert(b, xrows):
    """xrows fp32 [n, 2560] -> out fp32 [n, 2560] for one expert blob b (uint8 [BLOB]), n <= 8 rows per call (loops)."""
    n = xrows.shape[0]
    res = np.zeros((n, H), np.float32)
    tr = {k: b[O[k]:O[k] + 614400].view(np.uint16) for k in ("gt", "ut", "dt")}
    for r0 in range(0, n, 8):
        xr = np.ascontiguousarray(xrows[r0:r0 + 8]); m = xr.shape[0]
        xf = xr.astype(np.float16)
        def had(a, suhname, k):
            out = np.zeros((m, k), np.float16)
            cpu.exl3_cpu_had_in(P(a), P(fld(b, suhname, np.uint16, k)), m, k, P(out))
            return out
        def xq_of(a, k):
            xq = np.zeros(m * k * 2 + 64, np.uint16); xq = xq[(-xq.ctypes.data // 2) % 32:][:m * k * 2]
            cpu.exl3_cpu_prep_xq(P(a), m, k, P(xq)); return xq
        def gemv(xq, trn, svh, k, nn):
            C = np.zeros((m, nn), np.float32)
            for blk in range(nn // 128):
                assert cpu.exl3_cpu_gemv_blocks(P(xq), P(tr[trn]), P(fld(b, svh, np.uint16, nn)), m, k, nn, 3, 2, blk, blk + 1, P(C), None) == 0
            return C
        xh = had(xf, "gsuh", H); g = gemv(xq_of(xh, H), "gt", "gsvh", H, FF)
        xh = had(xf, "usuh", H); u = gemv(xq_of(xh, H), "ut", "usvh", H, FF)
        z = (g / (np.float32(1) + np.exp(-g)) * u).astype(np.float16)
        zh = had(z, "dsuh", FF)
        res[r0:r0 + m] = gemv(xq_of(zh, FF), "dt", "dsvh", FF, H)
    return res

def numpy_expert(b, xrows):
    """float64 reference from the fp16 operands (exl3_ref)."""
    gt = b[O["gt"]:O["gt"] + 614400].view(np.uint16).reshape(160, 40, 48)
    ut = b[O["ut"]:O["ut"] + 614400].view(np.uint16).reshape(160, 40, 48)
    dt = b[O["dt"]:O["dt"] + 614400].view(np.uint16).reshape(40, 160, 48)
    xf = xrows.astype(np.float16)
    f = lambda n, d: fld(b, n, np.float16, d)
    g = ref.gemv_f64(ref.had_in(xf, f("gsuh", H)), ref.reconstruct(gt, 3, 2), f("gsvh", FF))
    u = ref.gemv_f64(ref.had_in(xf, f("usuh", H)), ref.reconstruct(ut, 3, 2), f("usvh", FF))
    z = (g / (1 + np.exp(-g)) * u).astype(np.float16)
    return ref.gemv_f64(ref.had_in(z, f("dsuh", FF)), ref.reconstruct(dt, 3, 2), f("dsvh", H))

def stats(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt((d * d).sum() / max((b.astype(np.float64) ** 2).sum(), 1e-30))), float(np.abs(d).max())

def neon_moe(blobs, x, eidx, ridx, w):
    y = np.zeros_like(x, dtype=np.float32)
    outs = {}
    for e in sorted(set(eidx)):
        sel = [p for p in range(len(eidx)) if eidx[p] == e]
        o = neon_expert(blobs[e], x[[ridx[p] for p in sel]])
        for p, row in zip(sel, o): outs[p] = row
    for p in range(len(eidx)): y[ridx[p]] += np.float32(w[p]) * outs[p]
    return y

class Blobs:
    """n distinct real blobs in (a) page-aligned anonymous memory registered NoCopy, (b) plain numpy arrays (staged path)."""
    def __init__(self, specs):
        self.arr = [blob_u8(l, e) for l, e in specs]
        self.mm = mmap.mmap(-1, len(specs) * BLOB + 65536)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        for i, a in enumerate(self.arr): self.mm[i * BLOB:(i + 1) * BLOB] = a.tobytes()
        self.reg = mtl.exl3m_register(ctx, ctypes.c_void_p(self.base), len(specs) * BLOB + 65536)
        assert self.reg > 0, mtl.exl3m_last_error()
        self.ptr_reg = (vp * len(specs))(*[self.base + i * BLOB for i in range(len(specs))])
        self.ptr_stage = (vp * len(specs))(*[a.ctypes.data for a in self.arr])

def run_decode(bl, x, eidx, ridx, w, registered=True):
    m = x.shape[0]; y = np.zeros((m, H), np.float32)
    ea, ra, wa = np.array(eidx, np.int32), np.array(ridx, np.int32), np.array(w, np.float32)
    rc = mtl.exl3m_moe_rows(ctx, P(np.ascontiguousarray(x)), m, len(eidx), P(ea), P(ra), P(wa), bl.ptr_reg if registered else bl.ptr_stage, P(y))
    assert rc == 0, mtl.exl3m_last_error()
    return y

def run_prefill(bl, x, eidx, ridx, w, registered=True):
    n = x.shape[0]; y = np.zeros((n, H), np.float32)
    ea, ra, wa = np.array(eidx, np.int32), np.array(ridx, np.int32), np.array(w, np.float32)
    rc = mtl.exl3m_moe_prefill(ctx, P(np.ascontiguousarray(x)), n, len(eidx), P(ea), P(ra), P(wa), bl.ptr_reg if registered else bl.ptr_stage, P(y))
    assert rc == 0, mtl.exl3m_last_error()
    return y

rng = np.random.default_rng(1234)
R = {"gate_rel_rms": GATE, "tests": []}
ok_all = True
def record(name, ok, **kw):
    global ok_all
    ok_all &= bool(ok)
    R["tests"].append(dict(name=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name, {k: (round(v, 9) if isinstance(v, float) else v) for k, v in kw.items()}, flush=True)

specs = [(int(l), int(e)) for l, e in zip(rng.integers(0, 48, 24), rng.integers(0, 512, 24))]
specs[0] = (24, 5)
bl = Blobs(specs)
x8 = (rng.standard_normal((8, H)) * 1.0).astype(np.float32)

# 1. numpy-reference sanity of the NEON pipeline itself (one expert, 2 rows)
t = time.time()
yn = neon_expert(bl.arr[0], x8[:2]); y64 = numpy_expert(bl.arr[0], x8[:2])
rel, mx = stats(yn, y64)
record("neon_vs_numpy_f64", rel < 2e-3, rel_rms=rel, max_abs=mx, secs=round(time.time() - t, 1))

# 2. decode shape parity (m rows x E experts), registered NoCopy + staged
for (m, E, per) in [(1, 10, 1), (2, 13, 2), (4, 13, 3), (4, 20, 2), (8, 10, 8), (3, 7, 3)]:
    es = list(rng.choice(len(specs), E, replace=False))
    eidx, ridx, w = [], [], []
    for e in es:
        for r in rng.choice(m, min(per, m), replace=False):
            eidx.append(int(e)); ridx.append(int(r)); w.append(float(rng.uniform(0.05, 0.6)))
    x = x8[:m]
    yref = neon_moe([a for a in bl.arr], x, eidx, ridx, w)
    for registered in (True, False):
        y = run_decode(bl, x, eidx, ridx, w, registered)
        rel, mx = stats(y, yref)
        record(f"decode m={m} E={E} pairs={len(eidx)} {'nocopy' if registered else 'staged'}", rel <= GATE, rel_rms=rel, max_abs=mx, ref_max=float(np.abs(yref).max()))

# 3. deliberate-mismatch negative tests: the gate must FAIL
m, E = 2, 10
es = list(range(E)); eidx = [e for e in es for _ in range(2)]; ridx = [r for _ in es for r in (0, 1)]; w = [0.3] * len(eidx)
x = x8[:m]
yref = neon_moe(bl.arr, x, eidx, ridx, w)
y = run_decode(bl, x, eidx, ridx, w)
rel_ok, _ = stats(y, yref)
# (a) metal runs a different expert for one pair
eidx_bad = list(eidx); eidx_bad[3] = (eidx_bad[3] + 5) % len(specs)
y_bad = run_decode(bl, x, eidx_bad, ridx, w); rel, mx = stats(y_bad, yref)
record("negative wrong-expert detected", rel > GATE * 10 and rel_ok <= GATE, rel_rms=rel, max_abs=mx, baseline_rel=rel_ok)
# (b) one flipped bit in a trellis tile of one expert (metal side only)
saved = bytes(bl.mm[3 * BLOB + 5000:3 * BLOB + 5004]); bl.mm[3 * BLOB + 5000:3 * BLOB + 5004] = bytes(b ^ 0xFF for b in saved)
y_bad = run_decode(bl, x, eidx, ridx, w); rel, mx = stats(y_bad, yref)
bl.mm[3 * BLOB + 5000:3 * BLOB + 5004] = saved
y_back = run_decode(bl, x, eidx, ridx, w); rel_back, _ = stats(y_back, yref)
record("negative corrupted-trellis detected", rel > GATE and rel_back <= GATE, rel_rms=rel, max_abs=mx, restored_rel=rel_back)
# (c) wrong weight on one pair
w_bad = list(w); w_bad[0] = 0.31
y_bad = run_decode(bl, x, eidx, ridx, w_bad); rel, mx = stats(y_bad, yref)
record("negative weight-mismatch detected", rel > GATE, rel_rms=rel, max_abs=mx)

# 4. determinism (bitwise repeat), decode and prefill
ys = [run_decode(bl, x, eidx, ridx, w) for _ in range(30)]
record("determinism decode x30 bitwise", all(np.array_equal(ys[0], q) for q in ys[1:]))

# 5. prefill parity: n_tok rows, top-k over many experts
n_tok, k = 160, 4
eidx, ridx, w = [], [], []
for r in range(n_tok):
    for e in rng.choice(len(specs), k, replace=False):
        eidx.append(int(e)); ridx.append(r); w.append(float(rng.uniform(0.05, 0.5)))
xp = (rng.standard_normal((n_tok, H))).astype(np.float32)
t = time.time(); yref = neon_moe(bl.arr, xp, eidx, ridx, w); tn = time.time() - t
for registered in (True, False):
    y = run_prefill(bl, xp, eidx, ridx, w, registered); rel, mx = stats(y, yref)
    record(f"prefill n_tok={n_tok} topk={k} experts={len(specs)} {'nocopy' if registered else 'staged'}", rel <= GATE, rel_rms=rel, max_abs=mx, ref_max=float(np.abs(yref).max()))
# decode path vs prefill path on a decode-shaped call
m = 4; eidx = [int(e) for e in range(9) for _ in range(2)]; ridx = [int(r) for _ in range(9) for r in (0, 3)]; w = [0.2] * len(eidx)
yd = run_decode(bl, x8[:m], eidx, ridx, w); yp = run_prefill(bl, x8[:m], eidx, ridx, w); rel, mx = stats(yp, yd)
record("prefill path vs decode path (m=4, 9 experts; two fp32 accumulation orders)", rel <= GATE, rel_rms=rel, max_abs=mx)
eidx2, ridx2, w2 = [], [], []
for r in range(n_tok):
    for e in range(k):
        eidx2.append(int((r + e * 5) % len(specs))); ridx2.append(r); w2.append(0.25)
ys = [run_prefill(bl, xp, eidx2, ridx2, w2) for _ in range(6)]
record("determinism prefill x6 bitwise", all(np.array_equal(ys[0], q) for q in ys[1:]))
# prefill chunking independence: tiny chunks give the same bits? (order of chunk accumulation changes the fp32 sum order -> compare numerically)
mtl.exl3m_set_param(ctx, b"pf_chunk_experts", 3); ychunk = run_prefill(bl, xp, eidx2, ridx2, w2); mtl.exl3m_set_param(ctx, b"pf_chunk_experts", 16)
rel, mx = stats(ychunk, ys[0]); record("prefill chunk size independence (3 vs 16 experts)", rel < 1e-6, rel_rms=rel, max_abs=mx)

# 6. multi-piece registration (EXL3M_PIECE_MB test hook): blobs straddling piece boundaries must still resolve in place
os.environ["EXL3M_PIECE_MB"] = "5"
bl2 = Blobs(specs)
del os.environ["EXL3M_PIECE_MB"]
eidx = [int(e) for e in range(24) for _ in range(1)]; ridx = [e % 4 for e in range(24)]; w = [0.07] * 24
mtl.exl3m_get_stats.argtypes = [vp, vp]
class Stats(ctypes.Structure): _fields_ = [("dc", ctypes.c_uint64), ("pc", ctypes.c_uint64), ("mapped", ctypes.c_uint64), ("staged", ctypes.c_uint64), ("gpu", ctypes.c_double), ("wall", ctypes.c_double), ("enc", ctypes.c_double), ("cp", ctypes.c_double)]
s0 = Stats(); mtl.exl3m_get_stats(ctx, ctypes.byref(s0))
ya = run_decode(bl, x8[:4], eidx, ridx, w, True); yb = run_decode(bl2, x8[:4], eidx, ridx, w, True)
s1 = Stats(); mtl.exl3m_get_stats(ctx, ctypes.byref(s1))
record("multi-piece NoCopy registration == single piece (bitwise), all 48 blobs in place", np.array_equal(ya, yb) and s1.staged == s0.staged, staged_delta=int(s1.staged - s0.staged), mapped_delta=int(s1.mapped - s0.mapped))

# 7. thread safety + async submit/wait: 6 threads x 12 calls through 2 scratch slots, results bitwise equal to serial
import threading
mtl.exl3m_moe_rows_submit.argtypes = [vp, vp, i32, i32, vp, vp, vp, vp, vp, vp]
mtl.exl3m_wait.argtypes = [vp, vp]
jobs = []
for t_ in range(6):
    mm = [1, 2, 4][t_ % 3]; E = 9 + t_
    ee, rr, ww = [], [], []
    for e in range(E):
        for r in range(min(mm, 2)): ee.append(e); rr.append((e + r) % mm); ww.append(0.1 + 0.01 * e)
    jobs.append((x8[:mm].copy(), ee, rr, ww))
serial = [run_decode(bl, *j) for j in jobs]
bad = []
def worker(t_):
    for k in range(12):
        y = run_decode(bl, *jobs[t_], registered=bool(k % 2))
        if not np.array_equal(y, serial[t_]): bad.append((t_, k))
th = [threading.Thread(target=worker, args=(t_,)) for t_ in range(6)]
[t_.start() for t_ in th]; [t_.join() for t_ in th]
record("thread safety 6 threads x 12 calls (2 slots) bitwise == serial", not bad, mismatches=len(bad))
# async: two jobs in flight, waited out of order
outs = [np.zeros((jobs[i][0].shape[0], H), np.float32) for i in range(2)]; hs = []
arrs = []
for i in range(2):
    xx, ee, rr, ww = jobs[i]; ea, ra, wa = np.array(ee, np.int32), np.array(rr, np.int32), np.array(ww, np.float32); arrs.append((xx, ea, ra, wa))
    h = vp(); rc = mtl.exl3m_moe_rows_submit(ctx, P(xx), xx.shape[0], len(ee), P(ea), P(ra), P(wa), bl.ptr_reg, P(outs[i]), ctypes.byref(h)); assert rc == 0; hs.append(h)
assert mtl.exl3m_wait(ctx, hs[1]) == 0 and mtl.exl3m_wait(ctx, hs[0]) == 0
record("async submit/wait, two jobs in flight, out-of-order wait", all(np.array_equal(outs[i], serial[i]) for i in range(2)))

R["all_pass"] = bool(ok_all)
print("ALL PASS" if ok_all else "SOME FAILED")
if args.receipt: json.dump(R, open(args.receipt, "w"), indent=1)
sys.exit(0 if ok_all else 1)
