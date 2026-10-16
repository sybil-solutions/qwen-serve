"""M4 GPU test of libstrata_exl3 (decode pre/post + prefill dequant) on real pack blobs (layer 24), no GPU in this process.
Per case: S experts of layer 24 copied into "VRAM slots" (res[e] = slot), the rest misses (res -1); random x, routing
over a pool that mixes hits and misses, random bo (Strata's combine output).  Expected: bo + sum over HITS of w * expert(x)
(float64, x rounded to fp16 as exl3_moe_coop reads it); misses must contribute exactly nothing.  Gate (M1/moe gate):
rel RMS of the added part <= 2e-3, max err <= 2e-2 x RMS; both scratch groups identical; repeat bit-identical.
Dequant: dq_gu / dq_d of slot 0 == exl3_ref W_orig (fp16 of the float64 reconstruction) within 1e-3 rel RMS, layout
interleaved gate/up rows.  Run on the mini with the card free (engine + converter stopped):
  source ~/.chestnut_env; DEV=NV:PTX PYTHONPATH=~/exl3:~/shim:~/tg-strata:$SITE PYTHONNOUSERSITE=1 python test_exl3s.py
"""
import json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.expanduser("~/exl3"))
import exl3_ref as ref, pack_exl3 as px

HERE = os.path.dirname(os.path.abspath(__file__))
BITS = int(os.environ.get("BITS", "3"))   # BITS=2: the mixed-precision K 2 blobs of LAYER (default 19) from pack_mixed.py
L = int(os.environ.get("LAYER", "24" if BITS == 3 else "19"))
mm = np.memmap(px.BIN, np.uint8, "r")
if BITS == 2:
    import pack_mixed as pm
    POS = {w[1]: w[2] for w in pm.mman()["layers"][str(L)]["wrote"]}   # original id -> file position
    BLOBB, OFFB, FL = pm.BLOB2, pm.OFF2, pm.F2
else:
    POS, BLOBB, OFFB, FL = {e: e for e in range(512)}, px.BLOB, px.OFF, px.FIELDS
blob = lambda e: np.asarray(mm[L * px.LAYER + POS[e] * px.BLOB:L * px.LAYER + POS[e] * px.BLOB + BLOBB])
off = [OFFB[n] for n in px.COOP_ORDER]
Wcache = {}
def W(e):
    if e not in Wcache:
        base = L * px.LAYER + POS[e] * px.BLOB
        f = {n: np.ndarray(shp, dt, mm, base + OFFB[n]) for n, dt, shp in FL}
        Wcache[e] = {q: ref.weight_orig_f64(ref.reconstruct(f[f"{q}_trellis"], BITS, px.CB), f[f"{q}_suh"], f[f"{q}_svh"])
                     for q in ("gate", "up", "down")}
    return Wcache[e]
def expert(e, x):
    w = W(e); g, u = x @ w["gate"], x @ w["up"]
    return (g / (1 + np.exp(-g)) * u) @ w["down"]

rng = np.random.default_rng(20261011)
cases, ok_all = [], True
for rows, n_hit_pool, n_miss_pool in ((1, 12, 4), (3, 16, 6), (5, 20, 8), (8, 24, 8)):
    pool = rng.choice(sorted(POS), n_hit_pool + n_miss_pool, replace=False)
    hits, misses = pool[:n_hit_pool], pool[n_hit_pool:]
    res = np.full(512, -1, np.int32); res[hits] = np.arange(len(hits))
    slots = np.concatenate([blob(int(e)) for e in hits])
    x = (rng.standard_normal((rows, 2560)) * 0.5).astype(np.float32)
    ids = np.stack([rng.choice(pool, 10, replace=False) for _ in range(rows)]).astype(np.int32)
    lg = rng.standard_normal((rows, 10)); w = (np.exp(lg) / np.exp(lg).sum(1, keepdims=True)).astype(np.float32)
    bo = (rng.standard_normal((rows, 2560)) * 0.1).astype(np.float32)
    x16 = x.astype(np.float16).astype(np.float64)
    add = np.zeros((rows, 2560))
    for r in range(rows):
        for k in range(10):
            e = int(ids[r, k])
            if res[e] >= 0: add[r] += float(np.float16(w[r, k])) * expert(e, x16[r:r + 1])[0]
    d = tempfile.mkdtemp(prefix="exl3s_")
    for n, a in (("slots", slots), ("res", res), ("x", x), ("ids", ids), ("w", w), ("bo", bo)): a.tofile(f"{d}/{n}.bin")
    open(f"{d}/meta.txt", "w").write(f"512 {rows} {len(hits)} {BLOBB} 200 " + " ".join(map(str, off)) + f" {BITS}\n")
    p = subprocess.run([f"{HERE}/test_exl3s", f"{HERE}/libstrata_exl3.dylib", d], capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        print("binary failed", p.returncode, p.stderr[-2000:]); sys.exit(1)
    t = json.loads(p.stdout.strip().splitlines()[-1])
    o0, o1, orep = (np.fromfile(f"{d}/bo_out{s}.bin", np.float32).reshape(rows, 2560) for s in ("0", "1", "_rep"))
    got = o0.astype(np.float64) - bo
    rms = float(np.sqrt((add ** 2).mean()))
    rel, mx = float(np.sqrt(((got - add) ** 2).mean())) / rms, float(np.abs(got - add).max()) / rms
    c = dict(rows=rows, hits=int((res[ids] >= 0).sum()), misses=int((res[ids] < 0).sum()), rel_rms=rel, max_err_over_rms=mx,
             groups_identical=bool(np.array_equal(o0.view(np.uint32), o1.view(np.uint32))),
             repeat_identical=bool(np.array_equal(o0.view(np.uint32), orep.view(np.uint32))), us_per_layer=t["us_per_layer"])
    c["pass"] = bool(np.isfinite(o0).all() and rel <= 2e-3 and mx <= 2e-2 and c["groups_identical"] and c["repeat_identical"])
    if rows == 1:   # dequant of slot 0 (expert hits[0]) vs float64 W_orig, Strata layout
        w0 = W(int(hits[0]))
        gu = np.fromfile(f"{d}/dq_gu.bin", np.float16).reshape(1280, 2560).astype(np.float64)
        dd = np.fromfile(f"{d}/dq_d.bin", np.float16).reshape(2560, 640).astype(np.float64)
        exp_gu = np.empty((1280, 2560)); exp_gu[0::2] = w0["gate"].T; exp_gu[1::2] = w0["up"].T
        rr = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
        c["dequant_gu_rel_rms"], c["dequant_d_rel_rms"] = rr(gu, exp_gu), rr(dd, w0["down"].T)
        c["pass"] &= c["dequant_gu_rel_rms"] <= 1e-3 and c["dequant_d_rel_rms"] <= 1e-3
    ok_all &= c["pass"]; cases.append(c)
    print(json.dumps(c), flush=True)
json.dump(dict(layer=L, bits=BITS, cases=cases, ok=ok_all), open(f"{HERE}/exl3s_receipt{'' if BITS == 3 else '_k2'}.json", "w"), indent=1)
print("ALL PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
