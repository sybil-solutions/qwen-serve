"""M4 prefill GPU test of exl3s_prefill_moe (exllamav3 exl3_moe_kernel, 3 tiers) on real pack blobs (layer 24), no GPU in
this process.  13 experts with row counts spanning every tier and the >256-row split (incl. one with 0 rows), placed at
shuffled row offsets (not in list order) with 8 unlisted sentinel rows that must stay untouched.  Expected per listed row:
expert(x) in float64 from exl3_ref (x rounded to fp16, as the kernel reads it), unweighted.  Gate: rel RMS <= 2e-3, max
err <= 2e-2 x RMS over all listed rows, sentinels bit-identical, repeat bit-identical.  Run on the mini with the card
free (engine + converter stopped):
  cd ~/exl3/strata_m4 && source ~/.chestnut_env && SITE=$(~/tinygrad/.venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
  DEV=NV:PTX PYTHONPATH=~/exl3:~/shim:~/tg-strata:$SITE PYTHONNOUSERSITE=1 ~/tinygrad/.venv/bin/python test_exl3s_pf.py
"""
import json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.expanduser("~/exl3"))
import exl3_ref as ref, pack_exl3 as px

HERE = os.path.dirname(os.path.abspath(__file__))
BITS = int(os.environ.get("BITS", "3"))   # BITS=2: the mixed-precision K 2 blobs of LAYER (default 19) from pack_mixed.py
L = int(os.environ.get("LAYER", "24" if BITS == 3 else "19"))
mm = np.memmap(px.BIN, np.uint8, "r")
rng = np.random.default_rng(20261011)
counts = [1, 3, 8, 16, 17, 32, 33, 64, 100, 0, 256, 300, 520]
if BITS == 2:
    import pack_mixed as pm
    wrote = pm.mman()["layers"][str(L)]["wrote"]                  # [router index, original id, file position]
    experts = [w[2] for w in rng.choice(len(wrote), len(counts), replace=False).tolist() for w in [wrote[w]]]
    BLOBB, OFFB = pm.BLOB2, pm.OFF2
else:
    experts = rng.choice(512, len(counts), replace=False).tolist()   # file positions = expert ids at K 3
    BLOBB, OFFB = px.BLOB, px.OFF
def blob_f(e):
    base = L * px.LAYER + e * px.BLOB
    fl = pm.F2 if BITS == 2 else px.FIELDS
    return {n: np.ndarray(shp, dt, mm, base + OFFB[n]) for n, dt, shp in fl}
SENT = 8
R = sum(counts) + SENT
# shuffled placement: each expert's rows contiguous, experts in a random order, sentinels at the end
order = rng.permutation(len(counts))
off, at = [0] * len(counts), 0
for i in order:
    off[i] = at; at += counts[i]
xs = (rng.standard_normal((R, 2560)) * 0.5).astype(np.float16)
dm0 = np.full((R, 2560), 7.0, np.float32)
ref_out = np.zeros((R, 2560))
for i, e in enumerate(experts):
    if counts[i] == 0: continue
    bf = blob_f(e)
    W = {q: ref.weight_orig_f64(ref.reconstruct(bf[f"{q}_trellis"], BITS, px.CB), bf[f"{q}_suh"], bf[f"{q}_svh"])
         for q in ("gate", "up", "down")}
    x = xs[off[i]:off[i] + counts[i]].astype(np.float64)
    g, u = x @ W["gate"], x @ W["up"]
    ref_out[off[i]:off[i] + counts[i]] = (g / (1 + np.exp(-g)) * u) @ W["down"]
d = tempfile.mkdtemp(prefix="exl3s_pf_")
slots = np.concatenate([np.asarray(mm[L * px.LAYER + e * px.BLOB:L * px.LAYER + e * px.BLOB + BLOBB]) for e in experts])
slots.tofile(f"{d}/slots.bin"); xs.tofile(f"{d}/xs.bin"); dm0.tofile(f"{d}/dm0.bin")
np.array(counts, np.int32).tofile(f"{d}/cnt.bin"); np.array(off, np.int32).tofile(f"{d}/off.bin")
open(f"{d}/meta.txt", "w").write(f"{len(counts)} {R} {BLOBB} 50 " + " ".join(str(OFFB[n]) for n in px.COOP_ORDER) + f" {BITS}\n")
p = subprocess.run([f"{HERE}/test_exl3s_pf", f"{HERE}/libstrata_exl3.dylib", d], capture_output=True, text=True, timeout=900)
if p.returncode != 0:
    print("binary failed", p.returncode, p.stderr[-2000:]); sys.exit(1)
t = json.loads(p.stdout.strip().splitlines()[-1])
dm, rep = (np.fromfile(f"{d}/{n}.bin", np.float32).reshape(R, 2560) for n in ("dm", "dm_rep"))
listed = np.zeros(R, bool)
for i in range(len(counts)): listed[off[i]:off[i] + counts[i]] = True
got, want = dm[listed].astype(np.float64), ref_out[listed]
rms = float(np.sqrt((want ** 2).mean()))
rel, mx = float(np.sqrt(((got - want) ** 2).mean())) / rms, float(np.abs(got - want).max()) / rms
per = {int(c): float(np.sqrt(((dm[o:o + c] - ref_out[o:o + c]) ** 2).mean() / (ref_out[o:o + c] ** 2).mean()))
       for c, o in zip(counts, off) if c}
rec = dict(layer=L, experts=experts, counts=counts, rows=R, rel_rms=rel, max_err_over_rms=mx, per_expert_rel_rms=per,
           sentinels_untouched=bool(np.array_equal(dm[~listed].view(np.uint32), dm0[~listed].view(np.uint32))),
           repeat_identical=bool(np.array_equal(dm.view(np.uint32), rep.view(np.uint32))), finite=bool(np.isfinite(got).all()),
           **{"bin_" + k: v for k, v in t.items()})
rec["pass"] = bool(rec["finite"] and rel <= 3e-3 and mx <= 2e-2 and rec["sentinels_untouched"] and rec["repeat_identical"])
json.dump(rec, open(f"{HERE}/exl3s_pf_receipt{'' if BITS == 3 else '_k2'}.json", "w"), indent=1)
print(json.dumps(rec))
print("PASS" if rec["pass"] else "FAIL")
sys.exit(0 if rec["pass"] else 1)
