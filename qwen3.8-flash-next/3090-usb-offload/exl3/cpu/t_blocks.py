import ctypes, numpy as np, time, os
from concurrent.futures import ThreadPoolExecutor
lib = ctypes.CDLL(os.path.expanduser("~/exl3/cpu/libexl3cpu.dylib"))
A = lambda a: ctypes.c_void_p(a.ctypes.data)
rng = np.random.default_rng(0)
ok = True
for (m, k, n) in [(1, 2560, 640), (3, 2560, 640), (8, 640, 2560), (1, 640, 2560)]:
    K, cb = 3, 2
    tr = rng.integers(0, 65536, (k // 16) * (n // 16) * 16 * K, dtype=np.uint16)
    sv = (rng.standard_normal(n) * 0.1).astype(np.float16).view(np.uint16)
    x = rng.standard_normal((m, k)).astype(np.float16)
    C0, y = np.zeros((m, n), np.float32), np.zeros((m, n), np.float32)
    xq = np.zeros(m * k * 2 + 64, np.uint16); xq = xq[(-xq.ctypes.data // 2) % 32:][:m * k * 2]
    assert lib.exl3_cpu_gemv(A(x), A(tr), A(sv), m, k, n, K, cb, A(C0), A(y), A(xq), 1) == 0
    xq2 = np.zeros(m * k * 2 + 64, np.uint16); xq2 = xq2[(-xq2.ctypes.data // 2) % 32:][:m * k * 2]
    lib.exl3_cpu_prep_xq(A(x), m, k, A(xq2))
    C1 = np.zeros((m, n), np.float32)
    nb = n // 128
    for b in range(nb):
        assert lib.exl3_cpu_gemv_blocks(A(xq2), A(tr), A(sv), m, k, n, K, cb, b, b + 1, A(C1), None) == 0
    eq = np.array_equal(C0, C1); ok &= eq
    # timing: blocks over T threads (ctypes releases the GIL)
    for T in (1, 4):
        per = [(b * nb // T, (b + 1) * nb // T) for b in range(T)]
        with ThreadPoolExecutor(T) as ex:
            t = time.perf_counter()
            for _ in range(50):
                list(ex.map(lambda r: lib.exl3_cpu_gemv_blocks(A(xq2), A(tr), A(sv), m, k, n, K, cb, r[0], r[1], A(C1), None), per))
            dt = (time.perf_counter() - t) / 50 * 1e6
        print(f"m{m} {k}x{n} T{T}: {dt:.0f} us", flush=True)
    print(f"m{m} {k}x{n} bitwise == gemv: {eq}")
print("ALL OK" if ok else "MISMATCH")
