#!/usr/bin/env python3
"""pipe_bench.py - stock vs pipelined h2d/d2h on the tgcu AsyncCopy (env TGCU_PIPE_MIN=0 = stock)."""
import os, time, statistics, json
import numpy as np
import tgcu_api as api, tgcu as T
api.init(); ctx = api.ctx; ace = ctx.ace
dev = ctx.malloc(48 << 20)
rng = np.random.default_rng(11)
print("PIPE_MIN", ace.PIPE_MIN, "SLOTS", ace.PIPE_SLOTS, "DOWN", ace.PIPE_DOWN, flush=True)
res = {}
for n in (1 << 20, 4 << 20, 10_485_760, 10_485_760 + 12345):
    d = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    ts_u, ts_d, ok = [], [], True
    for rep in range(6):
        t = time.perf_counter(); ace.h2d(dev, d); ts_u.append(time.perf_counter() - t)
        t = time.perf_counter(); b = ace.d2h(dev, n); ts_d.append(time.perf_counter() - t)
        ok &= (b == d)
    # small ops afterwards must still work
    ace.h2d(dev + (40 << 20), d[:5000]); ok &= ace.d2h(dev + (40 << 20), 5000) == d[:5000]
    ok &= ace.d2h(dev + 77, 1 << 18) == d[77:77 + (1 << 18)] if n >= (1 << 18) + 77 else True
    mu, md = statistics.median(ts_u[1:]), statistics.median(ts_d[1:])
    print(f"{n:9d} B  h2d {mu*1e3:7.2f} ms {n/mu/1e6:6.1f} MB/s | d2h {md*1e3:7.2f} ms {n/md/1e6:6.1f} MB/s | exact {ok}", flush=True)
    res[n] = dict(h2d_MBps=round(n / mu / 1e6, 1), d2h_MBps=round(n / md / 1e6, 1), exact=bool(ok))
json.dump(res, open(os.path.expanduser(f"~/shim/pipe_bench_{os.environ.get('TAG','x')}.json"), "w"), indent=1)
