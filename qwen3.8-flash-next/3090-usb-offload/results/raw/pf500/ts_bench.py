#!/usr/bin/env python3
"""ts_bench.py - does GPU compute slow the USB copy path (and vice versa) through channel-group time slicing? env TS_G (graphics us), TS_C (copy us)."""
import os, sys, time, statistics, json
import numpy as np
import tgcu_api as api, tgcu as T
api.init(); ctx = api.ctx; ace = ctx.ace; dev = ctx.dev
nvg = T.onv.nv_gpu
def get_ts(grp):
    try: return dev.iface.rm_control(grp, nvg.NVA06C_CTRL_CMD_GET_TIMESLICE, nvg.NVA06C_CTRL_GET_TIMESLICE_PARAMS()).timesliceUs
    except Exception as e: return f"err {e}"
def set_ts(grp, us):
    dev.iface.rm_control(grp, nvg.NVA06C_CTRL_CMD_SET_TIMESLICE, nvg.NVA06C_CTRL_SET_TIMESLICE_PARAMS(timesliceUs=us))
gg, cg = dev.channel_group, ace.group
print("default timeslice us: graphics", get_ts(gg), "copy", get_ts(cg), flush=True)
if os.getenv("TS_G"): set_ts(gg, int(os.environ["TS_G"]))
if os.getenv("TS_C"): set_ts(cg, int(os.environ["TS_C"]))
print("timeslice us now: graphics", get_ts(gg), "copy", get_ts(cg), flush=True)
buf = ctx.malloc(1 << 30); buf2 = ctx.malloc(1 << 30)
N = 8 << 20
host = np.random.default_rng(1).integers(0, 256, N, dtype=np.uint8)
xf = ctx.malloc(N + (1 << 20))
def gpu_work(k):
    for _ in range(k): api.d2d(buf2, buf, 1 << 29)   # ~1.1 ms each
def sync(): api.synchronize()
def timeit(f):
    t = time.perf_counter(); r = f(); return time.perf_counter() - t
K = int(os.getenv("K", "60"))
res = {}
for rep in range(3):
    a = timeit(lambda: (gpu_work(K), sync()))                       # gpu alone
    b = timeit(lambda: ace.h2d(xf, host.tobytes()))                 # h2d alone
    c = timeit(lambda: ace.d2h(xf, N))                              # d2h alone
    def both(kind):
        gpu_work(K); api.flush()                                    # GPU busy now
        t = time.perf_counter()
        if kind == "h2d": ace.h2d(xf, host.tobytes())
        else: ace.d2h(xf, N)
        tx = time.perf_counter() - t
        sync()
        return tx, time.perf_counter() - t
    tx_u, tt_u = (None, None); 
    t0 = time.perf_counter(); r1 = both("h2d"); r2 = both("d2h")
    print(f"rep {rep}: gpu alone {a*1e3:.1f} ms | h2d alone {b*1e3:.1f} ms ({N/b/1e6:.0f} MB/s) d2h alone {c*1e3:.1f} ms ({N/c/1e6:.0f} MB/s) | with GPU busy: h2d {r1[0]*1e3:.1f} ms (total {r1[1]*1e3:.1f}), d2h {r2[0]*1e3:.1f} ms (total {r2[1]*1e3:.1f})", flush=True)
