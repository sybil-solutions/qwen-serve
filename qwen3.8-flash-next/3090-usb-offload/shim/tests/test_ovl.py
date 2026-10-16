# test_ovl.py : can the host issue GPU launches while a bulk push is in flight, and does a busy compute stream slow the push?
import sys, os, struct, time, threading
sys.path.insert(0, os.path.expanduser("~/shim"))
import numpy as np
import tgcu, tgcu_api as api
api.init(); ctx = api.ctx
dev = ctx.malloc(256 << 20)
data = os.urandom(8 << 20)
k = api.builtin["tgcu_memset32"]
def mk(nbytes_words, grid=65535):
    return ctx.encode(k, (grid, 1, 1), (256, 1, 1), 0, struct.pack("<QIIQ", dev + (64 << 20), 7, 0, nbytes_words))
small = mk(256, 1)
big = mk(48 << 20, 65535)   # ~192 MB memset: a GPU kernel of a few ms
def issue(n, kern):
    t = time.perf_counter()
    for _ in range(n): ctx.pending.append(kern)
    t1 = time.perf_counter()
    ctx.flush()
    return (t1 - t) * 1e3, (time.perf_counter() - t1) * 1e3
def sync(): ctx.synchronize()
def log(*a): print(*a, flush=True)
# warm
issue(4, small); sync(); ctx.ace.h2d(dev, data)
# 0: alone
t = time.perf_counter(); ctx.ace.h2d(dev, data); push_alone = (time.perf_counter() - t) * 1e3
t = time.perf_counter(); issue(76, small); sync(); launch_alone = (time.perf_counter() - t) * 1e3
t = time.perf_counter(); issue(1, big); sync(); big_alone = (time.perf_counter() - t) * 1e3
log(f"alone: push 8MB {push_alone:.1f} ms ({8*1000/push_alone:.0f} MB/s); 76 small launches+flush+sync {launch_alone:.1f} ms; big kernel {big_alone:.1f} ms")
# 1: launches while a push runs in another thread
res = {}
def pusher():
    t = time.perf_counter(); ctx.ace.h2d(dev, data); res["push"] = (time.perf_counter() - t) * 1e3
th = threading.Thread(target=pusher); th.start()
time.sleep(0.002)
t = time.perf_counter(); a, b = issue(76, small); res["issue"] = (a, b)
th.join(); sync()
log(f"launches during push: queue {res['issue'][0]:.1f} ms flush {res['issue'][1]:.1f} ms; push {res['push']:.1f} ms")
# 2: big kernel queued FIRST (flushed), then the push: does the push slow down? does the kernel finish in time?
t = time.perf_counter(); issue(1, big)
t1 = time.perf_counter(); ctx.ace.h2d(dev, data); push_busy = (time.perf_counter() - t1) * 1e3
sync(); tot = (time.perf_counter() - t) * 1e3
log(f"big kernel then push: push {push_busy:.1f} ms (alone {push_alone:.1f}); total {tot:.1f} ms (sum {push_alone + big_alone:.1f})")
