# test_wrap.py old|new N : the transient-ring wrap of tgcu.Context.  `old` replays the pre-2026-10-12 _transient (wait for sem_value, which already counts the pushbuffer
# being submitted) on a small ring so that the pushbuffer allocation crosses the end of the ring; `new` runs the shipped segmented ring on the same geometry.
import sys, os, struct, time
sys.path.insert(0, os.path.expanduser("~/shim"))
mode, n = sys.argv[1], int(sys.argv[2])
# geometry: one launch = QMD record (rec bytes) + one pushbuffer (0x100); the ring ends right after a QMD allocation, so the pushbuffer allocation crosses it
K = 40
import tgcu, tgcu_api as api
api.init()
ctx = api.ctx
va = ctx.malloc(1 << 20)
k = api.builtin["tgcu_memset32"]
params = struct.pack("<QIIQ", va, 7, 0, 256)
rec = ctx.encode(k, (1, 1, 1), (256, 1, 1), 0, params)
REC = tgcu.round_up(ctx._rec_size(rec), 0x100)
print("record bytes", hex(ctx._rec_size(rec)), "allocation", hex(REC), flush=True)
ctx.synchronize()
if mode == "new":
    ctx.seg_size = K * (REC + 0x100) + REC
    ctx.seg, ctx.ctrl_head = 0, 0x1000
if mode == "old":
    LIMIT = 0x1000 + K * (REC + 0x100) + REC
    def _transient(self, size):
        if tgcu.round_up(self.ctrl_head, 0x100) + size > LIMIT:
            self._wait(self.sem_value)
            self.ctrl_head = 0x1000
        off = tgcu.round_up(self.ctrl_head, 0x100)
        self.ctrl_head = off + tgcu.round_up(size, 0x100)
        return off
    tgcu.Context._transient = _transient
    ctx.ctrl_head = 0x1000
t0 = time.time()
try:
    for i in range(n):
        ctx.pending.append(ctx.encode(k, (1, 1, 1), (256, 1, 1), 0, params))
        ctx.flush()
    ctx.synchronize()
    print(f"{mode}: {n} flushes ok in {time.time() - t0:.1f} s, sem {ctx.sem_value}", flush=True)
except Exception as e:
    print(f"{mode}: FAILED after {time.time() - t0:.1f} s: {str(e)[:200]}", flush=True)
    sys.exit(1)
