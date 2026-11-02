#!/usr/bin/env python3
"""duplex_bench.py - is the USB link full duplex through the tgcu copy engines? (PrefillPipeline 2026-10-12)
Run via duplex_bench.sh on the mini with the server stopped."""
import os, sys, time, struct, threading, statistics, json
import numpy as np
import tgcu_api as api, tgcu
import tgcu as T
api.init(); ctx = api.ctx; ace0 = ctx.ace
SLOT = ace0.slot_size; round_up = T.round_up
MB = 1e6
dev = ctx.malloc(64 << 20)
rng = np.random.default_rng(7)
res = {}
def rep(k, v): res[k] = v; print(k, v, flush=True)

# ---------- generic single-round primitives with explicit SRAM slot placement ----------
def up_round(ac, dst, data, slot, prof=None):
    wire = round_up(len(data) + 8, 512)
    ac.gate_seq = (ac.gate_seq + 1) & 0xffffffff
    gate = 0x5A17C0DE00000000 | ac.gate_seq
    buf = bytearray(wire); buf[:len(data)] = data; buf[wire - 8:] = struct.pack("<Q", gate)
    base = ac.sram_base + (slot - 5) * SLOT
    def send():
        ac.usb.usb.control_write(0xF2, value=wire // 512, index=slot | (((wire + SLOT - 1) // SLOT) << 8))
        ac.usb.usb.bulk_write(bytes(buf))
    ac.sram_va = base
    ac._copy_and_signal(dst, base, len(data), gate=(base + wire - 8, gate), between=send)

def down_round(ac, src, n, slot, from_zero=False):
    # start-slot read: wire = n only (from_zero=False) or the stock path (start slot 0, prefix = slot*SLOT)
    base = ac.sram_base + (slot - 5) * SLOT
    off = slot * SLOT if from_zero else 0
    wire = round_up(off + n, 512)
    start = 0 if from_zero else slot
    ac.usb.usb.control_write(0xF2, value=(wire // 512) | 0x8000, index=start | (((wire + SLOT - 1) // SLOT) << 8))
    ac._copy_and_signal(base, src, n, ring_cq=True)
    return bytes(ac.usb.usb.bulk_read(wire, timeout=6000))[off:off + n]

ace0.sram_base = ace0.sram_va
def timed(fn, reps=1):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

N = 10 << 20
payload_u = rng.integers(0, 256, N, dtype=np.uint8).tobytes()
payload_d = rng.integers(0, 256, N, dtype=np.uint8).tobytes()

# ---- 0. baseline stock paths
ace0.h2d(dev, payload_u); ace0.h2d(dev + N, payload_d)
t = timed(lambda: ace0.h2d(dev, payload_u), 3); rep("stock.h2d.10MB.MBps", round(N / t / MB, 1))
t = timed(lambda: ace0.d2h(dev, N), 3); rep("stock.d2h.10MB.MBps", round(N / t / MB, 1))
assert ace0.d2h(dev, N) == payload_u

# ---- 1. our round primitives, single stream, sweep round sizes / start slot
def up_stream(ac, dst, data, rbytes, slot):
    for o in range(0, len(data), rbytes): up_round(ac, dst + o, data[o:o + rbytes], slot)
def down_stream(ac, src, n, rbytes, slot, from_zero=False):
    out = bytearray()
    for o in range(0, n, rbytes): out += down_round(ac, src + o, min(rbytes, n - o), slot, from_zero)
    return bytes(out)

for rb in (64 << 10, 128 << 10, 256 << 10, 392 << 10):
    nsl = 25
    t = timed(lambda: up_stream(ace0, dev, payload_u, rb, 5), 2); rep(f"up.1ch.round{rb>>10}K.MBps", round(N / t / MB, 1))
    assert ace0.d2h(dev, N) == payload_u
for rb in (64 << 10, 128 << 10, 256 << 10):
    t = timed(lambda: down_stream(ace0, dev, N, rb, 5, True), 2); rep(f"down.1ch.fromslot0.round{rb>>10}K.MBps", round(N / t / MB, 1))
# ---- 2. two outstanding same-direction rounds: the bulk OUT of round k+1 while round k's gated CE job still runs
def up_pipe(ac, dst, data, rbytes, nbuf=2, slot0=5, span=int(os.environ.get("SPAN","12"))):
    # nbuf alternating SRAM regions of `span` slots; CE jobs queued gated; wait only before reusing a region
    pend = [None] * nbuf
    k = 0
    def wait_sem(v):
        t0 = time.perf_counter()
        while ((struct.unpack("<I", ac.mv[ac.sem_off:ac.sem_off + 4])[0] - v) & 0xffffffff) >= 0x80000000:
            if time.perf_counter() - t0 > 5: raise RuntimeError("timeout")
    for o in range(0, len(data), rbytes):
        r = k % nbuf; slot = slot0 + r * span; k += 1
        if pend[r] is not None:
            try: wait_sem(pend[r])
            except RuntimeError: print("pipe timeout at k", k, "r", r, "slot", slot, flush=True); raise
        chunk = data[o:o + rbytes]
        wire = round_up(len(chunk) + 8, 512)
        ac.gate_seq = (ac.gate_seq + 1) & 0xffffffff
        gate = 0x5A17C0DE00000000 | ac.gate_seq
        buf = bytearray(wire); buf[:len(chunk)] = chunk; buf[wire - 8:] = struct.pack("<Q", gate)
        base = ac.sram_base + (slot - 5) * SLOT
        nvg, nvm = T.onv.nv_gpu, T.onv.nvm
        ac.sem_value += 1; sem = ac.va + ac.sem_off
        cmds = nvm(0, nvg.NVC56F_SEM_ADDR_LO, (base + wire - 8) & 0xffffffff, (base + wire - 8) >> 32, gate & 0xffffffff, gate >> 32,
                   T.onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
        cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, base >> 32, base & 0xffffffff, (dst + o) >> 32, (dst + o) & 0xffffffff)
        cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, len(chunk))
        cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))
        cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, ac.sem_value & 0xffffffff)
        cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
        ac._submit(cmds)
        ac.usb.usb.control_write(0xF2, value=wire // 512, index=slot | (((wire + SLOT - 1) // SLOT) << 8))
        ac.usb.usb.bulk_write(bytes(buf))
        pend[r] = ac.sem_value
    for p in pend:
        if p is not None: wait_sem(p)

try:
    for rb in (64 << 10, 128 << 10, 190 << 10):
        ace0.h2d(dev, bytes(N))   # clear
        up_pipe(ace0, dev, payload_u, rb)
        ok = ace0.d2h(dev, N) == payload_u
        t = timed(lambda: up_pipe(ace0, dev, payload_u, rb), 2)
        rep(f"up.pipe2.round{rb>>10}K.MBps", round(N / t / MB, 1)); rep(f"up.pipe2.round{rb>>10}K.exact", ok)
except Exception as e: rep("up.pipe.err", repr(e)); raise

# ---- 3. DUPLEX: two copy channels (COPY0 + second AsyncCopy on another CE), one thread per direction, disjoint SRAM slots
def make_ace2():
    for eng in ("NV2080_ENGINE_TYPE_COPY1", "NV2080_ENGINE_TYPE_COPY2", None):
        try:
            a = T.AsyncCopy(ctx, engine_type=(getattr(T.onv.nv_gpu, eng) if eng else ctx.ace.engine))
            a.sram_base = a.sram_va; a.hold_va, a.hold_off = ctx.ace.hold_va, ctx.ace.hold_off
            return a, eng
        except Exception as e: print("ace2", eng, "failed", repr(e)[:200], flush=True)
    return None, None
ace1, eng = make_ace2(); rep("ace2.engine", str(eng))
if ace1 is not None:
    ace1.sram_base = ace1.sram_va
    # upload on ace0 slots 5..16 (12 slots = 192K - 512), download on ace1 slots 17..29 (start-slot read)
    URB, DRB = 128 << 10, 128 << 10
    ace1.sram_base = ace0.sram_base
    out = {}
    def U():
        t = time.perf_counter(); up_stream(ace0, dev, payload_u, URB, 17); out["u"] = time.perf_counter() - t
    # data to download lives at dev+32M (written beforehand)
    ace0.h2d(dev + (32 << 20), payload_d)
    def D():
        t = time.perf_counter(); out["d_data"] = down_stream(ace1, dev + (32 << 20), N, DRB, 5, True); out["d"] = time.perf_counter() - t
    # single-direction references in the same partition
    U(); rep("duplex.ref.up.MBps", round(N / out["u"] / MB, 1))
    D(); rep("duplex.ref.down.MBps", round(N / out["d"] / MB, 1)); rep("duplex.ref.down.exact", out["d_data"] == payload_d)
    for rep_i in range(3):
        ace0.sram_va = ace0.sram_base; ace0.h2d(dev, bytes(N))
        th = [threading.Thread(target=U), threading.Thread(target=D)]
        t0 = time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]; wall = time.perf_counter() - t0
        ace0.sram_va = ace0.sram_base; ok_d = out["d_data"] == payload_d; ok_u = ace0.d2h(dev, N) == payload_u
        rep(f"duplex.run{rep_i}.up.MBps", round(N / out["u"] / MB, 1)); rep(f"duplex.run{rep_i}.down.MBps", round(N / out["d"] / MB, 1))
        rep(f"duplex.run{rep_i}.combined.MBps", round(2 * N / wall / MB, 1)); rep(f"duplex.run{rep_i}.exact", (ok_u, ok_d))
if os.environ.get("STARTSLOT"):
  try:
      for rb in (64 << 10, 128 << 10, 256 << 10, 392 << 10):
          out = down_stream(ace0, dev, N, rb, 5, False)
          ok = out == payload_u
          t = timed(lambda: down_stream(ace0, dev, N, rb, 5, False), 2)
          rep(f"down.1ch.startslot5.round{rb>>10}K.MBps", round(N / t / MB, 1)); rep(f"down.1ch.startslot5.round{rb>>10}K.exact", ok)
  except Exception as e: rep("down.startslot5.err", repr(e))
json.dump(res, open(os.path.expanduser("~/shim/duplex_bench_res.json"), "w"), indent=1)
print("DONE", flush=True)
