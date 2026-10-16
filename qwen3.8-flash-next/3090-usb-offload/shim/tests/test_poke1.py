# test_poke1.py: do single-TLP (one control transfer) 4-byte writes land? compare with the stream write; then run AsyncCopy submits with POKE1 on
import sys, os, struct, time
sys.path.insert(0, os.path.expanduser("~/shim"))
import tgcu, tgcu_api as api
api.init()
ctx = api.ctx; ace = ctx.ace
off = 0x30000
mv = ace.mv
print("mv type", type(mv).__name__, hex(mv.addr), flush=True)
mv[off:off + 4] = struct.pack("<I", 0x11111111)
print("stream readback", hex(int.from_bytes(bytes(mv[off:off + 4]), "little")), flush=True)
tgcu._w32(mv, off, 0x22222222)
print("w32 readback", hex(int.from_bytes(bytes(mv[off:off + 4]), "little")), flush=True)
t = time.perf_counter()
for i in range(200): tgcu._w32(mv, off, 0x33330000 + i)
print("w32 each ms", (time.perf_counter() - t) / 200 * 1e3, "readback", hex(int.from_bytes(bytes(mv[off:off + 4]), "little")), flush=True)
t = time.perf_counter()
for i in range(200): mv[off:off + 4] = struct.pack("<I", 0x44440000 + i)
print("stream each ms", (time.perf_counter() - t) / 200 * 1e3, flush=True)
# a real copy with each poke variant
va = ctx.malloc(1 << 20)
for bits in (1, 2, 3, 0):
    tgcu._POKE1 = bits
    data = os.urandom(65536)
    try:
        ace.h2d(va, data)
        back = ace.d2h(va, 65536)
        print("POKE1 bits", bits, "roundtrip ok:", back == data, flush=True)
    except Exception as e:
        print("POKE1 bits", bits, "FAILED:", repr(e)[:100], flush=True)
        break
print("gpu_mmio addr", hex(ctx.dev.gpu_mmio.addr), "mapped_mv addr", hex(ctx.mapped_mv.addr), flush=True)
# compute-channel submit with single-transfer GPPut + doorbell (bit 8)
k = api.builtin["tgcu_memset32"]
dv = ctx.malloc(1 << 20)
for bits in (15, 0):
    tgcu._POKE1 = bits
    try:
        for i in range(50):
            ctx.pending.append(ctx.encode(k, (4, 1, 1), (256, 1, 1), 0, struct.pack("<QIIQ", dv, 0x1234 + i, 0, 1024)))
            ctx.flush()
        ctx.synchronize()
        got = struct.unpack("<I", ctx.read_vram(dv, 4))[0]
        print("compute submit, POKE1 bits", bits, "ok:", got == 0x1234 + 49, hex(got), flush=True)
    except Exception as e:
        print("compute submit bits", bits, "FAILED", repr(e)[:120], flush=True); break
# deferred free
tgcu._FREE_DEFER = True
a = ctx.malloc(1 << 20); ctx.pending.append(ctx.encode(k, (4, 1, 1), (256, 1, 1), 0, struct.pack("<QIIQ", a, 5, 0, 1024)))
ctx.free(a); b = ctx.malloc(1 << 20); ctx.synchronize(); ctx._reap()
print("deferred free: deferred left", len(ctx.deferred), "a in allocs", a in ctx.allocs, "b==a", b == a, flush=True)
