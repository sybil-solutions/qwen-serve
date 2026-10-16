# The Python half of libcudart-tgcu: every function here is called from tgcudart.cpp with plain ints and host addresses.
from __future__ import annotations
import ctypes, os, re, struct, sys, time
import numpy as np
import tgcu

ctx: tgcu.Context | None = None
modules: list[tgcu.Module] = []
kernels: list[tgcu.Kernel] = []
builtin: dict[str, tgcu.Kernel] = {}
capture: list | None = None             # launch records while a stream is being captured
graphs: list[list] = []                 # captured launch lists
execs: list[int] = []                   # instantiated graphs: first QMD address
mapped: dict[int, tuple[int, int]] = {} # host address -> (device mirror, size)
MAPPED_SYNC = os.getenv("TGCU_MAPPED_SYNC", "1") != "0"
MAPPED_MAX = int(os.getenv("TGCU_MAPPED_MAX", str(8 << 20)))

def _mv(addr:int, size:int) -> memoryview: return memoryview((ctypes.c_char * size).from_address(addr)).cast("B")

def _helper(script:str):
  import subprocess
  py = os.getenv("TGCU_PYTHON", os.path.expanduser("~/tinygrad/.venv/bin/python"))
  r = subprocess.run([py, os.path.join(os.path.dirname(__file__), script)], capture_output=True, text=True, timeout=120)
  if r.returncode != 0: print(f"[tgcu] {script} failed: {r.stderr.strip()[-300:]}", flush=True)

def _reset_usb():
  # a process killed mid-transfer leaves stale state in the ASM2464 (shifted reads, bulk IN overflows): drain and clear
  # the bulk endpoints and reset the bridge's USB state from a fresh process, before tinygrad opens the device
  _helper("usb_drain.py")

def _recover():
  # deeper: power-cycle PCIe through the bridge (the same off->on step its firmware takes at boot), then the USB reset
  _helper("pcie_cycle.py"); _helper("usb_drain.py")

def init() -> int:
  global ctx
  # every start: PCIe power cycle + USB reset from fresh processes, before this process opens the device (a crashed run
  # leaves the bridge with stale transfer state, and tinygrad's first failed attempt would keep the device open)
  if ctx is None and not os.getenv("TGCU_NO_USB_RESET"): _recover()
  for attempt in range(3):
    if ctx is not None: break
    try: ctx = tgcu.Context()
    except Exception as e:
      if attempt == 2: raise
      print(f"[tgcu] GPU init failed ({type(e).__name__}: {e}); retrying", flush=True)
      time.sleep(3)
  if ctx is not None and not builtin:
    m = tgcu.Module(ctx, tgcu.fatbin_cubins(open(os.path.join(os.path.dirname(__file__), "tgcu_builtin.fatbin"), "rb").read())[0][1])
    builtin.update(m.kernels)
    ctx.zero_fn = _zero
    for ch in ctx.chunks: _zero(ch["base"], ch["size"] + ctx.GUARD)   # any made before the kernel was loaded
    libc = tgcu.Module(ctx, open(os.path.join(os.path.dirname(__file__), "tgcu_libc.cubin"), "rb").read())
    ctx.externs.update(libc.funcs)
  return 0

# *** memory ***
def _zero(va:int, n:int):
  # a fresh chunk reads as zeros, finished before it is handed out (copy-engine writes into it are not queued); the
  # launches already queued stay queued - they cannot touch memory that did not exist
  saved, ctx.pending = ctx.pending, [ctx.encode(builtin["tgcu_memset32"], _grid(n // 4), (256, 1, 1), 0, struct.pack("<QIIQ", va, 0, 0, n // 4))]
  try: ctx.flush(); ctx._wait(ctx.sem_value)
  finally: ctx.pending = saved + ctx.pending
def malloc(size:int) -> int: return ctx.malloc(size)
def free(va:int) -> int:
  if dev_watch: _dev_forget(va)   # (before ctx.free: it needs the block's size)
  ctx.free(va)
  return 0
def mem_info() -> tuple[int, int]:
  total = 24 << 30
  used = sum(ch["size"] + ctx.GUARD for ch in ctx.chunks) - sum(n for ch in ctx.chunks for _, n in ch["free"])
  return total - used - (1 << 30), total

# TGCU_H2D_RELAX=1: a bulk host->device push (>= PIPE_MIN, the prompt lane's y rows) waits only for the launches that name its destination range as a kernel
# argument, not for the whole GPU (cudaMemcpy is stream-ordered, and the engine's lockstep keeps the next chunk's front running behind it).  A range pushed
# for the third time is watched (the one-off pushes of startup are not): that push is ordered as before, and from then on launches whose parameter words fall
# inside it are tracked.  Kernels that reach the range through a pointer they were not given (an arena base + offsets) are not seen: the prompt lane's
# combine kernels take the pointer directly.
RELAX = os.getenv("TGCU_H2D_RELAX", "0") != "0"
dev_seen: dict[int, list[int]] = {}          # bulk push destination -> [pushes so far, largest size]
dev_watch: dict[int, tuple[int, int]] = {}   # watched range key (its lowest address) -> (lo, hi); disjoint
dev_pend: set[int] = set()                   # watched ranges named by launches still in ctx.pending (or a graph)
dev_sem: dict[int, int] = {}                 # watched range -> submission of the last launch that named it
_wlo = np.zeros(0, np.uint64); _whi = np.zeros(0, np.uint64); _wkey: list[int] = []
def _dev_rebuild():
  global _wlo, _whi, _wkey
  items = sorted(dev_watch.items(), key=lambda kv: kv[1][0])
  _wlo = np.array([lo for _, (lo, _h) in items], np.uint64); _whi = np.array([hi for _, (_l, hi) in items], np.uint64); _wkey = [k for k, _ in items]
def _dev_forget(va:int):   # a freed block takes every watched range inside it with it
  end = va + (ctx.allocs[va][1] if va in ctx.allocs else 1 << 40)
  gone = [k for k, (lo, hi) in dev_watch.items() if va <= lo < end]
  for k in gone: dev_watch.pop(k); dev_pend.discard(k); dev_sem.pop(k, None)
  for d in [d for d in dev_seen if va <= d < end]: dev_seen.pop(d)
  if gone: _dev_rebuild()
dev_pend_idx: dict[int, int] = {}            # watched range -> number of ctx.pending launches up to and including the last one that names it
def _dev_refs(cb:bytes, n:int):
  if len(cb) < 8 or not len(_wlo): return
  w = np.frombuffer(cb, np.uint64, len(cb) // 8)
  i = np.searchsorted(_wlo, w, side="right") - 1
  ok = (i >= 0) & (w < _whi[np.maximum(i, 0)])
  if ok.any():
    for j in set(i[ok].tolist()):
      key = _wkey[j]; dev_pend.add(key); dev_pend_idx[key] = n
def _dev_find(dst:int, size:int):   # the watched range that covers [dst, dst + size), else None
  for k, (lo, hi) in dev_watch.items():
    if lo <= dst and dst + size <= hi: return k
  return None
def _dev_register(dst:int, size:int):
  lo, hi = dst, dst + size
  for k, (l2, h2) in list(dev_watch.items()):   # keep the ranges disjoint: a new one that overlaps merges (the ordered push that registers it has synchronized everything)
    if l2 < hi and lo < h2:
      lo, hi = min(lo, l2), max(hi, h2); dev_watch.pop(k); dev_pend.discard(k); dev_sem.pop(k, None)
  dev_watch[lo] = (lo, hi)
  _dev_rebuild()
def h2d(dst:int, src:int, size:int) -> int:
  # the pipelined path (>= PIPE_MIN) reads the host buffer in place: the call is synchronous, a copy of 10 MB is 3 ms
  if size and RELAX and ctx.ace.PIPE_MIN and size >= ctx.ace.PIPE_MIN:
    k = _dev_find(dst, size) if dev_watch else None
    if k is not None:
      need = dev_sem.get(k, 0)
      if k in dev_pend:   # the launches that name it go first; the ones queued after the last of them are submitted too and run during the push
        idx = dev_pend_idx.get(k, len(ctx.pending))
        if ctx.pending: _push_mapped()
        if idx < len(ctx.pending):
          rest = ctx.pending[idx:]; ctx.pending = ctx.pending[:idx]
          ctx.flush(); need = ctx.sem_value
          ctx.pending = rest
          for j in [j for j in dev_pend if dev_pend_idx.get(j, 0) <= idx]: dev_sem[j] = need; dev_pend.discard(j)
          for j in dev_pend: dev_pend_idx[j] = dev_pend_idx.get(j, idx) - idx
          flush()   # the rest: mapped-region bookkeeping and the stamps of the ranges they name
        else:
          flush(); need = dev_sem.get(k, ctx.sem_value)
      elif ctx.pending: flush()   # nothing here names the range: the queued launches still go now, so the GPU works while the rows come in
      if _completed() < need: ctx._wait(need)
      ctx.write_vram(dst, _mv(src, size), ordered=None)   # concurrently with whatever else runs
      return 0
    s = dev_seen.setdefault(dst, [0, 0]); s[0] += 1; s[1] = max(s[1], size)
    if s[0] >= 3: _dev_register(dst, s[1])   # this push is still ordered
  if size: flush(); ctx.write_vram(dst, _mv(src, size) if ctx.ace.PIPE_MIN and size >= ctx.ace.PIPE_MIN else bytes(_mv(src, size)))
  return 0
def d2h(dst:int, src:int, size:int) -> int:
  if size: flush(); _mv(dst, size)[:] = ctx.read_vram(src, size)
  return 0

def _grid(n:int, per:int=256) -> tuple: return (max(1, min((n + per - 1) // per, 65535)), 1, 1)
def _queue(rec, cap:int=0):
  # cap: issued on a stream that belongs to the active capture (else it runs now, as on CUDA)
  if cap and capture is not None: capture.append(rec)
  else:
    if PUSH_DEBUG: _push_debug(rec)
    refs = _refs(rec[1])
    if refs and "wait_flag" in rec[2]: live_regions.update(refs)
    if refs and MAPPED_SYNC: _order_host_writes(refs)
    if dev_watch: _dev_refs(rec[1], len(ctx.pending) + 1)
    ctx.pending.append(rec); touched.update(refs); pend_refs.update(refs)
    if len(ctx.pending) >= AUTO_FLUSH: flush()   # keep the GPU fed while the host queues more
AUTO_FLUSH = int(os.getenv("TGCU_AUTO_FLUSH", "64"))
# A launch reads a mapped region as the host left it when the launch was made (on CUDA the kernel usually runs at once;
# Strata's prefill rewrites its expert-grouping buffer between launches that read it). So host changes to a region
# that queued or running launches read wait for those launches, and go to the GPU before the new launch is queued.
pend_refs: set[int] = set()          # regions read by ctx.pending
live_regions: set[int] = set()       # regions a graph that spins on a flag reads: the host feeds them while it runs
live_execs: set[int] = set()         # graphs that spin on a flag
live_sem = [0]                       # the last such graph's submission
def _completed() -> int: return struct.unpack("<Q", ctx.ctrl_mv[0:8])[0]
# lazy downloads (tgcudart.cpp): not while a spinning graph may run (its copy would queue behind it), and a fetch waits
# for its own copy only, never for the whole GPU (which may be waiting for this host)
def lazy_ok() -> int: return int(live_sem[0] <= _completed())
def lazy_submit() -> int: flush(); return ctx.sem_value
def lazy_fetch(dst:int, src:int, n:int, sem:int) -> int:
  if _completed() < sem: ctx._wait(sem)
  if ctx.ace.PIPE_MIN and n >= ctx.ace.PIPE_MIN: ctx.ace.d2h_into(src, n, _mv(dst, n))
  else: _mv(dst, n)[:] = ctx.ace.d2h(src, n)
  return 0
ref_sem: dict[int, int] = {}         # region -> the submission whose launches last read it
def _order_host_writes(refs:set[int]):
  for h in refs:
    m = mapped.get(h)
    if m is None or m[3] or _unchanged(h, m[1]): continue
    cur, snap = _host_arr(h, m[1]), shadow[h]
    mask = cur != snap
    busy = h in pend_refs or ref_sem.get(h, 0) > struct.unpack("<Q", ctx.ctrl_mv[0:8])[0]
    # a spinning graph waits for these (a handshake buffer): straight to the mirror
    if h in live_regions: _push_region(h); continue
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1) if not (h in span_ok or (not busy and _span_safe(h))) else np.zeros(0, np.int64)
    # nobody reads the old bytes and one run covers the change: straight to the mirror
    if not busy and len(breaks) == 0: _push_region(h); continue
    if len(breaks) + 1 > MAX_RUNS:                               # scattered: the readers finish, then a merged push
      if h in pend_refs: ctx.flush(); _submitted()
      ctx._wait(ref_sem.get(h, ctx.sem_value)); _push_region(h); continue
    # readers are queued or running: the new bytes go through a staging slot and a copy queued in order, so the
    # earlier launches still see the old bytes and the host does not wait for them
    starts = np.concatenate(([idx[0]], idx[breaks + 1])); ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    runs = list(zip(starts.tolist(), ends.tolist()))
    # one upload for all the runs (each upload is a USB round trip), then a queued copy per run out of it
    pos, offs = 0, []
    for a0, e0 in runs: offs.append(pos); pos += (e0 - a0 + 15) & ~15
    blob = bytearray(pos)
    for (a0, e0), o in zip(runs, offs): blob[o:o + e0 - a0] = _mv(h + a0, e0 - a0)
    base = _stage(bytes(blob))
    for (a0, e0), o in zip(runs, offs):
      ctx.pending.append(ctx.encode(builtin["tgcu_memcpy"], _grid((e0 - a0) // 16 + 1), (256, 1, 1), 0,
                                    struct.pack("<QQQ", m[0] + a0, base + o, e0 - a0)))
      snap[a0:e0] = cur[a0:e0]
    pend_refs.add(h); touched.add(h)
# staging rings for those copies: small runs in the BAR-visible arena, larger ones in VRAM through the copy engine.
# A slot is free again once the copy reading it has run: wrapping submits everything and waits for it.
STAGE_BAR_MAX, STAGE_BAR, STAGE_VRAM = 4096, 1 << 20, 32 << 20
_stg = {}
def _stage(data:bytes) -> int:
  n = (len(data) + 255) & ~255
  small = len(data) <= STAGE_BAR_MAX
  if not _stg:
    bva, boff = ctx.mapped_alloc(STAGE_BAR)
    _stg.update(bar=(bva, boff), bar_head=0, vram=ctx.malloc(STAGE_VRAM), vram_head=0)
  key, cap = ("bar_head", STAGE_BAR) if small else ("vram_head", STAGE_VRAM)
  if n > cap: raise RuntimeError(f"tgcu: a {len(data)}-byte mapped update exceeds the staging ring")
  if _stg[key] + n > cap:                                        # submit what is queued (copies included), drain
    ctx.flush(); _submitted(); ctx._wait(ctx.sem_value); _stg["bar_head"] = _stg["vram_head"] = 0
  at = _stg[key]; _stg[key] = at + n
  if small:
    bva, boff = _stg["bar"]; ctx.mapped_mv[boff + at:boff + at + len(data)] = data; return bva + at
  ctx.ace.h2d(_stg["vram"] + at, data); return _stg["vram"] + at
PUSH_DEBUG = os.getenv("TGCU_PUSH_DEBUG") == "1"
_push_seen: dict = {}
def _push_debug(rec):
  # debug: a region the host changed while launches that read it are still queued (they would see the new bytes)
  queued = {}
  for r in ctx.pending:
    for h in _refs(r[1]): queued.setdefault(h, r[2])
  for h, (dva, sz, aoff, managed) in mapped.items():
    if managed or h not in queued: continue
    diff = np.flatnonzero(_host_arr(h, sz) != shadow[h])
    if len(diff):
      key = (aoff, queued[h][:60], rec[2][:60])
      _push_seen[key] = _push_seen.get(key, 0) + 1
      if _push_seen[key] in (1, 100, 10000):
        print(f"[tgcu] push-debug x{_push_seen[key]}: region off {aoff} size {sz}: host changed bytes [{diff[0]}, {diff[-1]}] "
              f"while queued {queued[h][:70]} reads it; next launch {rec[2][:70]}", flush=True)

# A kernel writes a mapped region only through a pointer it is handed: every 8-byte word of a launch's constant
# buffer (its parameters, structs of pointers included) that points into the mapped arena marks that region, and a
# pull reads only the regions marked since the last one. TGCU_PULL_ALL=1 pulls every region as before.
PULL_ALL = os.getenv("TGCU_PULL_ALL") == "1"
touched: set[int] = set()
_reg_lo = _reg_hi = _reg_h = np.zeros(0, np.uint64)
def _reindex():
  global _reg_lo, _reg_hi, _reg_h
  rs = sorted((m[0], m[0] + m[1], h) for h, m in mapped.items())
  _reg_lo = np.array([r[0] for r in rs], np.uint64); _reg_hi = np.array([r[1] for r in rs], np.uint64)
  _reg_h = np.array([r[2] for r in rs], np.uint64)
def _refs(cb:bytes) -> set[int]:
  if not len(_reg_lo) or len(cb) < 8: return set()
  w = np.frombuffer(cb, np.uint64, len(cb) // 8)
  w = w[(w >= ctx.mapped_va) & (w < ctx.mapped_va + ctx.MAPPED_SIZE)]
  if not len(w): return set()
  i = np.searchsorted(_reg_lo, w, side="right") - 1
  ok = (i >= 0) & (w < _reg_hi[np.maximum(i, 0)])
  return {int(h) for h in _reg_h[i[ok]]}

def d2d(dst:int, src:int, size:int, cap:int=0) -> int:
  if size: _queue(ctx.encode(builtin["tgcu_memcpy"], _grid(size // 16 + 1), (256, 1, 1), 0, struct.pack("<QQQ", dst, src, size)), cap)
  return 0
def d2d_2d(dst:int, dpitch:int, src:int, spitch:int, width:int, height:int, cap:int=0) -> int:
  if width and height:
    _queue(ctx.encode(builtin["tgcu_memcpy2d"], _grid(width * height), (256, 1, 1), 0, struct.pack("<QQQQQQ", dst, dpitch, src, spitch, width, height)), cap)
  return 0
def memset(dst:int, value:int, size:int, cap:int=0) -> int:
  if size: _queue(ctx.encode(builtin["tgcu_memset8"], _grid(size // 4 + 1), (256, 1, 1), 0, struct.pack("<QIIQ", dst, value & 0xff, 0, size)), cap)
  return 0
def memset32(dst:int, value:int, count:int) -> int:
  if count: _queue(ctx.encode(builtin["tgcu_memset32"], _grid(count), (256, 1, 1), 0, struct.pack("<QIIQ", dst, value, 0, count)))
  return 0

# *** mapped host memory: a mirror per region in the BAR-visible arena ***
# small transfers go over BAR1 (~0.2 ms), larger ones through the async copy engine (~1 ms); both work while a graph runs.
# "managed" regions are synced by the program itself (tgcuMappedPush/Pull at its handshake points), the rest
# are pushed before every submission and pulled at every synchronization.
BAR_MAX = int(os.getenv("TGCU_BAR_MAX", "1024"))
STATIC_MAX = int(os.getenv("TGCU_STATIC_MAX", str(1536 << 20)))
static_mirrors: dict[int, tuple[int, int]] = {}  # host -> (VRAM copy, size): large tables the GPU only reads
def map_host(hptr:int, size:int) -> int:
  if size > MAPPED_MAX:
    # a large mapped region (the token embedding table): one VRAM copy at registration, never synced again;
    # beyond STATIC_MAX (an expert arena) there is no device alias and the program takes its no-alias path
    if size > STATIC_MAX: return 0
    dva = ctx.malloc(size)
    t = time.perf_counter()
    for o in range(0, size, 64 << 20): ctx.write_vram(dva + o, bytes(_mv(hptr + o, min(64 << 20, size - o))))
    if os.getenv("TGCU_VERBOSE"): print(f"[tgcu] static mirror {size / 1e6:.0f} MB in {time.perf_counter() - t:.1f} s", flush=True)
    static_mirrors[hptr] = (dva, size)
    return dva
  dva, off = ctx.mapped_alloc(size)
  mapped[hptr] = [dva, size, off, False]
  _reindex()
  _push(hptr, 0, size)
  shadow[hptr] = np.frombuffer(bytes(_mv(hptr, size)), np.uint8).copy()
  return dva
def unmap_host(hptr:int) -> int:
  mapped.pop(hptr, None)  # bump arena: the mirror is not reused
  shadow.pop(hptr, None)
  touched.discard(hptr); _reindex()
  if (m:=static_mirrors.pop(hptr, None)) is not None: ctx.free(m[0])
  return 0
def mapped_manage(hptr:int) -> int:
  if hptr in mapped: mapped[hptr][3] = True
  return 0
def _push(h:int, off:int, n:int):
  dva, size, aoff, _ = mapped[h]
  a, e = off & ~3, min(size, (off + n + 3) & ~3)
  if e - a <= BAR_MAX: ctx.mapped_mv[aoff + a:aoff + e] = bytes(_mv(h + a, e - a))
  else: ctx.ace.h2d(dva + a, bytes(_mv(h + a, e - a)))
def _pull(h:int, off:int, n:int):
  dva, size, aoff, _ = mapped[h]
  a, e = off & ~3, min(size, (off + n + 3) & ~3)
  _mv(h + a, e - a)[:] = ctx.mapped_mv[aoff + a:aoff + e] if e - a <= BAR_MAX else ctx.ace.d2h(dva + a, e - a)
# *** pre-armed handshake (STRATA_HS_PRE; tgcudart.cpp tgcuHsArm / tgcuHsWait / tgcuHsPushPre / tgcuHsPushGo) ***
_hs = {}
def _seg(b:int, off:int, n:int) -> tuple[int, int, int]:   # (region base, aligned start, aligned end) of a host range
    size = mapped[b][1]
    return b, off & ~3, min(size, (off + n + 3) & ~3)
def hs_arm(segs:list, gb:int, goff:int, want:int) -> int:
    ss = [_seg(*s) for s in segs]
    lo = min(mapped[b][2] + a for b, a, e in ss); hi = max(mapped[b][2] + e for b, a, e in ss)
    if hi - lo > ctx.ace.CHUNK: return -2
    ctx.ace.d2h_arm(ctx.mapped_va + lo, hi - lo, mapped[gb][0] + goff, want)
    _hs["rd"] = (lo, ss)
    return 0
def hs_wait(timeout_ms:int) -> int:
    lo, ss = _hs.pop("rd")
    try: data = ctx.ace.d2h_collect(timeout_ms)
    except Exception as e:
        print(f"[tgcu] hs_wait failed: {e!r}", file=sys.stderr, flush=True); return -1
    for b, a, e in ss: _mv(b + a, e - a)[:] = data[mapped[b][2] + a - lo:mapped[b][2] + e - lo]
    return 0
def hs_push_pre(segs:list) -> int:
    ss = [_seg(*s) for s in segs]
    ctx.ace.h2d_pre([(mapped[b][0] + a, e - a) for b, a, e in ss])
    _hs["wr"] = ss
    return 0
def hs_push_pre_next(segs:list, nsegs:list, gb:int, goff:int, want:int) -> int:
    # the rows + flag job AND the next ring's pull (held until hs_release), in one submission
    ss = [_seg(*s) for s in segs]; ns = [_seg(*s) for s in nsegs]
    lo = min(mapped[b][2] + a_ for b, a_, e in ns); hi = max(mapped[b][2] + e for b, a_, e in ns)
    if hi - lo > ctx.ace.CHUNK: return -2
    ctx.ace.h2d_pre([(mapped[b][0] + a_, e - a_) for b, a_, e in ss], (ctx.mapped_va + lo, hi - lo, mapped[gb][0] + goff, want))
    _hs["wr"] = ss; _hs["rd"] = (lo, ns)
    return 0
def hs_release() -> int:
    ctx.ace.d2h_release()
    return 0
def hs_push_go() -> int:
    ss = _hs.pop("wr")
    ctx.ace.h2d_go([_mv(b + a, e - a) for b, a, e in ss])   # views: h2d_go copies each into the wire buffer
    return 0
def mapped_push(h:int, off:int, n:int) -> int: _push(h, off, n); return 0
def mapped_pull(h:int, off:int, n:int) -> int:
  if ctx.pending: flush()   # a host spin on a mapped word must not wait on launches still queued here
  _pull(h, off, n); return 0
# Auto-synced regions keep a snapshot of their last synced bytes, so host and GPU writes merge instead of overwriting:
# a push sends the bytes the host changed; a pull takes the GPU's bytes except where the host changed them since.
shadow: dict[int, np.ndarray] = {}
def _host_arr(h:int, n:int) -> np.ndarray: return np.frombuffer(_mv(h, n), np.uint8)
_libc = ctypes.CDLL(None)
_libc.memcmp.restype = ctypes.c_int; _libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
def _unchanged(h:int, sz:int) -> bool: return _libc.memcmp(h, shadow[h].ctypes.data, sz) == 0   # C speed, no temporaries
def _changed_span(mask:np.ndarray) -> tuple[int, int] | None:
  idx = np.flatnonzero(mask)
  return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else None
MAX_RUNS = 256
gpu_written: set[int] = set()   # regions a pull has seen GPU writes in
clean_pulls: dict[int, int] = {}
span_ok: set[int] = set()       # pulled 8 times without a GPU write: one span per push; the rest exact byte runs
_synced_all = [0]               # the submission all launches had finished by at the last pull
def _span_safe(h:int) -> bool:   # no launch that reads (or writes) this region ran since the last pull
  return h in span_ok or ref_sem.get(h, 0) <= _synced_all[0] and h not in pend_refs
def _push_mapped(exclude:int|None=None):
  # before a submission: every auto region's host changes, ordered after the queued launches that read the old bytes
  if not MAPPED_SYNC: return
  _order_host_writes({h for h, m in mapped.items() if not m[3] and h != exclude})
def _push_region(h:int):
    dva, sz, aoff, managed = mapped[h]
    if _unchanged(h, sz): return
    cur, snap = _host_arr(h, sz), shadow[h]
    mask = cur != snap
    idx = np.flatnonzero(mask)
    if not len(idx): return
    breaks = np.flatnonzero(np.diff(idx) > 1) if not _span_safe(h) else np.zeros(0, np.int64)
    if len(breaks) + 1 <= MAX_RUNS:
      starts = np.concatenate(([idx[0]], idx[breaks + 1])); ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
      for a, e in zip(starts.tolist(), ends.tolist()): _push(h, a, e - a)
    else:
      # many scattered changes: after the GPU is done with this buffer, merge its bytes in and send the whole span
      lo, hi = int(idx[0]) & ~3, min(sz, (int(idx[-1]) + 4) & ~3)
      if ctx.sem_value: ctx._wait(ctx.sem_value)
      dev = np.frombuffer(ctx.ace.d2h(dva + lo, hi - lo), np.uint8)
      merged = np.where(mask[lo:hi], cur[lo:hi], dev)
      _mv(h + lo, hi - lo)[:] = merged.tobytes()
      _push(h, lo, hi - lo)
      snap[lo:hi] = merged
      return
    snap[mask] = cur[mask]
# the mirrors are bump-allocated side by side in the arena: one copy over their span costs one USB round trip
# (~3 ms) where a copy per region cost one each
dev_writes: dict | None = {} if os.getenv("TGCU_PULL_CHECK") == "1" else None   # host -> [pulls, pulls the GPU had changed]
PULL_SPAN_MAX = int(os.getenv("TGCU_PULL_SPAN_MAX", str(4 << 20)))
def _pull_mapped():
  if not MAPPED_SYNC: return
  _synced_all[0] = ctx.sem_value   # every launch so far has finished: after this pull each mirror equals host + shadow
  sel = None if (PULL_ALL or dev_writes is not None) else touched
  auto = [(h, m) for h, m in mapped.items() if not m[3] and (sel is None or h in sel)]
  seen = set(touched); touched.clear()
  if not auto: return
  lo, hi = min(m[2] for _, m in auto), max(m[2] + m[1] for _, m in auto)
  blob = (bytes(ctx.mapped_mv[lo:hi]) if hi - lo <= BAR_MAX else ctx.ace.d2h(ctx.mapped_va + lo, hi - lo)) \
    if hi - lo <= PULL_SPAN_MAX else None
  for h, (dva, sz, aoff, managed) in auto:
    cur, snap = _host_arr(h, sz).copy(), shadow[h]
    if blob is not None: dev = np.frombuffer(blob, np.uint8, sz, aoff - lo)
    else: dev = np.frombuffer(ctx.mapped_mv[aoff:aoff + sz] if sz <= BAR_MAX else ctx.ace.d2h(dva, sz), np.uint8)
    host_changed = cur != snap
    merged = np.where(host_changed, cur, dev)
    if h not in gpu_written:
      if ((dev != snap) & ~host_changed).any(): gpu_written.add(h); span_ok.discard(h)
      else:
        clean_pulls[h] = clean_pulls.get(h, 0) + 1
        if clean_pulls[h] >= 8: span_ok.add(h)
    if dev_writes is not None:
      c = dev_writes.setdefault(h, [0, 0, 0]); gpu = bool(((dev != snap) & ~host_changed).any())
      c[0] += 1; c[1] += gpu; c[2] += gpu and h not in seen    # [2]: GPU writes the pointer scan did not predict
    _mv(h, sz)[:] = merged.tobytes()
    snap[:] = merged
    if host_changed.any():   # the GPU's copy needs the host's newer bytes too
      span = _changed_span(host_changed)
      _push(h, span[0], span[1] - span[0])

# *** modules and launches ***
def module_load(addr:int) -> int:
  head = bytes(_mv(addr, 16))
  blob = bytes(_mv(addr, tgcu.fatbin_size(head)))
  cubins = [c for a, c in tgcu.fatbin_cubins(blob) if a == 86]
  if not cubins: return -1
  m = tgcu.Module(ctx, cubins[0])
  modules.append(m)
  return len(modules) - 1

def kernel_lookup(mod:int, name:str) -> int:
  m = modules[mod]
  k = m.kernels.get(name)
  if k is None:
    key = tgcu.anon_key(name)
    k = next((v for n, v in m.kernels.items() if tgcu.anon_key(n) == key), None)
  if k is None: return -1
  kernels.append(k)
  return len(kernels) - 1

def kernel_params(kid:int) -> list[tuple[int, int]]: return [(off, sz) for _, off, sz in kernels[kid].params]
def kernel_attrs(kid:int) -> tuple[int, int, int, int]:
  k = kernels[kid]
  return k.regs, k.shared, k.stack, k.param_size
def symbol(mod:int, name:str) -> tuple[int, int]:
  g = modules[mod].globals
  if name in g: return g[name]
  key = tgcu.canon(name)
  return next((v for n, v in g.items() if tgcu.canon(n) == key), (0, 0))

SYNC_EACH = os.getenv("TGCU_SYNC_EACH")   # debug: sync after every eager launch to find the one that hangs or faults
_bad_launch: set = set()
def _check_launch(k, g, b, shmem) -> str:
  # the checks the CUDA driver makes before a launch (it fails those with an error; the GPU would misbehave silently)
  thr = b[0] * b[1] * b[2]
  if thr > 1024 or b[0] > 1024 or b[1] > 1024 or b[2] > 64: return f"block {b} ({thr} threads)"
  if g[0] > 0x7fffffff or g[1] > 65535 or g[2] > 65535 or 0 in g or 0 in b: return f"grid {g}"
  if max(k.regs, 1) * ((thr + 31) // 32 * 32) > 65536: return f"{k.regs} registers x {thr} threads"
  if k.shared + shmem > 99 << 10: return f"{k.shared + shmem} bytes of shared memory"
  return ""
def launch(kid:int, gx:int, gy:int, gz:int, bx:int, by:int, bz:int, shmem:int, params:int, psize:int, cap:int=0, coop:int=0) -> int:
  # 0 queued, 1 invalid launch, 2 cooperative grid larger than the co-resident block count (cudaErrorCooperativeLaunchTooLarge)
  k = kernels[kid]
  if (why:=_check_launch(k, (gx, gy, gz), (bx, by, bz), shmem)):
    if (k.name, why) not in _bad_launch:
      _bad_launch.add((k.name, why))
      print(f"[tgcu] invalid launch (CUDA would refuse it): {k.name[:100]}: {why}", flush=True)
    return 1
  prm = bytes(_mv(params, psize)) if psize else b""
  if coop:
    if gx * gy * gz > ctx.coresident_blocks(k, bx * by * bz, shmem): return 2
    if not hasattr(ctx, "grid_ws"): flush()   # encode_cooperative zeroes the workspace with a GPU-ordered write
    _queue(ctx.encode_cooperative(k, (gx, gy, gz), (bx, by, bz), shmem, prm), cap)
  else: _queue(ctx.encode(k, (gx, gy, gz), (bx, by, bz), shmem, prm), cap)
  if SYNC_EACH and not (cap and capture is not None) and "wait_flag" not in k.name:
    _push_mapped(); ctx.flush()
    try: ctx._wait(ctx.sem_value, 5.0)
    except RuntimeError:
      print(f"[tgcu] SYNC_EACH: kernel did not finish: {k.name} grid {(gx, gy, gz)} block {(bx, by, bz)} shmem {shmem} "
            f"regs {k.regs} stack {k.stack} params {psize}", flush=True)
      raise
  return 0

def synchronize() -> int:
  if ctx.pending: _push_mapped()
  ctx.synchronize()
  _submitted()
  _pull_mapped()
  return 0
def flush(exclude:int|None=None) -> int:
  if ctx.pending: _push_mapped(exclude)
  ctx.flush()
  _submitted()
  return 0
def _submitted():
  for h in pend_refs: ref_sem[h] = ctx.sem_value
  pend_refs.clear()
  if dev_pend:
    for k in dev_pend: dev_sem[k] = ctx.sem_value
    dev_pend.clear()
def query() -> int:
  # 1 when all submitted work is done. As on CUDA, a completed query makes the GPU's writes to mapped memory visible:
  # programs spin on mapped words and break out on a successful query (Strata's MTP drafter does).
  done = not ctx.pending and struct.unpack("<Q", ctx.ctrl_mv[0:8])[0] >= ctx.sem_value
  if done: _pull_mapped()
  return int(done)

# *** GPU-timed events (TGCU_GPU_EVENTS=1) ***
def event_record() -> int: return ctx.event_record()   # submits only under TGCU_EVENT_TIMING with nothing queued
def event_elapsed_ns(a:int, b:int) -> int: return ctx.event_time(b) - ctx.event_time(a)

# *** graphs ***
def begin_capture() -> int:
  global capture
  capture = []
  return 0
def end_capture() -> int:
  global capture
  graphs.append(capture or [])
  capture = None
  return len(graphs) - 1
def graph_nodes(g:int) -> int: return len(graphs[g])
GRAPH_DUMP = os.getenv("TGCU_GRAPH_DUMP")
def graph_instantiate(g:int) -> int:
  recs = graphs[g]
  if GRAPH_DUMP:
    names = [r[2] for r in recs]
    short = [re.sub(r"_GLOBAL__N__\w+?_cu_\w{8}", "", n)[:70] for n in names]
    print(f"[tgcu] graph {g}: {len(recs)} launches; first 25:", flush=True)
    for i, n in enumerate(short[:25]): print(f"[tgcu]   {i:4d} {n}", flush=True)
  execs.append(ctx.graph_instantiate(recs) if recs else 0)
  exec_recs[len(execs) - 1] = recs
  exec_refs[len(execs) - 1] = set().union(*(_refs(r[1]) for r in recs)) if recs else set()
  if any("wait_flag" in r[2] for r in recs): live_regions.update(exec_refs[len(execs) - 1]); live_execs.add(len(execs) - 1)
  return len(execs) - 1
exec_recs: dict[int, list] = {}
exec_refs: dict[int, set] = {}
def graph_launch(e:int) -> int:
  if SYNC_EACH and e in exec_recs:   # debug: one launch per submission up to the first spin-wait, the rest as a chain
    recs = exec_recs[e]
    _push_mapped(); ctx.flush()
    for i, r in enumerate(recs):
      if "wait_flag" in r[2]:
        ctx.pending.extend(recs[i:]); ctx.flush(); return 0
      ctx.pending.append(r); ctx.flush()
      try: ctx._wait(ctx.sem_value, 5.0)
      except RuntimeError:
        print(f"[tgcu] SYNC_EACH: graph {e} launch {i} did not finish: {r[2]}", flush=True)
        raise
    return 0
  if execs[e]:
    _push_mapped()
    touched.update(exec_refs[e]); pend_refs.update(exec_refs[e])
    if dev_watch: dev_pend.update(dev_watch)   # a graph names its buffers in its own records: every watched range is assumed read
    ctx.graph_launch(execs[e])
    _submitted()
    if e in live_execs: live_sem[0] = ctx.sem_value
  return 0

def builtin_kernel(name:str) -> int:
  if name not in builtin: return -1
  kernels.append(builtin[name])
  return len(kernels) - 1

def h2d_raw(dst:int, src:int, size:int) -> int:
  # dirty pages of a tracked mapped region: lands before the queued launches, after the submitted ones
  if size: ctx.write_vram(dst, _mv(src, size) if ctx.ace.PIPE_MIN and size >= ctx.ace.PIPE_MIN else bytes(_mv(src, size)), ordered=False)
  return 0

def _hang_hook():   # TGCU_HANG_DUMP=1: the handshake words a spinning graph waits for, host copy vs device mirror
  try:
    print(f"[tgcu HANG] live_regions {len(live_regions)} pend_refs {len(pend_refs)} pending {len(ctx.pending)} ctrl_head {ctx.ctrl_head:#x} ref_sem_max {max(ref_sem.values(), default=0)}", file=sys.stderr, flush=True)
    for h in sorted(live_regions)[:12]:
      m = mapped.get(h)
      if m is None: continue
      n = min(m[1], 64)
      host = struct.unpack(f"<{n // 4}I", bytes(_mv(h, n)))
      try: dev = struct.unpack(f"<{n // 4}I", ctx.ace.d2h(m[0], n))
      except Exception as e: dev = f"d2h failed: {e}"
      print(f"[tgcu HANG] live region {h:#x} size {m[1]} host {host} dev {dev}", file=sys.stderr, flush=True)
  except Exception as e: print(f"[tgcu HANG] hook failed: {e!r}", file=sys.stderr, flush=True)

if os.getenv("TGCU_HANG_DUMP") == "1": tgcu.HANG_HOOK = _hang_hook

if os.getenv("TGCU_PROF"):
  import tgcu_prof
  tgcu_prof.install(globals(), tgcu)
  prof_reset = tgcu_prof.reset
