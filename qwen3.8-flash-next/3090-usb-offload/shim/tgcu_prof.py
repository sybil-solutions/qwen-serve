# TGCU_PROF=1: inclusive wall time and call counts of the shim's API functions (top level, "api.") and of the driver
# internals they call ("ctx.", "ace."), written to $TGCU_PROF_OUT (default ~/shim/prof.txt) every 5 s. "outside" is
# the wall time no API call was running: the program's own host work.
import os, time, threading, functools
stats: dict[str, list] = {}
SIZED = {"api.h2d": lambda a: a[2], "api.d2h": lambda a: a[2], "api.h2d_raw": lambda a: a[2],
         "ace.h2d": lambda a: len(a[2]), "ace.d2h": lambda a: a[2], "api._stage": lambda a: len(a[0]), "api._push": lambda a: a[2]}   # bytes moved, by argument
depth = threading.local()
t_start = time.perf_counter()
api_busy = [0.0]
OUT = os.path.expanduser(os.getenv("TGCU_PROF_OUT", "~/shim/prof.txt"))

def _wrap(name, fn, top):
  @functools.wraps(fn)
  def w(*a, **k):
    d = getattr(depth, "d", 0); depth.d = d + 1
    t = time.perf_counter()
    try: return fn(*a, **k)
    finally:
      dt = time.perf_counter() - t
      depth.d = d
      s = stats.setdefault(name, [0, 0.0, 0, 0.0, 0]); s[0] += 1; s[1] += dt
      if dt > s[3]: s[3] = dt
      if dt > 0.005: s[4] += 1
      if name in SIZED: s[2] += SIZED[name](a)
      if top and d == 0: api_busy[0] += dt
  return w

def dump():
  wall = time.perf_counter() - t_start
  lines = [f"wall {wall:.1f} s  api {api_busy[0]:.1f} s  outside {wall - api_busy[0]:.1f} s"]
  for n, (c, t, b, mx, slow) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
    lines.append(f"{n:28s} {c:9d} calls {t:9.3f} s {1e3 * t / max(c, 1):8.3f} ms/call  max {1e3 * mx:8.1f} ms  >5ms {slow:6d}" +
                 (f"  {b / 1e6:9.1f} MB {b / max(c, 1) / 1e3:9.1f} KB/call {b / 1e6 / max(t, 1e-9):7.1f} MB/s" if b else ""))
  import sys
  tg = sys.modules.get("tgcu")
  if tg is not None and tg.CB_STATS:
    lines.append("launches by kernel (count, cb0 bytes, cbuf0_init bytes, param base, param size):")
    for n, st in sorted(tg.CB_STATS.items(), key=lambda kv: -kv[1][0] * kv[1][1])[:15]:
      lines.append(f"  {st[0]:7d} x {st[1]:6d} B  init {st[2]:6d}  params {st[3]:#x}+{st[4]}  {n[:90]}")
  api = sys.modules.get("tgcu_api")
  if api is not None:
    dw = api.dev_writes or {}
    regs = sorted((v[2], v[1], v[3], dw.get(h, [0, 0, 0])) for h, v in list(api.mapped.items()))
    lines.append(f"mapped regions {len(regs)} (arena offset, size, managed, pulls, pulls with GPU changes, of those unpredicted):")
    lines += [f"  {o:10d} {sz:10d} {'managed' if m else 'auto':8s} {c[0]:6d} {c[1]:6d} {c[2]:6d}" for o, sz, m, c in regs]
  open(OUT, "w").write("\n".join(lines) + "\n")

def reset():
  global t_start
  stats.clear(); api_busy[0] = 0.0; t_start = time.perf_counter()

def install(api_globals, tgcu):
  skip = {"init", "_mv", "_helper", "_reset_usb", "_recover", "_grid", "_host_arr", "_changed_span"}
  for n, f in list(api_globals.items()):
    if callable(f) and getattr(f, "__module__", None) == api_globals["__name__"] and not isinstance(f, type) and n not in skip:
      api_globals[n] = _wrap("api." + n, f, top=True)
  for cls, pre in ((tgcu.Context, "ctx."), (tgcu.AsyncCopy, "ace.")):
    for n in ("_wait", "flush", "_submit", "write_vram", "read_vram", "encode", "_layout", "graph_launch", "synchronize",
              "h2d", "d2h", "_copy_and_signal", "_pushbuf", "event_record"):
      if n in cls.__dict__: setattr(cls, n, _wrap(pre + n, cls.__dict__[n], top=False))
  def loop():
    while True:
      time.sleep(5)
      try:
        if os.path.exists(OUT + ".reset"): os.unlink(OUT + ".reset"); reset()   # touch prof.txt.reset to zero the counters
        dump()
      except Exception: pass
  threading.Thread(target=loop, daemon=True).start()
