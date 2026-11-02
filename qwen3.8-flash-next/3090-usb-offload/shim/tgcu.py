# tgcu: a minimal CUDA-runtime backend on top of tinygrad's NV device (RTX 3090 behind an ASM2464 USB3 bridge).
# tinygrad boots the GPU (GSP, VA space, channels); this module loads nvcc cubins itself, builds QMDs, chains launches into
# batches and submits them on tinygrad's compute ring. Bulk copies go through tinygrad's USB copy path.
from __future__ import annotations
import struct, re, os, time, array
_WAIT_S = float(os.getenv("TGCU_WAIT_S", "0"))       # diagnostic: raise the GPU-wait timeout (the default is the caller's, 10 s)
HANG_HOOK = None                                    # set by tgcu_api: prints the handshake regions
_HANG_DUMP = os.getenv("TGCU_HANG_DUMP") == "1"      # diagnostic: dump every thread + a native `sample` 8 s into a GPU wait
_FREE_DEFER = os.getenv("TGCU_FREE_DEFER", "0") != "0"   # cudaFree queues the pending work and hands the block back once it has completed, instead of a full synchronize per free (the lazy x shadows of the prompt lane: 6.9 ms x 1 per chunk-layer)
from tinygrad.runtime.support import usb as _usbmod
from dataclasses import dataclass, field
from tinygrad import Device
from tinygrad.device import Buffer, BufferSpec, BufferStorage
from tinygrad.helpers import round_up, data64_le
import tinygrad.runtime.ops_nv as onv

DEBUG = int(os.getenv("TGCU_DEBUG", "0"))

# *** cubin loading ***

SHT_PROGBITS, SHT_SYMTAB, SHT_NOBITS, SHT_REL, SHT_RELA, SHF_ALLOC = 1, 2, 8, 9, 4, 0x2
# .nv.info attribute ids (EIATTR_*), checked against cuobjdump -elf
EI_PARAM_CBANK, EI_KPARAM_INFO, EI_MIN_STACK_SIZE, EI_FRAME_SIZE, EI_MAX_STACK_SIZE, EI_REGCOUNT = 0x0a, 0x17, 0x12, 0x11, 0x23, 0x2f
EI_NUM_BARRIERS = 0x4c

@dataclass
class Section: idx:int; name:str; type:int; flags:int; off:int; size:int; link:int; info:int; align:int; entsize:int; addr:int=-1 # noqa: E702

@dataclass
class Kernel:
  name: str
  module: "Module"
  text_off: int
  text_size: int
  regs: int = 0
  shared: int = 0                      # static shared memory
  params: list = field(default_factory=list)   # [(ordinal, offset, size)]
  param_base: int = 0x160
  param_size: int = 0
  cbuf0_init: bytes = b""
  constbufs: dict = field(default_factory=dict) # bank -> (image offset, size), banks > 0
  stack: int = 0
  barriers: int = 1
  max_dyn_shared: int = 48 << 10

class Module:
  def __init__(self, ctx:"Context", elf:bytes):
    self.ctx, self.elf = ctx, elf
    assert elf[:4] == b"\x7fELF", "not an ELF cubin"
    shoff, = struct.unpack_from("<Q", elf, 0x28); shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3a)
    raw = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize) for i in range(shnum)]
    strtab = raw[shstrndx][4]
    def cstr(o): return elf[o:elf.index(0, o)].decode()
    self.secs = [Section(i, cstr(strtab + r[0]), r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9]) for i, r in enumerate(raw)]
    self.by_name = {s.name: s for s in self.secs}

    # lay out every allocated section (code, constants, globals, nobits) in one image
    img = bytearray()
    for s in self.secs:
      if not (s.flags & SHF_ALLOC) or s.type not in (SHT_PROGBITS, SHT_NOBITS) or s.size == 0: continue
      img += b"\0" * (round_up(len(img), max(s.align, 128)) - len(img))
      s.addr = len(img)
      img += elf[s.off:s.off + s.size] if s.type == SHT_PROGBITS else b"\0" * s.size
    self.image = bytearray(img.ljust(round_up(len(img), 0x1000) + 0x1000, b"\0"))  # slack for instruction prefetch

    symsec = next(s for s in self.secs if s.type == SHT_SYMTAB)
    symstr = self.secs[symsec.link]
    self.syms = []
    for i in range(symsec.size // 24):
      st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", elf, symsec.off + i * 24)
      self.syms.append((cstr(symstr.off + st_name), st_info, st_shndx, st_value, st_size))

    # the image goes to VRAM first, so relocations can use the final addresses
    self.size = len(self.image)
    self.base = ctx.code_alloc(self.size)
    self._relocate()
    ctx.write_vram(self.base, bytes(self.image))

    self.kernels: dict[str, Kernel] = {}
    self.funcs: dict[str, int] = {}  # global functions by address (what other modules may call)
    self.globals: dict[str, tuple[int, int]] = {}  # name -> (device address, size) for __device__/__constant__ variables
    for name, info, shndx, value, size in self.syms:
      if shndx == 0 or shndx >= len(self.secs) or not name: continue
      sec = self.secs[shndx]
      if (info & 0xf) == 1 and sec.addr >= 0: self.globals[name] = (self.base + sec.addr + value, size)  # STT_OBJECT
      if (info & 0xf) == 2 and (info >> 4) in (0, 1) and sec.name == f".text.{name}": self.kernels[name] = self._kernel(name, sec)  # global FUNC
      if (info & 0xf) == 2 and (info >> 4) == 1 and sec.addr >= 0: self.funcs[name] = self.base + sec.addr + value

  def sym_addr(self, symidx:int) -> int:
    name, info, shndx, value, size = self.syms[symidx]
    if shndx == 0:  # a driver-provided function (vprintf, __assertfail): tgcu's own libc module
      if name in self.ctx.externs: return self.ctx.externs[name]
      raise RuntimeError(f"cubin references undefined symbol {name}")
    sec = self.secs[shndx]
    assert sec.addr >= 0, f"symbol {name} in unplaced section {sec.name}"
    return self.base + sec.addr + value

  def _relocate(self):
    for s in self.secs:
      if s.type not in (SHT_REL, SHT_RELA): continue
      tgt = self.secs[s.info]
      if tgt.addr < 0: continue # debug sections
      esz = 24 if s.type == SHT_RELA else 16
      for i in range(s.size // esz):
        r_off, r_info = struct.unpack_from("<QQ", self.elf, s.off + i * esz)
        addend = struct.unpack_from("<q", self.elf, s.off + i * esz + 16)[0] if s.type == SHT_RELA else 0
        typ, S = r_info & 0xffffffff, self.sym_addr(r_info >> 32) + addend
        p = tgt.addr + r_off
        if typ == 1: struct.pack_into("<I", self.image, p, S & 0xffffffff)                            # R_CUDA_32
        elif typ == 2: struct.pack_into("<Q", self.image, p, S)                                      # R_CUDA_64
        elif typ == 0x38: struct.pack_into("<I", self.image, p + 4, S & 0xffffffff)                  # R_CUDA_ABS32_LO_32
        elif typ == 0x39: struct.pack_into("<I", self.image, p + 4, (S >> 32) & 0xffffffff)          # R_CUDA_ABS32_HI_32
        elif typ == 0x3a:                                                                            # R_CUDA_ABS47_34: (S >> 2) at bits [34, 81)
          assert S < (1 << 49) and S % 4 == 0, f"ABS47_34 target {S:#x}"
          w0, w1 = struct.unpack_from("<QQ", self.image, p)
          w0 = (w0 & ~(((1 << 30) - 1) << 34)) | (((S >> 2) & ((1 << 30) - 1)) << 34)
          w1 = (w1 & ~((1 << 17) - 1)) | ((S >> 32) & ((1 << 17) - 1))
          struct.pack_into("<QQ", self.image, p, w0, w1)
        else: raise RuntimeError(f"unsupported CUDA relocation type {typ:#x} in {s.name}")

  def _info(self, sec:Section):
    o, end = sec.off, sec.off + sec.size
    while o < end:
      fmt, attr, sz = struct.unpack_from("<BBH", self.elf, o)
      if fmt == 0x04: yield attr, self.elf[o + 4:o + 4 + sz]; o += 4 + sz  # EIFMT_SVAL: sized payload
      else: yield attr, struct.pack("<H", sz); o += 4                      # EIFMT_NVAL / HVAL: value in the size field

  def _kernel(self, name:str, text:Section) -> Kernel:
    k = Kernel(name, self, text.addr, text.size)
    symidx = next(i for i, s in enumerate(self.syms) if s[0] == name and s[2] == text.idx)
    if (g:=self.by_name.get(".nv.info")) is not None:
      for attr, d in self._info(g):
        if len(d) >= 8 and struct.unpack_from("<I", d)[0] == symidx:
          v = struct.unpack_from("<I", d, 4)[0]
          if attr == EI_REGCOUNT: k.regs = v
          elif attr in (EI_MIN_STACK_SIZE, EI_MAX_STACK_SIZE, EI_FRAME_SIZE): k.stack = max(k.stack, v)
    if (s:=self.by_name.get(f".nv.info.{name}")) is not None:
      for attr, d in self._info(s):
        if attr == EI_KPARAM_INFO:
          _, ordinal, off, flags = struct.unpack_from("<IHHI", d)
          k.params.append((ordinal, off, (flags >> 18) & 0x3fff))
        elif attr == EI_PARAM_CBANK: _, k.param_base, k.param_size = struct.unpack_from("<IHH", d)
        elif attr == EI_NUM_BARRIERS and len(d) == 2: k.barriers = max(1, struct.unpack("<H", d)[0])
    k.params.sort()
    if (s:=self.by_name.get(f".nv.shared.{name}")) is not None: k.shared = s.size
    if (s:=self.by_name.get(f".nv.constant0.{name}")) is not None: k.cbuf0_init = self.elf[s.off:s.off + s.size]
    for s in self.secs:
      if (m:=re.fullmatch(r"\.nv\.constant(\d+)(?:\.(.+))?", s.name)) and int(m.group(1)) != 0 and m.group(2) in (None, name) and s.addr >= 0:
        k.constbufs[int(m.group(1))] = (s.addr, s.size)
    return k

# *** context: memory, submission ***

class Context:
  CTRL_SIZE, CODE_SIZE, MAPPED_SIZE = 32 << 20, 256 << 20, 48 << 20   # BAR1-visible pool is small
  TRANSIENT = int(os.getenv("TGCU_TRANSIENT_BYTES", str(CTRL_SIZE // 2)))   # the transient half of the control arena (test knob: a small ring wraps often)

  def __init__(self):
    self.dev = Device["NV"]
    _ = self.dev.fifos  # brings up the compute and copy channels
    self.fifo = self.dev.fifos["COMPUTE:0"]
    self.iface = self.dev.iface
    self.nvg = onv.nv_gpu
    # cpu-visible control arena: queued QMDs + constant buffers, pushbuffers, semaphores
    self.ctrl = self.iface.alloc(self.CTRL_SIZE, cpu_access=True)
    self.ctrl_va, self.ctrl_mv = self.ctrl.buf, self.ctrl.host
    self.sem_va, self.sem_value = self.ctrl_va, 0
    self.ctrl_mv[0:8] = bytes(8)
    self.ctrl_head = 0x1000
    self.seg_size = ((self.TRANSIENT - 0x1000) // 2) & ~0xff      # the transient ring is two segments used in turn
    self.seg, self.seg_last, self.seg_touched = 0, [0, 0], set()   # current segment; the last submission (semaphore value) that read each; segments the submission being built has written
    self.code = self.iface.alloc(self.CODE_SIZE)
    self.code_va, self.code_head = self.code.buf, 0
    self.allocs: dict[int, tuple[dict, int]] = {}   # va -> (chunk, bytes)
    self.deferred: list[tuple[int, int]] = []       # TGCU_FREE_DEFER: freed blocks (va, submission that last could read them) not yet handed back to the free lists
    self.chunks: list[dict] = []
    self.externs: dict[str, int] = {}  # symbols the driver would resolve, from tgcu_libc.cubin
    self.pending: list = []  # launches not yet submitted
    self.last_launch = None
    self.shmem_window, self.lmem_window = self.dev.shared_mem_window, self.dev.local_mem_window
    self.qmd_sz = round_up(onv.QMD(self.dev).sz * 4, 256)
    self.ace = AsyncCopy(self)
    self.mapped_mem = self.iface.alloc(self.MAPPED_SIZE, cpu_access=True)
    self.mapped_va, self.mapped_mv, self.mapped_head = self.mapped_mem.buf, self.mapped_mem.host, 0
    self.ace.hold_va, self.ace.hold_off = self.mapped_alloc(0x1000)   # the deferred-pull hold word (zero)

  # memory, CUDA-like: 256-byte aligned blocks carved from big chunks (a chunk's tail has a 2 MB guard), so a kernel
  # that reads or writes a little past its buffer touches mapped memory as it would on CUDA instead of faulting
  CHUNK, GUARD, ALIGN = 256 << 20, 2 << 20, 256
  zero_fn = None   # set by tgcu_api once its memset kernel is loaded
  _enc_cache: dict = {}
  def _release(self, va:int):
    ch, sz = self.allocs.pop(va)
    free, off = ch["free"], va - ch["base"]
    free.append((off, sz)); free.sort()
    merged = []
    for o, n in free:  # coalesce neighbours
      if merged and merged[-1][0] + merged[-1][1] == o: merged[-1] = (merged[-1][0], merged[-1][1] + n)
      else: merged.append((o, n))
    ch["free"][:] = merged
  def _reap(self, block:bool=False):   # hand back the deferred blocks whose submission has completed (block: after a full synchronize)
    if not self.deferred: return
    if block: self.synchronize()
    done = struct.unpack("<Q", self.ctrl_mv[0:8])[0]
    keep = []
    for va, sem in self.deferred:
      if sem <= done: self._release(va)
      else: keep.append((va, sem))
    self.deferred = keep
  def malloc(self, size:int) -> int:
    sz = round_up(max(size, 1), self.ALIGN)
    if self.deferred: self._reap()
    for attempt in (0, 1):
      for ch in self.chunks:
        free = ch["free"]
        for i, (off, n) in enumerate(free):
          if n >= sz:
            if n == sz: free.pop(i)
            else: free[i] = (off + sz, n - sz)
            va = ch["base"] + off
            self.allocs[va] = (ch, sz)
            return va
      if attempt == 0 and self.deferred: self._reap(True)   # no room: wait for the freed blocks before a new chunk is carved
      else: break
    csize = max(self.CHUNK, round_up(sz, 2 << 20))
    st = self.iface.alloc(csize + self.GUARD)
    # as from the CUDA driver, fresh device memory reads as zeros (kernels that reset their own counters rely on it)
    # (finished before the block is handed out: copy-engine writes into it, e.g. a graph's launch chain, are not queued)
    if self.zero_fn is not None: self.zero_fn(st.buf, csize + self.GUARD)
    ch = {"st": st, "base": st.buf, "size": csize, "free": [(sz, csize - sz)] if csize > sz else []}
    self.chunks.append(ch)
    self.allocs[st.buf] = (ch, sz)
    return st.buf
  def free(self, va:int):
    if _FREE_DEFER:   # cudaFree without the stall: the work submitted so far is queued now, the block is reusable once it has completed
      self.flush()
      self.deferred.append((va, self.sem_value))
      self._reap()
      return
    self.synchronize()   # cudaFree semantics: the block may be reused right away
    self._release(va)
  def mapped_alloc(self, size:int) -> tuple[int, int]:  # (device va, offset in the BAR-visible mapped arena)
    off = self.mapped_head
    self.mapped_head = round_up(off + size, 0x1000)
    if self.mapped_head > self.MAPPED_SIZE: raise MemoryError("tgcu mapped arena full")
    return self.mapped_va + off, off
  def code_alloc(self, size:int) -> int:
    va, self.code_head = self.code_va + self.code_head, round_up(self.code_head + size, 0x1000)
    assert self.code_head <= self.CODE_SIZE, "code arena full"
    return va
  def _buf(self, va:int, size:int) -> Buffer: # external: tinygrad must never free it
    return Buffer("NV", size, opaque=BufferStorage(va), options=BufferSpec(external_ptr=va))
  def write_vram(self, va:int, data:bytes, ordered:bool|None=True):
    # ordered=True: after everything queued so far. False: before queued (unsubmitted) launches, after submitted work.
    # None: concurrently with running GPU work (a handshake into a graph that is spinning on a flag)
    if not data: return
    if ordered: self.synchronize()
    elif ordered is False and self.sem_value: self._wait(self.sem_value)
    self.ace.h2d(va, data)
  def read_vram(self, va:int, size:int, ordered:bool=True) -> bytes:
    if ordered: self.synchronize()
    return self.ace.d2h(va, size)
  def copy_d2d(self, dst:int, src:int, size:int):  # through the host for now
    self.write_vram(dst, self.read_vram(src, size))

  # launches
  def launch(self, k:Kernel, grid:tuple, block:tuple, dyn_shared:int, params:bytes):
    self.pending.append(self.encode(k, grid, block, dyn_shared, params))

  # cooperative launches (cudaLaunchCooperativeKernel): cooperative_groups grid.sync() reads the grid workspace pointer
  # from the driver words c[0][0x8c] (high 32 bits) and c[0][0x90] (low 32 bits) (sm_86 SASS: null -> BPT.TRAP; then
  # ATOM.ADD.STRONG.GPU on [ws + 4] with a top-bit generation flip, so one zeroed word serves every serialized launch)
  # and needs every block co-resident
  GRID_WS_HI, GRID_WS_LO = 0x8c, 0x90
  def num_sms(self) -> int:
    if not hasattr(self, "_num_sms"):
      nvg, dev = self.nvg, self.dev
      self._num_sms = dev.num_sm_per_tpc * sum(bin(self.iface.rm_control(dev.subdevice, nvg.NV2080_CTRL_CMD_GR_GET_TPC_MASK,
        nvg.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS(gpcId=i)).tpcMask).count("1") for i in range(dev.num_gpcs))
    return self._num_sms
  def coresident_blocks(self, k:Kernel, threads:int, dyn_shared:int=0) -> int:
    warps = (threads + 31) // 32
    by_regs = 65536 // (warps * round_up(max(k.regs, 1) * 32, 256))
    by_smem = (100 << 10) // round_up(0x400 + k.shared + dyn_shared, 128)
    return min(16, self.dev.max_warps_per_sm // warps, by_regs, by_smem) * self.num_sms()
  def encode_cooperative(self, k:Kernel, grid:tuple, block:tuple, dyn_shared:int, params:bytes) -> tuple[onv.QMD, bytes, str]:
    # the record of a cooperative launch: encode() plus the grid workspace pointer (allocated and zeroed on first use,
    # which waits for the GPU: callers with mapped regions in flight push them first)
    blocks, cap = grid[0] * grid[1] * grid[2], self.coresident_blocks(k, block[0] * block[1] * block[2], dyn_shared)
    if blocks > cap: raise RuntimeError(f"{k.name}: cooperative grid {blocks} > {cap} co-resident blocks")
    if not hasattr(self, "grid_ws"):
      self.grid_ws = self.malloc(256)
      self.write_vram(self.grid_ws, bytes(256))
    q, cb0, name = self.encode(k, grid, block, dyn_shared, params)
    cb0 = bytearray(cb0)
    struct.pack_into("<I", cb0, self.GRID_WS_HI, self.grid_ws >> 32)
    struct.pack_into("<I", cb0, self.GRID_WS_LO, self.grid_ws & 0xffffffff)
    return q, bytes(cb0), name
  def launch_cooperative(self, k:Kernel, grid:tuple, block:tuple, dyn_shared:int, params:bytes):
    self.pending.append(self.encode_cooperative(k, grid, block, dyn_shared, params))

  def encode(self, k:Kernel, grid, block, dyn_shared:int, params:bytes) -> tuple[onv.QMD, bytes]:
    # a launch's QMD and constant buffer depend on the kernel and its dimensions only, but for the parameters: built
    # once per shape (a prompt makes ~11,000 launches per chunk), then copied with the parameters dropped in
    self.last_launch = (k.name, grid, block, dyn_shared)
    key = (id(k), grid, block, dyn_shared, self.dev.slm_per_thread)
    if (t:=self._enc_cache.get(key)) is None:
      q, cb0, _ = self._encode(k, grid, block, dyn_shared, b"")
      t = self._enc_cache[key] = (bytes(q.mv), cb0)
    cb0 = bytearray(t[1])
    cb0[k.param_base:k.param_base + len(params)] = params
    if CB_STATS is not None:
      st = CB_STATS.setdefault(k.name, [0, len(cb0), len(k.cbuf0_init), k.param_base, k.param_size]); st[0] += 1
    return onv.QMD(self.dev, blob=bytearray(t[0])), bytes(cb0), k.name
  def _encode(self, k:Kernel, grid, block, dyn_shared:int, params:bytes) -> tuple[onv.QMD, bytes]:
    dev = self.dev
    shmem = round_up(0x400 + k.shared + dyn_shared, 128)
    if shmem > 100 << 10: raise RuntimeError(f"{k.name}: {shmem} bytes of shared memory")
    if k.stack: dev._ensure_has_local_memory(k.stack + 0x240)
    smem_cfg = min(c * 1024 for c in [32, 64, 100] if c * 1024 >= shmem) // 4096 + 1
    cb0_size = round_up(max(len(k.cbuf0_init), k.param_base + k.param_size), 256)
    cb0 = bytearray(k.cbuf0_init.ljust(cb0_size, b"\0"))
    # CUDA driver params: c[0][0x0] blockDim xyz, c[0][0xc] gridDim xyz, then the shared/local memory windows
    struct.pack_into("<6I", cb0, 0, *block, *grid)
    struct.pack_into("<6I", cb0, 24, *data64_le(self.shmem_window), *data64_le(self.lmem_window), *data64_le(0xfffdc0))
    cb0[k.param_base:k.param_base + len(params)] = params
    q = onv.QMD(dev)
    q.write(qmd_major_version=3, sm_global_caching_enable=1, shared_memory_size=shmem, register_count_v=max(k.regs, 1),
      shader_local_memory_high_size=dev.slm_per_thread, qmd_group_id=0x3f, invalidate_texture_header_cache=1,
      invalidate_texture_sampler_cache=1, invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1,
      sampler_index=1, barrier_count=k.barriers, cwd_membar_type=self.nvg.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR,
      constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg, target_sm_config_shared_mem_size=smem_cfg,
      max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(k.text_size >> 8, 0x1ff), sass_version=dev.sass_version)
    q.write(cta_raster_width=grid[0], cta_raster_height=grid[1], cta_raster_depth=grid[2],
            cta_thread_dimension0=block[0], cta_thread_dimension1=block[1], cta_thread_dimension2=block[2])
    prog = k.module.base + k.text_off
    q.write(program_address_lower=prog & 0xffffffff, program_address_upper=prog >> 32,
            program_prefetch_addr_lower_shifted=(prog >> 8) & 0xffffffff, program_prefetch_addr_upper_shifted=prog >> 40)
    q.write(constant_buffer_size_shifted4_0=cb0_size, constant_buffer_valid_0=1)
    for i, (off, sz) in k.constbufs.items():
      a = k.module.base + off
      q.write(**{f"constant_buffer_addr_lower_{i}": a & 0xffffffff, f"constant_buffer_addr_upper_{i}": a >> 32,
                 f"constant_buffer_size_shifted4_{i}": round_up(sz, 16), f"constant_buffer_valid_{i}": 1})
    return q, bytes(cb0), k.name

  def _rec_size(self, r) -> int: return self.qmd_sz + round_up(len(r[1]), 256)
  def _layout(self, recs:list[tuple[onv.QMD, bytes]], base_va:int) -> bytes:
    # launches as [QMD | cbuf0] blocks at base_va, each block as long as its own cbuf0 (256-byte aligned), chained by
    # dependent-QMD pointers
    blob, off = bytearray(), 0
    for i, (q, cb, *_) in enumerate(recs):
      qva, size = base_va + off, self._rec_size((q, cb))
      q.write(constant_buffer_addr_lower_0=(qva + self.qmd_sz) & 0xffffffff, constant_buffer_addr_upper_0=(qva + self.qmd_sz) >> 32)
      if i + 1 < len(recs):
        q.write(dependent_qmd0_pointer=(qva + size) >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)
      else: q.write(dependent_qmd0_enable=0)
      blob += bytes(q.mv).ljust(self.qmd_sz, b"\0") + cb.ljust(size - self.qmd_sz, b"\0")
      off += size
    return bytes(blob)

  VRING_SIZE, BAR_BATCH = 64 << 20, 4096
  def _vring_alloc(self, size:int) -> int:
    # VRAM ring for eager launch batches too big for BAR writes; wrapping waits for everything submitted
    if not hasattr(self, "vring_va"): self.vring_va, self.vring_head = self.malloc(self.VRING_SIZE), 0
    if self.vring_head + size > self.VRING_SIZE:
      self._wait(self.sem_value)
      self.vring_head = 0
    off, self.vring_head = self.vring_head, round_up(self.vring_head + size, 0x100)
    return self.vring_va + off

  def alloc_ctrl(self, size:int) -> int:
    if not hasattr(self, "persist_head"): self.persist_head = self.CTRL_SIZE // 2
    off, self.persist_head = self.persist_head, round_up(self.persist_head + size, 0x100)
    if self.persist_head > self.CTRL_SIZE: raise MemoryError(f"tgcu control arena full ({self.persist_head >> 20} MB of graphs)")
    return off

  def _pushbuf(self, first_qmd:int|None) -> list[int]:
    nvg, nvm = self.nvg, onv.nvm
    cmds = []
    if first_qmd is not None:
      # constants, globals and code may have been written by copies since the last batch
      cmds += nvm(1, nvg.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
                  onv.nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"))
      cmds += nvm(1, nvg.NVC6C0_SEND_PCAS_A, first_qmd >> 8)
      cmds += nvm(1, nvg.NVC6C0_SEND_SIGNALING_PCAS2_B, nvg.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE)
    self.sem_value += 1
    flags = onv.nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit")
    cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, self.sem_va & 0xffffffff, self.sem_va >> 32, self.sem_value & 0xffffffff, self.sem_value >> 32, flags)
    return cmds

  def _transient(self, size:int) -> int:
    # The transient ring is two segments used in turn.  Entering a segment waits only for the submissions that last read it - all queued a whole segment ago, so
    # never for the submission being built (its pushbuffer allocation lands here AFTER _pushbuf took its semaphore value: waiting for `sem_value` was a wait for
    # a release that cannot be queued until this returns, a 10 s GPU timeout whenever a pushbuffer crossed the end of the ring) and never for a graph that
    # still spins on a host handshake.
    off = round_up(self.ctrl_head, 0x100)
    if off + size > 0x1000 + (self.seg + 1) * self.seg_size:
      assert size <= self.seg_size, f"transient allocation of {size} bytes does not fit a {self.seg_size} byte segment"
      nxt = self.seg ^ 1
      self._wait(self.seg_last[nxt])
      self.seg, off = nxt, 0x1000 + nxt * self.seg_size
    self.seg_touched.add(self.seg)
    self.ctrl_head = off + round_up(size, 0x100)
    return off

  def _submit(self, cmds:list[int]):
    pb = array.array("I", cmds).tobytes()
    off = self._transient(len(pb))
    self.ctrl_mv[off:off + len(pb)] = pb
    fifo = self.fifo
    put = fifo.put_value.as_memoryview(allow_zero_copy=True).cast("Q")
    p = put[0]
    entry = (self.ctrl_va + off) | ((len(pb) // 4) << 42) | (1 << 41)
    ring = fifo.ring.host
    ring[(p % fifo.entries) * 8:(p % fifo.entries) * 8 + 8] = struct.pack("<Q", entry)
    put[0] = p + 1
    if _POKE1 & 8:   # GPPut and doorbell: one USB control transfer each instead of control + bulk
      _w32(fifo.gpput.host, 0, (p + 1) % fifo.entries)
      _w32(self.dev.gpu_mmio, 0x90, fifo.token)
    else:
      fifo.gpput.host[0:4] = struct.pack("<I", (p + 1) % fifo.entries)
      self.dev.gpu_mmio[0x90 // 4] = fifo.token # usermode doorbell in BAR0, a PCIe write over USB
    for s in self.seg_touched: self.seg_last[s] = self.sem_value   # this submission (or a later one) is the last to read the segments it wrote
    self.seg_touched.clear()

  def flush(self, extra:list[int]|None=None):
    if not self.pending:
      if extra: self._submit(extra)
      return
    recs, self.pending = self.pending, []
    if SPLIT_BATCH and len(recs) > 1:   # debug: one pushbuffer (and a wait for idle) per launch
      for r in recs[:-1]:
        self.pending = [r]; self.flush()
      recs = recs[-1:]
    size = sum(self._rec_size(r) for r in recs)
    if size <= self.BAR_BATCH:   # a few launches: straight into the BAR-visible arena
      off = self._transient(size)
      first = self.ctrl_va + off
      self.ctrl_mv[off:off + size] = self._layout(recs, first)
    else:                        # a long batch: VRAM ring, through the async copy engine
      first = self._vring_alloc(size)
      self.ace.h2d(first, self._layout(recs, first))
    self._submit(self._pushbuf(first) + (extra or []))

  # GPU-timed events: a semaphore release that also writes the GPU's timer (ns) after all work submitted before it
  EV_SLOTS = 4096
  EVENT_SUBMIT = os.getenv("TGCU_EVENT_TIMING") == "1"   # profiling: a GPU stamp for every event
  # (on the last queued launch's own QMD release when there is one: no submission; else a pushbuffer of its own)
  ev_last = None
  def event_record(self) -> int:
    if not hasattr(self, "ev_off"): self.ev_off, self.ev_next = self.alloc_ctrl(16 * self.EV_SLOTS), 0
    if self.pending and self.ev_last is not None and self.ev_last[0] is self.pending[-1][0]: return self.ev_last[1]
    slot, self.ev_next = self.ev_next, (self.ev_next + 1) % self.EV_SLOTS
    va = self.ctrl_va + self.ev_off + 16 * slot
    if self.pending and self._qmd_timestamp(self.pending[-1][0], va):
      self.ev_last = (self.pending[-1][0], slot)
      return slot
    # nothing queued (or the launch's two releases are taken): host time - a submission per event costs ~0.65 ms
    if self.pending or not self.EVENT_SUBMIT: return -1
    flags = onv.nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit", release_timestamp="en")
    self.flush(onv.nvm(0, self.nvg.NVC56F_SEM_ADDR_LO, va & 0xffffffff, va >> 32, 1, 0, flags))
    return slot
  @staticmethod
  def _qmd_timestamp(q, va:int) -> bool:   # a release (payload + GPU timer) when the grid completes, in a free slot
    for i in range(2):
      if q.read(f"release{i}_enable"): continue
      q.write(**{f"release{i}_address_lower": va & 0xffffffff, f"release{i}_address_upper": va >> 32,
                 f"release{i}_payload_lower": 1, f"release{i}_payload_upper": 0, f"release{i}_enable": 1,
                 f"release{i}_structure_size": 0, f"release{i}_payload64b": 1})
      return True
    return False
  def event_time(self, slot:int) -> int: return struct.unpack_from("<Q", self.ctrl_mv[self.ev_off + 16 * slot + 8:self.ev_off + 16 * slot + 16])[0]

  def _hang_dump(self, value:int):   # TGCU_HANG_DUMP=1 (diagnostic): the Python stacks of every thread, the semaphores, and a native `sample` of this process
    import faulthandler, subprocess, sys
    self.hang_n = getattr(self, "hang_n", 0) + 1
    try:
      ce = struct.unpack("<I", self.ace.mv[self.ace.sem_off:self.ace.sem_off + 4])[0]
      print(f"[tgcu HANG] compute sem {struct.unpack('<Q', self.ctrl_mv[0:8])[0]} want {value} (submitted {self.sem_value}); CE sem {ce} (submitted {self.ace.sem_value}); last {self.last_launch}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] state read failed: {e}", file=sys.stderr, flush=True)
    try:
      off = self.ace.ENTRIES * 8 + getattr(onv.nv_gpu.AmpereAControlGPFifo, 'GPGet').offset
      print(f"[tgcu HANG] CE gpget {struct.unpack_from('<I', self.ace.mv, off)[0]} put {self.ace.put % self.ace.ENTRIES}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] CE gpget failed: {e!r}", file=sys.stderr, flush=True)
    try:
      f = self.fifo; gp = onv.nv_gpu.AmpereAControlGPFifo
      goff = f.gpput.offset - getattr(gp, 'GPPut').offset + getattr(gp, 'GPGet').offset
      print(f"[tgcu HANG] compute gpget {struct.unpack_from('<I', self.dev.gpfifo_buf.host, goff)[0]} gpput {struct.unpack_from('<I', f.gpput.host, 0)[0]} put_value {f.put_value.as_memoryview(allow_zero_copy=True).cast('Q')[0]}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] compute gpget failed: {e!r}", file=sys.stderr, flush=True)
    try:
      t0 = time.perf_counter(); self.ace.d2h(self.ctrl_va, 16)
      print(f"[tgcu HANG] CE probe (d2h 16 B of the control arena) ok in {(time.perf_counter() - t0) * 1e3:.1f} ms", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] CE probe failed: {e!r}", file=sys.stderr, flush=True)
    try:
      nvg, dev = onv.nv_gpu, self.dev
      se = dev.iface.rm_control(dev.debugger, nvg.NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES, nvg.NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS(hTargetChannel=dev.debug_channel, numSMsToRead=100))
      print(f"[tgcu HANG] mmuFault.valid {se.mmuFault.valid}", file=sys.stderr, flush=True)
      if se.mmuFault.valid:
        mm = dev.iface.rm_control(dev.debugger, nvg.NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO, nvg.NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS())
        for i in range(mm.count):
          pf = mm.mmuFaultInfoList[i]
          print(f"[tgcu HANG] MMU fault {i}: addr {pf.faultAddress:#x} type {pf.faultType} access {pf.accessType} engine {getattr(pf, 'faultEngine', '?')}", file=sys.stderr, flush=True)
      for i, e in enumerate(se.smErrorStateArray):
        if e.hwwGlobalEsr or e.hwwWarpEsr: print(f"[tgcu HANG] SM {i}: esr {e.hwwGlobalEsr} warp_esr {e.hwwWarpEsr:#x} pc {e.hwwWarpEsrPc64:#x}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] fault readout failed: {e!r}", file=sys.stderr, flush=True)
    if HANG_HOOK is not None: HANG_HOOK()
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    subprocess.Popen(["/usr/bin/sample", str(os.getpid()), "4", "-file", f"/tmp/tgcu_hang_{os.getpid()}_{self.hang_n}.sample"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  def _wait(self, value:int, timeout_s:float=10.0):
    timeout_s = max(timeout_s, _WAIT_S)
    t0 = time.perf_counter()
    dumped = False
    while struct.unpack("<Q", self.ctrl_mv[0:8])[0] < value:
      el = time.perf_counter() - t0
      if el > 8.0 and not dumped and _HANG_DUMP:
        dumped = True
        self._hang_dump(value)
      if el > timeout_s:
        report = ""
        try: self.dev.on_device_hang()   # raises with the MMU fault / SM error report read through GSP
        except Exception as e: report = str(e)
        raise RuntimeError(f"GPU timeout waiting for semaphore {value} (last {self.last_launch}): {report}")

  def synchronize(self, timeout_s:float=10.0):
    self.flush()
    if self.sem_value: self._wait(self.sem_value, timeout_s)

  # graphs: a fixed chain of launches written once, replayed with one pushbuffer
  def graph_instantiate(self, recs:list[tuple[onv.QMD, bytes]]) -> int:
    # a graph's launch chain lives in plain VRAM, written once through the async copy engine
    first = self.malloc(sum(self._rec_size(r) for r in recs))
    self.ace.h2d(first, self._layout(recs, first))
    return first
  def graph_launch(self, first_qmd:int):
    self.flush()
    self._submit(self._pushbuf(first_qmd))

CB_STATS: dict | None = {} if os.getenv("TGCU_CB_STATS") else None
SPLIT_BATCH = os.getenv("TGCU_SPLIT_BATCH") == "1"

# *** fatbin ***

def fatbin_cubins(blob:bytes) -> list[tuple[int, bytes]]:
  magic, ver, hsz, fsz = struct.unpack_from("<IHHQ", blob)
  assert magic == 0xBA55ED50, f"bad fatbin magic {magic:#x}"
  out, o, end = [], hsz, hsz + fsz
  while o < end:
    kind, _, ehsz, psz = struct.unpack_from("<HHIQ", blob, o)
    arch = struct.unpack_from("<I", blob, o + 28)[0]
    flags = struct.unpack_from("<Q", blob, o + 40)[0] if ehsz >= 48 else 0
    payload = blob[o + ehsz:o + ehsz + psz]
    if kind == 2:
      if payload[:4] != b"\x7fELF": raise RuntimeError(f"compressed fatbin entry (flags {flags:#x}); build with nvcc -no-compress")
      out.append((arch, payload))
    o += ehsz + psz
  return out

def fatbin_size(blob_ptr_bytes:bytes) -> int:
  _, _, hsz, fsz = struct.unpack_from("<IHHQ", blob_ptr_bytes)
  return hsz + fsz

# Host stubs (clang, macOS) and device code (nvcc, Linux) name the same kernel differently in two ways:
#  - anonymous namespaces: clang "12_GLOBAL__N_1", nvcc "41_GLOBAL__N__<hash>_<file>_<hash>"
#  - 64-bit integers: uint64_t/int64_t are unsigned long long/long long on macOS ("y"/"x"), unsigned long/long on Linux ("m"/"l")
# canon() rewrites both into one form; identifiers (length-prefixed) are copied verbatim.
def canon(mangled:str) -> str:
  if not mangled.startswith("_Z"): return mangled
  out, i, n = ["_Z"], 2, len(mangled)
  if mangled.startswith("_ZL") and len(mangled) > 3 and mangled[3].isdigit(): i = 3  # clang marks static kernels _ZL, nvcc does not
  remap = {"y": "m", "x": "l"}
  while i < n:
    c = mangled[i]
    if c.isdigit():
      j = i
      while j < n and mangled[j].isdigit(): j += 1
      k = int(mangled[i:j])
      ident = mangled[j:j + k]
      out.append("@A@" if ident.startswith("_GLOBAL__N_") else f"{k}{ident}")
      i = j + k
    elif c in "ST" and i + 1 < n and (mangled[i + 1] == "_" or mangled[i + 1].isdigit() or mangled[i + 1].isupper()):
      j = i + 1                                              # back-reference S<seq-id>_ / template parameter T<seq-id>_
      while j < n and mangled[j] != "_": j += 1
      out.append(mangled[i:j + 1]); i = j + 1
    elif c == "L" and i + 1 < n and mangled[i + 1] != "_":  # template literal: L <type> <value> E
      j = i + 1
      t = mangled[j]; out.append("L" + remap.get(t, t)); j += 1
      while j < n and mangled[j] != "E": out.append(mangled[j]); j += 1
      i = j
    else:
      out.append(remap.get(c, c)); i += 1
  return "".join(out)

def anon_key(mangled:str) -> str: return canon(mangled)

# *** async copy engine: a channel group of its own, so copies run while a compute graph spins on a host flag ***

_PROF = bool(os.getenv("TGCU_PROF"))
_POKE1 = int(os.getenv("TGCU_POKE1", "0"))   # bit 0: GPPut, 1: doorbell, 2: hold word as ONE USB control transfer instead of control + bulk (the stream write)
def _w32(mmio, off:int, val:int):
  a = mmio.addr + off
  mmio.usb.pcie_request(0x60 if a >> 32 else 0x40, a, val & 0xffffffff)   # 3DW header for a 32-bit address, 4DW above
_PROF_ACC = [[0.0, 0.0, 0, 0.0, 0.0], [0.0, 0.0, 0, 0.0, 0.0]]   # [h2d, d2h]: submit s, ce-wait s, calls, control s, bulk s
_SUB_ACC = [[0.0, 0.0, 0.0, 0.0, 0.0]]
def _prof_dump():
  import sys
  for name, p in zip(("h2d", "d2h"), _PROF_ACC):
    if p[2]: print(f"[tgcu prof] {name}: n={p[2]} control {p[3]/p[2]*1e3:.3f} ms, bulk {p[4]/p[2]*1e3:.3f} ms, submit {p[0]/p[2]*1e3:.3f} ms, ce-wait {p[1]/p[2]*1e3:.3f} ms", file=sys.stderr, flush=True)
    for i in range(5): p[i] = 0 if i == 2 else 0.0

class AsyncCopy:
  # host <-> bridge SRAM over USB bulk (0xF2), SRAM <-> VRAM with a copy engine outside the graphics channel group
  ENTRIES, CHUNK = 0x400, 0x40000   # gpfifo entries; bytes per SRAM round (slots 5.. of the staging window)

  def __init__(self, ctx:"Context", engine_type:int|None=None):
    self.ctx, dev, iface, nvg = ctx, ctx.dev, ctx.iface, onv.nv_gpu
    # COPY0 in a channel group of its own: time-sliced against the graphics group (the async CE runlists do not start here)
    self.engine = engine_type if engine_type is not None else nvg.NV2080_ENGINE_TYPE_COPY0
    self.usb = dev.iface.pci_dev.usb                       # the ASM2464 controller
    self.sram_va = dev.iface.ctrl.get_storage().buf + 0x5000  # GPU VA of the staging window (TRANSFER_PADDR)
    self.cq_va = dev.iface.ctrl.get_storage().buf + 0x100c    # the bridge's completion queue word (usb_cq)
    self.slot0, self.slot_size = iface.TRANSFER_START_SLOT, iface.SLOT_SIZE
    self.prefix = iface.TRANSFER_PADDR - iface.SRAM_PADDR
    self.CHUNK = min(self.CHUNK, iface.TRANSFER_SIZE)
    self.iface_transfer_size = iface.TRANSFER_SIZE          # the whole SRAM staging window, h2d rounds
    # ring + USERD + pushbuffers in one cpu-visible allocation
    self.mem = iface.alloc(0x100000, contiguous=True, cpu_access=True)
    self.va, self.mv, self.paddr = self.mem.buf, self.mem.host, self.mem.meta.hMemory
    self.mv[0:0x10000] = bytes(0x10000)
    self.notifier = iface.alloc(0x1000, uncached=True)
    # a channel group of its own on the copy engine, sharing the device's VA space, like tinygrad's graphics group
    self.group = iface.rm_alloc(dev.nvdevice, nvg.KEPLER_CHANNEL_GROUP_A, nvg.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=self.engine))
    self.ctxshare = iface.rm_alloc(self.group, nvg.FERMI_CONTEXT_SHARE_A,
      nvg.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=dev.vaspace, flags=nvg.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC))
    params = nvg.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=self.va, gpFifoEntries=self.ENTRIES,
      hObjectError=self.notifier.meta.hMemory, hObjectBuffer=self.paddr, hUserdMemory=(onv.ctypes.c_uint32*8)(self.paddr),
      userdOffset=(onv.ctypes.c_uint64*8)(self.ENTRIES * 8), engineType=self.engine, hContextShare=self.ctxshare)
    self.chan = iface.rm_alloc(self.group, iface.gpfifo_class, params)
    iface.rm_alloc(self.chan, iface.dma_class)
    iface.rm_control(self.group, nvg.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nvg.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
    self.token = iface.rm_control(self.chan, nvg.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nvg.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)).workSubmitToken
    self.gpput_off = self.ENTRIES * 8 + getattr(nvg.AmpereAControlGPFifo, 'GPPut').offset
    self.put, self.pb_head = 0, 0x10000
    self.sem_off = 0x8000                                   # semaphore word inside the same allocation
    self.sem_value = 0
    self.gate_seq = 0
    self.hold_va, self.hold_off, self.hold_tok = 0, 0, 0   # set by Context once its mapped arena exists
    self._submit(onv.nvm(4, nvg.NVC6C0_SET_OBJECT, iface.dma_class))
    # TGCU_TS_G / TGCU_TS_C (microseconds): the graphics/compute group's and the copy group's RM time slice (both 1024 by default).  A short compute slice lets the copy
    # engine run at full rate while a kernel runs (prompt lane: the y push beside the next chunk's front)
    for _grp, _ev in ((dev.channel_group, "TGCU_TS_G"), (self.group, "TGCU_TS_C")):
      if os.getenv(_ev): iface.rm_control(_grp, nvg.NVA06C_CTRL_CMD_SET_TIMESLICE, nvg.NVA06C_CTRL_SET_TIMESLICE_PARAMS(timesliceUs=int(os.environ[_ev])))

  def _submit(self, cmds:list[int]):
    pb = array.array("I", cmds).tobytes()
    if self.pb_head + len(pb) > 0x100000: self.pb_head = 0x10000  # callers wait on each submission, so the ring is free
    off, self.pb_head = self.pb_head, round_up(self.pb_head + len(pb), 0x100)
    _s0 = time.perf_counter()
    self.mv[off:off + len(pb)] = pb
    _s1 = time.perf_counter()
    entry = (self.va + off) | ((len(pb) // 4) << 42) | (1 << 41)
    slot = self.put % self.ENTRIES
    self.mv[slot * 8:slot * 8 + 8] = struct.pack("<Q", entry)
    _s2 = time.perf_counter()
    self.put += 1
    if _POKE1 & 1: _w32(self.mv, self.gpput_off, self.put % self.ENTRIES)
    else: self.mv[self.gpput_off:self.gpput_off + 4] = struct.pack("<I", self.put % self.ENTRIES)
    _s3 = time.perf_counter()
    if _POKE1 & 2: _w32(self.ctx.dev.gpu_mmio, 0x90, self.token)
    else: self.ctx.dev.gpu_mmio[0x90 // 4] = self.token
    if _PROF:
      q = _SUB_ACC[0]; q[0] += _s1 - _s0; q[1] += _s2 - _s1; q[2] += _s3 - _s2; q[3] += time.perf_counter() - _s3; q[4] += 1
      if q[4] % 400 == 0:
        import sys; print(f"[tgcu prof] submit parts (ms, avg of {int(q[4])}): pushbuf {q[0]/q[4]*1e3:.3f} ring {q[1]/q[4]*1e3:.3f} gpput {q[2]/q[4]*1e3:.3f} doorbell {q[3]/q[4]*1e3:.3f}", file=sys.stderr, flush=True)
        for i in range(5): q[i] = 0.0

  def _copy_and_signal(self, dst:int, src:int, n:int, ring_cq:bool=False, gate:tuple[int, int]|None=None, between=None):
    # gate = (sram address, 64-bit value): the copy engine channel first acquires that word (equal), so the job can be queued
    # before the payload exists; the payload's own last bytes release it. between() runs after the job is queued.
    nvg, nvm = onv.nv_gpu, onv.nvm
    self.sem_value += 1
    sem = self.va + self.sem_off
    cmds = []
    if gate is not None:
      ga, gv = gate
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, ga & 0xffffffff, ga >> 32, gv & 0xffffffff, gv >> 32,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
    cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, src >> 32, src & 0xffffffff, dst >> 32, dst & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                         src_memory_layout="pitch", dst_memory_layout="pitch"))
    if ring_cq:  # a download: the data is in SRAM, the bridge may now stream the armed bulk IN
      cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, self.cq_va >> 32, self.cq_va & 0xffffffff, 0)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    _t1 = time.perf_counter()
    self._submit(cmds)
    _t2 = time.perf_counter()
    if between is not None: between()
    t0 = time.perf_counter()
    while struct.unpack("<I", self.mv[self.sem_off:self.sem_off + 4])[0] != self.sem_value:
      if time.perf_counter() - t0 > 5: raise RuntimeError("async copy timeout")
    if _PROF:
      _t3 = time.perf_counter(); p = _PROF_ACC[1 if ring_cq else 0]
      p[0] += _t2 - _t1; p[1] += _t3 - _t2; p[2] += 1

  # ---- pipelined bulk transfers (PrefillPipeline 2026-10-12). Measured on the idle link (duplex_prof.py): one 392 KB upload round
  # costs 1.69 ms = CE-job submit 0.65 (4 BAR pokes) + arm 0.08 + bulk OUT 0.44 + CE wait 0.50; the stock loop pays all four serially per round.
  # Here ONE pushbuffer holds the copy-engine jobs of the whole message (one doorbell), and the uploads alternate between two SRAM
  # regions, so the copy engine drains region k while the bulk OUT fills the other. Downloads queue one hold-gated job per round
  # (a single doorbell for the message; the host releases a job with one poke after the arm). TGCU_PIPE_MIN (bytes, default 1 MiB; 0 = off).
  PIPE_MIN = int(os.getenv("TGCU_PIPE_MIN", str(1 << 20)))
  PIPE_SLOTS = int(os.getenv("TGCU_PIPE_SLOTS", "12"))     # slots per upload region (2 regions of 12 of the 25 transfer slots)
  PIPE_DOWN = int(os.getenv("TGCU_PIPE_DOWN", str(400 << 10)))   # download payload bytes per round (the SRAM read starts at slot 0: wire = prefix + payload)

  def _wait_sem(self, value:int, timeout_s:float=5.0):
    t0 = time.perf_counter()
    while ((struct.unpack("<I", self.mv[self.sem_off:self.sem_off + 4])[0] - value) & 0xffffffff) >= 0x80000000:
      if time.perf_counter() - t0 > timeout_s: raise RuntimeError("async copy timeout (pipelined)")

  def h2d(self, dst:int, data:bytes):
    if self.PIPE_MIN and len(data) >= self.PIPE_MIN: return self._h2d_pipe(dst, data)
    return self._h2d_rounds(dst, data)

  def d2h(self, src:int, n:int) -> bytes:
    if self.PIPE_MIN and n >= self.PIPE_MIN: return self._d2h_pipe(src, n)
    return self._d2h_rounds(src, n)

  def _h2d_pipe(self, dst:int, data:bytes):
    nvg, nvm = onv.nv_gpu, onv.nvm
    SL = self.slot_size
    cap = self.PIPE_SLOTS * SL                              # bytes per region
    rb = cap - 512                                          # payload per round (wire == cap)
    n = len(data)
    rounds = (n + rb - 1) // rb
    mv = memoryview(data) if not isinstance(data, memoryview) else data
    sem = self.va + self.sem_off
    cmds, plan = [], []
    for i in range(rounds):
      o = i * rb
      m = min(rb, n - o)
      wire = round_up(m + 8, 512)
      reg = i & 1
      base = self.sram_va + reg * cap                       # region 0 = slots 5.., region 1 = 5 + PIPE_SLOTS..
      self.gate_seq = (self.gate_seq + 1) & 0xffffffff
      gate = 0x5A17C0DE00000000 | self.gate_seq
      self.sem_value += 1
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, (base + wire - 8) & 0xffffffff, (base + wire - 8) >> 32, gate & 0xffffffff, gate >> 32,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
      d = dst + o
      cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, base >> 32, base & 0xffffffff, d >> 32, d & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, m)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                           src_memory_layout="pitch", dst_memory_layout="pitch"))
      cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
      plan.append((o, m, wire, reg, gate, self.sem_value))
    self._submit(cmds)                                      # the jobs wait on their sentinels; the pushbuffer is free again once the last one has run
    usb = self.usb.usb
    ub = self.usb.usb
    for i, (o, m, wire, reg, gate, sv) in enumerate(plan):
      if i >= 2: self._wait_sem(plan[i - 2][5])             # this region's previous job has drained it
      if wire > len(ub._bulk_mv): ub._bulk_buf, ub._bulk_mv = _usbmod.alloc_cbuffer(wire)
      ub._bulk_mv[:m] = mv[o:o + m]                         # straight into libusb's buffer (stock bulk_write copies twice more)
      ub._bulk_mv[wire - 8:wire] = struct.pack("<Q", gate)
      ub.control_write(0xF2, value=wire // 512, index=(self.slot0 + reg * self.PIPE_SLOTS) | (((wire + SL - 1) // SL) << 8))
      _usbmod.checked(_usbmod.libusb.libusb_bulk_transfer, "bulk OUT 0x02 failed")(ub.handle, 0x02, ub._bulk_buf, wire, ub._transferred, 1000)
      assert ub._transferred.value == wire, f"bulk OUT short write: {ub._transferred.value}/{wire} bytes"
    self._wait_sem(plan[-1][5])

  def _d2h_pipe(self, src:int, n:int, into=None) -> bytes|None:
    nvg, nvm = onv.nv_gpu, onv.nvm
    SL = self.slot_size
    R = min(self.PIPE_DOWN, self.iface_transfer_size)
    rounds = (n + R - 1) // R
    sem, ha = self.va + self.sem_off, self.hold_va
    base_tok = self.hold_tok
    cmds, plan = [], []
    for i in range(rounds):
      o = i * R
      m = min(R, n - o)
      wire = round_up(self.prefix + m, 512)
      self.sem_value += 1
      tok = base_tok + 1 + i
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, ha & 0xffffffff, ha >> 32, tok & 0xffffffff, 0,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acq_strict_geq", payload_size="32bit"))   # released by the host after its arm
      s = src + o
      cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, s >> 32, s & 0xffffffff, self.sram_va >> 32, self.sram_va & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, m)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                           src_memory_layout="pitch", dst_memory_layout="pitch"))
      cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, self.cq_va >> 32, self.cq_va & 0xffffffff, 0)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
      cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
      plan.append((o, m, wire, tok, self.sem_value))
    self.hold_tok = base_tok + rounds
    self._submit(cmds)
    usb = self.usb.usb
    hmv = self.ctx.mapped_mv
    out = into if into is not None else bytearray(n)
    for o, m, wire, tok, sv in plan:
      usb.control_write(0xF2, value=(wire // 512) | 0x8000, index=0 | (((wire + SL - 1) // SL) << 8))
      hmv[self.hold_off:self.hold_off + 4] = struct.pack("<I", tok & 0xffffffff)
      got = usb.bulk_read(wire, timeout=30000)
      out[o:o + m] = got[self.prefix:self.prefix + m]
    self._wait_sem(plan[-1][4])
    return bytes(out) if into is None else None

  def d2h_into(self, src:int, n:int, into):   # download n bytes straight into a writable buffer (one copy fewer than d2h)
    self._d2h_pipe(src, n, into)

  def _h2d_rounds(self, dst:int, data:bytes):
    # tinygrad's upload order: the copy engine job is queued FIRST, gated by an acquire on a sentinel (the last 8 bytes of the
    # bulk payload), then the bridge is armed (0xF2) and the bulk OUT sent; the payload releases the copy engine, so nothing
    # is written to the BAR after the payload moves. The arm MUST follow the submission: a BAR write between the arm and
    # the bulk OUT drops the transfer (measured: the gate never opens). Measured against the old order (payload, then
    # submit): same cost idle, 1.45 vs 2.11 ms for 256 KB while a kernel runs, -10..12% decode ms/window in the split run.
    step = self.iface_transfer_size - 512                  # payload bytes per round: the whole SRAM window minus the gate block
    for o in range(0, len(data), step):
      chunk = data[o:o + step]
      wire = round_up(len(chunk) + 8, 512)
      self.gate_seq = (self.gate_seq + 1) & 0xffffffff
      gate = 0x5A17C0DE00000000 | self.gate_seq
      buf = bytearray(wire)
      buf[:len(chunk)] = chunk
      buf[wire - 8:] = struct.pack("<Q", gate)
      _t = [0.0, 0.0, 0.0]
      def send():
        _t[0] = time.perf_counter()
        self.usb.usb.control_write(0xF2, value=wire // 512, index=self.slot0 | (((wire + self.slot_size - 1) // self.slot_size) << 8))
        _t[1] = time.perf_counter(); self.usb.usb.bulk_write(bytes(buf)); _t[2] = time.perf_counter()
      self._copy_and_signal(dst + o, self.sram_va, len(chunk), gate=(self.sram_va + wire - 8, gate), between=send)
      if _PROF:
        p = _PROF_ACC[0]; p[3] += _t[1] - _t[0]; p[4] += _t[2] - _t[1]
        if p[2] % 200 == 0 and p[2]: _prof_dump()

  def _d2h_rounds(self, src:int, n:int) -> bytes:
    out = bytearray()
    for o in range(0, n, self.CHUNK):
      m = min(self.CHUNK, n - o)
      wire = round_up(self.prefix + m, 512)
      # arm the read first; the copy engine fills SRAM and rings the completion queue, then the bulk IN streams it
      _ta = time.perf_counter()
      self.usb.usb.control_write(0xF2, value=(wire // 512) | 0x8000, index=0 | (((wire + self.slot_size - 1) // self.slot_size) << 8))
      _tb = time.perf_counter()
      self._copy_and_signal(self.sram_va, src + o, m, ring_cq=True)
      _tc = time.perf_counter()
      out += bytes(self.usb.usb.bulk_read(wire, timeout=30000))[self.prefix:self.prefix + m]
      if _PROF:
        p = _PROF_ACC[1]; p[3] += _tb - _ta; p[4] += time.perf_counter() - _tc
        if p[2] % 200 == 0 and p[2]: _prof_dump()
      pass
    return bytes(out)

  # ---- pre-armed handshake transfers (STRATA_HS_PRE): the copy engine pulls the doorbell payload the moment the GPU raises
  # the ring word, with no host poll / arm / submit after the ring; the y push + flag is one gated job submitted before the
  # CPU computes, so only the arm and the bulk OUT are left once the rows exist.
  def _d2h_cmds(self, src:int, n:int, gate_va:int, want:int, hold:int|None=None) -> list[int]:
    nvg, nvm = onv.nv_gpu, onv.nvm
    self.sem_value += 1
    sem = self.va + self.sem_off
    cmds = []
    if hold is not None:   # the host releases this job (after its own USB arm) by writing `hold` to the word; keeps the cq ring behind the arm
      ha = self.hold_va
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, ha & 0xffffffff, ha >> 32, hold & 0xffffffff, 0,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="32bit"))
    cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, gate_va & 0xffffffff, gate_va >> 32, want & 0xffffffff, 0,
                onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acq_strict_geq", payload_size="32bit"))
    cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, src >> 32, src & 0xffffffff, self.sram_va >> 32, self.sram_va & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                         src_memory_layout="pitch", dst_memory_layout="pitch"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, self.cq_va >> 32, self.cq_va & 0xffffffff, 0)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    return cmds

  def _rd_arm(self, wire:int):
    self.usb.usb.control_write(0xF2, value=(wire // 512) | 0x8000, index=0 | (((wire + self.slot_size - 1) // self.slot_size) << 8))

  def d2h_arm(self, src:int, n:int, gate_va:int, want:int):
    assert n <= self.CHUNK
    wire = round_up(self.prefix + n, 512)
    self._rd_arm(wire)
    self._submit(self._d2h_cmds(src, n, gate_va, want))
    self._rd = (wire, n)

  def d2h_release(self):
    # arm the bulk IN, then open the hold word: the queued (deferred) pull may now ring the bridge
    wire, n = self._rd
    self._rd_arm(wire)
    self.hold_tok += 1
    if _POKE1 & 4: _w32(self.ctx.mapped_mv, self.hold_off, self.hold_tok)
    else: self.ctx.mapped_mv[self.hold_off:self.hold_off + 4] = struct.pack("<I", self.hold_tok)

  def d2h_collect(self, timeout:int=20000) -> memoryview:
    wire, n = self._rd
    return self.usb.usb.bulk_read(wire, timeout=timeout)[self.prefix:self.prefix + n]   # a view of the USB buffer, valid until the next bulk transfer: the caller copies it out at once

  def h2d_pre(self, dsts:list[tuple[int, int]], nxt:tuple|None=None) -> list[int]:
    # dsts = [(vram va, nbytes multiple of 4)]; the payload is their concatenation. Queue ONE gated job (acquire on the sentinel,
    # then one copy per destination, ordered by a flushing release in between); returns each part's offset in the payload.
    nvg, nvm = onv.nv_gpu, onv.nvm
    pos, o = [], 0
    for _, n in dsts: pos.append(o); o += n
    assert o <= self.iface_transfer_size - 512
    wire = round_up(o + 8, 512)
    self.gate_seq = (self.gate_seq + 1) & 0xffffffff
    gate = 0x5A17C0DE00000000 | self.gate_seq
    self.sem_value += 1
    sem = self.va + self.sem_off
    cmds = nvm(0, nvg.NVC56F_SEM_ADDR_LO, (self.sram_va + wire - 8) & 0xffffffff, (self.sram_va + wire - 8) >> 32, gate & 0xffffffff, gate >> 32,
               onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
    for idx, ((dva, n), p) in enumerate(zip(dsts, pos)):
      src = self.sram_va + p
      cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, src >> 32, src & 0xffffffff, dva >> 32, dva & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                           src_memory_layout="pitch", dst_memory_layout="pitch"))
      if idx + 1 < len(dsts):   # the next copy (the flag) must not pass this one (the rows)
        d = sem + 8
        cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, d >> 32, d & 0xffffffff, idx + 1)
        cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    if nxt is not None:   # (src va, nbytes, seq gate va, want): the next ring's pull, held until d2h_release()
      src, n, gva, want = nxt
      assert n <= self.CHUNK
      cmds += self._d2h_cmds(src, n, gva, want, hold=self.hold_tok + 1)
      self._rd = (round_up(self.prefix + n, 512), n)
    self._submit(cmds)
    self._wr = (wire, gate, o)
    return pos

  def h2d_go(self, parts:list[bytes]):
    wire, gate, o = self._wr
    buf = bytearray(wire)
    p = 0
    for b in parts: buf[p:p + len(b)] = b; p += len(b)
    buf[wire - 8:] = struct.pack("<Q", gate)
    self.usb.usb.control_write(0xF2, value=wire // 512, index=self.slot0 | (((wire + self.slot_size - 1) // self.slot_size) << 8))
    self.usb.usb.bulk_write(buf)   # a bytearray: bulk_write copies it into its C buffer, no extra bytes() copy

# *** TGCU_HSPROF=1 (diagnostic, HostTailAndPrefill 2026-10-12): per-call wall time of the handshake path + USB transfer counts, printed every 480 rings ***
if os.getenv("TGCU_HSPROF"):
  import sys as _sys
  _HP: dict[str, list] = {}
  _ring_n = [0]
  def _wrap(obj, name, label=None):
    fn = getattr(obj, name); lab = label or name
    def w(*a, **k):
      t = time.perf_counter()
      try: return fn(*a, **k)
      finally:
        e = _HP.setdefault(lab, [0.0, 0]); e[0] += time.perf_counter() - t; e[1] += 1
    setattr(obj, name, w)
  for _n in ("h2d_pre", "h2d_go", "d2h_release", "d2h_arm", "_submit", "_rd_arm"): _wrap(AsyncCopy, _n)
  def _collect(self, timeout=20000, _f=AsyncCopy.d2h_collect):
    t = time.perf_counter()
    try: return _f(self, timeout)
    finally:
      e = _HP.setdefault("d2h_collect", [0.0, 0]); e[0] += time.perf_counter() - t; e[1] += 1
      _ring_n[0] += 1
      if _ring_n[0] % 480 == 0:
        print("[tgcu hsprof] per call ms (n): " + " | ".join(f"{k} {v[0]/v[1]*1e3:.3f} ({v[1]})" for k, v in sorted(_HP.items())), file=_sys.stderr, flush=True)
        _HP.clear()
  AsyncCopy.d2h_collect = _collect
  for _n in ("control_write", "bulk_write", "bulk_read", "control_read"): _wrap(_usbmod.USB3, _n, "usb." + _n)
