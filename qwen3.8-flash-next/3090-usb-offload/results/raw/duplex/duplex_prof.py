#!/usr/bin/env python3
"""duplex_prof.py - per-step cost of one USB copy round (up / down) at several sizes."""
import os, time, struct, statistics, json
import numpy as np
import tgcu_api as api, tgcu as T
api.init(); ctx = api.ctx; ace = ctx.ace
SLOT = ace.slot_size; rup = T.round_up
dev = ctx.malloc(8 << 20)
nvg, nvm = T.onv.nv_gpu, T.onv.nvm
def pc(): return time.perf_counter()
def med(x): return round(statistics.median(x) * 1e3, 3)
def up_once(n):
    data = os.urandom(n); wire = rup(n + 8, 512)
    ace.gate_seq = (ace.gate_seq + 1) & 0xffffffff; gate = 0x5A17C0DE00000000 | ace.gate_seq
    t0 = pc()
    buf = bytearray(wire); buf[:n] = data; buf[wire - 8:] = struct.pack("<Q", gate); bb = bytes(buf)
    t1 = pc()
    ace.sem_value += 1; sem = ace.va + ace.sem_off
    base = ace.sram_va
    cmds = nvm(0, nvg.NVC56F_SEM_ADDR_LO, (base + wire - 8) & 0xffffffff, (base + wire - 8) >> 32, gate & 0xffffffff, gate >> 32,
               T.onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
    cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, base >> 32, base & 0xffffffff, dev >> 32, dev & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, ace.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    ace._submit(cmds)
    t2 = pc()
    ace.usb.usb.control_write(0xF2, value=wire // 512, index=ace.slot0 | (((wire + SLOT - 1) // SLOT) << 8))
    t3 = pc()
    ace.usb.usb.bulk_write(bb)
    t4 = pc()
    while struct.unpack("<I", ace.mv[ace.sem_off:ace.sem_off + 4])[0] != ace.sem_value: pass
    t5 = pc()
    return (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)
def down_once(n):
    wire = rup(ace.prefix + n, 512)
    t0 = pc()
    ace.usb.usb.control_write(0xF2, value=(wire // 512) | 0x8000, index=0 | (((wire + SLOT - 1) // SLOT) << 8))
    t1 = pc()
    ace.sem_value += 1; sem = ace.va + ace.sem_off
    cmds = nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, dev >> 32, dev & 0xffffffff, ace.sram_va >> 32, ace.sram_va & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, ace.cq_va >> 32, ace.cq_va & 0xffffffff, 0)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, ace.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    ace._submit(cmds)
    t2 = pc()
    while struct.unpack("<I", ace.mv[ace.sem_off:ace.sem_off + 4])[0] != ace.sem_value: pass
    t3 = pc()
    got = ace.usb.usb.bulk_read(wire, timeout=30000)
    t4 = pc()
    b = bytes(got); t5 = pc()
    return (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)
out = {}
for n in (16 << 10, 64 << 10, 128 << 10, 256 << 10, 392 << 10):
    r = [up_once(n) for _ in range(30)]
    cols = list(zip(*r)); out[f"up{n>>10}K"] = dict(pack=med(cols[0]), submit=med(cols[1]), ctrl=med(cols[2]), bulk=med(cols[3]), cewait=med(cols[4]), total=med(cols[5]))
    print(f"up   {n>>10:4d}K pack {med(cols[0])} submit {med(cols[1])} ctrl {med(cols[2])} bulk_out {med(cols[3])} ce_wait {med(cols[4])} total {med(cols[5])} ms", flush=True)
for n in (16 << 10, 64 << 10, 128 << 10, 256 << 10, 392 << 10):
    r = [down_once(n) for _ in range(30)]
    cols = list(zip(*r)); out[f"down{n>>10}K"] = dict(ctrl=med(cols[0]), submit=med(cols[1]), cewait=med(cols[2]), bulk=med(cols[3]), copy=med(cols[4]), total=med(cols[5]))
    print(f"down {n>>10:4d}K ctrl {med(cols[0])} submit {med(cols[1])} ce_to_sram {med(cols[2])} bulk_in {med(cols[3])} bytes() {med(cols[4])} total {med(cols[5])} ms", flush=True)
json.dump(out, open(os.path.expanduser("~/shim/duplex_prof_res.json"), "w"), indent=1)
