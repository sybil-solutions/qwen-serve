#!/usr/bin/env python3
"""duplex_lockstep.py - arm an upload AND a download back to back (controls serialized in the main thread), then run bulk OUT and
bulk IN concurrently in two threads. Tests whether the ASM2464 bridge streams both directions at once."""
import os, sys, time, struct, threading, statistics, json
import numpy as np
import tgcu_api as api, tgcu as T
api.init(); ctx = api.ctx; ace0 = ctx.ace
SLOT = ace0.slot_size; round_up = T.round_up; MB = 1e6
dev = ctx.malloc(64 << 20)
rng = np.random.default_rng(3)
N = 8 << 20
pu = rng.integers(0, 256, N, dtype=np.uint8).tobytes(); pd = rng.integers(0, 256, N, dtype=np.uint8).tobytes()
ace0.h2d(dev + (32 << 20), pd)
ace1 = T.AsyncCopy(ctx, engine_type=getattr(T.onv.nv_gpu, os.environ.get("ENG2", "NV2080_ENGINE_TYPE_COPY1")))
base0 = ace0.sram_va
UP_SLOT = 17
res = {}
def log(k, v): res[k] = v; print(k, v, flush=True)
def wait_sem(ac, v):
    t0 = time.perf_counter()
    while ((struct.unpack("<I", ac.mv[ac.sem_off:ac.sem_off + 4])[0] - v) & 0xffffffff) >= 0x80000000:
        if time.perf_counter() - t0 > 5: raise RuntimeError("sem timeout")
def up_prep(ac, dst, data):
    wire = round_up(len(data) + 8, 512)
    ac.gate_seq = (ac.gate_seq + 1) & 0xffffffff
    gate = 0x5A17C0DE00000000 | ac.gate_seq
    buf = bytearray(wire); buf[:len(data)] = data; buf[wire - 8:] = struct.pack("<Q", gate)
    base = base0 + (UP_SLOT - 5) * SLOT
    nvg, nvm = T.onv.nv_gpu, T.onv.nvm
    ac.sem_value += 1; sem = ac.va + ac.sem_off
    cmds = nvm(0, nvg.NVC56F_SEM_ADDR_LO, (base + wire - 8) & 0xffffffff, (base + wire - 8) >> 32, gate & 0xffffffff, gate >> 32,
               T.onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
    cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, base >> 32, base & 0xffffffff, dst >> 32, dst & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, len(data))
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, ac.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    ac._submit(cmds)
    return wire, bytes(buf), ac.sem_value
def arm_up(ac, wire): ac.usb.usb.control_write(0xF2, value=wire // 512, index=UP_SLOT | (((wire + SLOT - 1) // SLOT) << 8))
def down_prep(ac, src, n):
    wire = round_up(5 * SLOT + n, 512)
    nvg, nvm = T.onv.nv_gpu, T.onv.nvm
    ac.sem_value += 1; sem = ac.va + ac.sem_off
    cmds = nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, src >> 32, src & 0xffffffff, base0 >> 32, base0 & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, ac.cq_va >> 32, ac.cq_va & 0xffffffff, 0)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, ac.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, T.onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    return wire, cmds
def arm_down(ac, wire): ac.usb.usb.control_write(0xF2, value=(wire // 512) | 0x8000, index=0 | (((wire + SLOT - 1) // SLOT) << 8))

def run(mode, rb, nrounds):
    """mode: 'seq' = up then down (one at a time), 'both_ud' arm up, arm down, concurrent bulks, 'both_du' reversed arm order"""
    outd = bytearray(); err = []
    ace0.h2d(dev, bytes(N))
    t0 = time.perf_counter()
    for r in range(nrounds):
        uo = r * rb
        wire_u, buf_u, sv = up_prep(ace0, dev + uo, pu[uo:uo + rb])
        wire_d, cmds_d = down_prep(ace1, dev + (32 << 20) + uo, rb)
        got = {}
        def bw():
            try: ace0.usb.usb.bulk_write(buf_u)
            except Exception as e: err.append(("w", repr(e)))
        def br():
            try: got["d"] = bytes(ace0.usb.usb.bulk_read(wire_d, timeout=4000))[5 * SLOT:5 * SLOT + rb]
            except Exception as e: err.append(("r", repr(e)))
        if mode == "seq":
            arm_up(ace0, wire_u); bw(); wait_sem(ace0, sv)
            arm_down(ace1, wire_d); ace1._submit(cmds_d); br(); wait_sem(ace1, ace1.sem_value)
        elif mode in ("stag_ud", "stag_du"):
            # one direction's bulk is already streaming when the other direction is armed (200 us later)
            ta, tb = threading.Thread(target=bw), threading.Thread(target=br)
            if mode == "stag_ud":
                arm_up(ace0, wire_u); ta.start(); time.sleep(0.0002)
                arm_down(ace1, wire_d); ace1._submit(cmds_d); tb.start()
            else:
                arm_down(ace1, wire_d); ace1._submit(cmds_d); tb.start(); time.sleep(0.0002)
                arm_up(ace0, wire_u); ta.start()
            ta.join(); tb.join()
            if err: return None, err
            st = []
            for nm, a, v in (("up", ace0, sv), ("down", ace1, ace1.sem_value)):
                try: wait_sem(a, v); st.append((nm, "ok"))
                except RuntimeError: st.append((nm, "TIMEOUT"))
            print(mode, "round", r, st, "down data exact:", got.get("d") == pd[uo:uo + rb], flush=True)
            if any(x[1] != "ok" for x in st): return None, st
        else:
            if mode == "both_ud": arm_up(ace0, wire_u); arm_down(ace1, wire_d)
            else: arm_down(ace1, wire_d); arm_up(ace0, wire_u)
            ace1._submit(cmds_d)
            ta, tb = threading.Thread(target=bw), threading.Thread(target=br)
            ta.start(); tb.start(); ta.join(); tb.join()
            if err: return None, err
            st = []
            for nm, a, v in (("up", ace0, sv), ("down", ace1, ace1.sem_value)):
                try: wait_sem(a, v); st.append((nm, "ok"))
                except RuntimeError: st.append((nm, "TIMEOUT"))
            print(mode, "round", r, st, "down data exact:", got.get("d") == pd[uo:uo + rb], flush=True)
            if any(x[1] != "ok" for x in st): return None, st
        if err: return None, err
        outd += got["d"]
    wall = time.perf_counter() - t0
    n = nrounds * rb
    ok_d = bytes(outd) == pd[:n]
    ok_u = ace0.d2h(dev, n) == pu[:n]
    return (n, wall, ok_u, ok_d), None

for rb in (128 << 10,):
    for mode in os.environ.get("MODES", "seq,both_ud,both_du").split(","):
        nr = N // rb
        try:
            out, err = run(mode, rb, nr)
        except Exception as e: out, err = None, repr(e)
        if out is None: log(f"{mode}.round{rb>>10}K", f"FAIL {err}"); 
        else:
            n, wall, ok_u, ok_d = out
            log(f"{mode}.round{rb>>10}K", dict(combined_MBps=round(2 * n / wall / MB, 1), exact=(ok_u, ok_d), wall_s=round(wall, 3)))
        if out is None: break
json.dump(res, open(os.path.expanduser("~/shim/duplex_lockstep_res.json"), "w"), indent=1)
