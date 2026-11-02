"""levers_cal.py - moetier levers table CALIBRATED to the first full-512 end-to-end run (F1, 2026-10-11) of the 3090-over-USB + Mac mini box.

    python3 levers_cal.py [--moetier /tmp/moetier]      (trace_p256.npy + the registry/ records here)

Calibration = registry/engine cpu lane (per_layer 3.5 ms handshake with SLIM=1 + spin0, per_expert 0.25 ms effective pool wall per distinct cold
expert) + recipe fixed_ms 19 ms at 3 tokens + draft 9 ms; 17 whole-resident layers, 8770 slots.  The sim then gives 5.68 tok/s against 5.4-5.6 measured
(F1: 334/376 ms per window at 1.81/2.12 accepted tokens).  The routing is still the P256 calibration trace with a q=0.2 non-kept tail (no full-512 trace
exists); prefill is NOT modelled here (measured link-bound: 9.5-12 tok/s)."""
import argparse, copy, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--moetier", default="/tmp/moetier"); a = ap.parse_args()
sys.path.insert(0, a.moetier)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
src = open("levers_strata.py").read().split("if a.q is None:")[0].replace("a = ap.parse_args()", "a = ap.parse_args(['trace_p256.npy','--q','0.2','--moetier','%s'])" % a.moetier)
exec(src)
tr = leak(0.2); order_f, f = freq_order(tr)
R0 = "qwen38fn-3090usb-mini16-full512"
BASE_SLOTS = 8770


def run(ov, ngl, placement="conc"):
    R = spec.resolve(reg, R0, **ov)
    if placement == "conc": SeedLedger.order, _ = conc_order(tr, f, ngl)
    else: SeedLedger.order = order_f
    return sim.run(R, [tr], conc=1, warm=60, max_steps=400, window=3, accept=1.95, draft_ms=9.0)


cum = {}
rows = []
def step(name, ov, ngl, measured="", placement="conc"):
    cum.update(ov)
    r = run(dict(cum), ngl, placement)
    rows.append((name, r))
    print(f"| {name} | {r['tok_s']} | {r['ms_per_step']} | {r['vram_hit']} | {r['nvme_per_tok']} | {measured} |", flush=True)

print("| lever (cumulative) | model tok/s c1 (MTP, 1.95 acc/window) | ms/window | VRAM hit | cold experts (RAM/page cache) per token | measured (3 prompts, tok/s) |")
print("|---|---|---|---|---|---|")
step("baseline = F1: 17 whole-resident layers, 8770 slots, 1.6 GiB locked RAM, 3.5 ms handshake, 0.25 ms/expert", {}, 17, "5.1 avg (4.4 / 5.6 / 5.4)")
step("+ slots 9108 (vram reserve 700->100 MiB) and RAM tier 4.3 GiB locked (headroom 4->1.5 GiB), 17 whole layers", {"budget.vram_expert_slots": 9108}, 17, "not run servable; E4 with prefill-borrow (9964 slots, 19 layers) 5.4 avg - borrow crashes on a 2.5K prefill")
step("+ frequency-ranked placement (every layer rings, 3.2-3.5 ms each; the hottest 9108 pairs resident)", {}, 17, "F5 (servable): 5.9 avg (4.9 / 7.4 / 5.5); E5 with borrow 6.1 avg", placement="freq")
step("+ RAM tier doubled again (12 GB for experts; not reachable on 16 GB with the engine resident)", {"budget.runtime_gb": 2.0, "budget.other_gb": 2.0}, 17, "not run", placement="freq")
step("+ Metal lane next to NEON (compute share 0.155->0.089 ms: 0.25 -> 0.184 effective)", {"lane_overrides": {"cpu": {"per_expert_ms": 0.184, "per_extra_token_ms": 0.016, "per_layer_ms": 3.5}}}, 17, "not built", placement="freq")
step("+ handshake hidden behind the previous layer (3.5 -> 0.5 ms/layer)", {"lane_overrides": {"cpu": {"per_expert_ms": 0.184, "per_extra_token_ms": 0.016, "per_layer_ms": 0.5}}}, 17, "not built (3.5 ms is the measured floor of the current protocol; USB DMA floor 1 ms)", placement="freq")
step("+ 4000 more VRAM slots (no 200K KV / vision, or a 2nd card)", {"budget.vram_expert_slots": 9108 + 4000}, 25, "not run", placement="freq")
