"""levers_strata.py - moetier levers table for Qwen3.8-Flash-Next EXL3 3.05bpw (all 512 experts) on the 3090-over-USB + Mac mini M4 box.

    python3 levers_strata.py <trace.npy> [--q 0.0..1] [--moetier /tmp/moetier]

Trace = calibration capture of the P256 model (every pick is one of the 256 kept experts). To emulate the full-512 router, each pick is
replaced with probability q by a uniformly random NON-kept expert of the same layer; q is calibrated so that frequency seeding at 36% of
pairs gives the 70-72% VRAM hit measured on the full IQ3_S 512 model (EXL3_PLAN.md). Placement 'concentrated' = whole layers resident
(zero-doorbell graph), 'freq' = global frequency ranking (every layer rings)."""
import argparse, json, os, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--q", type=float, default=None); ap.add_argument("--moetier", default="/tmp/moetier")
a = ap.parse_args()
sys.path.insert(0, a.moetier)
from moetier import spec, sim, ledger

HERE = os.path.dirname(os.path.abspath(__file__))
reg = spec.load(os.path.join(HERE, "registry"))
tr0 = np.load(a.trace).astype(np.int32)
L, E = 48, 512
kept = [np.unique(tr0[:, l, :]) for l in range(L)]
rng = np.random.default_rng(0)


def leak(q):
    tr = tr0.copy()
    if q <= 0:
        return tr
    for l in range(L):
        non = np.setdiff1d(np.arange(E), kept[l])
        m = rng.random(tr[:, l, :].shape) < q
        tr[:, l, :][m] = rng.choice(non, int(m.sum()))
    return tr


def freq_order(tr):
    f = np.zeros(L * E)
    for l in range(L):
        np.add.at(f, l * E + tr[:, l, :].reshape(-1), 1)
    return [int(k) for k in np.argsort(-f, kind="stable")], f


class SeedLedger(ledger.Ledger):
    order = None
    def seed(self, ranked):
        super().seed(self.order if self.order is not None else ranked)


sim.Ledger = SeedLedger


def conc_order(tr, f, n_gpu_layers):
    """whole layers resident, spaced evenly (no layer 0), then the rest by frequency"""
    step = L / max(1, n_gpu_layers)
    gl = sorted({min(L - 1, int(step * i + step / 2)) for i in range(n_gpu_layers)})
    first = [l * E + e for l in gl for e in np.argsort(-f[l * E:(l + 1) * E], kind="stable")]
    rest = [int(k) for k in np.argsort(-f, kind="stable") if int(k) // E not in gl]
    return first + rest, gl


def hit_at_36pct(tr):
    order, f = freq_order(tr)
    n = int(0.36 * L * E)
    top = set(order[:n])
    cnt = sum(1 for l in range(L) for e in tr[:, l, :].reshape(-1)[::7] if l * E + int(e) in top)
    return cnt / (L * tr.shape[0] * tr.shape[2] / 7)


if a.q is None:
    best = None
    for q in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        h = hit_at_36pct(leak(q))
        print(f"q={q}: freq-seeded VRAM hit at 36% of pairs = {h:.3f}", flush=True)
        if best is None or abs(h - 0.71) < abs(best[1] - 0.71): best = (q, h)
    a.q = best[0]
    print("calibrated q =", a.q)
tr = leak(a.q)
order_f, f = freq_order(tr)
streams = [tr]
R0 = "qwen38fn-3090usb-mini16-full512"
slots = reg["recipe"][R0]["budget"]["vram_expert_slots"]
ngl = slots // E


def run(ov, placement, label, window=3, accept=2.7, steps=400, draft_ms=9.0):
    R = spec.resolve(reg, R0, **ov)
    if placement == "conc":
        SeedLedger.order, gl = conc_order(tr, f, ov.get("_ngl", ngl))
    else:
        SeedLedger.order = order_f
    r = sim.run(R, streams, conc=1, warm=60, max_steps=steps, window=window, accept=accept, draft_ms=draft_ms)
    return r


rows = []
acc = {}
def step(name, ov, placement):
    acc.update(ov)
    o = {k: v for k, v in acc.items() if not k.startswith("_")}
    o2 = dict(o); o2["_ngl"] = acc.get("_ngl", ngl)
    r = run(o2, placement, name)
    rows.append((name, r))
    print(f"| {name} | {placement} | {r['tok_s']} | {r['vram_hit']} | {r['ms_per_step']} | {r['ms_fixed']} + max(gpu {r['ms_gpu_moe']}, cpu {r['ms_cpu_moe']}) | {r['nvme_per_tok']} |", flush=True)


print("\n| lever (cumulative) | placement | decode tok/s c1 (MTP window 3, 2.7 accepted) | VRAM hit | ms/step | fixed + max(gpu, cpu) | NVMe/token |")
print("|---|---|---|---|---|---|---|")
step("baseline: freq-ranked 9296 slots, every layer rings (handshake 6.5 ms, NEON 0.155)", {}, "freq")
step("concentrated: 18 whole layers resident (zero-doorbell), 30 CPU layers", {}, "conc")
step("+ Metal lane in parallel with NEON (0.089 ms/expert)", {"lane_overrides": {"cpu": {"per_expert_ms": 0.089, "per_extra_token_ms": 0.016}}}, "conc")
step("+ one-roundtrip handshake (6.5 -> 2.5 ms/layer)", {"lane_overrides": {"cpu": {"per_expert_ms": 0.089, "per_extra_token_ms": 0.016, "per_layer_ms": 2.5}}}, "conc")
step("+ layer-ahead start of CPU experts (handshake hidden: 0.5 ms/layer)", {"lane_overrides": {"cpu": {"per_expert_ms": 0.089, "per_extra_token_ms": 0.016, "per_layer_ms": 0.5}}}, "conc")
step("+ mini RAM tier doubled (12 GB for experts, NVMe misses halve)", {"budget.runtime_gb": 2.0, "budget.other_gb": 2.0}, "conc")
step("+ 4000 more VRAM slots (no 200K KV / vision; or 2 cards)", {"budget.vram_expert_slots": slots + 4000, "_ngl": (slots + 4000) // E}, "conc")
print("\nR9-style fit check (P256 stand-in)  see EXL3_PLAN.md")
