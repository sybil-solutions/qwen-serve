#!/usr/bin/env python3
"""make_placement512.py - whole-resident placement for the full 48x512 pack (no keep mask).

--out DIR --slots S [--n-gpu M | --gpu-layers l,l,...] [--base shipped 512 profile]
The profile ranks every pair of the GPU layers first, LAYER BY LAYER (a slot shortfall only truncates the last GPU layer, the
others stay 100% resident -> per-layer zero-doorbell graph), then the CPU layers' experts hottest first (the leftover slots), then the rest.
"""
import argparse, os, sys
sys.path.insert(0, os.path.expanduser("~/strata/tools"))
from make_profile import read_profile, write_profile
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--slots", type=int, required=True)
ap.add_argument("--n-gpu", type=int, default=-1)
ap.add_argument("--gpu-layers", default="")
ap.add_argument("--base", default=os.path.expanduser("~/strata/data/expert-profile.bin"))
a = ap.parse_args()
NL, NE = 48, 512
base = read_profile(a.base, NE)
assert len(base) == NL * NE
rank = {l: [e for ll, e in base if ll == l] for l in range(NL)}
assert all(len(rank[l]) == NE for l in range(NL))
if a.gpu_layers:
    gpu = sorted({int(x) for x in a.gpu_layers.split(",")})
else:
    m = a.n_gpu if a.n_gpu >= 0 else a.slots // NE
    step = NL / m
    gpu = sorted({min(NL - 1, int(step * i + step / 2)) for i in range(m)})
cpu = [l for l in range(NL) if l not in gpu]
ranked, seen = [], set()
def take(l, e):
    if (l, e) not in seen:
        seen.add((l, e)); ranked.append((l, e))
for l in gpu:
    for e in rank[l]: take(l, e)
for r in range(NE):
    for l in cpu: take(l, rank[l][r])
os.makedirs(a.out, exist_ok=True)
write_profile(os.path.join(a.out, "profile.bin"), ranked, NE)
assert read_profile(os.path.join(a.out, "profile.bin"), NE) == ranked
left = a.slots - len(gpu) * NE
print(f"gpu layers ({len(gpu)}): {gpu}\ncpu layers ({len(cpu)})\nslots {a.slots} -> leftover {left} hottest experts of the CPU layers; cold experts total {len(cpu)*NE - max(0,left)} of {NL*NE}")
