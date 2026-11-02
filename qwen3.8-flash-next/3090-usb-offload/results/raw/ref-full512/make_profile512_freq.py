#!/usr/bin/env python3
"""make_profile512_freq.py OUT - frequency-ranked 512-expert profile: the P256 calibration ranking (router index -> original expert id via the
pack_p256 keep lists) first, then every other pair.  Every layer gets its hottest experts (no whole-resident layers)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/strata/tools"))
from make_profile import read_profile, write_profile
base = read_profile(os.path.expanduser("~/models/qwen38fn/q2p256/expert-profile-p256.bin"), 256)
keep = {}
for line in open(os.path.expanduser("~/exl3/pack_p256/exl3_experts.txt")):
    if line.startswith("keep "):
        p = line.split(); keep[int(p[1])] = [int(x) for x in p[2:]]
assert len(keep) == 48 and all(len(v) == 256 for v in keep.values())
ranked, seen = [], set()
for l, i in base:
    k = (l, keep[l][i])
    if k not in seen: seen.add(k); ranked.append(k)
print("p256 pairs", len(ranked))
for e in range(512):
    for l in range(48):
        if (l, e) not in seen: seen.add((l, e)); ranked.append((l, e))
assert len(ranked) == 48 * 512
os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)
write_profile(sys.argv[1], ranked, 512)
assert read_profile(sys.argv[1], 512) == ranked
print("wrote", sys.argv[1], len(ranked))
