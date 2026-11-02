#!/usr/bin/env python3
"""make_placement_w.py OUT W [slots] - W whole-resident layers (highest cold-traffic cost first, order below) at the head of the profile,
then every other pair in the shipped frequency order (f512c).  A slot shortfall truncates the LAST whole layer only (its least frequent experts)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/strata/tools"))
from make_profile import read_profile, write_profile
# cold pairs per chunk-layer, mean of 34 chunk-layers of the 33K needle prompt (G=4 run, pf-mac stats): highest first
ORDER = [29, 32, 30, 31, 27, 28, 26, 33, 22, 23, 34, 35, 25, 24, 41, 40, 39, 16, 19, 18, 17, 13]
out, W = sys.argv[1], int(sys.argv[2])
slots = int(sys.argv[3]) if len(sys.argv) > 3 else 7640
base = read_profile(os.path.expanduser("~/exl3/placement/f512c/profile.bin"), 512)
assert len(base) == 48 * 512
whole = ORDER[:W]
ranked = []
for l in whole:
    ranked += [(ll, e) for ll, e in base if ll == l]
ranked += [(ll, e) for ll, e in base if ll not in whole]
assert len(ranked) == 48 * 512 and len(set(ranked)) == len(ranked)
os.makedirs(out, exist_ok=True)
write_profile(os.path.join(out, "profile.bin"), ranked, 512)
assert read_profile(os.path.join(out, "profile.bin"), 512) == ranked
full = min(W, slots // 512)
print(f"W={W} whole layers {whole}; slots {slots}: {full} fully resident, last {slots - full * 512 if W > full else 0} of the next; leftover for freq {max(0, slots - W * 512)}")
