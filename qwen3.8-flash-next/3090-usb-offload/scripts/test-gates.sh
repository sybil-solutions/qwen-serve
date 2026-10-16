#!/bin/bash
# Regression test for the reproducibility gates: a deliberately wrong hash (or a corrupted byte) must make them exit NONZERO, a right one zero.
# Card-free, offline, a few KiB of fixtures under a temp dir. Needs python3 with numpy (PYTHON=... to choose another).
#   scripts/test-gates.sh          exit 0 = every gate behaved
# Covers: exl3/pack_td.py verify (--sha-only: section sha256 vs the manifest) and compare (fresh manifest vs the committed one), the usage
# exit code, and scripts/verify-card-free.sh aborting nonzero when a step fails (a STRATA_DIR without the v0.1.41 tag).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); PYTHON=${PYTHON:-python3}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fails=0
expect() {   # expect <want: zero|nonzero> <label> <cmd...>
  local want=$1 label=$2; shift 2
  "$@" >"$T/out.txt" 2>&1; local rc=$?
  if { [ "$want" = zero ] && [ $rc -eq 0 ]; } || { [ "$want" = nonzero ] && [ $rc -ne 0 ]; }; then echo "PASS  $label (exit $rc)"
  else echo "FAIL  $label: wanted $want, got exit $rc"; sed 's/^/      /' "$T/out.txt" | tail -8; fails=$((fails + 1)); fi
}
# pack_td.py with a 2-layer, 1 KiB-per-layer pack in place of the real 45.8 GB one
driver() {   # driver <ok|wronghash|corrupt|compare-ok|compare-wrong|usage>
  "$PYTHON" - "$T" "$1" <<'PY'
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.environ["REPO"], "exl3"))
import pack_exl3 as px, pack_td as td
d, mode = sys.argv[1], sys.argv[2]
px.N_LAYERS, px.LAYER = 2, 1024
px.BIN, px.MANIFEST = os.path.join(d, "experts.bin"), os.path.join(d, "exl3_pack.json")
data = np.random.default_rng(1).integers(0, 256, 2 * px.LAYER, dtype=np.uint8).tobytes()
sha = lambda l: hashlib.sha256(data[l * px.LAYER:(l + 1) * px.LAYER]).hexdigest()
layers = {str(l): dict(sha256=sha(l), via="turboderp/fixture") for l in range(2)}
if mode == "wronghash": layers["1"]["sha256"] = "0" * 64
if mode == "corrupt": data = data[:5] + bytes([data[5] ^ 1]) + data[6:]
open(px.BIN, "wb").write(data)
json.dump(dict(layers=layers), open(px.MANIFEST, "w"))
if mode.startswith("compare"):
    fresh = dict(layers={str(l): dict(sha256=sha(l)) for l in range(2)})
    if mode == "compare-wrong": fresh["layers"]["0"]["sha256"] = "f" * 64
    json.dump(fresh, open(os.path.join(d, "fresh.json"), "w"))
    sys.exit(td.main(["compare", os.path.join(d, "fresh.json"), px.MANIFEST]))
if mode == "usage": sys.exit(td.main(["nonsense"]))
sys.exit(td.main(["verify", "--sha-only"]))
PY
}
export REPO
expect zero    "pack_td verify --sha-only, right hashes"                 driver ok
expect nonzero "pack_td verify --sha-only, deliberately wrong hash"      driver wronghash
grep -q "VERIFY FAILED" "$T/out.txt" || { echo "FAIL  wrong-hash run did not print VERIFY FAILED"; fails=$((fails + 1)); }
expect nonzero "pack_td verify --sha-only, corrupted byte"               driver corrupt
expect zero    "pack_td compare, equal manifests"                        driver compare-ok
expect nonzero "pack_td compare, one layer hash differs"                 driver compare-wrong
expect nonzero "pack_td unknown command"                                 driver usage
# verify-card-free.sh must abort nonzero on a failing step (set -euo pipefail), not print "done" after it
git init -q "$T/fakestrata" && (cd "$T/fakestrata" && git -c user.email=t@t -c user.name=t commit -q --allow-empty -m x)   # a repo without the v0.1.41 tag
expect nonzero "verify-card-free.sh patches with a STRATA_DIR lacking the tag" env WORK="$T/work" STRATA_DIR="$T/fakestrata" zsh "$REPO/scripts/verify-card-free.sh" patches
if grep -q "^done:" "$T/out.txt"; then echo "FAIL  verify-card-free.sh printed 'done' after a failing step"; fails=$((fails + 1)); fi
[ $fails -eq 0 ] && echo "ALL GATES OK" || { echo "$fails GATE CHECK(S) FAILED"; exit 1; }
