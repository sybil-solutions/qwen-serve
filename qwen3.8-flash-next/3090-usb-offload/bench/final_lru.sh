#!/bin/zsh
# final_lru.sh : restart on the final LRU config; decode x2 (right after the restart), prefill rows, 32K needle, vision -> ~/shim/final_lru.log
{
echo "== final_lru $(date -u +%FT%TZ)"
~/shim/ls_srv.sh ~/shim/strata-exl3-full512-final-g4-lru.json || { echo "SERVER FAILED"; exit 1; }
echo "-- mem at READY"; vm_stat | egrep "free|wired|File-backed|compressor"; ps -o rss= -p $(pgrep -x strata)
echo "-- decode run 1"; python3 ~/shim/split_probe.py FL_a 300
echo "-- decode run 2"; python3 ~/shim/split_probe.py FL_b 300
echo "-- mem after decode"; vm_stat | egrep "free|wired|File-backed|compressor|Pageouts"; ps -o rss= -p $(pgrep -x strata)
echo "-- prefill rows"
for c in 10000 10000 24000 24000 25000 55000; do python3 ~/shim/ls_pf.py $c; done
echo "-- needle 32000"; python3 ~/shim/final_long.py needle 32000 50
echo "-- vision"; python3 ~/shim/final_vision.py
echo "-- decode after prefill"; python3 ~/shim/split_probe.py FL_c 300
echo "-- mem after all"; vm_stat | egrep "free|wired|File-backed|compressor|Pageouts"
echo "== done $(date -u +%FT%TZ)"
} > ~/shim/final_lru.log 2>&1
