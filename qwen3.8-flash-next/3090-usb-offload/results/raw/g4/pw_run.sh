#!/bin/zsh
# pw_run.sh W... : for each W: restart on ~/shim/pw<W>.json, decode x2 (2 warm runs, right after the restart), prefill 3.8K x2, 8.2K x2, 17.6K -> ~/shim/pw_<W>.log
for w in "$@"; do
out=~/shim/pw_$w.log
{
echo "== W=$w $(date -u +%FT%TZ)"
~/shim/ls_srv.sh ~/shim/pw$w.json || { echo "SERVER FAILED"; continue; }
echo "-- decode run 1"
python3 ~/shim/split_probe.py PW${w}_a 300
echo "-- decode run 2"
python3 ~/shim/split_probe.py PW${w}_b 300
echo "-- prefill rows"
for c in 10000 10000 25000 25000 55000; do python3 ~/shim/ls_pf.py $c; done
echo "== done $(date -u +%FT%TZ)"
} > $out 2>&1
done
