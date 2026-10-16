#!/bin/zsh
name=$1; cfg=$2
{
echo "== $name $(date -u +%FT%TZ) cfg=$cfg"
~/shim/ls_srv.sh $cfg || { echo "SERVER FAILED"; exit 1; }
echo "-- decode run 1"; python3 ~/shim/split_probe.py DC_${name}_a 300
echo "-- decode run 2"; python3 ~/shim/split_probe.py DC_${name}_b 300
echo "-- prefill rows"
for c in 10000 10000 24000 24000 25000 25000 55000; do python3 ~/shim/ls_pf.py $c; done
echo "== done $(date -u +%FT%TZ)"
} > ~/shim/dc_$name.log 2>&1
