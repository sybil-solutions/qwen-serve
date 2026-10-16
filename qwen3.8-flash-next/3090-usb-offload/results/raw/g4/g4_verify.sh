#!/bin/zsh
# g4_verify.sh NAME CFG : restart on CFG; decode x2 (right after the restart); prefill rows 3.8K x2, 7.9K x2, 8.2K x2, 17.6K; needle 33K; needle 100K -> ~/shim/g4v_NAME.log
name=$1; cfg=$2
{
echo "== $name $(date -u +%FT%TZ) cfg=$cfg"
~/shim/ls_srv.sh $cfg || { echo "SERVER FAILED"; exit 1; }
echo "-- decode run 1"; python3 ~/shim/split_probe.py G4V_${name}_a 300
echo "-- decode run 2"; python3 ~/shim/split_probe.py G4V_${name}_b 300
echo "-- prefill rows"
for c in 10000 10000 24000 24000 25000 25000 55000; do python3 ~/shim/ls_pf.py $c; done
echo "-- needle 32000"; python3 ~/shim/final_long.py needle 32000 50
echo "-- needle 100000"; python3 ~/shim/final_long.py needle 100000 25
echo "== done $(date -u +%FT%TZ)"
} > ~/shim/g4v_$name.log 2>&1
