#!/bin/zsh
# ht_ex2.sh CFG TAG : the test config only: exact texts vs the ref's, timing, prof -> ~/shim/ht_ex2_TAG.log
cfg=$1; tag=$2
pl=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['env'].get('TGCU_PROF_OUT','/dev/null'))" $cfg)
{
echo "== ht_ex2 $tag $(date -u +%FT%TZ)"
~/shim/ls_srv.sh $cfg || { echo "SERVER FAILED"; exit 1; }
python3 ~/shim/ht_exact.py ~/shim/hx_${tag}_exact8k.txt 24000
python3 ~/shim/ht_exact.py ~/shim/hx_${tag}_exact4k.txt 10000
sleep 8; cp $pl ~/shim/hx_${tag}_prof0.txt
python3 ~/shim/ls_pf.py 24000 24000
sleep 8; cp $pl ~/shim/hx_${tag}_prof1.txt
for f in exact8k exact4k; do cmp ~/shim/hx_ref_$f.txt ~/shim/hx_${tag}_$f.txt && echo "$f IDENTICAL"; done
echo "== done $(date -u +%FT%TZ)"
} > ~/shim/ht_ex2_$tag.log 2>&1
