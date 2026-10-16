#!/bin/zsh
# runsplit.sh NAME CFG [max_tokens] : restart the server on CFG (env passes through), run split_probe.py (3 prompts), log engine lines.
name=$1; cfg=$2; n=${3:-300}
log=~/shim/split_$name.log
{ echo "== $name $(date -u +%FT%TZ) cfg=$cfg"; env | grep -E '^(STRATA_|TGCU)' | sort; } > $log
start=$(( $(wc -l < ~/shim/serve-engine.log) + 1 ))
t0=$(date +%s)
~/shim/srv.sh $cfg >> $log 2>&1 || { echo "SRV FAILED" >> $log; exit 1; }
echo "ready after $(( $(date +%s) - t0 )) s" >> $log
python3 ~/shim/split_probe.py $name $n >> $log 2>&1
if [ -n "$ROUND2" ]; then python3 ~/shim/split_probe.py ${name}_r2 $n >> $log 2>&1; fi
echo "-- engine log" >> $log
tail -n +$start ~/shim/serve-engine.log | grep -v 'tgcu prof' | cut -c1-600 >> $log
echo DONE >> $log
