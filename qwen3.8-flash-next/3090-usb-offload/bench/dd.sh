#!/bin/zsh
# dd.sh NAME CFG : restart on CFG, decode probe x2, timing + lane stats + memory snapshot -> ~/shim/dd_NAME.log
name=$1; cfg=$2
elog=$(python3 -c "import json; print(json.load(open('$cfg')).get('log') or '')")
{
echo "== $name $(date -u +%FT%TZ) cfg=$cfg"
start=$(( $(wc -l < $elog) + 1 ))
~/shim/ls_srv.sh $cfg || { echo "SERVER FAILED"; exit 1; }
echo "-- mem at READY"; vm_stat | egrep "free|wired|File-backed|compressor"; ps -o rss= -p $(pgrep -x strata)
echo "-- decode run 1"; python3 ~/shim/split_probe.py DD_${name}_a 300
echo "-- decode run 2"; python3 ~/shim/split_probe.py DD_${name}_b 300
echo "-- mem after"; vm_stat | egrep "free|wired|File-backed|compressor|Pageouts"; ps -o rss= -p $(pgrep -x strata)
echo "-- engine log"; tail -n +$start $elog | grep -a -E "decode timing|exl3-metal: [0-9]+ calls|drafts accepted|expert cache auto|tier|pf-mac: [0-9] sets" | cut -c1-420 | tail -30
echo "== done $(date -u +%FT%TZ)"
} > ~/shim/dd_$name.log 2>&1
