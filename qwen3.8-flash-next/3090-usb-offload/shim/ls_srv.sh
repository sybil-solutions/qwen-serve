#!/bin/zsh
# ls_srv.sh <config.json> : lockstep test launcher (srv.sh's twin): stops the running server + engine (either binary), starts <config>, waits for ready.
cfg=$1; log=~/shim/serve_ls.log
elog=$(python3 -c "import json,sys,os; print(json.load(open('$cfg')).get('log') or '')")
[ -n "$elog" ] || elog=~/shim/serve-engine.log
if [ -f ~/shim/server.pid ]; then kill -TERM $(cat ~/shim/server.pid) 2>/dev/null; fi
for i in {1..90}; do pgrep -x strata >/dev/null || break; sleep 1; done
pgrep -x strata >/dev/null && { echo "engine still alive after 90 s"; exit 2; }
sleep 1
start=$(( $(wc -l < $elog 2>/dev/null || echo 0) + 1 ))
cd ~/strata && STRATA_PREFILL_CPU_SHARE=1 nohup .venv/bin/python serve/server.py --engine strata --config $cfg --port 8080 > $log 2>&1 < /dev/null &
echo $! > ~/shim/server.pid
for i in {1..120}; do
  grep -q "^ready" $log && break
  kill -0 $(cat ~/shim/server.pid) 2>/dev/null || { echo DIED; tail -20 $log; exit 1; }
  sleep 5
done
grep -q "^ready" $log || { echo TIMEOUT; tail -20 $log; exit 1; }
tail -n +$start $elog | grep -i -E "expert cache auto|VRAM free with|expert_slots|slots|kv |context|ERR|prompt path|pf-mac: 2 sets" | grep -v "reading the prompt" | tail -14
echo READY
