#!/bin/zsh
# srv.sh <config.json> : stop the running Strata server (wait for the engine to exit), start one with <config>, wait for
# "ready" (max 600 s), print the expert-cache / VRAM / batch lines of the new engine. Exit 1 if it dies or times out.
cfg=$1; log=~/shim/serve_q2.log
if [ -f ~/shim/server.pid ]; then kill -TERM $(cat ~/shim/server.pid) 2>/dev/null; fi
for i in {1..60}; do pgrep -f "strata-mac/strata" >/dev/null || break; sleep 1; done
pkill -KILL -f "strata-mac/strata" 2>/dev/null; pkill -KILL -f "serve/server.py" 2>/dev/null; sleep 1
start=$(( $(wc -l < ~/shim/serve-engine.log) + 1 ))
cd ~/strata && STRATA_PREFILL_CPU_SHARE=1 nohup .venv/bin/python serve/server.py --engine strata --config $cfg --port 8080 \
  > $log 2>&1 < /dev/null &
echo $! > ~/shim/server.pid
for i in {1..120}; do
  grep -q "^ready" $log && break
  kill -0 $(cat ~/shim/server.pid) 2>/dev/null || { echo DIED; tail -20 $log; exit 1; }
  sleep 5
done
grep -q "^ready" $log || { echo TIMEOUT; tail -20 $log; exit 1; }
tail -n +$start ~/shim/serve-engine.log | grep -i -E "expert cache auto|VRAM free with|batch|kv |context|ERR" | grep -v "reading the prompt" | tail -12
echo READY
