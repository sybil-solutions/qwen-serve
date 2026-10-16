# Concurrent decode benchmark against the local Strata server: N clients send different prompts at once (greedy,
# reasoning off, fixed max_tokens). Aggregate tok/s = all completion tokens / wall time of the round.
# usage: bench_conc.py N [max_tokens] [rounds]
import json, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8080/v1/chat/completions"
QS = ["Explain in detail how a B-tree works.", "Write a Python function that merges overlapping intervals, then explain it.",
      "Compare TCP and QUIC: handshake, congestion control, multiplexing.", "Tell the history of computing pi, from Archimedes to today.",
      "What were the main causes of the French Revolution?", "Explain how a transformer's attention works, step by step.",
      "Give an apple pie recipe, step by step.", "Describe the life cycle of a star like the Sun."]
n = int(sys.argv[1]); mt = int(sys.argv[2]) if len(sys.argv) > 2 else 200; rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1

def one(i, out):
  body = json.dumps({"model": "x", "messages": [{"role": "user", "content": QS[i % len(QS)]}], "max_tokens": mt,
                     "reasoning_effort": "none", "temperature": 0}).encode()
  t0 = time.time()
  try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600))
    out[i] = (r["usage"]["completion_tokens"], time.time() - t0, r["timings"].get("predicted_per_second"))
  except Exception as e:
    print(f"request {i} failed: {e}", flush=True)
    out[i] = (0, time.time() - t0, None)

for rd in range(rounds):
  out = [None] * n
  ths = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
  t0 = time.time()
  for t in ths: t.start()
  for t in ths: t.join()
  wall = time.time() - t0
  toks = sum(o[0] for o in out)
  print(f"c{n} round {rd}: {toks} tokens in {wall:.1f} s = {toks / wall:.1f} tok/s aggregate; "
        f"per-request engine tok/s {[round(o[2] or 0, 1) for o in out]}; latency s {[round(o[1], 1) for o in out]}", flush=True)
