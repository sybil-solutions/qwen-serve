# Prefill and decode benchmark against the local server. Each prefill prompt gets a unique first line so no cache is
# reused; decode runs greedy with reasoning off.
import json, sys, time, urllib.request, glob, os, random
URL = "http://127.0.0.1:8080/v1/chat/completions"
def req(msgs, n=None):
  body = json.dumps({"model": "x", "messages": msgs, "reasoning_effort": "none", "temperature": 0}).encode()
  r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800))
  return r["choices"][0]["message"]["content"], r["timings"]
fs = sorted(glob.glob(os.path.expanduser("~/strata/docs/*.md")), key=os.path.getsize, reverse=True)
doc = "".join(open(f).read() for f in fs[:3])
rows = []
for chars in (6000, 12000, 18000, 24000):
  for rep in range(2):
    tag = f"Session {random.randrange(10**9)}. Here are some notes. Reply with OK only.\n\n"
    text, tm = req([{"role": "user", "content": tag + doc[:chars]}], 4)
    rows.append(("prefill", tm["prompt_n"], tm["prompt_ms"], tm["prompt_per_second"], text.strip()[:20]))
for prompt, n in (("Explain in three short paragraphs how a mixture-of-experts language model decides which experts to run for each token.", 256),
                  ("Write a Python function that checks whether a number is prime, then explain it briefly.", 256),
                  ("Write a short story about a lighthouse keeper, about 200 words.", 300)):
  text, tm = req([{"role": "user", "content": prompt}], n)
  rows.append(("decode", tm["predicted_n"], tm["predicted_ms"], tm["predicted_per_second"], f"drafts {tm.get('draft_n_accepted')}/{tm.get('draft_n')}"))
print(f"{'kind':8s} {'tokens':>7s} {'ms':>8s} {'tok/s':>8s}  note")
for k, n, ms, tps, note in rows: print(f"{k:8s} {n:7d} {ms:8.0f} {tps:8.1f}  {note!r}")
