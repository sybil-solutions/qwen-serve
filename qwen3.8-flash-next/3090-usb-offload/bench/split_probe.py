#!/usr/bin/env python3
"""split_probe.py NAME [max_tokens] - greedy chat on the prime prompt + 2 others against :8080; saves outputs, prints timings.
Outputs: ~/shim/split_<NAME>_<i>.txt ; one JSON line per prompt on stdout (timings from the server)."""
import json, sys, time, urllib.request, os, subprocess
def vm():
    d={}
    for l in subprocess.run(['vm_stat'],capture_output=True,text=True).stdout.splitlines():
        if l.startswith('Pageins') or l.startswith('Pageouts'): d[l.split(':')[0]]=int(l.split(':')[1].strip().strip('.'))
    return d
URL = "http://127.0.0.1:8080/v1/chat/completions"
name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
PROMPTS = [
    "Write a Python function that checks whether a number is prime, then explain it briefly.",
    "Explain in three short paragraphs how a mixture-of-experts language model decides which experts to run for each token.",
    "Write a short story about a lighthouse keeper, about 200 words.",
]
for i, p in enumerate(PROMPTS):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": n,
                       "reasoning_effort": "none", "temperature": 0}).encode()
    t = time.time(); v0 = vm()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800))
    v1 = vm()
    text, tm = r["choices"][0]["message"]["content"], r["timings"]
    open(os.path.expanduser(f"~/shim/split_{name}_{i}.txt"), "w").write(text)
    print(json.dumps({"i": i, "wall_s": round(time.time() - t, 1), "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                      "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
                      "tok_s": round(tm.get("predicted_per_second", 0), 2), "drafts": f"{tm.get('draft_n_accepted')}/{tm.get('draft_n')}",
                      "pageins_mib": (v1["Pageins"]-v0["Pageins"])*16//1024, "pageouts_mib": (v1["Pageouts"]-v0["Pageouts"])*16//1024, "head": text[:90].replace("\n", " ")}), flush=True)
