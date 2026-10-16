#!/usr/bin/env python3
"""pf_probe.py NAME chars,chars,... [max_tokens] - prefill probe against :8080 (docs-based prompts, unique first line)."""
import json, sys, time, urllib.request, glob, os, random, subprocess
URL = "http://127.0.0.1:8080/v1/chat/completions"
name = sys.argv[1]; sizes = [int(x) for x in sys.argv[2].split(",")]; ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 4
fs = sorted(glob.glob(os.path.expanduser("~/strata/docs/*.md")), key=os.path.getsize, reverse=True)
doc = "".join(open(f).read() for f in fs[:3])
def pageins():
    for l in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines():
        if l.startswith("Pageins"): return int(l.split()[1].strip("."))
for chars in sizes:
    tag = f"Session {random.randrange(10**9)}. Here are some notes. Reply with OK only.\n\n"
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": tag + doc[:chars]}], "max_tokens": ntok,
                       "reasoning_effort": "none", "temperature": 0}).encode()
    p0 = pageins(); t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600))
    tm = r["timings"]; p1 = pageins()
    print(json.dumps({"chars": chars, "wall_s": round(time.time() - t, 1), "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                      "prompt_tps": round(tm.get("prompt_per_second", 0), 1), "pageins_16k": p1 - p0,
                      "text": r["choices"][0]["message"]["content"][:20]}), flush=True)
