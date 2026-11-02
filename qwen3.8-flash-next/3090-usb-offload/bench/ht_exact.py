#!/usr/bin/env python3
# ht_exact.py OUT CHARS : one fixed (deterministic) long prompt, greedy 24 tokens -> OUT (text) + prints prompt tokens, tok/s, md5
import json, sys, urllib.request, glob, os, time, hashlib
out, c = sys.argv[1], int(sys.argv[2])
fs = sorted(glob.glob("/Users/sero/strata/docs/*.md"), key=os.path.getsize, reverse=True)
doc = "".join(open(f).read() for f in fs[:4])
body = json.dumps({"model": "x", "messages": [{"role": "user", "content": "Fixed session 7. Read the notes and name their main topic in one short sentence.\n\n" + doc[:c]}],
                   "max_tokens": 24, "reasoning_effort": "none", "temperature": 0}).encode()
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=3600))
tm = r["timings"]; text = r["choices"][0]["message"]["content"]
open(out, "w").write(text)
print(c, "chars", tm["prompt_n"], "tokens", round(tm["prompt_ms"]), "ms", round(tm["prompt_n"] / (tm["prompt_ms"] / 1000), 1), "tok/s md5", hashlib.md5(text.encode()).hexdigest()[:10], repr(text[:60]), flush=True)
