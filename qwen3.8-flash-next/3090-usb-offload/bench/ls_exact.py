# ls_exact.py NAME : 48 greedy tokens on 2 multi-chunk prompts (~4.7K and ~7K tokens: 3 and 4 chunks of 2048) -> ~/shim/lsx_NAME.json
import json, sys, urllib.request, glob, os, time
name = sys.argv[1]
fs = sorted(glob.glob(os.path.expanduser("~/strata/docs/*.md")), key=os.path.getsize, reverse=True)
docs = [open(f).read() for f in fs[:3]]
prompts = [("p1", docs[0][:16000]), ("p2", docs[1][1000:25000])]
out = {}
for k, d in prompts:
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": "Read this document, then reply with a one sentence summary and its first heading.\n\n" + d}],
                       "max_tokens": 48, "reasoning_effort": "none", "temperature": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=3600))
    out[k] = {"prompt_n": r["timings"]["prompt_n"], "prompt_ms": r["timings"]["prompt_ms"], "text": r["choices"][0]["message"]["content"], "wall": round(time.time() - t, 1)}
    print(k, out[k]["prompt_n"], round(out[k]["prompt_ms"]), "ms", repr(out[k]["text"][:200]), flush=True)
json.dump(out, open(os.path.expanduser(f"~/shim/lsx_{name}.json"), "w"), indent=1)
