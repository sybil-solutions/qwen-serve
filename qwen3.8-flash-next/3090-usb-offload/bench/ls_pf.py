# ls_pf.py CHARS... : one uncached prefill per size (unique first line), prints prompt_n, ms, tok/s
import json, sys, urllib.request, glob, os, random, time
fs = sorted(glob.glob(os.path.expanduser("~/strata/docs/*.md")), key=os.path.getsize, reverse=True)
doc = "".join(open(f).read() for f in fs[:4])
for c in sys.argv[1:]:
    tag = f"Session {random.randrange(10**9)}. Here are some notes. Reply with OK only.\n\n"
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": tag + doc[:int(c)]}], "max_tokens": 2,
                       "reasoning_effort": "none", "temperature": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=3600))
    tm = r["timings"]
    print(c, "chars", tm["prompt_n"], "tokens", round(tm["prompt_ms"]), "ms", round(tm["prompt_n"] / (tm["prompt_ms"] / 1000), 1), "tok/s", "reply", repr(r["choices"][0]["message"]["content"][:20]), "wall", round(time.time() - t, 1), flush=True)
