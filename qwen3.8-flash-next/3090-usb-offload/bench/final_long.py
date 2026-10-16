#!/usr/bin/env python3
"""final_long.py MODE ARGS: prefill rows / needle at length. 
  pf CHARS...            uncached prefill (unique first line), max_tokens 2
  needle TOKENS DEPTH    haystack of TOKENS tokens (3.2 chars/token), code word at DEPTH %, answer + ~100-word summary (decode speed)"""
import json, sys, time, random, urllib.request, os
sys.path.insert(0, os.path.expanduser("~/strata/tools"))
import needle_bench as nb
URL = "http://127.0.0.1:8080/v1/chat/completions"
def ask(prompt, mt):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "reasoning_effort": "none", "temperature": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=7200))
    return r, time.time() - t
mode = sys.argv[1]
if mode == "pf":
    doc = nb.haystack(60000)
    for c in sys.argv[2:]:
        tag = f"Session {random.randrange(10**9)}. Here are some notes. Reply with OK only.\n\n"
        r, w = ask(tag + doc[:int(c)], 2); tm = r["timings"]
        print(json.dumps({"chars": int(c), "prompt_n": tm["prompt_n"], "prompt_ms": tm["prompt_ms"], "tok_s": round(tm["prompt_n"] / tm["prompt_ms"] * 1000, 1), "wall_s": round(w, 1)}), flush=True)
else:
    tokens, depth = int(sys.argv[2]), int(sys.argv[3])
    rnd = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 7)
    text = nb.haystack(int(tokens * nb.CHARS_PER_TOKEN))
    word = f"{rnd.choice(nb.WORDS)}-{rnd.choice(nb.WORDS)}-{rnd.randint(100, 999)}"
    cut = int(len(text) * depth / 100); cut = text.rfind("\n", 0, cut) + 1 or cut
    needle = f"\nThe secret code word for this text is: {word}. Remember it.\n"
    prompt = (f"Session {random.randrange(10**9)}.\n" + text[:cut] + needle + text[cut:] + "\n\nWhat is the secret code word mentioned in the text above? Put the code word alone on the first line, then on the following lines write a summary of about 100 words of what kind of text this is.")
    r, w = ask(prompt, 160); tm = r["timings"]; ans = r["choices"][0]["message"]["content"] or ""
    print(json.dumps({"tokens": tokens, "depth": depth, "word": word, "found": word in ans, "first_line_ok": word in ans.strip().split("\n")[0],
                      "prompt_n": tm["prompt_n"], "cache_n": tm.get("cache_n"), "prompt_ms": tm["prompt_ms"], "prefill_tok_s": round(tm["prompt_n"] / tm["prompt_ms"] * 1000, 1),
                      "ttft_s": round(tm["prompt_ms"] / 1000, 1), "predicted_n": tm["predicted_n"], "decode_tok_s": round(tm.get("predicted_per_second", 0), 2),
                      "drafts": f"{tm.get('draft_n_accepted')}/{tm.get('draft_n')}", "wall_s": round(w, 1), "answer": ans.strip()[:160]}), flush=True)
