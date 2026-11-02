# One chat request to the local Strata server; prints the text and the server's timings.
import json, sys, time, urllib.request
prompt, max_tokens = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 128
body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                   "reasoning_effort": "none", "temperature": 0}).encode()
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body,
                                                           {"Content-Type": "application/json"}), timeout=1800))
tm = r.get("timings", {})
print(json.dumps(r["choices"][0]["message"]["content"][:300]))
print(f"prompt {tm.get('prompt_n')} tok in {tm.get('prompt_ms', 0) / 1000:.1f} s ({tm.get('prompt_per_second')} tok/s) | "
      f"decode {tm.get('predicted_n')} tok in {tm.get('predicted_ms', 0) / 1000:.1f} s ({tm.get('predicted_per_second')} tok/s) | "
      f"drafts {tm.get('draft_n_accepted')}/{tm.get('draft_n')} | wall {time.time() - t:.1f} s")
