import json, os, time, urllib.request
idx = json.load(open(os.path.expanduser("~/exl3/td/experts_index.json")))
REV = "69e33439ae950f17bcbe95c98f117d80f759ab6d"
BASE = f"https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/{REV}/"
# runs: coalesce by shard order
sp = sorted((v["down_suh"][0], min(x[1] for x in v.values()), max(x[2] for x in v.values()), k) for k, v in idx.items())
runs = []
for sh, a, b, k in sp:
    if runs and runs[-1][0] == sh and runs[-1][2] == a: runs[-1][2] = b; runs[-1][3] += 1
    else: runs.append([sh, a, b, 1])
print("runs", len(runs), "sizes MB", sorted(((r[2] - r[1]) >> 20, r[3]) for r in runs)[-5:], "min", sorted(((r[2] - r[1]) >> 20, r[3]) for r in runs)[:3])
sh, a, b, n = runs[0]
t = time.time(); got = 0
req = urllib.request.Request(BASE + sh, headers={"Range": f"bytes={a}-{a + (128 << 20) - 1}"})
with urllib.request.urlopen(req, timeout=120) as r:
    while True:
        c = r.read(1 << 20)
        if not c: break
        got += len(c)
print(f"{got / 1e6:.0f} MB in {time.time() - t:.1f} s = {got / 1e6 / (time.time() - t):.1f} MB/s")
